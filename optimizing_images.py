import os
import sys
import argparse
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import dino_utils as utils
import vision_transformer as vits


def max_accuracy(responses, labels):
    """
    max one vs rest accuracy over all classes

    Returns:
    --------
    max accuracy, index of max accuracy class
    """
    accs = []
    num_labels = np.amax(labels) + 1
    for l in range(num_labels):
        one = responses[labels == l]
        rest = responses[labels != l]
        accs.append(one_vs_rest_accuracy(one, rest))

    return np.amax(accs), np.argmax(accs)

def one_vs_rest_accuracy(one, rest):
    """
    compute one vs rest accuracy: probability a randomly selected pair will be correctly classified
    
    Returns:
    --------
    accuracy of discriminating one vs rest
    """
    accs = []
    for o in one:
        accs.append(np.mean(o > rest))
    m = np.mean(accs)    
    acc = max(m, 1.-m)
    return acc

def find_optimizing_imgs(args, feature_idxs):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)

    model.cuda()
    model.eval()

    # load weights to evaluate
    #utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")

    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_dataset = ImageFolder(args.val_data_path, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print('Data loaded: dataset contains {} images, and takes {} training iterations per epoch.'.format(len(val_dataset), len(val_loader)))

    print(val_dataset.classes)

    # forward prop all images, return last layer att. activations + labels
    preds, labels = forward_imgs(val_loader, model, args)

    # # compute best one vs rest classification accuracy for each feature
    # n_features = preds.shape[-1]
    # max_accs, max_labels = [], []
    # for f in range(n_features):
    #     resp = preds[:, f]
    #     max_acc, max_label = max_accuracy(resp, labels)
    #     max_accs.append(max_acc)
    #     max_labels.append(max_label)
    # print('Mean (std) max accs:', np.mean(max_accs), np.std(max_accs))

    # save preds + labels + max_accs + max_labels for further analysis
    save_path = os.path.join(args.output_dir, args.save_prefix + '_preds_labels.npz')
    np.savez(save_path, preds=preds, labels=labels)
    # np.savez(save_path, preds=preds, labels=labels, max_accs=max_accs, max_labels=max_labels)

    sorted_pred_idx = np.argsort(preds, axis=0)

    # plot optimizing images for some individual features
    for f_idx in feature_idxs:
        worst_idx = sorted_pred_idx[:args.top_k_imgs, f_idx]  
        best_idx = sorted_pred_idx[-args.top_k_imgs:, f_idx]  

        worst_imgs = []
        for i in worst_idx:
            worst_imgs.append(val_dataset[i][0].unsqueeze(0))

        best_imgs = []
        for i in best_idx[::-1]:
            best_imgs.append(val_dataset[i][0].unsqueeze(0))

        worst_imgs = torch.cat(worst_imgs, 0)
        best_imgs = torch.cat(best_imgs, 0)

        # save images
        worst_path = os.path.join(args.output_dir, args.save_prefix + '_fidx_' + str(f_idx) + '_worst_imgs.pdf')
        best_path = os.path.join(args.output_dir, args.save_prefix + '_fidx_' + str(f_idx) + '_best_imgs.pdf')

        save_image(worst_imgs, worst_path, nrow=int(np.sqrt(args.top_k_imgs)), normalize=True)
        save_image(best_imgs, best_path, nrow=int(np.sqrt(args.top_k_imgs)), normalize=True)

@torch.no_grad()
def forward_imgs(val_loader, model, args):

    preds = []
    labels = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    for inp, target in metric_logger.log_every(val_loader, len(val_loader) // 1, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            else:
                # TODO: handle this part separately for resnext models.
                output = model(inp)

        preds.append(output)
        labels.append(target)

    preds = torch.cat(preds, 0)
    labels = torch.cat(labels, 0)

    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    print('Preds shape:', preds.shape)
    print('Labels shape:', labels.shape)

    return preds, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Find maximizing images given a dataset and feature index')
    parser.add_argument('--arch', default='vit_base', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--batch_size', default=1024, type=int, help='Total batch size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--top_k_imgs', default=100, type=int, help='Number of optimizing images to show.')

    # dataset arguments
    parser.add_argument('--val_data_path', default='', type=str)

    # misc
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument("--save_prefix", default="", type=str, help="""prefix for saving checkpoint and log files""")

    args = parser.parse_args()
    print(args)

    feature_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    find_optimizing_imgs(args, feature_idxs)