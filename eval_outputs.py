import os
import sys
import argparse
import numpy as np

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import dino_utils as utils
import vision_transformer as vits

def compute_outputs(args):
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_small, vit_base, vit_large)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
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
    if not args.save_prefix.startswith("random"): 
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
        print(f"Model {args.arch} built. Loaded checkpoint at {args.pretrained_weights}.")
    else:
        print(f"Model {args.arch} built. Using random (untrained) weights.")

    # ============ preparing data ... ============
    # data transforms
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if args.split:
        from torch.utils.data.sampler import SubsetRandomSampler

        val_dataset = ImageFolder(args.val_data_path, transform=val_transform)

        num_data = len(val_dataset)
        print('Total data size is', num_data)

        indices = list(range(num_data))
        np.random.shuffle(indices)

        if args.subsample:
            num_subsampled = int(0.1 * num_data)
            test_idx = indices[:(num_subsampled // 2)]
        else:
            split = int(np.floor(0.5 * num_data))  # split 50-50, change here if you need to do sth else
            test_idx = indices[:split]

        test_sampler = SubsetRandomSampler(test_idx)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)

        print(f"Data loaded with {len(test_idx)} imgs.")
        print(f"{len(val_loader)} iterations to go thru the whole data.")
    else:
        val_dataset = ImageFolder(args.val_data_path, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        print(f"Data loaded with {len(val_dataset)} imgs.")
        print(f"{len(val_loader)} iterations to go thru the whole data.")

    print('Class names:', val_dataset.classes)

    # ============ run fwd prop ============
    fwd_prop(val_loader, model, val_dataset.classes, args)


@torch.no_grad()
def fwd_prop(val_loader, model, class_names, args):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    labels = []
    outputs = []

    task = os.path.split(args.output_dir)[-1]
    if  task == 'places365':
        places365_val_labels = torch.from_numpy(np.load('places365_val_labels.npz')['labels'])
        it = 0

    for inp, target in metric_logger.log_every(val_loader, len(val_loader) // 1, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        if task== 'places365':
            target = places365_val_labels[it*target.size(0):(it+1)*target.size(0)]
            target = target.cuda(non_blocking=True)
            it += 1
        else:
            target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = model(inp)

        labels.append(target)
        outputs.append(output)
    
    # save trial by trial accuracy
    labels = torch.cat(labels, 0)
    outputs = torch.cat(outputs, 0)
    labels = labels.cpu().numpy()
    outputs = outputs.cpu().numpy()
    print('Labels shape:', labels.shape)
    print('Outputs shape, min, max:', outputs.shape, outputs.min(), outputs.max())
    np.savez(os.path.join(args.output_dir, args.save_prefix + "_outputs.npz"), labels=labels, outputs=outputs, class_names=class_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('compute embeddings')
    parser.add_argument('--arch', default='vit_large', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--batch_size', default=1024, type=int, help='total batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # dataset arguments
    parser.add_argument('--val_data_path', default='', type=str)
    parser.add_argument('--split', default=False, action='store_true', help='whether to manually split dataset into train-val')
    parser.add_argument('--subsample', default=False, action='store_true', help='whether to subsample the data')

    # misc
    parser.add_argument('--num_workers', default=1, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument("--save_prefix", default="", type=str, help="""prefix for saving checkpoint and log files""")

    args = parser.parse_args()
    compute_outputs(args)