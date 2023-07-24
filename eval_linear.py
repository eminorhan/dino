import os
import sys
import argparse
import json
from pathlib import Path
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
import vision_mlps as vimlps

def eval_linear(args):
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        embed_dim = model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    elif args.arch in vimlps.__dict__.keys():
        model = vimlps.__dict__[args.arch]()
        embed_dim = model.embed_dim
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

    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DataParallel(linear_classifier)

    # ============ preparing data ... ============
    # validation transforms
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # training transforms
    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if args.split:
        from torch.utils.data.sampler import SubsetRandomSampler

        val_dataset = ImageFolder(args.train_data_path, transform=val_transform)
        train_dataset = ImageFolder(args.train_data_path, transform=train_transform)

        num_train = len(train_dataset)
        print('Total data size is', num_train)

        indices = list(range(num_train))
        np.random.shuffle(indices)

        if args.subsample:
            num_data = int(0.1 * num_train)
            train_idx, test_idx = indices[:(num_data // 2)], indices[(num_data // 2):num_data]
        else:
            split = int(np.floor(0.5 * num_train))  # split 50-50, change here if you need to do sth else
            train_idx, test_idx = indices[:split], indices[split:]

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)

        print(f"Data loaded with {len(train_idx)} train and {len(test_idx)} val imgs.")
        print(f"{len(train_loader)} train and {len(val_loader)} val iterations per epoch.")
    else:
        val_dataset = ImageFolder(args.val_data_path, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        train_dataset = ImageFolder(args.train_data_path, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        
        print(f"Data loaded with {len(train_dataset)} train and {len(val_dataset)} val imgs.")
        print(f"{len(train_loader)} train and {len(val_loader)} val iterations per epoch.")
    # ============ done data ... ============

    print('Class names:', train_dataset.classes)

    # set optimizer
    optimizer = torch.optim.Adam(linear_classifier.parameters(), lr=args.lr)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, args.save_prefix + "_checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer    
        )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    # start training
    for epoch in range(start_epoch, args.epochs):

        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, model, linear_classifier, args)
            print(f"Accuracy at epoch {epoch} of the network on test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()}, **{f'test_{k}': v for k, v in test_stats.items()}}
        
        if utils.is_main_process():
            with (Path(args.output_dir) / (args.save_prefix + "_log.txt")).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, args.save_prefix + "_checkpoint.pth.tar"))

    print("Training of the linear classifier on frozen features completed.\n Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(model, linear_classifier, optimizer, loader, epoch, args):

    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, len(loader) // 1, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            else:
                output = model(inp)
        output = linear_classifier(output)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, args):

    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    labels = []
    choices = []

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
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            else:
                output = model(inp)
        output = linear_classifier(output)
        loss = nn.CrossEntropyLoss()(output, target)

        choice = torch.argmax(output, dim=1)
        labels.append(target)
        choices.append(choice)

        if linear_classifier.module.num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        else:
            acc1, = utils.accuracy(output, target, topk=(1,))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

        if linear_classifier.module.num_labels >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # print results
    if linear_classifier.module.num_labels >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'.format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'.format(top1=metric_logger.acc1, losses=metric_logger.loss))
    
    # save trial by trial accuracy
    labels = torch.cat(labels, 0)
    choices = torch.cat(choices, 0)
    labels = labels.cpu().numpy()
    choices = choices.cpu().numpy()
    print('val. labels shape:', labels.shape)
    print('val. choices shape:', choices.shape)
    np.savez(os.path.join(args.output_dir, args.save_prefix + "_val_accs.npz"), labels=labels, choices=choices) 

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Linear probe evaluation')
    parser.add_argument('--arch', default='vit_large', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the beginning of training (highest LR used during training).""")
    parser.add_argument('--batch_size', default=1024, type=int, help='total batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # dataset arguments
    parser.add_argument('--train_data_path', default='', type=str)
    parser.add_argument('--val_data_path', default='', type=str)
    parser.add_argument('--split', default=False, action='store_true', help='whether to manually split dataset into train-val')
    parser.add_argument('--subsample', default=False, action='store_true', help='whether to subsample the data')

    # misc
    parser.add_argument('--num_workers', default=1, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument("--save_prefix", default="", type=str, help="""prefix for saving checkpoint and log files""")

    args = parser.parse_args()
    eval_linear(args)