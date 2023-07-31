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
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=args.num_labels)
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        model.fc = nn.Linear(2048, args.num_labels)
    elif args.arch in vimlps.__dict__.keys():
        model = vimlps.__dict__[args.arch]()
        embed_dim = model.embed_dim
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)

    # load weights to evaluate
    if not args.save_prefix.startswith("random"): 
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
        print(f"Model {args.arch} built. Loaded checkpoint at {args.pretrained_weights}.")
    else:
        print(f"Model {args.arch} built. Using random (untrained) weights.")

    if args.arch in vimlps.__dict__.keys():
        model = nn.Sequential(model, nn.Linear(embed_dim, args.num_labels))
        
    model.cuda()
    model = nn.parallel.DataParallel(model)
    print('Model:', model)

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

    # ============ prepare data pipeline ... ============
    val_dataset = ImageFolder(args.val_data_path, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=4*args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)  # note we use a larger batch size for eval

    train_dataset = ImageFolder(args.train_data_path, transform=train_transform)
    # few-shot finetuning
    if args.frac_retained < 1.0:
        print('Fraction of train data retained:', args.frac_retained)
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        train_idx = indices[:int(args.frac_retained * num_train)]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
        print(f"Data loaded with {len(train_idx)} train and {len(val_dataset)} val imgs.")
    else:
        print('Using all of train data')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, sampler=None)    
        print(f"Data loaded with {len(train_dataset)} train and {len(val_dataset)} val imgs.")

    print(f"{len(train_loader)} train and {len(val_loader)} val iterations per epoch.")

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    best_acc_1 = 0.0
    best_acc_5 = 0.0

    # start training
    for epoch in range(0, args.epochs):
        # train for one epoch
        train_stats = train(model, optimizer, train_loader, epoch, args)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, model, args)
            print(f"Accuracy at epoch {epoch} of the network on test images: {test_stats['acc1']:.1f}% top-1 - {test_stats['acc5']:.1f}% top-5")
            best_acc_1 = max(best_acc_1, test_stats["acc1"])
            best_acc_5 = max(best_acc_5, test_stats["acc5"])

            print(f'Max accuracy so far: {best_acc_1:.2f}% top-1 - {best_acc_5:.2f}% top-5')
            log_stats = {**{k: v for k, v in log_stats.items()}, **{f'test_{k}': v for k, v in test_stats.items()}}
        
        if utils.is_main_process():
            with (Path(args.output_dir) / (args.save_prefix + "_{}_log.txt".format(args.frac_retained))).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # save_dict = {
            #     "epoch": epoch + 1,
            #     "state_dict": model.state_dict(),
            #     "optimizer": optimizer.state_dict(),
            #     "best_acc_1": best_acc_1,
            #     "best_acc_5": best_acc_5,
            # }

            # torch.save(save_dict, os.path.join(args.output_dir, args.save_prefix + "_{}_checkpoint.pth.tar".format(args.frac_retained)))

    print("Finetuning of the model completed.\n Top-1 val accuracy: {acc1:.1f}".format(acc1=best_acc_1))
    print("Finetuning of the model completed.\n Top-5 val accuracy: {acc5:.1f}".format(acc5=best_acc_5))

def train(model, optimizer, loader, epoch, args):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, len(loader) // 1, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        output = model(inp)

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
def validate_network(val_loader, model, args):

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for inp, target in metric_logger.log_every(val_loader, len(val_loader) // 1, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = model(inp)

        loss = nn.CrossEntropyLoss()(output, target)
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # print results
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'.format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Finetuning evaluation')
    parser.add_argument('--arch', default='vit_large', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="student", type=str, help='Key to use in the checkpoint (example: "teacher")')
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
    parser.add_argument("--frac_retained", default=0.0005, type=float, choices=[0.010147, 0.02, 0.03, 0.05, 0.1, 1.0], help="""Fraction of train data retained for finetuning""")

    args = parser.parse_args()
    eval_linear(args)