# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import webdataset as wds
from torchvision import transforms
from torchvision import models as torchvision_models

import dino_utils
import vision_transformer as vits
from vision_transformer import DINOHead

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

GLOBAL_ITER = 0

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_small', 'vit_base', 'vit_large', 'deit_tiny', 'deit_small'] + torchvision_archs, help="""Name of architecture to train. For quick experiments with ViTs, we recommend using vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels of input square patches - default 16 (for 16x16 patches). If <16, we recommend disabling mixed precision training (--use_fp16 false) to avoid instabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=dino_utils.bool_flag, help="""Whether or not to weight normalize the last layer of the DINO head. Not normalizing leads to better performance but can make the training unstable. In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.9995, type=float, help="""Base EMA parameter for teacher update. The value is increased to 1 during training with cosine schedule. We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=dino_utils.bool_flag, help="Whether to use batch normalizations in projection head (Default: False)")

    # Teacher temperature parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float, help="""Initial value for the teacher temperature: 0.04 works well in most cases. Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup) of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int, help='Number of warmup epochs for the teacher temperature')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=dino_utils.bool_flag, default=False, help="""Whether or not to use half precision for training. We recommend disabling for small patch sizes and bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="""Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.0, help="""Final value of the weight decay. We use a cosine schedule for WD and using a larger decay by the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=1.0, help="""Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int, help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=0, type=int, help="""Number of epochs during which we keep the output layer fixed.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of linear warmup (highest LR used during training).""")
    parser.add_argument("--warmup_epochs", default=0, type=int, help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=0.0005, help="""Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.), help="""Scale range of the cropped image before resizing, relatively to the origin image. Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small local views to generate. Set this parameter to 0 to disable multi-crop training.""")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4), help="""Scale range of the cropped image before resizing, relatively to the origin image.""")

    # Misc
    parser.add_argument("--data_path", default="/scratch/eo41/data/saycam/SAY_5fps_300s_{000000..000009}.tar", type=str, help="""path to dataset""")
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=10000, type=int, help='Save checkpoint every x iterations.')
    parser.add_argument('--print_freq', default=10000, type=int, help='Print progress every x iterations.')
    parser.add_argument('--seed', default=1, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help='url used to set up distributed training')
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--save_prefix", default="", type=str, help="""prefix for saving checkpoint and log files""")

    return parser

def preprocess(sample):
    return sample[0]

def train_dino(args):
    dino_utils.init_distributed_mode(args)
    dino_utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(dino_utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(args.global_crops_scale, args.local_crops_scale, args.local_crops_number)

    dataset = (wds.WebDataset(args.data_path, resampled=True).shuffle(10000, initial=10000).decode("pil").to_tuple("jpg").map(preprocess).map(transform))
    data_loader = wds.WebLoader(dataset, shuffle=False, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers)

    print(f"Data loaded.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](patch_size=args.patch_size, drop_path_rate=0.1)
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
        print('Embedding dimension of student & teacher nets', embed_dim)
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = dino_utils.MultiCropWrapper(student, DINOHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head, norm_last_layer=args.norm_last_layer))
    teacher = dino_utils.MultiCropWrapper(teacher, DINOHead(embed_dim, args.out_dim, args.use_bn_in_head))

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if dino_utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = dino_utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = dino_utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = dino_utils.cosine_scheduler(args.lr, args.min_lr, args.epochs, 1000, warmup_epochs=args.warmup_epochs)
    wd_schedule = dino_utils.cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, 1000)
    
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = dino_utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, 1000)
    
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    dino_utils.restart_from_checkpoint(
        os.path.join(args.output_dir, args.save_prefix + "_checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, fp16_scaler, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, fp16_scaler, args):
    metric_logger = dino_utils.MetricLogger(delimiter="  ")

    global GLOBAL_ITER

    for _, images in enumerate(data_loader):
        # update weight decay and learning rate according to their schedule

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[0]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[0]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = dino_utils.clip_gradients(student, args.clip_grad)
            dino_utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = dino_utils.clip_gradients(student, args.clip_grad)
            dino_utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[GLOBAL_ITER // 10]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    
        if GLOBAL_ITER % args.saveckp_freq == 0:
            # ============ writing logs ... ============
            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'it': GLOBAL_ITER,
                'args': args,
                'dino_loss': dino_loss.state_dict(),
            }
            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()

            dino_utils.save_on_master(save_dict, os.path.join(args.output_dir, args.save_prefix + '_checkpoint.pth'))
            dino_utils.save_on_master(save_dict, os.path.join(args.output_dir, args.save_prefix + f'checkpoint{GLOBAL_ITER:04}.pth'))
            
            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)
            train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'it': GLOBAL_ITER}
            if dino_utils.is_main_process():
                with (Path(args.output_dir) / (args.save_prefix + "_log.txt")).open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            # start a fresh logger to wipe off old stats
            metric_logger = dino_utils.MetricLogger(delimiter="  ")

        GLOBAL_ITER += 1

    return train_stats


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs, nepochs, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.5)], p=0.9),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            dino_utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            dino_utils.GaussianBlur(0.1),
            dino_utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            dino_utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)