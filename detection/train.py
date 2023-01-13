r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os
import sys
import time

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from torchvision.models.detection.anchor_utils import AnchorGenerator
import utils
from coco_utils import get_coco, get_coco_kp
from engine import evaluate, train_one_epoch
from group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from torchvision.transforms import InterpolationMode
from transforms import SimpleCopyPaste
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNHeads
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead

sys.path.insert(0, os.path.abspath('..'))
import vision_transformer as vits

### vit pretrained model loading utilities
def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("There is no reference weights available for this model => We use random weights.")

class vit_maskrcnn_backbone(torch.nn.Module):
    def __init__(self, arch, patch_size, pretrained_weights, checkpoint_key):
        super().__init__()

        self.patch_size = patch_size
        self.backbone = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        if pretrained_weights is not None:
            load_pretrained_weights(self.backbone, pretrained_weights, checkpoint_key)
        self.out_channels = self.backbone.embed_dim

    def forward(self, x):
        o = self.backbone.get_intermediate_layers(x)[0]
        o = torch.permute(o, (0, 2, 1))  # batch x channel x flattened spatial dimensions
        o = o[:, :, 1:]
        w_fmap = x.shape[-2] // self.patch_size
        h_fmap = x.shape[-1] // self.patch_size
        o = o.reshape((o.shape[0], o.shape[1], w_fmap, h_fmap))  # batch x channel x width x height
        return o

class LinearHead(torch.nn.Module):
    """
    Linear head for FPN-based models
    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """
    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = torch.nn.Linear(in_channels, representation_size, bias=True)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.fc6(x)
        return x

### ###################################################################################################

def copypaste_collate_fn(batch):
    copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
    return copypaste(*utils.collate_fn(batch))


def get_dataset(name, image_set, transform, data_path):
    paths = {"coco": (data_path, get_coco, 91), "coco_kp": (data_path, get_coco_kp, 2)}
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train, args):
    if train:
        return presets.DetectionPresetTrain(data_augmentation=args.data_augmentation)
    else:
        return presets.DetectionPresetEval()


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    # basics
    parser.add_argument("--data-path", default="", type=str, help="dataset path")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=26, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)")
    parser.add_argument("--print-freq", default=600, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument("--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone")
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only.")
    parser.add_argument("--output_dir", default=".", help='Path to save logs and checkpoints')
    parser.add_argument("--save_prefix", default="", type=str, help="""prefix for saving checkpoint and log files""")

    # model params
    parser.add_argument("--arch", default='vit_large', type=str, help='Architecture')
    parser.add_argument("--patch_size", default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--pretrained_weights", default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="student", type=str, help='Key to use in the checkpoint (example: "teacher")')

    # optimizer parameters
    parser.add_argument("--opt", default="Adam", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.0001, type=float, help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu")
    parser.add_argument("--weight-decay", default=0.0, type=float, metavar="W", help="weight decay (default: none)", dest="weight_decay")
    parser.add_argument("--norm-weight-decay", default=None, type=float, help="weight decay for Normalization layers (default: None, same value as --wd)")

    # Mixed precision parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # data augmentation parameters
    parser.add_argument("--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)")
    parser.add_argument("--use-copypaste", action="store_true", help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'")

    return parser


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # load data
    print("Loading data")
    dataset, num_classes = get_dataset(args.dataset, "train", get_transform(True, args), args.data_path)
    dataset_test, _ = get_dataset(args.dataset, "val", get_transform(False, args), args.data_path)

    print("Creating data loaders")
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    train_collate_fn = utils.collate_fn
    if args.use_copypaste:
        if args.data_augmentation != "lsj":
            raise RuntimeError("SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies")

        train_collate_fn = copypaste_collate_fn

    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=train_collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn)
    print('Number of training images:', len(dataset), 'Number of training iterations per epoch:', len(data_loader), 'Number of classes:', num_classes)
    print('Number of test images:', len(dataset_test), 'Number of test iterations per epoch:', len(data_loader_test))

    # set up model
    print("Creating model")
    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if args.rpn_score_thresh is not None:
        kwargs["rpn_score_thresh"] = args.rpn_score_thresh

    # model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2()

    # frozen backbone
    backbone = vit_maskrcnn_backbone(args.arch, args.patch_size, args.pretrained_weights, args.checkpoint_key)
    for p in backbone.parameters():
        p.requires_grad = False

    # anchor generator 
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

    # rpn_head = RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0])
    # box_head = FastRCNNConvFCHead((backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=torch.nn.BatchNorm2d)
    # mask_head = MaskRCNNHeads(backbone.out_channels, [256, 256, 256, 256], 1, norm_layer=torch.nn.BatchNorm2d)

    rpn_head = RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0])
    box_head = FastRCNNConvFCHead((backbone.out_channels, 7, 7), [256], [1024], norm_layer=None)
    mask_head = MaskRCNNHeads(backbone.out_channels, [256], 1, norm_layer=None)

    box_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)

    # # rpn head
    # rpn_head = RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], conv_depth=0)

    # # mask head
    # mask_layers = (256,)
    # mask_dilation = 1
    # mask_head = MaskRCNNHeads(backbone.out_channels, mask_layers, mask_dilation)

    # # box_head 
    # resolution = box_roi_pooler.output_size[0]
    # representation_size = 1024
    # box_head = LinearHead(backbone.out_channels * resolution**2, representation_size)

    model = MaskRCNN(backbone, 
                     num_classes=91, 
                     rpn_anchor_generator=anchor_generator,
                     rpn_head=rpn_head,
                     box_head=box_head,
                     mask_head=mask_head,
                     box_roi_pool=box_roi_pooler, 
                     mask_roi_pool=mask_roi_pooler
                     )

    print(model)

    model.to(device)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    model = torch.nn.parallel.DataParallel(model)
    model_without_ddp = model.module

    if args.norm_weight_decay is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
        nums = sum([p.numel() for p in model.parameters() if p.requires_grad])
        print('Trainable parameters:', nums)
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    optimizer = torch.optim.__dict__[args.opt](parameters, args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        torch.backends.cudnn.deterministic = True
        evaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, scaler)
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, args.save_prefix + "_checkpoint.pth"))

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)