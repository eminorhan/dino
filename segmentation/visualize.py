import os
import sys
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision.models.segmentation import FCN
from torchvision import models as torchvision_models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models._meta import _VOC_CATEGORIES

sys.path.insert(0, os.path.abspath('..'))
import vision_transformer as vits


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        layers = [nn.Conv2d(in_channels, channels, 1)]

        super().__init__(*layers)


class vit_fcn_backbone(nn.Module):
    def __init__(self, arch, patch_size):
        super().__init__()

        self.patch_size = patch_size
        self.backbone = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        self.out_channels = self.backbone.embed_dim

    def forward(self, x):
        o = self.backbone.get_intermediate_layers(x)[0]
        o = torch.permute(o, (0, 2, 1))  # batch x channel x flattened spatial dimensions
        o = o[:, :, 1:]
        w_fmap = x.shape[-2] // self.patch_size
        h_fmap = x.shape[-1] // self.patch_size
        o = o.reshape((o.shape[0], o.shape[1], w_fmap, h_fmap))  # batch x channel x width x height
        y = {"out": o}
        return y


class vit_fcn_backbone_aux(nn.Module):
    def __init__(self, arch, patch_size):
        super().__init__()

        self.patch_size = patch_size
        self.backbone = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        self.out_channels = self.backbone.embed_dim

    def forward(self, x):
        l = self.backbone.get_intermediate_layers(x, n=2)
        y = {"out": [], "aux": []}

        o, a = l[-1], l[-2]

        o = torch.permute(o, (0, 2, 1))  # batch x channel x flattened spatial dimensions
        o = o[:, :, 1:]
        w_fmap = x.shape[-2] // self.patch_size
        h_fmap = x.shape[-1] // self.patch_size
        o = o.reshape((o.shape[0], o.shape[1], w_fmap, h_fmap))  # batch x channel x width x height
        y["out"] = o

        a = torch.permute(a, (0, 2, 1))  # batch x channel x flattened spatial dimensions
        a = a[:, :, 1:]
        a = a.reshape((a.shape[0], a.shape[1], w_fmap, h_fmap))  # batch x channel x width x height
        y["aux"] = a
        return y


class resnext_fcn_backbone(nn.Module):
    def __init__(self, arch, aux_loss):
        super().__init__()

        self.backbone = torchvision_models.__dict__[arch]()
        self.out_channels = self.backbone.fc.weight.shape[1]
        if aux_loss:
            self.backbone = IntermediateLayerGetter(self.backbone, return_layers={"layer4": "out", "layer3": "aux"})
        else:
            self.backbone = IntermediateLayerGetter(self.backbone, return_layers={"layer4": "out"})

    def forward(self, x):
        return self.backbone(x)


# some params, argparse these later on
arch, patch_size, aux_loss = "", 14, False
data_path = "/scratch/eo41/dino/segmentation/dog"
pretrained_weights = "/scratch/eo41/dino/segmentation/evals/coco/imagenet_100_vitb14_checkpoint.pth"
state_dict = torch.load(pretrained_weights, map_location="cpu")

# set up model
if arch in vits.__dict__.keys():
    if aux_loss:
        backbone = vit_fcn_backbone_aux(arch, patch_size)
    else:
        backbone = vit_fcn_backbone(arch, patch_size)
elif arch in torchvision_models.__dict__.keys():
    backbone = resnext_fcn_backbone(arch, aux_loss)
else:
    print(f"Unknown architecture: {arch}")
    sys.exit(1)

classifier = FCNHead(backbone.out_channels, 21)
model = FCN(backbone, classifier, None)

msg = model.load_state_dict(state_dict, strict=True)
print('Pretrained weights found at {} and successfully loaded with msg: {}'.format(pretrained_weights, msg))

model.eval()

sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(_VOC_CATEGORIES)}

images = ImageFolder(data_path)
