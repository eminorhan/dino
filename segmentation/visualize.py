import os
import sys
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision.models.segmentation import FCN
from torchvision import models as torchvision_models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models._meta import _VOC_CATEGORIES
from torchvision.utils import draw_segmentation_masks, save_image
from torchvision.io import read_image
import torchvision.transforms as T

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

# ---------------------------------------------------------------------------------------------------------------
# some params, argparse these later on
workers = 16
batch_size = 32
arch, patch_size, aux_loss = "vit_large", 16, False
data_path = "/scratch/eo41/dino/segmentation/cat"
# pretrained_weights = "/scratch/eo41/dino/segmentation/evals/coco/random_vitb14_checkpoint.pth"
pretrained_weights = "/scratch/eo41/dino/segmentation/evals/coco/say_vitl16_checkpoint.pth"
# pretrained_weights = "/scratch/eo41/dino/segmentation/evals/coco/imagenet_100_vitb14_checkpoint.pth"

state_dict = torch.load(pretrained_weights, map_location="cpu")
device = torch.device("cuda")

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

msg = model.load_state_dict(state_dict["model"], strict=True)
print('Pretrained weights found at {} and successfully loaded with msg: {}'.format(pretrained_weights, msg))

model = model.to(device)
model.eval()

# data
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(_VOC_CATEGORIES)}
trans = T.Compose(
            [
                T.Resize(256),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
images = ImageFolder(data_path, transform=trans)
data_loader = torch.utils.data.DataLoader(images, batch_size=32, sampler=None, shuffle=False, num_workers=workers)

outputs = []
imgs = []
# prop images through model
with torch.no_grad():
    for _, (image, _) in enumerate(data_loader):
        image = image.to(device)
        output = model(image)
        output = output["out"]
        outputs.append(output)
        imgs.append(image)

outputs = torch.cat(outputs)
imgs = torch.cat(imgs)
print(outputs.shape, outputs.min().item(), outputs.max().item())

normalized_masks = torch.nn.functional.softmax(0.9*outputs, dim=1)
print(normalized_masks.shape, normalized_masks.min().item(), normalized_masks.max().item())

cat_mask = normalized_masks[:, 8, :, :]  
person_mask = normalized_masks[:, 15, :, :]  
sofa_mask = normalized_masks[:, 18, :, :]  

for i in range(len(imgs)):
    imgs[i] = (imgs[i] - imgs[i].min()) / imgs[i].max()

imgs[:, 0, :, :] = (0.0*cat_mask + 0.1*imgs[:, 0, :, :])
imgs[:, 1, :, :] = (0.0*person_mask + 0.1*imgs[:, 1, :, :]) 
imgs[:, 2, :, :] = (0.9*sofa_mask + 0.1*imgs[:, 2, :, :])

for i in range(len(imgs)):
    save_image(imgs[i].unsqueeze(0), "cat-say-vitl16/sofa_{}.jpg".format(i), padding=0, normalize=True)

# dog_masks = normalized_masks[:, sem_class_to_idx['person'], :, :]
# print(dog_masks.shape)

# boolean_dog_masks = (normalized_masks.argmax(1) == sem_class_to_idx['person'])
# print(boolean_dog_masks.float().mean())

# img_list = [trans2(read_image(os.path.join('person/frames_person', i))) for i in os.listdir('person/frames_person')]

# dogs_with_masks = [draw_segmentation_masks(img, masks=mask, alpha=1.0) for img, mask in zip(img_list, boolean_dog_masks.cpu())]
# for i in range(len(dogs_with_masks)):
#     x = dogs_with_masks[i].unsqueeze(0)
#     x = x.float() / 255.
#     print(x.dtype, x.min(), x.max())
#     save_image(x, f"person-mask/person-mask-{i:03}.jpg", padding=0, normalize=False)