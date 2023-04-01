'''Plots spatial attention maps'''
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torchvision.models import resnext50_32x4d
from torch.nn import Linear
import matplotlib.cm as cm
import numpy as np

def load_pretrained_resnext(pretrained_backbone, pretrained_fc, n_out):
    model = resnext50_32x4d()
    model.fc = Linear(in_features=2048, out_features=n_out, bias=True)

    state_dict_backbone = torch.load(pretrained_backbone, map_location="cpu")["teacher"]
    state_dict_fc = torch.load(pretrained_fc, map_location="cpu")["state_dict"]

    # remove `module.` prefix
    state_dict_backbone = {k.replace("module.", ""): v for k, v in state_dict_backbone.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict_backbone = {k.replace("backbone.", ""): v for k, v in state_dict_backbone.items()}
    # load backbone onto model
    msg_backbone = model.load_state_dict(state_dict_backbone, strict=False)
    print('Pretrained backbone found at {} and loaded with msg: {}'.format(pretrained_backbone, msg_backbone))

    # remove `module.` prefix
    state_dict_fc = {k.replace("module.", ""): v for k, v in state_dict_fc.items()}
    # remove `linear.` prefix
    state_dict_fc = {k.replace("linear.", ""): v for k, v in state_dict_fc.items()}
    # load fc onto model
    msg_fc = model.fc.load_state_dict(state_dict_fc, strict=True)
    print('Pretrained fc found at {} and loaded with msg: {}'.format(pretrained_fc, msg_fc))

    return model


def extract_7x7_map_layer(model):
    layer_list = list(model.children())[:-2]
    new_model = torch.nn.Sequential(*layer_list)

    return new_model


def load_data(data_dir, args):
    train_dataset = ImageFolder(data_dir, Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, sampler=None)

    return train_loader


def predict(data_loader, model, weights, batch_size):
    # move to GPU
    model = model.cuda()
    weights = weights.cuda()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            images = images.cuda()

            print(images.size())

            # compute predictions
            pred = model(images)

            if i == 0:
                break

    linear_combination_map = torch.einsum('ijkl,j->ikl', pred, weights)

    x = torch.zeros(batch_size, 3, 7, 7)
    x[:, 0, :, :] = linear_combination_map
    x[:, 1, :, :] = linear_combination_map
    x[:, 2, :, :] = linear_combination_map

    m = torch.nn.Upsample(scale_factor=32, mode='bicubic')
    mm = m(x).cuda()
    print(torch.mean(mm))
    mm = 255 * torch.sigmoid((mm - torch.mean(mm)) / torch.std(mm))
    mm = mm.int()
    mm = mm.cpu().numpy()

    # jet colors
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]

    mm = jet_colors[mm[:, 0, :, :]]
    mm = np.transpose(mm, (0, 3, 1, 2))

    composite_map = 0.8*mm + 0.2*images.cpu().numpy()
    composite_map = torch.from_numpy(composite_map)

    return composite_map, images


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot class activation maps')
    parser.add_argument('--data_path', default='', type=str, help='path to dataset')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=12, type=int, help='total batch size on all GPUs')
    parser.add_argument('--pretrained_backbone', default='s_5fps_resnext50_checkpoint.pth', type=str, help='backbone checkpoint')
    parser.add_argument('--pretrained_fc', default='s_resnext50_checkpoint.pth.tar', type=str, help='fc checkpoint')
    parser.add_argument('--n_out', default=26, type=int, help='output dim of pre-trained model')
    parser.add_argument('--class_idx', default=0, type=int, help='class index for which the maps will be computed')
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    args = parser.parse_args()
    print(args)

    model = load_pretrained_resnext(args.pretrained_backbone, args.pretrained_fc, args.n_out)
    map_layer = extract_7x7_map_layer(model)

    weights = model.fc.weight.data[args.class_idx, :]

    data_loader = load_data(args.data_path, args)
    preds, imgs = predict(data_loader, map_layer, weights, args.batch_size)
    print('Preds shape:', preds.shape)

    save_image(preds, 'cam_class_{}_{}.jpg'.format(args.class_idx, args.seed), nrow=12, padding=1, normalize=True)
    save_image(imgs, 'org_imgs_{}_{}.jpg'.format(args.class_idx, args.seed), nrow=12, padding=1, normalize=True)