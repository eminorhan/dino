import torch.nn as nn


class Block(nn.Module):
    def __init__(self, hidden_features, drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, residual=True):
        super().__init__()
        self.residual = residual
        self.norm = norm_layer(hidden_features)
        self.fc = nn.Linear(hidden_features, hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.residual:
            return x + self.drop(self.act(self.fc(self.norm(x))))
        else:
            return self.drop(self.act(self.fc(self.norm(x))))
        

class VisionMLP(nn.Module):
    """ Vision MLP """
    def __init__(self, input_size, depth, hidden_features, drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, residual=True):
        super().__init__()
        self.flatten = nn.Flatten()
        self.first_layer = nn.Linear(input_size, hidden_features)
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.blocks = nn.ModuleList([Block(hidden_features, drop, act_layer, norm_layer, residual) for i in range(depth)])
        self.norm = norm_layer(hidden_features)
        self.embed_dim = hidden_features

    def forward(self, x):
        x = self.act_layer(self.norm_layer(self.first_layer(self.flatten(x))))
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


class PyramidMLP(nn.Module):
    """ Pyramid MLP """
    def __init__(self, input_size, drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.flatten = nn.Flatten()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

        self.layer_1 = nn.Linear(input_size, 100000)
        self.norm_1 = norm_layer(100000)

        self.layer_2 = nn.Linear(100000, 25000)
        self.norm_2 = norm_layer(25000)
    
        self.layer_3 = nn.Linear(25000, 10000)
        self.norm_3 = norm_layer(10000)

        self.layer_4 = nn.Linear(10000, 10000)
        self.norm_4 = norm_layer(10000)

        self.layer_5 = nn.Linear(10000, 10000)
        self.norm_5 = norm_layer(10000)

        self.layer_6 = nn.Linear(10000, 10000)
        self.norm_6 = norm_layer(10000)

        self.embed_dim = 10000

    def forward(self, x):
        x = self.layer_1(self.flatten(x))
        x = self.drop(self.act(self.layer_2(self.norm_1(x))))
        x = self.drop(self.act(self.layer_3(self.norm_2(x))))
        x = self.drop(self.act(self.layer_4(self.norm_3(x))))
        x = self.drop(self.act(self.layer_5(self.norm_4(x))))
        x = self.drop(self.act(self.layer_6(self.norm_5(x))))
        x = self.norm_6(x)
        return x


def vimlp_huge(**kwargs):
    model = VisionMLP(input_size=224*224*3, depth=16, hidden_features=9008, **kwargs)
    return model


def vimlp_giant(**kwargs):
    model = VisionMLP(input_size=128*128*3, depth=16, hidden_features=12150, **kwargs)
    return model


def vimlp_collosal(**kwargs):
    model = VisionMLP(input_size=64*64*3, depth=8, hidden_features=20000, **kwargs)
    return model