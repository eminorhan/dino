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
        self.blocks = nn.ModuleList([Block(hidden_features, drop, act_layer, norm_layer, residual) for i in range(depth)])
        self.norm = norm_layer(hidden_features)
        self.embed_dim = hidden_features

    def forward(self, x):
        x = self.first_layer(self.flatten(x))
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


def vimlp_huge(**kwargs):
    model = VisionMLP(input_size=224*224*3, depth=16, hidden_features=9008, **kwargs)
    return model


def vimlp_giant(**kwargs):
    model = VisionMLP(input_size=224*224*3, depth=16, hidden_features=9040, **kwargs)
    return model