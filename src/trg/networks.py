from builtins import super
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


def get_network(name):
    models = {
        "conv": ConvNet(),
        "inference": InferenceConv()
    }
    return models[name.lower()]


def load_network(path, device):
    """Construct the neural network and load parameters from the specified file.

    Args:
        path: Path to the model parameters. The name must conform to `vgn_name_[_...]`.

    """
    model_name = path.stem.split("_")[1]
    net = get_network(model_name).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net


def load_inference_network(path, device):
    net = get_network("inference").to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net


def load_calibration_inference_network(path, device):
    net_backbone = get_network("conv")
    net = InferenceSE(net_backbone).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    return net


def conv(in_channels, out_channels, kernel_size):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)


def conv_stride(in_channels, out_channels, kernel_size):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2
    )


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(1, [16, 32, 64], [5, 3, 3])
        self.decoder = Decoder(64, [64, 32, 16], [3, 3, 5])
        self.conv_qual = conv(16, 1, 5)
        self.conv_rot = conv(16, 4, 5)
        self.conv_width = conv(16, 1, 5)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        qual_out = self.conv_qual(x)
        rot_out = F.normalize(self.conv_rot(x), dim=1)
        width_out = self.conv_width(x)
        return qual_out, rot_out, width_out


class InferenceConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(1, [16, 32, 64], [5, 3, 3])
        self.decoder = Decoder(64, [64, 32, 16], [3, 3, 5])
        self.conv_qual = conv(16, 1, 5)
        self.conv_rot = conv(16, 4, 5)
        self.conv_width = conv(16, 1, 5)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        qual_out = self.conv_qual(x)
        rot_out = F.normalize(self.conv_rot(x), dim=1)
        width_out = self.conv_width(x)
        return qual_out, rot_out, width_out, qual_out, qual_out


class Encoder(nn.Module):
    def __init__(self, in_channels, filters, kernels):
        super().__init__()
        self.conv1 = conv_stride(in_channels, filters[0], kernels[0])
        self.conv2 = conv_stride(filters[0], filters[1], kernels[1])
        self.conv3 = conv_stride(filters[1], filters[2], kernels[2])

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, filters, kernels):
        super().__init__()
        self.conv1 = conv(in_channels, filters[0], kernels[0])
        self.conv2 = conv(filters[0], filters[1], kernels[1])
        self.conv3 = conv(filters[1], filters[2], kernels[2])

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = F.interpolate(x, 10)
        x = self.conv2(x)
        x = F.relu(x)

        x = F.interpolate(x, 20)
        x = self.conv3(x)
        x = F.relu(x)

        x = F.interpolate(x, 40)
        return x


def count_num_trainable_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


class ModelWithSE(nn.Module):

    def __init__(self, model):
        super(ModelWithSE, self).__init__()
        self.model = model
        self.conv1 = conv(1, 16, 5)
        self.conv2 = conv(16, 32, 3)
        self.conv3 = conv(32, 64, 3)
        self.conv4 = conv(64, 1, 3)

    def train(self, mode=True):
        super(ModelWithSE, self).train()
        self.model.eval()

    def forward(self, input):
        logits = self.model(input)
        y = self.conv1(input)
        y = F.relu(y)

        y = self.conv2(y)
        y = F.relu(y)

        y = self.conv3(y)
        y = F.relu(y)

        y = self.conv4(y)
        y = F.relu(y)

        return y * logits[0], logits[1], logits[2]


class InferenceSE(nn.Module):

    def __init__(self, model):
        super(InferenceSE, self).__init__()
        self.model = model
        self.conv1 = conv(1, 16, 5)
        self.conv2 = conv(16, 32, 3)
        self.conv3 = conv(32, 64, 3)
        self.conv4 = conv(64, 1, 3)

    def train(self, mode=True):
        super(InferenceSE, self).train()
        self.model.eval()

    def forward(self, input):
        logits = self.model(input)
        y = self.conv1(input)
        y = F.relu(y)

        y = self.conv2(y)
        y = F.relu(y)

        y = self.conv3(y)
        y = F.relu(y)

        y = self.conv4(y)
        y = F.relu(y)

        return y * logits[0], logits[1], logits[2], y, logits[0]
