import torch
import torch.nn as nn
import torch.nn.functional as F


class iden_block(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size = 3):
        
        super(iden_block, self).__init__()

        # get padding size
        assert kernel_size % 2 == 1, "kernel_size should be odd"
        p = (kernel_size - 1) / 2

        # register submodules
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size = kernel_size, padding = p)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size = kernel_size, padding = p)
        self.bn2 = nn.BatchNorm2d(output_channel)


    def forward(self, inputs):
        # two conv layers path
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # merge two paths
        assert x.shape == inputs.shape, "merge failed in iden_block"
        x = F.relu(x + inputs)

        return x


class conv_block(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size = 3, stride = 2, bias = True):
        super(conv_block, self).__init__()

        # get padding size
        p = (kernel_size - stride + 1) // 2

        # register submodules
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size = kernel_size, padding = p, stride = stride)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size = kernel_size, padding = p)
        self.bn2 = nn.BatchNorm2d(output_channel)
        # for shortcut
        self.conv3 = nn.Conv2d(input_channel, output_channel, kernel_size = 1, stride = stride, bias = bias)
        self.bn3 = nn.BatchNorm2d(output_channel)

        self.stride = stride


    def forward(self, inputs):
        # two conv layers path
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # shortcut path
        shortcut = self.conv3(inputs)
        shortcut = self.bn3(shortcut)

        # merge two paths
        assert x.shape == shortcut.shape, "merge failed in conv_block"
        x = F.relu(x + shortcut)

        return x


if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10("~/dataset/cifar10", transform = transform)
    x = trainset[0][0].unsqueeze(0)[:,:2,:16,:16]

    print(x.shape)
    cb = conv_block(2, 2)
    y = cb(x)
    print(cb)
    print(y.shape, y.max(), y.min())