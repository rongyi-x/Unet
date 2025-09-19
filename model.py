import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Unet(nn.Module):

    def __init__(self, in_channel=3, out_channel=1, channels: list = [64, 128, 256, 512]):
        super(Unet, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for middle_channel in channels:
            self.downs.append(DoubleConv(in_channel, middle_channel))
            in_channel = middle_channel

        for middle_channel in reversed(channels):
            # 上采样通道数减少
            self.ups.append(nn.ConvTranspose2d(middle_channel*2, middle_channel, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(middle_channel * 2, middle_channel))

        self.bottleneck = DoubleConv(channels[-1], channels[-1]*2)
        self.final_conv = nn.Conv2d(channels[0], out_channel, kernel_size=1)

    def forward(self, x):

        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # 反转
        skip_connections = skip_connections[::-1]

        x = self.bottleneck(x)

        for idx in range(0, len(self.ups), 2):
            # (Batch_Size, Features, Height/16, Width/16) -> # (Batch_Size, Features, Height/8, Width/8)
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            # if skip_connection.shape != x.shape:
            #     # 调整x大小
            # print(f'before resize:{x.shape}')

            x = TF.resize(x, size=skip_connection.shape[2:])

            # print(f'after resize:{x.shape}')
            x = torch.concat((x, skip_connection), dim=1)
            x = self.ups[idx+1](x)

        return self.final_conv(x)

# # test
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# x = torch.randn((2, 3, 512, 512)).to(device)
# model = Unet().to(device)
# output = model(x)
# print(output.shape)




