from base_nets import ResNet18Conv, R3MConv, ShallowConv
import torch
Batch = 64
T = 16
C, H, W = 3, 256, 256
fake_input = torch.randn(Batch, T, C, H, W)

from robomimic.utils.tensor_utils import join_dimensions
fake_input = join_dimensions(fake_input, 0, 1)  # (Batch * T, C, H, W)
net = ResNet18Conv(input_channel = C)
output = net(fake_input)
print(output.shape)  # torch.Size([1024, 512, 2, 2])
net = ShallowConv(input_channel = C)
output = net(fake_input)

print(output.shape)  # torch.Size([1024, 512, 2, 2])