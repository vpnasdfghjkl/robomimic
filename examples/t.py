from add_new_mlp_encoder import CustomMlpEncoderCore
import torch
mlp = CustomMlpEncoderCore(input_shape = (64,2, 7), welcome_str="welcome to the new encoder")
output = mlp(torch.randn(64,2,7))
print(output.shape)