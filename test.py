# check if custom layer and pytorch layer return the same output
from torch.nn import TransformerEncoder
import torch
import copy
from EncoderLayer import WrapperEncoderLayer
import torch.nn as nn


if __name__=="__main__":
    # set random seed
    torch.manual_seed(0)

    # initialize pytorch transformer model with Encoder layer.
    encoder_layer = nn.TransformerEncoderLayer(d_model=12, nhead=4, dim_feedforward=12)
    transformer = nn.TransformerEncoder(encoder_layer = encoder_layer, num_layers=2)

    # initialize custom transformer model with custom Encoder layer to extraxt attention weights.
    custom_layer  = WrapperEncoderLayer(d_model=12,nhead=4, dim_feedforward=12)
    custom_transformer = nn.TransformerEncoder(encoder_layer=custom_layer, num_layers=2)

    # copy state dict to custom transformer
    custom_transformer.load_state_dict(transformer.state_dict())

    # put on eval mode
    transformer.eval()
    custom_transformer.eval()

    # define input
    input = torch.randn(512, 32, 12)
    print(transformer(input))
    print(custom_transformer(input))