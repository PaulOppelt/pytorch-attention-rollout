# check if custom layer and pytorch layer return the same output
from torch.nn import TransformerEncoder
import torch
import copy
from custom import WrapperEncoderLayer
import torch.nn as nn
import torch.functional as F


if __name__=="__main__":
    # set random seed
    torch.manual_seed(0)

    seq_lenght = 32
    d_model = 12
    nhead = 4
    dim_feedforward = 12
    num_layers = 2

    # initialize pytorch transformer model with Encoder layer.
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
    transformer = nn.TransformerEncoder(encoder_layer = encoder_layer, num_layers=num_layers)

    # initialize custom transformer model with custom Encoder layer to extraxt attention weights.
    custom_layer  = WrapperEncoderLayer(d_model=d_model,nhead=nhead, dim_feedforward=dim_feedforward)
    custom_transformer = nn.TransformerEncoder(encoder_layer=custom_layer, num_layers=2)

    # copy state dict to custom transformer
    custom_transformer.load_state_dict(transformer.state_dict())

    # put on eval mode
    transformer.eval()
    custom_transformer.eval()

    bsz = 512
    device = "cpu"
    label_ind = 1 # label towards which we want to calculate the attention rollout.

    # check if custom layer and pytorch layer return the same output
    input_batch = torch.randn(512, 32, 12)
    print(transformer(input))
    print(custom_transformer(input))

    # do a forward pass though the custom transformer in order to store the attention weights.
    custom_transformer.zero_grad()

    out = custom_transformer(input_batch)
    # some kind of classification head on top of the transformer output. needs to be defined by the user.
    #out = outputlayer(out) <<< to implement by the user.
    mask = torch.zeros(out.shape).to(device)
    mask[:,label_ind] = 1
    loss = (mask*out).sum()
    # by calling loss backwards the attention weights and the gradients are stored in the custom layer.
    loss.backward()

    # calculate rollout from stored attention weights and gradients.
    with torch.no_grad():
        #rollout = torch.eye(seq_lenght)  init rollout is the identity martix. this models the residual connection.
        result = torch.eye(seq_lenght).to(device) # init rollout is the identity martix. this models the residual connection.
        for layer in custom_transformer.layers:
            # fetch attention weights and their gradients w.r.t the class loss.
            gradient = layer.gradients.pop().view(bsz, nhead, seq_lenght, seq_lenght)
            attention = layer.weight_attn.pop().view(bsz, nhead, seq_lenght, seq_lenght)
            # get the maximum elements across the attention heads.
            inter_ = torch.max(gradient*attention, dim=1)[0] 
            # apply relu activation in order to have only positive values.)
            inter_ = nn.ReLU()(inter_)                 
            # normalize the attention matrix. 

            rollout = (inter_ + torch.eye(seq_lenght))/2
            rollout = F.normalize(rollout, dim=1)
            result = torch.matmul(rollout, result)

            """NOTE: weights_attn and gradiens are initialized as lists even though they only store one
            element. this is because register hook does not support value assignment (as far as I know).
            Therefore i free the lists in order to save memory.
            """
            #layer.weight_attn = []
            #layer.gradients = []

        # map the attention rollout to the input tokens.
        token_att = result @ torch.ones(seq_lenght)
