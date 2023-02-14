import torch
from torch.nn import TransformerEncoderLayer
from torch import Tensor
import typing
import math
import torch.nn.functional as F

class WrapperEncoderLayer(TransformerEncoderLayer):
    r"""class to modify the transformer encoder layer such, that we have access to the
    attention weights in order to calculate the attention rollout. For this we redefine the _sa_block method
    in the TransformerEncoderLayer class. 
    
    Arguments:
        d_model (`int`): Embedding dimension of the features
        nhead (`int`): number of self attention heads
        dim_feedforward (`int`): dimension of the hidden linear layer.
    """
    def __init__(self, d_model, nhead, dim_feedforward):
        super(WrapperEncoderLayer, self).__init__(d_model, nhead, dim_feedforward)
        
        # define weights and gradient lists.
        self.weight_attn = []
        self.gradients = []

    def _sa_block(self, x: Tensor,
                  attn_mask: typing.Optional[Tensor],
                  key_padding_mask: typing.Optional[Tensor])-> Tensor:
        """refined _sa_block method to save the attention weights and gradients. We reimplement the attention calculation such
        that we can trace the weihts and their corresponding gradients. This method returns the same ouput as the method from the pytorch
        implementation."""
        
        # define some parameters
        sql, bsz, d_model = x.shape
        
        nheads = self.self_attn.num_heads
        
        assert d_model % nheads == 0, "Embedding dimension must be devisible by number of heads"
        d_k = d_model // nheads
        
        # manually calculate attention to inspect gradients.
        w_q, w_k, w_v = self.self_attn.in_proj_weight.chunk(3)
        b_q, b_k, b_v = self.self_attn.in_proj_bias.chunk(3)
        
        q, k, v = F.linear(x, w_q, b_q), F.linear(x, w_k, b_k), F.linear(x, w_v, b_v)
        
        # shape weight matrices into attention heads
        q = q.contiguous().view(sql, bsz * nheads ,d_k).transpose(0, 1)
        k = k.contiguous().view(sql, bsz * nheads ,d_k).transpose(0, 1)
        v = v.contiguous().view(sql, bsz * nheads ,d_k).transpose(0, 1)
        
        attn_output_weights = torch.bmm(q / math.sqrt(d_k), k.transpose(-2, -1))
        attn_output_weights = F.softmax(attn_output_weights,dim=-1)
        
        # save attention and gradient
        self.weight_attn.append(attn_output_weights)
        attn_output_weights.register_hook(lambda grad: self.gradients.append(grad))

        # calculate output
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(sql * bsz, d_model)
        attn_output = self.self_attn.out_proj(attn_output)
        attn_output = attn_output.view(sql, bsz, attn_output.size(1))
        
        # attention heads are still to be fused. The exact method is up the the user. We later use max fusion.
        return self.dropout1(attn_output) 