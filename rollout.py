import torch
import torch.nn as nn
import torch.nn.functional as F


class rollout:
    def __init__(self, model, device):
        self.model = model
        self.layers = self.model.bert.transformer_encoder.layers
        self.device = device
        self.nheads = model.bert.transformer_encoder.num_attention_heads


    def model_forward(self, input_batch, segments_batch):
        """Compute the attention rollout for the given input batch."""

        # calculate a forward pass to get the attention weights
        self.model.zero_grad()
        input_batch[input_batch == 0] = 1

        out = self.model(input_batch, segments_batch)
        mask = torch.zeros(out.shape).to(self.device)
        mask[:,1] = 1
        loss = (mask*out).sum()
        loss.backward()


    def compute_attention_rollout(self, bsz, seq_lenght):
        with torch.no_grad():
            #rollout = torch.eye(seq_lenght)  init rollout is the identity martix. this models the residual connection.
            result = torch.eye(seq_lenght).to(self.device) # init rollout is the identity martix. this models the residual connection.
            for layer in self.layers:
                # fetch attention weights and their gradients w.r.t the class loss.
                gradient = layer.gradients[-1].view(bsz, self.nheads, seq_lenght, seq_lenght)
                attention = layer.weight_attn[-1].view(bsz, self.nheads, seq_lenght, seq_lenght)
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
                layer.weight_attn = []
                layer.gradients = []

            token_att = result @ torch.ones(seq_lenght)

        return token_att



