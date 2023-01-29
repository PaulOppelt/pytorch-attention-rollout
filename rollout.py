import torch
import torch.nn as nn
import torch.nn.functional as F

# implement attention rollout.
def _compute_attention_rollout(model, input_batch, segments_batch, device):
        """Compute the attention rollout for the given input batch."""

        # calculate a forward pass to get the attention weights
        model.zero_grad()
        input_batch[input_batch == 0] = 1 # replace unknown tokens with padding

        out = model(input_batch, segments_batch)

        # calculate the loss. How do the label probabilities change with the attention weights. 
        mask = torch.zeros(out.shape).to(device)
        mask[:,1] = 1
        loss = (mask*out).sum()
        loss.backward()
        
        # get input dimensions
        bsz = input_batch.shape[0]
        seq_lenght = 128 #self.pretrained_model.window_size * self.pretrained_model.n_domains_incl
        nheads = 16#self.pretrained_model.bert.transformer_encoder.layers[0].self_attn.num_heads

        # compute the attention rollout without tracking the gradients of the computations.
        with torch.no_grad():
            #rollout = torch.eye(seq_lenght)  init rollout is the identity martix. this models the residual connection.
            result = torch.eye(seq_lenght).to(device) # init rollout is the identity martix. this models the residual connection.
            for layer in model.bert.transformer_encoder.layers:
                # fetch attention weights and their gradients w.r.t the class loss.
                gradient = layer.gradients[-1].view(bsz, nheads, seq_lenght, seq_lenght)
                attention = layer.weight_attn[-1].view(bsz, nheads, seq_lenght, seq_lenght)
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