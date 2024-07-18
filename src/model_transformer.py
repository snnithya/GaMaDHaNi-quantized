# 1. clean the xtransformers.py (which is currently a combinaton of 
    # a. attend.py
    # b. x_transformers.py
    # c. autoregressive_wrapper.py
    # d. xl_autoregressive_wrapper.py (maybe remove this?)

# 2. install the lucirains repo and note the main changes and OVERRIDE them in THIS script
    #    class TransformerWrapperMod(TransformerWrapper from lucidrains):
    
# 3. make it neatly configurable PLISSSSSSS

# 4. just the xtransformer_alibi.py script again here, but remove ununsed parts
# Intention: combine the 2 files we had (xtransformers and xtransformers_alibi)


# 1a. Attend.py
    # Intermediates
    # helper functions
    # attend class
        # - flash attention
        # - forward
    
    # -> intermediates and helper functions
    #     - no changes, just deletions, no need to override

    # -> attend
        # - not using flash attention
        # - no changes in forward, just structural changes, no need to override


# 1b. x_transformers.py
    #######################
    # LayerIntermediates

    # #######################
    # helper functions
    #     - structured dropout, more effective than traditional attention dropouts
    #     - apply_rotary_pos_emb
    # helper classes
    #     - activations
    # ReluSquared
    # Embedding classes
    #     TokenEmbedding
    #     AbsolutePositionalEmbedding
    #     ScaledSinusoidalEmbedding
    #     RelativePositionBias
    #     CoPE
    #     DynamicPositionBias
    #     AlibiPositionalBias
    #     RotaryEmbedding
    # Normalization classes
    #     Scale
    #     LayerNorm
    #     AdaptiveLayerNorm
    #     ScaleNorm
    #     RMSNorm
    #     AdaptiveRMSNorm
    #     SimpleRMSNorm
    # Residual Gate Classes
    #     Residual
    #     GRUGating
    # token shifting
    #     ShiftTokens
    # post branch operator
    #     LayerScale
    #     AdaptiveLayerScale

    # #######################
    # feedforward
    #     GLU
    #     FeedForward

    # #######################
    # Attention
        # -> seems like restructured only, no need to override
    # AttentionLayers
        # -> check TransformerWrapper
    # Encoder (inherits AttentionLayers)
    # Decoder (inherits AttentionLayers)
    # PrefixDecoder (inherits AttentionLayers)
    # CrossAttender (inherits AttentionLayers)

    # #######################
    # ViTransformerWrapper
    #     -> not using in the project at all, no need to override
    # TransformerWrapper
    #     -> added conditioning, yes need to override (not for now)
    #     -> layerNorm after the attn_layers, whereas this norm is included in attn layers in original, so no need to override
    # #######################
    # XTransformer
    #     -> useful only for the encoder+decoder usecase, hence my version does not have it, no need to override 

# 1c. autoregressive_wrapper.py
    #######################
    # sampling methods
    #     - nucleus is a bit different, but not being used in our experiments, so no need to override for now
    # AutoregressiveWrapper
    #     - generate
    #         -> different, yes need to override
    #     - forward
    #         -> yes need to override
import logging
import math
from typing import Callable, Dict, Optional, Sequence, Tuple
import gin

import pytorch_lightning as pl
import torch
torch.cuda.empty_cache()
import torch.nn as nn

#CHANGE THIS - should not need to add path
import sys
sys.path.append('/home/mila/k/krishna-maneesha.dendukuri/x-transformers')
from x_transformers.x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper
from x_transformers.x_transformers import eval_decorator, top_k
import pdb
TensorDict = Dict[str, torch.Tensor]


class extendedAutoregressiveWrapper(AutoregressiveWrapper):
    def __init__(self,
                 net,
                 ignore_index = -100,
                 pad_value = 0,
                 mask_prob = 0.):
        super().__init__(net, 
                        ignore_index, 
                        pad_value,
                        mask_prob)
    @torch.no_grad()
    @eval_decorator
    def sample_fn(self,
                  batch_size: int=1,
                  prime=None,
                  seq_len: int=2400,
                  temperature: float=1.,
                  filter_logits_fn = top_k,
                  filter_thres: float=0.9,
                  verbose: bool=False,
                  return_prime: bool=False,
                  device: str=None,
                  **kwargs):
        
        if type(prime)==tuple:
            prime, features = prime

        device = prime.device
        prime, ps = pack([prime], '* n')
        out = prime

        if verbose:    
            print("Generating sequence of max length:", seq_len)
        for s in tqdm(range(seq_len)):
            x = out[:, -self.max_seq_len:]
            logits = self.net((x, features), **kwargs)[:, -1]

            if filter_logits_fn == top_k:
                filtered_logits = filter_logits_fn(logits, thres = filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

        if return_prime:
            out =  out[:, :]  
        else:
            out =  out[:, t:]
        
        out, = unpack(out, ps, '* n')
        return out

    def forward(self, x, targets, labels = None, **kwargs):
        logits = self.net(x, **kwargs)
        
        if targets is not None:
            current_token = targets[..., 0] # teacher forcing - replacing quant_id with 0 using _forward function in MultivariatePredictor
        else:
            noise_added_logits = sample_from_logits(logits[:, -1], temperature=0) # (64, 256): (bs, seq_len)
            current_token = torch.argmax(noise_added_logits, -1).unsqueeze(0)       

        samples = list(current_token)
        all_logits = list(logits.unsqueeze(2))

        all_logits = torch.stack(all_logits, 0)  #[bs, seq_len, vocab_size]
        samples = torch.stack(samples, -1) # (64, 256, 16): (bs, seq_len, num_quantizers)

        dist = torch.log_softmax(all_logits, -1)
        entropy = -(dist * dist.exp()).sum(-1)
        perplexity = entropy.exp().mean(-1)

        return all_logits, samples, entropy

    def compute_accuracy(self, logits, labels):
        out = torch.argmax(logits, dim=-1) 
        out = out.flatten() 
        labels = labels.flatten() 

        mask = (labels != self.ignore_index) # can also be self.pad_value (your choice)
        out = out[mask] 
        labels = labels[mask] 

        num_right = (out == labels)
        num_right = torch.sum(num_right).type(torch.float32)

        acc = num_right / len(labels) 
        return acc


class XTransformerPrior(pl.LightningModule):
    def __init__(self,
                 num_tokens: int,
                seq_len: int,
                model_dim: int,
                head_dim: int,
                num_layers: int,
                num_heads: int,
                dropout_rate: float,
                emb_dim: Optional[int]=None,
                max_seq_len: int=None,):
        super().__init__()

        self.num_tokens = num_tokens
        self.seq_len = seq_len
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.emb_dim = emb_dim

        self.max_seq_len = max_seq_len if max_seq_len else seq_len

        self.model = extendedAutoregressiveWrapper(
                        TransformerWrapper(
                            num_tokens = self.num_tokens,
                            max_seq_len = self.seq_len,       #setting the same as the seq_len
                            emb_dim = self.emb_dim,
                            attn_layers = Decoder(
                                dim = self.model_dim,     #because we concat the condition with pitch
                                dim_head = self.head_dim,
                                depth = self.num_layers,
                                heads = self.num_heads,
                                attn_dropout=self.dropout_rate,  # dropout post-attention
                                ff_dropout=self.dropout_rate,     # feedforward dropout
                                alibi_pos_bias = True, # turns on ALiBi positional embedding
                                alibi_num_heads = self.num_heads # only use ALiBi for 4 out of the 8 heads, so other 4 heads can still attend far distances
                                )
                            ),
                        )

 