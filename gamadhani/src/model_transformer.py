import logging
from typing import Callable, Dict, Optional, Sequence, Tuple
from tqdm import tqdm

from einops import pack, unpack
import pytorch_lightning as pl
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F

#CHANGE THIS - should not need to add path
import sys
from x_transformers.x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper
from x_transformers.autoregressive_wrapper import top_p, top_k, eval_decorator

#debug xtransformers
# from msprior.x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper

import pdb
import gin
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
                  seq_len: int=1200,
                  temperature: float=1.,
                  filter_logits_fn = top_k,
                  filter_fn_param: float=40,  #k=40 for top_k sampling
                  **kwargs):
        
        if type(prime)==tuple:
            prime, features = prime

        prime, ps = pack([prime], '* n')
        out = prime

        print("Generating sequence of max length:", seq_len)
        for s in tqdm(range(seq_len)):
            x = out[:, -self.max_seq_len:]
            logits = self.net(x, **kwargs)[:, -1]

            if filter_logits_fn == top_k:
                filtered_logits = filter_logits_fn(logits, k = filter_fn_param)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

        out, = unpack(out, ps, '* n')
        return out

    def forward(self, x, targets, labels = None, **kwargs):
        x = x.squeeze(2)  #debug code
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

@gin.configurable
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
                max_seq_len: int=None,
                log_samples_every: int=10):
        super().__init__()

        self.num_tokens = num_tokens
        self.seq_len = seq_len
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.log_samples_every = log_samples_every
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.emb_dim = emb_dim

        self.max_seq_len = max_seq_len if max_seq_len else seq_len

        self.model = extendedAutoregressiveWrapper(
                        TransformerWrapper(
                            num_tokens = self.num_tokens,
                            max_seq_len = self.seq_len,       #setting the same as the seq_len
                            emb_dim = self.emb_dim,
                            # dim = self.model_dim,   #only old impln
                            attn_layers = Decoder(
                                dim = self.model_dim,     #because we concat the condition with pitch
                                attn_dim_head = self.head_dim,
                                depth = self.num_layers,
                                heads = self.num_heads,
                                attn_dropout=self.dropout_rate,  # dropout post-attention
                                ff_dropout=self.dropout_rate,     # feedforward dropout
                                alibi_pos_bias = True, # turns on ALiBi positional embedding
                                alibi_num_heads = self.num_heads # only use ALiBi for 4 out of the 8 heads, so other 4 heads can still attend far distances
                                )
                            ),
                        )
    def sample_fn(self,
                  **kwargs):
        return self.model.sample_fn(**kwargs)
    
    def loss(self, inputs: TensorDict) -> torch.Tensor:
        try:
            logits,_,_ = self.model(
                x=(inputs["decoder_inputs"]),
                targets=inputs["decoder_targets"]
            )   # (64, 256, 16, 1024)
        except Exception as e:
            print("failed with error: ",e)
            print(inputs["decoder_inputs"].max(), inputs["decoder_inputs"].min())
            logits = None

        targets_one_hot = nn.functional.one_hot(
            torch.clamp(inputs["decoder_targets"].long(), 0),
            logits.shape[-1],
        ).float()

        logits = torch.log_softmax(logits, -1)

        loss = -(logits * targets_one_hot).sum(-1)

        return loss.mean(), logits
    # def on_validation_epoch_end(self) -> None:
    #     #include the call to the sample function
    #     #call the plot and save functions to log the output
    #     #log on wandb

    # def validation_step(self, batch, batch_idx):

    # def training_step(self, batch, batch_idx): 
    #     #call forward and not sample_fn

    def training_step(self, batch, batch_idx):
        # pdb.set_trace()
        loss, logits = self.loss(batch)
        accuracies = self.accuracy(logits, batch["decoder_targets"]) #.squeeze(-1))
        # print("Train loss and accuracies: ", loss, accuracies)
        for topk, acc in accuracies:
            self.log(f'train_acc_top_{topk}', acc)

        self.log('cross_entropy', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self.loss(batch)
        # import pdb; pdb.set_trace()
        self.log('val_cross_entropy', loss)

        accuracies = self.accuracy(logits, batch["decoder_targets"])
        # print("Val loss and accuracies: ", loss, accuracies)
        for topk, acc in accuracies:
            self.log(f'val_acc_top_{topk}', acc)
    
    # def on_validation_epoch_end(self) -> None:
    #     if self.current_epoch % self.log_samples_every == 0:
    #         samples = self.sample_fn(batch_size=8).detach().cpu().numpy()
    #         if self.ckpt is not None:
    #             os.makedirs(os.path.join(self.ckpt, 'samples', str(self.current_epoch)), exist_ok=True)
    #         fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    #         for i in range(4):
    #             for j in range(4):
    #                 axs[i, j].plot(samples[i*4+j].squeeze())
    #                 pd.DataFrame(samples[i*4+j].squeeze(), columns=['normalized_pitch']).to_csv(os.path.join(self.ckpt, 'samples', str(self.current_epoch), f'sample_{i*4+j}.csv'))
    #         if self.logger:
    #             wandb.log({"samples": [wandb.Image(fig, caption="Samples")]})
    #         else:
    #             fig.savefig(os.path.join(self.ckpt, 'samples', str(self.current_epoch), 'samples.png'))
    #         plt.close(fig)

    @gin.configurable
    def configure_optimizers(
            self, optimizer_cls: Callable[[], torch.optim.Optimizer],
            scheduler_cls: Callable[[],
                                    torch.optim.lr_scheduler._LRScheduler]):
        optimizer = optimizer_cls(self.parameters())
        scheduler = scheduler_cls(optimizer)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def accuracy(self, prediction: torch.Tensor,
                 target: torch.Tensor) -> Sequence[Tuple[float, float]]:
        prediction = prediction.cpu()
        target = target.cpu()
        # import pdb; pdb.set_trace()
        top_10 = torch.topk(prediction, 10, -1).indices # sampling is not used to calculate this WHYYYYYYY???
        accuracies = (target[..., None] == top_10).long()
        k_values = [1, 3, 5, 10]
        k_accuracy = []
        for k in k_values:
            current = (accuracies[..., :k].sum(-1) != 0).float()
            k_accuracy.append(current.mean())
        return list(zip(k_values, k_accuracy))

    def on_fit_start(self):
        pass