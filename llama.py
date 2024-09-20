from contextlib import nullcontext
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from base_llama import LlamaPreTrainedModel, LlamaConfig
from rope import apply_rotary_emb
from utils import *

# Root Mean Square Layer Normalization (https://arxiv.org/abs/1910.07467)
# borrowed from the official Llama implementation:
# https://github.com/facebookresearch/llama/blob/main/llama/model.py
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Compute the root mean square normalization. Use Equation 4 under
        Section 4 of https://arxiv.org/abs/1910.07467 as a reference. Add 
        the given epsilon value (self.eps) to the tensor's norm (i.e. inside
        the square root in Equation 4) before normalizing the tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """

        '''
        4 RMSNorm
        A well-known explanation of the success of LayerNorm is its re-centering and re-scaling invariance
        property. The former enables the model to be insensitive to shift noises on both inputs and weights,
        and the latter keeps the output representations intact when both inputs and weights are randomly
        scaled. In this paper, we hypothesize that the re-scaling invariance is the reason for success of
        LayerNorm, rather than re-centering invariance.

        We propose RMSNorm which only focuses on re-scaling invariance and regularizes the summed
        inputs simply according to the root mean square (RMS) statistic:

        6.1 Machine Translation
        In short, both LayerNorm and RMSNorm outperform the Baseline by accelerating model convergence:
        they reduce the number of training steps until convergence by about 50%, and improve test accuracy,
        with RMSNorm being comparable to LayerNorm. This supports our hypothesis that re-scaling invariance
        is the core property of LayerNorm, and that RMSNorm is an effective substitute. Our results with
        L2-Norm show that it fails to improve the model.

        Due to their normalization properties, both RMSNorm and LayerNorm stabilize standard deviation.
        Although the mean in RMSNorm is not normalized, in practice it is more stable than the mean of the baseline.
        This supports our hypothesis that RMSNorm stabilizes recurrent activations without the need to explicitly
        normalize the mean.

        Results in Figure 4 show that LayerNorm becomes very unstable with abnormal initialization,
        but RMSNorm is more robust (both underperform the original initialization). Our empirical evidence
        so far suggests that RMSNorm is similarly robust as LayerNorm, or more.
        '''
        # todo
        '''
        cf. https://pytorch.org/docs/stable/_modules/torch/nn/modules/normalization.html#LayerNorm
        
        The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
        is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
        is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
        the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
        :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
        :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
        The standard-deviation is calculated via the biased estimator, equivalent to
        `torch.var(input, unbiased=False)`.
        '''
        # Tensor dim = 1, single vector (Eeasier to understand imo)
        # https://pytorch.org/docs/stable/generated/torch.sum.html
        # torch.sum(x)
        # https://stackoverflow.com/questions/52772534/what-does-the-1-mean-in-tensor-size-1
        # x.size(-1)
        # Assume we are dealing with a vector, not vectors.
        '''MS = torch.sum(torch.square(x))/x.size(-1) + self.eps'''
        # https://stackoverflow.com/questions/57727372/how-do-i-get-the-value-of-a-tensor-in-pytorch
        # https://stackoverflow.com/questions/47588682/how-to-cast-a-1-d-inttensor-to-int-in-pytorch
        '''RMS = math.sqrt(MS.item())'''
        # g ∈ ℝⁿ is the gain parameter used to re-scale the standardized summed inputs,
        # and is set to 1 at the beginning.
        '''normalized = x*self.weight/RMS'''
        
        # https://github.com/meta-llama/llama/blob/main/llama/model.py
        # Tensor dim >= 1, vector(s)
        # https://pytorch.org/docs/stable/generated/torch.mean.html
        # If keepdim is True, the output tensor is of the same size as input except in the dimension(s) dim
        # where it is of size 1. Otherwise, dim is squeezed (see torch.squeeze()),
        # resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).
        normalized = x*torch.rsqrt(torch.square(x).mean(-1, keepdim=True)+self.eps)
        return normalized

    def forward(self, x):
        """
        Apply the root mean square normalizer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Attention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        assert config.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = config.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.max_seq_len = config.max_seq_len
        self.compute_query = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.compute_key = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.compute_value = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.compute_output = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def compute_query_key_value_scores(self,
                                       query: torch.Tensor,
                                       key: torch.Tensor,
                                       value: torch.Tensor) -> torch.Tensor:
        '''
        Jointly compute Scaled Dot Product Attention (see Section 3.2.1 in
        https://arxiv.org/abs/1706.03762 for details). The query, key, and
        value tensors each have shape (bs, n_local_heads, seqlen, head_dim).
        An optimal implemention will jointly computing attention for multiple
        heads (n_local_heads of them) at once using matrix/tensor operations.

        Make sure to use attention_dropout (self.attn_dropout) on the computed
        attention matrix before applying it to the value tensor.
        '''
        # todo
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor
    ):
        '''
        Llama2 uses Grouped-Query Attention. The details of GQA are actually
        not critical to solving this assignment; you are simply asked to
        compute Scaled Dot Product Attention (see above for details). GQA is
        a memory optimization to compute multi-head attention efficiently. See
        Section 2.2 in https://arxiv.org/abs/2305.13245 or
        https://ai.plainenglish.io/understanding-llama2-kv-cache-grouped-query-attention-rotary-embedding-and-more-c17e5f49a6d7
        for details.
        '''
        batch_size, seqlen, _ = x.shape

        query = self.compute_query(x)
        key = self.compute_key(x)
        value = self.compute_value(x)
        query = query.view(batch_size, seqlen, self.n_local_heads, self.head_dim)
        key = key.view(batch_size, seqlen, self.n_local_kv_heads, self.head_dim)
        value = value.view(batch_size, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        query, key = apply_rotary_emb(query, key, self.head_dim, self.max_seq_len)

        # Grouped multiquery attention: expand out keys and values.
        # Convert both to:
        # (bs, seqlen, n_local_heads, head_dim)
        key = torch.repeat_interleave(key, dim=2, repeats=self.n_rep)
        value = torch.repeat_interleave(value, dim=2, repeats=self.n_rep)

        # make heads into a batch dimension
        query = query.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        output = self.compute_query_key_value_scores(query, key, value)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seqlen, -1)

        # final projection into the residual stream
        output = self.resid_dropout(self.compute_output(output))
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def SwiGLU(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Compute the SwiGLU activation function (see Section 2 in
        https://arxiv.org/abs/2204.02311
        '''
        return F.silu(self.w1(x)) * self.w3(x)

    def forward(self, x):
        return self.dropout(self.w2(self.SwiGLU(x)))


class LlamaLayer(nn.Module):
    def __init__(self, layer_id: int, config: LlamaConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.dim,
            hidden_dim=config.hidden_dim,
            multiple_of=config.multiple_of,
            dropout=config.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.layer_norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.layer_norm_eps)

    def forward(self, x):
        '''
        This is the forward pass of the basic transformer building block. This is a
        modernized version of the block shown on the left of Figure 1 on
        https://arxiv.org/pdf/1706.03762.pdf.

        The transformer block should consist of:
        1) layer normalization of the input (via Root Mean Square layer normalization)
        2) self-attention on the layer-normalized input
        3) a residual connection (i.e., add the input to the output of the self-attention)
        3) layer normalization on the output of the self-attention
        4) a feed-forward network on the layer-normalized output of the self-attention
        5) add a residual connection from the unnormalized self-attention output to the
           output of the feed-forward network
        '''
        # todo
        raise NotImplementedError

class Llama(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        '''
        You will probably never need to call this function, unless you decide
        to pretrain a Llama model from scratch.
        '''
        super().__init__(config)
        self.params = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(LlamaLayer(layer_id, config))
        self.norm = RMSNorm(config.dim, eps=config.layer_norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('compute_output.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _batch_size, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim

        return logits, h

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        We perform this generation using basic temperature sampling. Note that we are not using
        nucleus sampling (i.e. limiting ourselves to sampling from the top-k most probable tokens
        at each timestep), though this is often used in conjunction with temperature sampling,
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # crop to just the final time step
            # todo
            raise NotImplementedError

            if temperature == 0.0:
                # select the single most likely index
                idx_next = None
            else:
                '''
                Perform temperature sampling:
                1) identify  the logits at the final step.
                2) scale (divide) these probabilities by the given temperature.
                3) normalize the scaled logits with a softmax to obtain scaled probabilities.
                4) sample from the scaled probability distribution.

                Note that we are not using top-k sampling/nucleus sampling in this procedure.
                '''
                idx_next = None
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)


        return idx

def load_pretrained(checkpoint):
  device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
  #dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
  dtype = "float32"

  torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
  torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
  device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
  ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
  ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

  # init from a model saved in a specific directory
  checkpoint_dict = torch.load(checkpoint, map_location=device)
  config = LlamaConfig(**checkpoint_dict['model_args'])
  model = Llama(config)
  state_dict = checkpoint_dict['model']
  unwanted_prefix = '_orig_mod.'
  for k,v in list(state_dict.items()):
      if k.startswith(unwanted_prefix):
          state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
  model.load_state_dict(state_dict, strict=False)
  return model
