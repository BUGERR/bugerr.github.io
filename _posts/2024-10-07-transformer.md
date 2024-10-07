---
title: '手撕Transformer'
date: 2024-10-7
permalink: /posts/2024/10/transformer/
tags:
  - Deep Learning
  - NLP
  - Seq2seq
---

# 手撕Transformer

## 1. Tokenizer

## 2. Model details
Transformer 最初被提出用于解决 Seq2seq 翻译任务。如果要实现英文到德文的翻译，那么我们称英文为源语言，德文为目标语言。Transformer 的结构如下图所示，源文本的 embedding 与 positional encoding 相加后输入到 Encoder，经过 N 层 Encoder layer 后，输出在 Decoder 的 cross attention 中进行交互。目标文本的 embedding 同样与 positional encoding 相加后输入到 Decoder，Decoder 的输出通常会再经过一个线性层（具体取决于任务要求）。
<div style="text-align: center;">
  <img src="/images/transformer.png" alt="Transformer" style="width: 300px; height: auto;">
</div>

Encoder 和 Decoder 分别使用了两种 mask，`src_mask` 和 `tgt_mask`。`src_mask` 用于遮盖所有的 PAD token，避免它们在 attention 计算中产生影响。`tgt_mask` 除了遮盖所有 PAD token，还要防止模型在进行 next word prediction 时访问未来的词。


### 2.1 Embedding
#### 2.1.1 Token Embedding
嵌入层里，有一个 embedding 矩阵 ($$U\in R^{C\times d_{model}}$$), $$C$$ 为词典大小`vocab_size`，将输入的`src_token_id`转换成对应的 Vector，即嵌入空间的向量表示，其中语义相近的向量距离会更近。具体来说，Embedding one-hot 向量对 $$U$$ 的操作是“指定抽取”，即取出某个 Token 的向量行。
```python
self.tok_emb = nn.Embedding(vocab_size=vocab_size, d_model=d_model, padding_idx=pad_idx)
```
#### 2.1.2 Positional Encoding
[位置编码详解](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
由于 Transformer 不像 RNN 那样具有天然的序列特性，在计算 attention 时会丢失顺序信息，因此需要引入位置编码。原文使用固定的位置编码，用正余弦组合代表一个顺序。计算公式如下：

- 对于偶数维度:  
  
  $$\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

- 对于奇数维度:  
  
  $$\text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

为了数值稳定性，我们对 div term 取指数和对数，即：  
  
  $$\text{div-term} = 10000^{2i/d_{\text{model}}} = \exp\left(\frac{2i \cdot -\log(10000)}{d_{\text{model}}}\right)$$

***

类比二进制表示一个数字，本质是不同位置上数字的交替变换，其中变换频率不同。

- 同时使用正弦和余弦：

让模型获取相对位置，因为对任何固定偏移量 $$k$$，$$pos+k$$ 的位置编码可以被表示为 $$pos$$ 位置编码的线性方程。

***

- $$pos$$：单词在一句话中的位置，取值范围：`[0:max_len]`
- $$i$$：单词的维度序号，取值：`[0:d_model]`

位置编码和输入序列无关，只要模型的超参数确定，`pos_enc`就确定了，是一个大小为 `[max_len * d_model]`的矩阵 ，序列输入时，取前 `[seq_len , d_model]` 的部分。然后根据广播机制与 shape 为 `[batch_size, seq_len, d_model]` 的 `token_emb` 相加，得到 Encoder 的输入，记作 $x_0$。



固定的位置编码对任何序列都是相同的，虽然依靠 `max_len` 和 `d_model` 生成，但是具有可扩展性。相同位置的编码不会受 `max_len` 影响,如果输入序列长度超出 `max_len` ，只需加上对应长度的位置编码。d_model 8 —> 4，相当于保留前半部分。

`We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.`

***

- 代码实现：

位置矩阵和输入无关，只和该点的坐标有关，枚举 pos_enc 的行列坐标，计算赋值。

`max_len` 指数据集里最长的一句话的 token 数，每个 batch 之间的 `seq_len` 可以不同，只要保证 batch 内 padding 到一样长就行。

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super(PositionalEncoding, self).__init__()

        # Initialize position encoding matrix (shape: [max_len, d_model])
        pe = torch.zeros(max_len, d_model)

        # Create a tensor of shape [max_len, 1] with position indices
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )

        # Compute the div_term (shape: [d_model//2]) for the sin and cos functions
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        # Apply sin/cos to even/odd indices in the position encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register pe as a buffer, not a parameter (no gradients needed)
        self.register_buffer("pe", pe)

    def forward(self, x):
        batch_size, seq_len = x.size()
        # [seq_len, d_model]
        return self.pe[:seq_len, :]
```
GPT使用可学习的位置编码：
```python
# 给出两种写法
positions_embed=nn.Embedding(max_len, d_model)
# src:[seq_len, B, d_model]
self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)
```

#### 2.1.3 Transformer Embedding
`In the embedding layers, we multiply those weights by √d_model.`
```python
class TransformerEmbedding(nn.Module):
    def __init__(self, pad_idx, vocab_size, d_model, max_len, dropout=0.1):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = nn.Embedding(vocab_size=vocab_size, d_model=d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(max_len=max_len, d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x):
        '''
        enc_inputs: [batch_size, seq_len]
        '''
        # [batch_size, seq_len, d_model]
        tok_emb = self.tok_emb(x) * math.sqrt(self.d_model)

        # [seq_len, d_model]
        pos_enc = self.pos_enc(x)
        return self.dropout(tok_emb + pos_enc) # auto-broadcast
```
### 2.2 Layers
#### 2.2.1 Mask

<div style="text-align: center;">
  <img src="/images/mask.png" alt="mask" style="width: 300px; height: auto;">
</div>

掩码操作，给不参与运算的位置赋值负无穷大，softmax 后为 0.

在 Decoder 中, `dec_input_token` 因为输入有先后顺序，当前位置的 token 应当只能看到它自己和左侧的 token。等同于给右侧位置的每个元素加上负无穷，要保留的位置表现为一个值为1或True的下三角矩阵。

`torch.Tensor.masked_fill(mask == 0, value)`
将布尔矩阵mask中值为 0 或 False 元素在 Tensor 的对应位置填充为 value。即保留了mask矩阵中值为 1 的区域。

- src: pad_mask
- tgt: pad_mask & sub_mask

```python
def make_pad_mask(seq, pad_idx):
    # [batch_size, 1, 1, src_len]
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask.to(seq.device)


def make_causal_mask(seq):
    batch_size, seq_len = seq.size()
    # [seq_len, seq_len]
    mask = torch.tril(torch.ones((seq_len, seq_len), device=seq.device)).bool()
    return mask


def make_tgt_mask(tgt, pad_idx):
    batch_size, tgt_len = tgt.shape
    # [batch_size, 1, 1, tgt_len]
    tgt_pad_mask = make_pad_mask(tgt, pad_idx)
    # [tgt_len, tgt_len]
    tgt_sub_mask = make_causal_mask(tgt)
    # [batch_size, 1, tgt_len, tgt_len]
    tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0).unsqueeze(0)
    return tgt_mask
```

#### 2.2.2 Scaled Dot-Product Attention

<div style="text-align: center;">
  <img src="/images/attention.png" alt="Attention" style="width: 200px; height: auto;">
</div>

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_{key}}}\right) \cdot V$$

```python
# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.1) -> torch.Tensor:
    '''
    q: [batch_size, n_head, seq_len, d_k]
    
    mask: [batch_size, 1, 1, src_len]
    or [batch_size, 1, tgt_len, tgt_len]
    '''
    scale_factor = 1 / math.sqrt(query.size(-1))

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight = attn_weight.masked_fill_(attn_mask.logical_not(), float("-inf"))
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weight
    
```

#### 2.2.3 Multi-Head Attention

<div style="text-align: center;">
  <img src="/images/multihead_attention.png" alt="Multi-head Attention" style="width: 200px; height: auto;">
</div>

在 Encoder 的 self-attention 中，K、Q、V 均为上一层的输出经过不同线性层得到的。在 Decoder 的 cross-attention 中，K 和 V 来自 Encoder 最后一层的输出，而 Q 是 Decoder 上一层的输出。

为了使模型关注不同位置的不同特征子空间信息，我们需要使用多头注意力。具体来说，将 shape 为 `[batch_size, seq_len, d_model]` 的 K、Q、V 分为 `[batch_size, seq_len, n_head, d_key]`，再交换 `seq_len` 和 `n_head` 两个维度，以便进行 attention 机制中的矩阵乘法。计算了 attention 之后再将结果合并，并通过一个线性层映射到与输入相同的维度。算法的流程如下：

```python
# projection
K, Q, V = W_k(x), W_q(x), W_v(x)

# split
d_key = d_model // n_head
K, Q, V = (K, Q, V).view(batch_size, seq_len, n_head, d_key).transpose(1, 2)
out = scaled_dot_product_attention(K, Q, V)

# concatenate
out = out.transpose(1, 2).view(batch_size, seq_len, d_model)
out = W_cat(out)
```

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.dropout = dropout
        self.n_head = n_head
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_concat = nn.Linear(d_model, d_model)
        self.attn = None

    def forward(self, q, k, v, mask=None):
        '''
        q: [batch_size, seq_len, d_model]
        mask: [[batch_size, 1, seq_len, seq_len]]
        '''
        q, k, v = self.W_q(q), self.W_k(k), self.W_v(v)

        q, k, v = self.split(q), self.split(k), self.split(v)
        
        out, attention_score = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout)
        
        self.attn = attention_score

        out = self.concat(out)
        out = self.W_concat(out)
        return out
    
    def split(self, tensor):
        '''
        [batch_size, seq_len, d_mdoel] --> [batch_size, n_head, seq_len, d_k]
        '''
        batch_size, seq_len, d_model = tensor.size()
        tensor = tensor.view(batch_size, seq_len, self.n_head, -1).transpose(1, 2)
        return tensor
    
    def concat(self, tensor):
        '''
        [batch_size, n_head, seq_len, d_k]
        '''
        batch_size, n_head, seq_len, d_tensor = tensor.size()
        d_model = n_head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return tensor
```

#### 2.2.4 Position-wise Feed-Forward Networks
```python
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

#### 2.2.5 LayerNorm

<div style="text-align: center;">
  <img src="/images/layernorm.png" alt="LayerNorm" style="width: 500px; height: auto;">
</div>

```python

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            # Learnable parameters
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x):
        # x: [batch_size, ..., normalized_shape]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_normalized = (x - mean) / (std + self.eps)

        if self.elementwise_affine:
            x_normalized = self.gamma * x_normalized + self.beta

        return x_normalized
```

### 2.3 Encoder
#### 2.3.1 Encoder Layer
Encoder 包含多个相同的层。上一层的输出 $x_i$ 以如下途径经过该层：
```python
# attention mechanism
residual = x
x = multihead_attention(q=x, k=x, v=x, mask=src_mask)
x = dropout(x)
x = layer_norm(x + residual)

# position-wise feed forward
residual = x
x = feed_forward(x)
x = dropout(x)
x = layer_norm(x + residual)
```
### 2.4 Decoder
#### 2.4.1 Decoder Layer
Decoder 相较于 Encoder 除了多了一层 cross-attention 之外，还使用了 masked multi-head attention。由于模型在此处不能访问未来信息，因此这种注意力机制也称为 causal self-attention。
Decoder 同样包含多个相同的层，Encoder 最后一层的输出 `enc` 和 Decoder 上一层的输出 `dec` 以如下途径经过该层（省略了 dropout）：
```python
# causal self-attention
residual = dec
x = multihead_attention(q=dec, k=dec, v=dec, mask=tgt_mask)
x = layer_norm(x + residual)

# cross-attention
x = multihead_attention(q=x, k=enc, v=enc, mask=src_mask)
x = layer_norm(x + residual)

# position-wise feed forward
residual = x
x = feed_forward(x)
x = layer_norm(x + residual)
```
### 2.5 Transformer
## 3. Training Strategy
### 3.1 Training Data and Batching
[Attention is all you need](https://arxiv.org/pdf/1706.03762) Sec 5.1 提到，训练集使用的是 WMT 2014，每一个训练批次有大约 25k source tokens 和 25k target tokens，结果产生了 6,230 个批次。平均批次大小为 724，平均长度为 45 个 tokens。考虑到 GPU 显存不足，为了确保每个批次都有足够的 tokens，因此需要采取梯度累积策略，每 `update_freq` 轮才更新一次梯度。

论文还提到对 base transformer 进行了 100,000 次迭代训练，这应该对应于 16 个 epochs。

### 3.2 Optimizer
### 3.3 Label Smoothing

推荐阅读：  
<https://nn.labml.ai/transformers/models.html>  
<http://nlp.seas.harvard.edu/annotated-transformer/>  
<https://github.com/hyunwoongko/transformer>  
<https://github.com/jadore801120/attention-is-all-you-need-pytorch>  
<https://github.com/Kami-chanw/SeekDeeper>  
[d2l](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)  
[Transformer的细枝末节](https://zhuanlan.zhihu.com/p/60821628?utm_psn=1826383472404590594)  
[BPE](https://github.com/BrightXiaoHan/MachineTranslationTutorial/blob/master/tutorials/Chapter2/BPE.md)  
