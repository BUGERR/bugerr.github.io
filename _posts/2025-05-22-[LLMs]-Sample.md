# sample
- 采样目的：生成文本时，从预测结果中选出高概率的候选词，避免随机选到低概率词导致的语句不合理，同时保留一定多样性。
- 先用temperature 调整分布平滑度，然后 top-k + top-p 控制候选范围


## Temperature
   - 温度参数调整概率分布的平滑度（高温平滑概率分布，增加多样性；低温放大高概率词，更稳定）

## Top-k 采样
   - 固定选概率最高的 k 个词。（50～100）
   - 若概率分布陡峭（一个0.9，其他很低），可能包含无关词；若分布平坦，可能遗漏高概率词

## Top-p 采样（核采样）
   - 动态选概率累积到 p 的词。直到 >= p（0.7～0.95，越小生成越保守）
   - 根据上下文调整候选词数量，避免太死板或随机，适应不同概率分布
   - 平衡生成文本的创造性和连贯性：陡峭时选较少词，保证质量。平坦时选更多词，增加多样性。

## Top-k + Top-p（双重保险）
   - Top-k 先限制候选词数量，确保不遗漏潜在的高质量候选。
   - Top-p 进一步筛选，动态排除低概率的尾部词汇。
   - 单独 k，若 k 过大，可能包含过多低质量词。
   - 单独 p，若概率分布平坦，可能过早截断候选词。
   - 结果更稳定：小 k + 大 p。更多样化：大 k + 小 p

``` python
@torch.no_grad
def sample(
    model,
    memory,
    memory_mask,
    temperature,
    top_k,
    top_p,
    max_len,
    sos_idx,
    eos_idx,
    pad_idx,
):
    device = memory.device
    batch_size, seq_len, hidden_size = memory.shape
    ys = torch.ones(batch_size, 1, dtype=torch.long, device=device).fill_(sos_idx)
    ended = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for i in range(max_len - 1):
        logits = model.decode(ys, memory, memory_mask)[:, -1]
        logits = logits / temperature

        # Top-k sampling
        if top_k > 0:
            top_k_values, _ = torch.topk(logits, top_k)
            min_top_k_values = top_k_values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_top_k_values,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            indices_to_remove = cumulative_probs > top_p
            indices_to_remove[:, 1:] = indices_to_remove[:, :-1].clone()
            indices_to_remove[:, 0] = 0

            indices_to_remove = indices_to_remove.scatter(
                1, sorted_indices, indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        next_words = torch.multinomial(probs, num_samples=1).squeeze(1)

        ys = torch.cat([ys, next_words.unsqueeze(1)], dim=1)
        ended = ended | (next_words == eos_idx)

        ys[ended & (ys[:, -1] != eos_idx), -1] = pad_idx

        if ended.all():
            break

    if i == max_len - 2:  # reach max length
        ys[~ended, -1] = eos_idx
        ys[ended, -1] = pad_idx

    return ys

```