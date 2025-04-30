---
title: 'LLMs: Seq2seq'
date: 2025-03-13
permalink: /posts/2025/03/seq2seq/
tags:
  - LLMs
  - seq2seq
  - 
---

# Sequence to Sequence Learning with Neural Networks
本篇是用 spaCy 和 torchtext 库预处理 sequence text 的教程，主要关注处理流程和细节。以英语德语翻译为例，当然也可以扩展到其他seq2seq任务，例如内容总结，从一个序列到更短序列。

[TOC]

## 1. Introduction
最常用的seq2seq模型是 encoder-decoder 架构，通常把原始输入句子编码提取特征为 context vector，然后解码。\
早期用`<sos>`和`<eos>`标志句子的起始，

## 2. Preparing Data
### 1. Dataset
1. 数据集已经划分好了，每个划分是一个 list of dict，每个字典有两个索引，对应英文和德文。
2. 数据集是一个 Dataset 对象。

``` python
import datasets
dataset = datasets.load_dataset("bentrevett/multi30k")
'''
DatasetDict({
    train: Dataset({
        features: ['en', 'de'],
        num_rows: 29000
    })
    validation: Dataset({
        features: ['en', 'de'],
        num_rows: 1014
    })
    test: Dataset({
        features: ['en', 'de'],
        num_rows: 1000
    })
})
'''

train_data, valid_data, test_data = (
    dataset["train"],
    dataset["validation"],
    dataset["test"],
)

train_data[0]
# {'en': 'Two young, White males are outside near many bushes.',
# 'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'}
```

### 2. Tokenizers
核心思想：把句子看成 sequence of tokens 而不是 words。\
token 的含义更广，包括：words，punctuation，numbers，symbols。

``` python
import spacy

en_nlp = spacy.load("en_core_web_sm")
de_nlp = spacy.load("de_core_news_sm")

```
(这部分应该是在做预分词，用预训练的小模型按空格分词，再加上开始和结束token) \
1. 下面用 model.tokenizer 把每个数据划分中的句子转成 token。接收一个 Dataset 对象，把 token list 截成设定的最大长度，转小写，加上开始和结束token。
2. 然后用数据集 Dataset 对象的 map 方法，创建并保存新的映射条目：en_tokens 和 de_tokens。这时 train_data 就包含四个条目：英语句子，德语句子，对应的英语tokens，德语tokens。

``` python
def tokenize_datasplit(datasplit, en_nlp, de_nlp, max_length, lower, sos_token, eos_token):
    en_tokens = [token.text for token in en_nlp.tokenizer(datasplit["en"])][:max_length]
    de_tokens = [token.text for token in de_nlp.tokenizer(datasplit["de"])][:max_length]
    if lower:
        en_tokens = [token.lower() for token in en_tokens]
        de_tokens = [token.lower() for token in de_tokens]
    en_tokens = [sos_token] + en_tokens + [eos_token]
    de_tokens = [sos_token] + de_tokens + [eos_token]
    return {"en_tokens": en_tokens, "de_tokens": de_tokens}


max_length = 1_000
lower = True
sos_token = "<sos>"
eos_token = "<eos>"

fn_kwargs = {
    "en_nlp": en_nlp,
    "de_nlp": de_nlp,
    "max_length": max_length,
    "lower": lower,
    "sos_token": sos_token,
    "eos_token": eos_token,
}

train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(tokenize_example, fn_kwargs=fn_kwargs)
test_data = test_data.map(tokenize_example, fn_kwargs=fn_kwargs)

'''
train_data[0]

{'en': 'Two young, White males are outside near many bushes.',
 'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.',
 'en_tokens': ['<sos>',
  'two',
  'young',
  ',',
  'white',
  'males',
  'are',
  'outside',
  'near',
  'many',
  'bushes',
  '.',
  '<eos>'],
 'de_tokens': ['<sos>',
  'zwei',
  'junge',
  'weiße',
  'männer',
  'sind',
  'im',
  'freien',
  'in',
  'der',
  'nähe',
  'vieler',
  'büsche',
  '.',
  '<eos>']}
'''
```

### 3. Vocabularies
1. 为源语言和目标语言创建词典，将 token 和 index(ID) 相关联。原始输入 text string 在进入模型之前，先被切分成 tokens，然后通过词典这个查找表转成对应 numbers id。因为神经网络本质是数值运算。 \

简单例子：词典："hello" = 1, "world" = 2, "bye" = 3。输入 'hello world'，切成 ["hello", "world"]，转成 [1, 2]. \
   
2. 这里用 torchtext 提供的 `build_vocab_from_iterator`，从数据集中创建词典。通过训练集得到的这个词典不可能覆盖所有单词，因此只要是没见过的，就用特殊 token `<unk>` 表示 \

理想情况是，希望模型可以通过学习 unknow token 的上下文来理解含义。这就需要训练集里也有`<unk>`。因此在创建训练集词典时，通过控制 `min_freq` 参数，规定出现次数小于某个值的 token 会被替换成未知。

3. 特别注意：词典只能从训练集创建，绝对不能用验证或测试集，会信息泄露。
4. 通过 `specials` 参数来设定 special tokens。通常包括：开始，结束，未知，补0.

5. 词典是 token 和 ID 的哈希表，通过 vocab['token'] 得到 token 对应的 index。反过来要用 vocab.itos()[ID] 得到 index 对应的 token。

``` python
import torchtext

min_freq = 2
unk_token = "<unk>"
pad_token = "<pad>"

special_tokens = [
    unk_token,
    pad_token,
    sos_token,
    eos_token,
]

en_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["en_tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)

de_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["de_tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)

'''
en_vocab.get_itos()[:10]
['<unk>', '<pad>', '<sos>', '<eos>', 'a', '.', 'in', 'the', 'on', 'man']

en_vocab["the"]
7

len(en_vocab), len(de_vocab)
(5893, 7853)
'''
```
6. torchtext 要手动设置 OOV token 的 index，用 `set_default_index`
7. encode：`lookup_indices`，把 token list 转 ID。
8. 类比 decode 过程，用 `lookup_tokens` 将 ID list 转回 token。

``` python
assert en_vocab[unk_token] == de_vocab[unk_token]
assert en_vocab[pad_token] == de_vocab[pad_token]

unk_index = en_vocab[unk_token]
pad_index = en_vocab[pad_token]

en_vocab.set_default_index(unk_index)
de_vocab.set_default_index(unk_index)


tokens = ["i", "love", "watching", "crime", "shows"]
en_vocab.lookup_indices(tokens)
# [956, 2169, 173, 0, 821]
en_vocab.lookup_tokens(en_vocab.lookup_indices(tokens))
# ['i', 'love', 'watching', '<unk>', 'shows']
```

9. 现在可以在数据集里加入 ID 的条目了，同样用 map，类比上文如何加入 token。数据集处理后包括 6 个条目了。

``` python
def numericalize_example(example, en_vocab, de_vocab):
    en_ids = en_vocab.lookup_indices(example["en_tokens"])
    de_ids = de_vocab.lookup_indices(example["de_tokens"])
    return {"en_ids": en_ids, "de_ids": de_ids}

fn_kwargs = {"en_vocab": en_vocab, "de_vocab": de_vocab}

train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(numericalize_example, fn_kwargs=fn_kwargs)
test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs)

train_data[0]

'''
'en': 'Two young, White males are outside near many bushes.',
 'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.',
 'en_tokens': ['<sos>',
  'two',
  'young',
  ',',
  'white',
  'males',
  'are',
  'outside',
  'near',
  'many',
  'bushes',
  '.',
  '<eos>'],
 'de_tokens': ['<sos>',
  'zwei',
  'junge',
  'weiße',
  'männer',
  'sind',
  'im',
  'freien',
  'in',
  'der',
  'nähe',
  'vieler',
  'büsche',
  '.',
  '<eos>'],
 'en_ids': [2, 16, 24, 15, 25, 778, 17, 57, 80, 202, 1312, 5, 3],
 'de_ids': [2, 18, 26, 253, 30, 84, 20, 88, 7, 15, 110, 7647, 3171, 4, 3]}
'''
```

10. 最后把数据类型转成 tensor。datasets 库自带转类型的方法 `with_format`，指定数据类型，和要转换的列，并保留其他没参与转换的列。

``` python
data_type = "torch"
format_columns = ["en_ids", "de_ids"]

train_data = train_data.with_format(
    type=data_type, columns=format_columns, output_all_columns=True
)

valid_data = valid_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)

test_data = test_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)

train_data[0]

'''
{'en_ids': tensor([   2,   16,   24,   15,   25,  778,   17,   57,   80,  202, 1312,    5,
            3]),
 'de_ids': tensor([   2,   18,   26,  253,   30,   84,   20,   88,    7,   15,  110, 7647,
         3171,    4,    3]),
 'en': 'Two young, White males are outside near many bushes.',
 'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.',
 'en_tokens': ['<sos>',
  'two',
  'young',
  ',',
  'white',
  'males',
  'are',
  'outside',
  'near',
  'many',
  'bushes',
  '.',
  '<eos>'],
 'de_tokens': ['<sos>',
  'zwei',
  'junge',
  'weiße',
  'männer',
  'sind',
  'im',
  'freien',
  'in',
  'der',
  'nähe',
  'vieler',
  'büsche',
  '.',
  '<eos>']}
'''
```

## 3. Dataloader
1. 先构造一个 collate function，对 batch 内的数据做一些处理，因为原始输入数据包括6个条目，只用提取出英语和德语的ID部分，然后分别做padding。注意这里 `rnn.pad_sequence` 返回形状是 [max_len, batch_size]，通过 `batch_first=True`改顺序。
2. 一个 batch 内按最长的句子做 padding，使得 batch 内每个句子的 token 数量相同。
3. 这里用闭包的方式构造 `collate_fn`，可以很方便的把 pad_index 作为参数传进去，而避免用全局变量。

``` python
def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_en_ids = [example["en_ids"] for example in batch]
        batch_de_ids = [example["de_ids"] for example in batch]
        batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
        batch_de_ids = nn.utils.rnn.pad_sequence(batch_de_ids, padding_value=pad_index)
        batch = {
            "en_ids": batch_en_ids,
            "de_ids": batch_de_ids,
        }
        return batch

    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader

batch_size = 128

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)

```