---
title: 'LLMs: Tokenlizer'
date: 2025-03-12
permalink: /posts/2025/03/tokenizer/
tags:
  - LLMs
  - Tokenlizer
  - 
---

# Tokenlizer

[TOC]

## 2. spaCy
### 1. en_core_web_sm
1. func \
English pipeline optimized for CPU. \
Components: tok2vec, tagger, parser, senter, ner, attribute_ruler, lemmatizer. \

2. use \
安装：
```
$ python -m spacy install en_core_web_sm
```

demo:
``` python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

for token in doc:
    print(token.text, token.pos_, token.dep_)

'''
Apple is looking at buying U.K. startup for $1 billion
Apple PROPN nsubj
is AUX aux
looking VERB ROOT
at ADP prep
buying VERB pcomp
U.K. PROPN dobj
startup NOUN dep
for ADP prep
$ SYM quantmod
1 NUM compound
billion NUM pobj
'''

```


## 3. huggingface tokenizers
### 1. build tokenizer
1. 用 tokenizers 库时，构建顺序：
- 初始化分词器(BPE)
- 归一化(如果corpus预处理好就不用了)
- 定义预分词方式(Whitespace，确保token不是多个单词)
- 初始化训练器和迭代器(引入特殊token，顺序决定了id值。定义vocab_size或min_frequency)
- enable_padding：batch内自动按最长的句子补0
- truncation(max_len)
- 后处理(特殊token规则)
- 保存该分词器

2. demo: subword BPE tokenizer from Vanilla Transformer \
   数据已经预处理好了，不用再normalize。token 和其对应 index 的词典是已经集成在对应语言的 tokenizer 里了。
   - batch_iterator：数据集每个划分是一个 Dataset 对象。外层 for 按 batch_size 的步长遍历位置，循环内层用切片，先指定数据条目范围，再指定具体列。要的是这部分切片的translation 项，而不是 translation 属性的部分切片。
   - 返回键值对形式的元组 list，例如 [['en':'abc'], ['en:'def']]
``` python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

special_tokens = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN]

def build_tokenizer(dataset, lang, force_reload):
    tokenizer_path = config.dataset_dir / f"tokenizer-{lang}.json"
    if os.path.exists(tokenizer_path) and not force_reload:
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN)).from_file(str(tokenizer_path))
    else:
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            special_tokens=special_tokens, show_progress=True, min_frequency=2
        )
        print(f"Training tokenizer for {lang}...")

        def batch_iterator():
            for i in range(0, dataset["train"].num_rows, config.batch_size):
                batch = dataset["train"][i : i + config.batch_size]["translation"]
                yield [item[lang] for item in batch]

        tokenizer.train_from_iterator(batch_iterator(), trainer)
        tokenizer.enable_padding(
            pad_id=tokenizer.token_to_id(PAD_TOKEN), pad_token=PAD_TOKEN
        )
        tokenizer.enable_truncation(max_length=config.max_len)
        tokenizer.post_processor = TemplateProcessing(
            single=f"{SOS_TOKEN} $A {EOS_TOKEN}",
            pair=f"{SOS_TOKEN} $A {EOS_TOKEN} $B:1 {EOS_TOKEN}:1",  # not used
            special_tokens=[
                (SOS_TOKEN, tokenizer.token_to_id(SOS_TOKEN)),
                (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
            ],
        )
        tokenizer.save(str(tokenizer_path))
    return tokenizer

'''
huggingface iwslt2017 An example of 'train' looks as follows.
{
    "translation": {
        "de": "Die nächste Folie, die ich Ihnen zeige, ist eine Zeitrafferaufnahme was in den letzten 25 Jahren passiert ist.",
        "en": "The next slide I show you will be a rapid fast-forward of what's happened over the last 25 years."
    }
}
'''
```

3. demo：BERT with WordPiece

``` python

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer

bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
bert_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
bert_tokenizer.pre_tokenizer = Whitespace()
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

trainer = WordPieceTrainer(vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
bert_tokenizer.train(files, trainer)
bert_tokenizer.save("data/bert-wiki.json")

```

4. Training from memory
- demo_3 是用 text file 训练分词模型，但实际上可以用任意 Python Iteratior 作为训练数据。先初始化一个 Unigram tokenizer，使用了 ByteLevel 作为预分词和对应解码。

``` python 
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
tokenizer = Tokenizer(models.Unigram())
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.UnigramTrainer(
    vocab_size=20000,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<PAD>", "<BOS>", "<EOS>"],
)
```
- 用基本数据类型作为训练数据：List，Tuple，np.Array等作为迭代器来提供strings。

``` python
# First few lines of the "Zen of Python" https://www.python.org/dev/peps/pep-0020/
data = [
    "Beautiful is better than ugly."
    "Explicit is better than implicit."
    "Simple is better than complex."
    "Complex is better than complicated."
    "Flat is better than nested."
    "Sparse is better than dense."
    "Readability counts."
]
tokenizer.train_from_iterator(data, trainer=trainer)
```
- 用 datasets 库，先构造 dataset 的迭代器 batch_iterator。类似构造 dataloader 的 collate_fn，指定原始数据的某个 column 条目用作训练。通过 batch 训练分词器。
- demo：先指定了只用 text 列，按 batch_size 作为迭代步长。最后指定返回字典的索引名为 text。

``` python
import datasets
dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train+test+validation")

def batch_iterator(batch_size=1000):
    # Only keep the text column to avoid decoding the rest of the columns unnecessarily
    tok_dataset = dataset.select_columns("text")
    for batch in tok_dataset.iter(batch_size):
        yield batch["text"]

tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset))

'''
wikitext:
{
    "text": "\" The Sinclair Scientific Programmable was introduced in 1975 , with the same case as the Sinclair Oxford . It was larger than t..."
}
'''
```

- 用 gzip 文件：

``` python
import gzip
with gzip.open("data/my-file.0.gz", "rt") as f:
    tokenizer.train_from_iterator(f, trainer=trainer)

files = ["data/my-file.0.gz", "data/my-file.1.gz", "data/my-file.2.gz"]
def gzip_iterator():
    for path in files:
        with gzip.open(path, "rt") as f:
            for line in f:
                yield line
tokenizer.train_from_iterator(gzip_iterator(), trainer=trainer)
```

### 2. Using the tokenizer
1. encode text返回一个Encoding对象，属性包括：
   - tokens
   - ids：token在vocab里的index
   - offsets：将token对应回原文中的位置
encode demo: 

``` python
sentence = "Hello, y'all! How are you 😁 ?"
output = tokenizer.encode(sentence)

print(output.tokens)
# ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]

print(output.ids)
# [27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35]

print(output.offsets[9])
# (26, 27)
sentence[26:27]
# "😁"
```

2. encode_batch \
   批处理的情况，可以用 enable_padding 让batch内自动按最长的句子补0.并且自带 attention_mask.

demo: encode_batch
``` python
output = tokenizer.encode_batch(["Hello, y'all!", "How are you 😁 ?"])

# sentences pair 的情况：list 的 list
output = tokenizer.encode_batch(
    [["Hello, y'all!", "How are you 😁 ?"], ["Hello to you too!", "I'm fine, thank you!"]]
)
```
demo: enable_padding
``` python
tokenizer.enable_padding(
    pad_id=tokenizer.token_to_id(PAD_TOKEN), pad_token=PAD_TOKEN
)

output = tokenizer.encode_batch(["Hello, y'all!", "How are you 😁 ?"])
print(output[1].tokens)
# ["[CLS]", "How", "are", "you", "[UNK]", "?", "[SEP]", "[PAD]"]

print(output[1].attention_mask)
# [1, 1, 1, 1, 1, 1, 1, 0]
```

3. demo：用 IWSLT 2017 dataset 训练英语和德语分词器，并对数据集分词，然后制作语言模型的dataset，每个batch的item是转为tensor的分词token id。然后把src和tgt的batch padding到一样长。（为什么输入输出要补到一样长？不一样也可以啊。检验输出结果是否正确：把输入数据的id转回text，就能看到输入原文了）

``` python
def load_data(
    src_lang, tgt_lang, splits: Optional[Sequence[str]] = None, force_reload=False
):
    """
    Load IWSLT 2017 dataset..
    Args:
        src_lang (str): 
            Source language, which depends on which language pair you download.
        tgt_lang (str): 
            Target language, which depends on which language pair you download.
        splits (`Sequence[str]`, *optional*): 
            The splits you want to load. It can be arbitrary combination of "train", "test" and "validation".
            If not speficied, all splits will be loaded.
        force_reload (`bool`, defaults to `False`): 
            If set to `True`, it will re-train a new tokenizer with BPE.
    """
    if sorted((src_lang, tgt_lang)) != ["de", "en"]:
        raise ValueError("Available language options are ('de','en') and ('en', 'de')")
    all_splits = ["train", "validation", "test"]
    if splits is None:
        splits = all_splits
    elif not set(splits).issubset(all_splits):
        raise ValueError(f"Splits should only contain some of {all_splits}")

    dataset = datasets.load_dataset(
        "iwslt2017", f"iwslt2017-{src_lang}-{tgt_lang}", trust_remote_code=True
    )

    src_tokenizer = build_tokenizer(dataset, src_lang, force_reload)
    tgt_tokenizer = build_tokenizer(dataset, tgt_lang, force_reload)

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for item in batch:
            src_batch.append(item["translation"][src_lang])
            tgt_batch.append(item["translation"][tgt_lang])

        src_batch = src_tokenizer.encode_batch(src_batch)
        tgt_batch = tgt_tokenizer.encode_batch(tgt_batch)

        src_tensor = torch.LongTensor([item.ids for item in src_batch])
        tgt_tensor = torch.LongTensor([item.ids for item in tgt_batch])

        if src_tensor.shape[-1] < tgt_tensor.shape[-1]:
            src_tensor = F.pad(
                src_tensor,
                [0, tgt_tensor.shape[-1] - src_tensor.shape[-1]],
                value=src_tokenizer.token_to_id(PAD_TOKEN),
            )
        else:
            tgt_tensor = F.pad(
                tgt_tensor,
                [0, src_tensor.shape[-1] - tgt_tensor.shape[-1]],
                value=tgt_tokenizer.token_to_id(PAD_TOKEN),
            )

        return src_tensor, tgt_tensor

    dataloaders = [
        DataLoader(
            dataset[split],
            batch_size=config.batch_size,
            collate_fn=collate_fn,
            shuffle=split == "train",
        )
        for split in splits
    ]

    return (src_tokenizer, tgt_tokenizer, *dataloaders)
```

### 3. pipeline
- Encoding: text --> words --> tokens(subwords) --> IDs
- Decoding: IDs --> tokens --> text

### 4. Encoding pipeline
当调用 Tokenizer.encode 或 encode_batch 时，输入的 text 经过以下流程：
- normalization
- pre-tokenization
- model
- post-processig

1. 归一化 \
   使原始的 string 更少random和cleaner。\
   常用操作：
   - 去空格: Strip()
   - 去语气词: StripAccents()
   - 全小写: Lowercase()

    demo:
    ``` python
    from tokenizers import normalizers
    from tokenizers.normalizers import NFD, Lowercase, StripAccents

    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

    ```

2. 预分词 \
   先初步按单词分，最终的分词结果就是基于这些单词的细分。\
   常用：
   - 按空格和语气词：Whitespace()
   - 按数字：Digits()
   输出是tuple元组的list，包括了单词和它在原始句子中跨越的位置。

    demo:

    ``` python
    from tokenizers import pre_tokenizers
    from tokenizers.pre_tokenizers import Whitespace, Digits

    pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
    pre_tokenizer.pre_tokenize_str("Call 911!")
    # [("Call", (0, 4)), ("9", (5, 6)), ("1", (6, 7)), ("1", (7, 8)), ("!", (8, 9))]
    ```

3. Model \
   model这部分是pipeline里需要在corpus语料库上训练的。输入的text经过归一化和预分词后，model用学好的规则把words分成tokens。并在model的vocabulary里把token映射到对应id。
   - BPE
   - Unigram
   - WordLevel
   - WordPiece

4. Post-Processing \
   后处理是分词流程中的最后一步，对Encoding执行最后的变换，例如加special tokens。\
   - single：指定单句的模版，$A 代表句子
   - pair：指定 输入是成对句子 情况下的模版，一般用不到。$A 代表第一句，$B 代表第二句。':1'代表输入每个部分的 type IDs，默认为 :0，所以第一句和第一个分割符没有特指。第二句和最后的分割符被设为1.
   - special_tokens：指定特殊tokens和词库中对应的id。

   BERT demo：
   ``` python
    from tokenizers.processors import TemplateProcessing

    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
   ```
   改变一个分词器的normalizer或pre-tokenizer后都需要重新训练，但改变后处理不需要。

   demo：看下后处理效果
   ``` python
   # 输入为句子对
    output = tokenizer.encode("Hello, y'all!", "How are you 😁 ?")
    print(output.tokens)
    # ["[CLS]", "Hello", ",", "y", "'", "all", "!", "[SEP]", "How", "are", "you", "[UNK]", "?", "[SEP]"]
    print(output.type_ids)
    # [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
   ```

### 5. Decoding pipeline
把model生成的IDs转回text。会先把IDs根据词库转回tokens，然后去除所有special tokens，然后把剩下的tokens用空格连起来。 \
如果用的分词model里加了special characters来代表subtokens（例如BERT的WordPiece用了'##'来连接子词），解码时就需要用对应规则的解码器。

``` python
output = bert_tokenizer.encode("Welcome to the 🤗 Tokenizers library.")
print(output.tokens)
# ["[CLS]", "welcome", "to", "the", "[UNK]", "tok", "##eni", "##zer", "##s", "library", ".", "[SEP]"]
bert_tokenizer.decode(output.ids)
# "welcome to the tok ##eni ##zer ##s library ."


from tokenizers import decoders

bert_tokenizer.decoder = decoders.WordPiece()
bert_tokenizer.decode(output.ids)
# "welcome to the tokenizers library."

```

### 6. Using a pretrained tokenizer

``` python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file()

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
```
