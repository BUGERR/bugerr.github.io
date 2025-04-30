---
title: '论文阅读: BERT'
date: 2025-04-19
permalink: /posts/2025/04/BERT/
tags:
  - Deep Learning
  - BERT
  - NLP
---

# 读 CCV ｜ 用于情景医学图像分割的循环上下文验证

[TOC]

题目：Cycle Context Verification for In-Context  Medical Image Segmentation

作者：

链接：

代码：

推荐阅读: 

<div style="text-align: center;">
  <img src="/images/BERT.png" style="width: auto; height: auto;">
</div>

## what
1. 情景学习（ICL）：通用医学图像分割，用一个模型，跨模态分割多种兴趣对象。
2. 性能：做 query 的图像，和上下文的图像-掩码对要高度对齐。
3. 问题：标签图像少，导致难选上下文 pairs。且由于计算量和遗忘，不能微调 ICL 模型。
4. 方法：新框架 Cycle Context Verification（CCV），用自我验证的预测，和增强上下文的对齐，帮助 ICL 分割。
5. 具体实现：循环流程，模型先为查询图像生成一个分割掩码。然后反过来，预测原始上下文图像的掩码，用查询来验证。这个第二步辅助预测的准确性，侧面衡量了最初查询的分割好坏。
6. 在 query 上加了 prompt，用来**提高**query和in-context pairs之间的**对齐**，进而帮助分割。（传统 prompt 是针对整个数据集）



## Introduction
1. 通用医学图像分割，`Segment anything in medical images`，ICL 根据场景指导分割不同目标，最初在 LLMs，用上下文信息提高预测。
2. 图像分割的 ICL：用上下文的图像-掩码对，处理测试图像（query）。很多研究如何为每个测试查询选择最佳的上下文对，然而最佳可能不存在可用中。
3. 替代方法：增强上下文，而不是选择最佳对，或微调 ICL 模型。
4. 目前视觉情景的 ICL 有限，InMeMo 用了任务特定的提示，加到了上下文图像边缘，需要大量有标签数据来训提示。但医学图像缺少合适上下文对，用不了。
5. 为了克服这一限制，用查询特定的提示。增强每个查询图像的上下文，并在测试时用有限的上下文对优化。**但是很难定义测试时优化目标。**
6. LLMs 的 test time scaling 结合了约束，例如预算强制，来提示模型验证其输出并纠正推理错误。受此启发，answer verification。提出了一种循环上下文验证机制：使 ICL 模型二次检查预测 mask，并优化 query 特定的 prompt。
7. 具体：已知一个测试 query 图像，和一个关联的 in-context 对（任务图像和其mask的gt），ICL 模型为查询生成初始的分割 mask。第二步，把 query 图像和这个生成的 mask 作为新的 context 对，而任务图像作为新的query。
8. 这样，ICL 模型可以对最初生成的 query mask 二次检查，且额外生成了任务图像的 mask。任务图像 mask 和 mask gt 对比，该准确率也侧面说明了初始 query mask 的生成准确率。（原始 query 的预测准确率表明了 query 图像和 in context 对的对齐情况，这会影响预测任务图像 mask 的准确率）
9. 通过这种二次预测，优化 query 的 prompt，从而提高 in-context 预测准确率，最终帮助 query 分割。
10. **注意整套流程是用在测试时的**，用来验证预测的 query mask。prompt 是可学习的，只加在 query 上，用来提高二次预测过程中的 in-context mask 预测。

## 我的问题
1. prompt 是只在测试时更新的吗，训练时更新吗？（A：他这个 pipeline 不改模型，只训练引入的 prompt。防止模型过拟合这个附加的验证步骤。
2. 数据集的图像-掩码对，本质上不是用来微调模型的，而是作为一种基准映射规则，去学这个 prompt。每个 context 类别需要重新学一个新的对应 promt 吗？
3. 第一步没懂，怎么通过 query 图像找到对应的 context 对？
4. 说白了两个贡献：
   - 输出当输入
   - 只在 query 上加了个可学习的 prompt。
   - freeze 预训练ICL大模型，用 in-context 图像作为 rule-based 的指导，希望模型能够建模这种图像-掩码对的映射关系，然后将输入的 query 图像按这种映射关系得到对应 mask。
   - 有点像音频-嘴唇的对齐，根据 in-context 学到映射关系，输入音频 query，输出对应嘴唇。

5. 上下文学习：出自 GPT-3，不微调模型，而是修改 prompt，在句子或词元级别加上一些可学习参数。