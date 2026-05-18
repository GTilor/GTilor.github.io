---
title: nanoGPT技术详解
date: 2026-05-17 15:45:01
tags: [数学, AI]
categories: [AI]
---

# 前言
本篇博客旨在讲解如何从零搭建一个nanoGPT模型到理解nanoGPT的原理，帮助初入大模型的学习者快速学习一个大模型是如何构建、训练和推理。以及给没有算力的同学提供一些免费的算力的方法进行训练自己的nanogpt模型。AI时代我们不仅仅要学会使用大模型，更需要我们适当地了解一个大模型的原理。


nanoGPT是 Andrej Karpathy（前 OpenAI、Tesla AI 负责人）开源的极简 GPT实现，目标是用最少的代码（单文件约 500 行）完整覆盖 GPT的核心架构，去除所有工程复杂性，只保留最纯粹的模型定义和训练逻辑。本文基于对 nanoGPT 源码的逐行学习，记录自己从零理解和实现一个完整 GPT 语言模型的过程。模型的内部结构如下
![alt text](image.png)

本文的实现参考:
- 1.**Karpathy nanoGPT**: https://github.com/karpathy/nanoGPT
- 2.**OpenAI GPT-2**: https://github.com/openai/gpt-2/blob/master/src
- 3.**Hugging Face**: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2


# 代码部署与文件结构
**步骤一**：打开系统自带的终端(terminal),并进入要将代码部署的文件夹。以Mac为例，在终端输入以下代码： `cd ~/Documents/AI_Projects`。

**步骤二**：进入部署的文件夹打开，我们可以看到nanoGPT的目录结构如下:
```bash
nanoGPT/
  ├── model.py              # 模型定义
  ├── train.py              # 训练逻辑
  ├── sample.py             # 推理/生成
  ├── configurator.py       # 配置加载工具
  ├── bench.py              # 性能基准测试
  ├── config/               # 预定义训练/评估配置
  │   ├── train_gpt2.py
  │   ├── train_shakespeare_char.py
  │   └── ...
  ├── data/                 # 数据集
  │   ├── shakespeare_char/
  │   ├── shakespeare/
  │   └── openwebtext/
  ├── out/                  # 训练输出（checkpoint）
  └── assets/               # 图片资源
  ```

各个文件作用如下：
 | 文件 | 作用 | 服务对象 |
  |------|------|----------|
  | `model.py` | 定义 GPT 模型结构（Attention、MLP、Block） | 模型前向传播 |
  | `train.py` | 训练循环、优化器、学习率调度 | 训练阶段 |
  | `sample.py` | 加载 checkpoint，生成文本 | 推理阶段 |
  | `configurator.py` | 从 config 文件或命令行加载超参数 | 所有阶段 |
  | `config/*.py` | 预定义的超参数配置 | 训练/评估 |                                                             
  | `data/*/prepare.py` | 将原始文本转为二进制训练数据 | 数据预处理 |
  | `out/ckpt.pt` | 训练保存的模型权重 | 推理/续训 |


其中`model.py`是该项目的核心，后续章节我们将逐模块拆解其中的实现细节和数学原理。

# data/prepare.py——数据处理
data文件夹里有三个子文件夹包含着三类数据，我们这次仅以`nanoGPT/data/shakespeare_char`为例讲解我们是如何获取数据以及到如何处理数据使得模型可以对我们的数据进行学习。
原始文件只有：
```bash 
  shakespeare_char/
  ├── input.txt      # 原始文本数据
  └── prepare.py     # 数据预处理脚本

  运行 python data/shakespeare_char/prepare.py 会生成：                                                        
  
  ├── train.bin      # 训练集（二进制）
  ├── val.bin        # 验证集（二进制）
  └── meta.pkl       # 元数据（词表大小、编码/解码映射）
```

`input.txt`是原始文本——莎士比亚的戏剧作品。我们接下来将具体讲解`prepare.py`是如何处理文本以便于模型读取的。
```python
#拼接文件路径，找到数据集的文件路径
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
# 如果文件不存在则从 GitHub 下载（Tiny Shakespeare 数据集）
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)
# 读取全部文本数据到内存
with open(input_file_path, 'r') as f:
    data = f.read()
print(f"数据集字符总数: {len(data):,}")
```
经过上面代码:这样我们就将读取了数据集里面的数据转移到了data里面存储。
```python
# 提取所有不重复字符并排序，确保映射的一致性
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("所有唯一字符:", ''.join(chars))
print(f"词汇表大小: {vocab_size:,}")
```
经过上面代码:这里我们就可以知道整个数据集字符总数有1,115,394个。所有唯一字符包括空格、标点符号（如 `!$&',-.:;?`）、大写字母 A-Z、小写字母 a-z，共 65 个字符。词汇表大小为 65。
```python
# stoi: string-to-integer，字符 -> 整数编码（编码器使用）
stoi = { ch:i for i,ch in enumerate(chars) }
# itos: integer-to-string，整数 -> 字符解码（解码器使用）
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    """编码器：将字符串转换为整数列表"""
    return [stoi[c] for c in s]
def decode(l):
    """解码器：将整数列表转换回字符串"""
    return ''.join([itos[i] for i in l])
```
经过上面代码:我们就创建了不同字符对不同整数的映射的字典，方便后面解码和编码，将各种字符编码成0-64。

```python
n = len(data)
train_data = data[:int(n*0.9)]      # 前 90% 作为训练集
val_data = data[int(n*0.9):]        # 后 10% 作为验证集

train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"训练集 token 数: {len(train_ids):,}")
print(f"验证集 token 数: {len(val_ids):,}")

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

meta = {
    'vocab_size': vocab_size,   # 词汇表大小
    'itos': itos,               # 整数 -> 字符（解码用）
    'stoi': stoi,               # 字符 -> 整数（编码用）
}

with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

```
最后一段代码首先将数据按9:1划为训练集和验证集，然后利用上面的代码块的encode函数将文本编码为整数序列，
再将这些用65个数字转换为二进制格式，加快模型的读取能力，最后将编码和解码的规则保存在当地存为meta.pkl。

# model.py——模型结构
经过了prepare.py的处理，我们已经把我们的数据转化为了可供模型快速读取的二进制的数据了。此时每个字符对应的整数编号（0-64）我们称之为**token**。
接下来我们就开始搭建我们的模型框架了。nanoGPT的核心组件可以分为如下几部分：
- GPTConfig: 模型配置类（超参数定义）
- GPT: GPT 模型主类
- CausalSelfAttention: 因果自注意力机制
- MLP: 前馈神经网络
- Block: Transformer 块（Attention + MLP）
- LayerNorm: 层归一化
我们也将model.py分为上述6部分进行讲解。

## GPTConfig：
这个部分是设定模型的超参数的一个类，以便后面直接调用。
```python
@dataclass
class GPTConfig:
    """
    GPT 模型配置：定义模型的超参数。
    """
    block_size: int = 1024      # 最大序列长度（上下文窗口）
    vocab_size: int = 65     # 词汇表大小，即刚刚的65个映射
    n_layer: int = 12           # Transformer 层数
    n_head: int = 12            # 注意力头数
    n_embd: int = 768           # embedding 维度
    dropout: float = 0.0        # Dropout 率
    bias: bool = True           # 是否使用偏置（True 更像 GPT-2，False 更快）
```
这是一个定义模型超参数的类，block_size就是把一个长文本都切成1024个字符的串，然后再把一串一串的字符放进模型内。其他参数：`vocab_size` 是词表大小（Shakespeare 是 65），`n_layer` 是 Transformer 层数，`n_head` 是注意力头数，`n_embd` 是每个 token 的向量维度，`dropout` 是随机丢弃比例，`bias` 是是否加偏置。实际训练时 `vocab_size` 会被数据集的真实词表大小覆盖，`n_head` 必须能整除 `n_embd`。具体的参数作用会在后面进行更详细地解析。


## LayerNorm
正如前言的模型架构图所演示的，这是模型第一个经过的神经网络层——归一化层。其代码如下：
```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))   # 缩放参数
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # 偏移参数（可选）

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```
ndim是 n_embd，即768，他是将一个token转化为768维的一维向量。传进来告诉LayerNorm "我要归一化的向量有多长"。

假设一个 token 的向量是 $x = [x_1, x_2, ..., x_{768}]$；LayerNorm通过F.layer_norm函数做两件事：

1.算均值和方差：$\mu$，$\sigma^2$。归一化：$\hat{x}_i$，$\epsilon$=1e-5很小 防止除零

2.缩放和偏移：$y_i = \gamma_i \hat{x}_i + \beta_i$，其中 $\gamma$ 就是 self.weight，$\beta$ 就是 self.bias让模型自由调整每个维度的范围。
weight 初始全是1，bias 初始全是0，训练时通过反向传播自动学习。

为什么需要LayerNorm？ 每一层的输出值可能越来越大或越来越小，LayerNorm 把它们拉回到均值 0、方差 1 的范围，让训练更稳定。

## CausalSelfAttention




























# train.py——训练细节
# sample.py——推理生成

# configurator.py + config/ ——参数配置系统