---
title: nanoGPT技术详解
date: 2026-05-17 15:45:01
tags: [数学, AI]
categories: [AI]
math: true
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
    block_size: int = 1024
    vocab_size: int = 65
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
```
这是一个定义模型超参数的类，block_size就是把一个长文本都切成1024个字符的串，然后再把一串一串的字符放进模型内。其他参数：`vocab_size` 是词表大小（Shakespeare 是 65），`n_layer` 是 Transformer 层数，`n_head` 是注意力头数，`n_embd` 是每个 token 的向量维度，`dropout` 是随机丢弃比例，`bias` 是是否加偏置。实际训练时 `vocab_size` 会被数据集的真实词表大小覆盖，`n_head` 必须能整除 `n_embd`。具体的参数作用会在后面进行更详细地解析。


## LayerNorm
正如前言的模型架构图所演示的，这是模型第一个经过的神经网络层——归一化层。其代码如下：
```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

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
这一层是 nanoGPT 最重要的组成部分，来源于 2017 年 Google 团队的论文 *"Attention Is All You Need"*（Vaswani et al.）。这篇论文提出了 Transformer 架构。

自注意力机制的核心思想是：让序列中的每个 token 去"关注"其他 token，计算它们之间的关系强度，然后按关系强度加权求和，得到每个 token 的新表示。而"因果"（Causal）意味着每个 token 只能看到它前面的 token，不能看到后面的。

其数学表达式为：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中 Q、K、V 都是由输入 $X$ 通过不同的线性变换得到的：
$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

维度说明（以 nanoGPT 为例，$B$=batch size，$T$=序列长度，$d_{model}$=768，$h$=12 头，$d_k = d_{model}/h = 64$）：
- $X \in \mathbb{R}^{B \times T \times d_{model}}$：输入矩阵
- $W^Q, W^K \in \mathbb{R}^{d_{model} \times d_k}$：Q、K 的投影矩阵
- $W^V \in \mathbb{R}^{d_{model} \times d_v}$：V 的投影矩阵
- $Q, K \in \mathbb{R}^{B \times h \times T \times d_k}$：多头形式
- $QK^T \in \mathbb{R}^{B \times h \times T \times T}$：注意力分数矩阵，每个位置对每个位置的相似度
- $\text{softmax}(QK^T/\sqrt{d_k})V \in \mathbb{R}^{B \times h \times T \times d_k}$：加权求和结果

- $Q$（查询）：当前位置想要什么信息
- $K$（键）：每个位置提供什么信息
- $V$（值）：每个位置的实际内容
- $QK^T$：计算当前位置和每个位置的相似度
- $\sqrt{d_k}$：缩放防止数值过大
- softmax：把相似度变成概率（加起来为1）
- 乘 $V$：按概率加权求和

具体代码实现如下：
```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        return y
```
上述代码与前面模块的结构类似：首先在 `__init__` 中定义前向传播所需要的参数和函数，然后再在 `forward` 中定义前向传播的过程。

**`__init__` 中的参数：**
- **`self.c_attn`**：一个 Linear 层，输入 `n_embd` 维，输出 `3*n_embd` 维，一次性算出 Q、K、V 三个向量
- **`self.c_proj`**：输出投影层，把多头合并后的结果从 `n_embd` 投影回 `n_embd`，融合不同头的信息
- **`self.attn_dropout`**：在 softmax 之后随机丢弃一些注意力连接，防止过拟合
- **`self.resid_dropout`**：在输出投影之后随机丢弃一些特征

**前向传播 `forward` 的步骤：**
1. 通过 `self.c_attn` 算出 Q、K、V，拆分后 reshape 成多头形式 `(B, n_head, T, head_dim)`
2. 计算注意力分数 $QK^T / \sqrt{d_k}$，得到每个 token 对其他 token 的相似度
3. 因果掩码：通过下三角矩阵遮挡未来位置。
4. softmax 归一化：把相似度变成概率分布
5. 加权求和：用注意力概率对 V 加权求和
6. 合并多头 + 输出投影：拼接多个头的结果，通过 `self.c_proj` 投影回原始维度

下面用一个具体例子走一遍整个注意力计算过程，看看数据的维度是怎么变化的。假设输入数据的 shape 为 `(32, 1024, 768)`，即 32 个样本，每个样本有 1024 个 token，每个 token 用一个 768 维的向量表示。

**第一步：一次性计算 Q、K、V。** 输入 `(32, 1024, 768)` 经过 `c_attn` 这个 Linear 层，直接输出 `(32, 1024, 2304)`，也就是把 768 维映射到了 768×3=2304 维。然后沿着最后一维拆成三份，得到 Q、K、V，每个 shape 都是 `(32, 1024, 768)`。

**第二步：拆成 12 个头。** 多头注意力的意思是把 768 维拆成 12 份，每份 64 维（768÷12=64）。以 Q 为例，拆分后得到 $Q_1, Q_2, ..., Q_{12}$，每个 $Q_i$ 的 shape 都是 `(32, 1024, 64)`。K 和 V 也做同样的拆分。

**第三步：计算注意力分数。** 以其中一个头为例，$Q_i$ 和 $K_i$ 做矩阵乘法：`(32, 1024, 64) × (32, 64, 1024)` → `(32, 1024, 1024)`。这个 1024×1024 的矩阵表示"每个 token 对其他所有 token 的相似度"。然后经过因果掩码（遮挡未来位置）、softmax 归一化和 dropout，shape 不变，仍然是 `(32, 1024, 1024)`。

**第四步：加权求和。** 用上一步得到的注意力概率矩阵乘以 $V_i$：`(32, 1024, 1024) × (32, 1024, 64)` → `(32, 1024, 64)`。意思是：按每个 token 对其他 token 的"关注度"，对 V 进行加权求和，得到这个头的输出。

**第五步：合并 12 个头。** 把 12 个头的结果拼接起来：`(32, 1024, 64) × 12` → `(32, 1024, 768)`，恢复到原始维度。

**第六步：输出投影。** 通过 `c_proj` 这个 Linear 层，`(32, 1024, 768)` → `(32, 1024, 768)`，融合不同头的信息，输出 shape 和输入完全一样。

上述步骤会产生疑问：上面的Linear层的参数也就是$W^Q, W^K ，W^V$是怎么得来的呢，他们的初始值是随机的（N(0, 0.02)），训练时通过反向传播自动学习来不断优化的，这也就是网上开源模型所开源的具体参数。


## MLP
这一层神经网络是transformer blokc的最后一层了，其主要作用就是可以用下面式子理解：$$\text{MLP}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$$其中 $W_1 \in \mathbb{R}^{4d \times d}$（升维），$W_2 \in \mathbb{R}^{d \times 4d}$（降维)。其具体代码实现如下：
```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```
MLP的实现非常好理解，就是定义维度相互映射$W_1$和$W_2$与激活函数GELU，然后对经过注意力处理后的数据进行进一步变换。同理`nn.Linear`的维度变换的矩阵参数也是后面不断优化学习的。

## Block
这个代码块的目的就是将上述讲的神经网络层组合起来使用形成前言图中的模型内部的流程图。具体实现如下：
```python
class Block(nn.Module):
    # 结构: x → LayerNorm → Attention → +x → LayerNorm → MLP → +x
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)  # 第一个 LayerNorm
        self.attn = CausalSelfAttention(config)                  # 注意力层
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)  # 第二个 LayerNorm
        self.mlp = MLP(config)                                   # 前馈层

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # 注意力 + 残差
        x = x + self.mlp(self.ln_2(x))   # 前馈 + 残差
        return x
```
这个代码在`__init__`中首先将前面介绍的LayerNorm，CausalSelfAttention，MLP进行引入作为Block的组件。前向传播中使用了**残差连接**：`x = x + self.attn(self.ln_1(x))`——输入 `x` 经过 LayerNorm → Attention 后得到输出，再和原始输入 `x` 直接相加。这样做的好处是：即使 Attention 层没学到有用的东西（输出接近 0），原始信息也能通过这条"捷径"直接传过去，不会丢失。同时梯度也能通过这条捷径直接回传，避免深层网络的梯度消失问题。


## GPT(nn.Module)
这个代码是将前言中的图片从1到5的数据处理流程和模型等所有组件进行拼接的主类。我们将按以下顺序讲解：1.`__init__`  2.`forward`  3.`_init_weights`  4.`get_num_params`  5.辅助方法。

### 1. `__init__`：模型搭建
```python
def __init__(self, config):
    super().__init__()
    assert config.vocab_size is not None
    assert config.block_size is not None
    self.config = config

    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size, config.n_embd),
        wpe = nn.Embedding(config.block_size, config.n_embd),
        drop = nn.Dropout(config.dropout),
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ln_f = LayerNorm(config.n_embd, bias=config.bias),
    ))

    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    self.transformer.wte.weight = self.lm_head.weight

    self.apply(self._init_weights)

    for pn, p in self.named_parameters():
        if pn.endswith('c_proj.weight'):
            torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
```
这个代码就是串联起前面LayerNorm等等神经网络层的模型参数，但有独特操作：1.将Token Embedding层和输出头共享一份权重，wte 把 token ID 变成向量，lm_head 把向量变回 token ID，数学上相当于用 embedding 矩阵的转置做输出投影，好处是减少参数量，而且直觉上也合理——"相似的 token 应该有相似的向量表示"和"相似的向量应该输出相似的预测"是同一件事的两面。2.对模型的Linear和Embedding层调用权重初始(`_init_weights`,后面有具体介绍)为 $N(0, 0.02)$，偏置初始化为0。（只在模型创建时做一次）3.遍历所有的参数，找到名字以c_proj.weight结尾的参数，即 Attention 和 MLP 的输出投影层，用更小的标准差 $0.02/\sqrt{2 \times n\_layer}$ 重新初始化。因为`Block`的残差连接中会累加各层的输出，层数越多累加值越大，缩放后保证初始时各层输出的方差不会随层数增长而爆炸。

### 2. `forward`：前向传播
```python
def forward(self, idx, targets=None):
    device = idx.device
    b, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)

    tok_emb = self.transformer.wte(idx)
    pos_emb = self.transformer.wpe(pos)
    x = self.transformer.drop(tok_emb + pos_emb)

    for block in self.transformer.h:
        x = block(x)

    x = self.transformer.ln_f(x)

    if targets is not None:
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    else:
        logits = self.lm_head(x[:, [-1], :])
        loss = None

    return logits, loss
```
这个模型是数据在整个nanoGPT的处理过程，首先将数据进行token和pos的升维合并，然后经过12层block，然后再经过输出头进行输出最终的答案，如果有目标的预测token，则计算logits和loss值进行返回。如果没有则计算logits进行返回（推理模型）。

logits的维度是（32，1024，65），其中每个位置都在预测紧邻的下一个token：位置0预测位置1的token，位置1预测位置2的token，...，位置1023预测位置1024的token。每个预测从65个字符中选择，所以每条序列有1024×65个预测值，32条序列共有32×1024×65个预测值。训练时每个位置都计算loss，推理时只需要最后一个位置的预测来生成下一个token。loss计算采用的是交叉熵损失，具体计算流程 `F.cross_entropy` 进行了打包计算。

为什么是位置0预测位置1？这不是数学上的必然，而是训练目标的设计选择——targets是输入右移一位，所以cross_entropy优化的方向就是让每个位置预测紧邻的下一个token。如果改成右移两位，模型也能学会跳一个预测。但"预测下一个"最自然，因为自回归生成时就是一个接一个地生成。

### 3. `_init_weights`：权重初始化
```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```
在上一个代码块里我们看到了`_init_weights`的使用，这一节具体定义了如何权重初始化。对Linear层和Embedding层都进行了具体地初始化方案：权重为平均值为0，标准差为0.02；偏置初始化为0。
### 4. `get_num_params`：参数数量
```python
def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding:
        n_params -= self.transformer.wpe.weight.numel()
    return n_params
```
这个函数定义了对模型参数的统计方法。通过self.parameters()递归遍历了所有子模块的nn.Parameter，用p.numel()统计了每个参数张量多元素数量并求和。non_embedding=True时，排除位置编码（wpe）的参数，因为位置编码不参与实际的计算，排除后得到的数字更能反应模型的有效参数量，排除序列长度的影响对参数量的影响，其实没什么影响。
### 5. 辅助方法

**`crop_block_size`：裁剪上下文窗口**
```python
def crop_block_size(self, block_size):
    assert block_size <= self.config.block_size
    self.config.block_size = block_size
    self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
    for block in self.transformer.h:
        if hasattr(block.attn, 'bias'):
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
```
这个函数是想裁剪输入模型的序列长度，位置编码矩阵是自己裁掉多余的维度；然后是遍历每个block的注意力层，将他的因果掩码裁成blcok_size*block_size的矩阵。

**`configure_optimizers`：配置 AdamW 优化器**
```python
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer
```

**`from_pretrained`：加载 GPT-2 预训练权重**
```python
@classmethod
def from_pretrained(cls, model_type, override_args=None):
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    from transformers import GPT2LMHeadModel
    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
    }[model_type]
    config_args['vocab_size'] = 50257
    config_args['block_size'] = 1024
    config_args['bias'] = True
    config = GPTConfig(**config_args)
    model = GPT(config)
    sd = model.state_dict()
    sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()
    sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            sd[k].copy_(sd_hf[k].t())
        else:
            sd[k].copy_(sd_hf[k])
    return model
```
`@classmethod` 表示这是类方法，不需要创建实例就能调用，`cls` 指向类本身。这个函数的作用是从 HuggingFace 加载 OpenAI 预训练好的 GPT-2 权重，加载到 nanoGPT 的模型结构里。

函数的主要流程：首先根据 `model_type` 查表取出对应的模型配置（层数、头数、维度），然后用这些配置创建一个 nanoGPT 模型（随机初始化）。接着从 HuggingFace 下载 GPT-2 的预训练权重，拿到它的参数字典。最后逐个参数复制——因为 OpenAI 用 Conv1D 存储权重（，而 nanoGPT 用 PyTorch 的 Linear，所以对 4 个权重矩阵（`c_attn.weight`、`c_proj.weight`、`mlp.c_fc.weight`、`mlp.c_proj.weight`）需要做转置 `.t()` 再复制，其他参数直接复制。

**`generate`：自回归生成文本**
```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```
`@torch.no_grad()` 表示不计算梯度（推理不需要反向传播，节省显存）。这是进行推理的函数，`idx` 是喂给模型的初始序列，`max_new_tokens` 是最多生成的 token 数。每次循环：如果序列超过 1024 就截取最后 1024 个 token，然后进行前向传播，取最后一个位置的 logits，除以 `temperature` 控制概率分布的集中程度（越小越确定，越大越随机），`top_k` 只保留概率最高的 k 个 token，其余设为 $-\infty$（softmax 后变成 0）。

然后通过 softmax 把 logits 变成概率分布。softmax 不是简单的除以同一个数，而是先对每个值取指数 $e^{x_i}$，再除以所有指数的和。比如 logits = `[2.1, 0.5, 3.8, -1.0]`，所以除以一个小的temperature会导致输出的值更集中，更确定，因为经过 softmax 后变成 `[0.149, 0.030, 0.814, 0.007]`，加起来等于 1，原来最大的 3.8 对应了最高的概率 0.814。

最后通过 `torch.multinomial` 按概率采样——不是只选最大的，而是按概率随机抽取。81.4% 的概率选到下标 2，14.9% 选到下标 0，3.0% 选到下标 1，0.7% 选到下标 3。采样得到的 token 拼接到序列末尾，重复 `max_new_tokens` 次。

# train.py——训练细节

## 超参数设定
```python
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'

dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

backend = 'nccl'

device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) 
config = {k: globals()[k] for k in config_keys} 
```
这一段代码目的是把模型训练的超参数进行定义和管理，超参数分为六组。第一组控制输出和日志，第二组是数据配置，第三组是模型结构（和 model.py 一致），第四组是 AdamW 优化器参数，第五组是学习率调度，第六组是系统设置。其中第四、五组是训练特有的，下面重点讲解最后三行：第一行是收集上面定义的超参数作为字典，然后第二行是执行configurator.py，解析命令行参数进行替换全局变量，第三行把所有超参数存为字典，后面存checkpoint和日志。这三行实现了命令行参数覆盖机制，让你可以在不修改代码的情况下调整超参数。

## 分布式训练与设备精度
```python
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
```

这段代码分为两部分：分布式训练初始化和设备精度设置。

**分布式训练（DDP）：** `ddp = int(os.environ.get('RANK', -1)) != -1` 检查环境变量 `RANK` 是否存在，`torchrun` 启动多 GPU 时会自动设置。如果存在则进入 DDP 模式：`init_process_group(backend=backend)` 建立进程间通信（用 NVIDIA 的 nccl 库），每个进程获取自己的编号（`RANK`、`LOCAL_RANK`）、总进程数（`WORLD_SIZE`），绑定对应的 GPU。`master_process = ddp_rank == 0` 表示只有 rank 0 负责打印日志和保存 checkpoint。`seed_offset = ddp_rank` 让每个进程用不同的随机种子，保证各 GPU 采样到不同数据。`gradient_accumulation_steps //= ddp_world_size` 把梯度累积步数按 GPU 数等分，保持每步总 token 数不变。

**设备精度：** `torch.manual_seed(1337 + seed_offset)` 设随机种子。`torch.backends.cuda.matmul.allow_tf32 = True` 开启 tf32 格式（矩阵乘法更快，精度损失很小）。`ctx` 是混合精度上下文：CPU 时为空操作，CUDA 时用 `torch.amp.autocast` 自动把适合的操作转成 bfloat16/fp16 计算，对精度敏感的操作保持 float32。

## 数据加载与模型初始化
```python
data_dir = os.path.join('data', dataset)#拼成路径
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

iter_num = 0
best_val_loss = 1e9

# 尝试从数据集获取词表大小
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
```
数据加载和模型初始化部分主要是工程实现：get_batch 用 np.memmap 映射二进制文件并随机采样训练样本；模型初始化支持三种模式（从零训练、从 checkpoint 恢复、加载 GPT-2 预训练权重）。这些代码不涉及模型原理，读者可自行阅读源码。

## 优化器配置与评估调度
```python
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
```

**优化器配置：** `model.configure_optimizers` 是 model.py 里定义的方法，把参数分成两组：2D 参数（权重矩阵）应用 weight decay，1D 参数（bias、LayerNorm 的 γ/β）不应用 weight decay。恢复训练时加载优化器状态（包含动量 $m_t$ 和二阶矩 $v_t$），否则优化器"失忆"，训练会震荡。`checkpoint = None` 释放内存。

**模型编译与 DDP 包装：** `torch.compile` 把模型编译成高效的 GPU 内核，加速约 20-30%。`DDP` 包装模型，让 backward 自动触发梯度同步。

**评估函数 `estimate_loss`：** `@torch.no_grad()` 不计算梯度，节省显存。`model.train_mode` 切换为评估模式，关闭 Dropout。对训练集和验证集各跑 200 个 batch 的 loss 取均值，减少随机波动。评估完切回训练模式。

**学习率调度 `get_lr`：** 三段式调度。0~2000 步线性 warmup（学习率从 0 线性升到最大值），2000~600000 步余弦衰减（平滑降到最小值），600000 步后恒定最小值。warmup 的目的是训练初期梯度不稳定，用小学习率避免参数飞出去。


## 训练循环
```python
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
while True:

    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break
```

训练循环是整个 train.py 的核心，`while True` 循环执行 60 万次，每次迭代做以下事情：

**1. 设置学习率：** 根据当前步数 `iter_num` 调用 `get_lr` 计算当前学习率，写入优化器的每个参数组。

**2. 评估与保存 checkpoint：** 每隔 2000 步（`eval_interval`），调用 `estimate_loss` 评估训练集和验证集的 loss。如果验证 loss 创新低或 `always_save_checkpoint=True`，保存 checkpoint（模型权重、优化器状态、配置、当前步数等）。

**3. 梯度累积的 forward + backward：** 循环 40 次（`gradient_accumulation_steps`），每次处理 12 个样本的 forward + backward，梯度累加到 `param.grad`。`loss / 40` 保证最终梯度是平均值。`scaler.scale(loss).backward()` 对 loss 放大后做反向传播，防止 float16 梯度下溢。每次 forward 的同时异步预取下一批数据。

**4. 梯度裁剪 + 参数更新：** `scaler.unscale_` 还原梯度，`clip_grad_norm_` 把梯度范数限制在 1.0 以内防止梯度爆炸。`scaler.step(optimizer)` 用累积的梯度更新参数，`scaler.update()` 检查是否有 inf/nan 并调整放大因子。`optimizer.zero_grad(set_to_none=True)` 清零梯度释放显存。

**5. 计时与日志：** 记录每步耗时，打印 loss、时间和 GPU 利用率（MFU）。

**6. 终止条件：** 超过 60 万步跳出循环。

# sample.py——推理生成
训练完模型后，用 `sample.py` 让模型生成文本。完整代码如下：
```python
init_from = 'resume'
out_dir = 'out'
start = "\n"
num_samples = 10
max_new_tokens = 500
temperature = 0.8
top_k = 200
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read())
```
推理参数：`init_from` 决定加载 checkpoint 还是 GPT-2 预训练权重；`start` 是起始文本；`temperature` 控制随机性（越小越确定）；`top_k` 只保留概率最高的 k 个 token。

```python
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
```
加载模型：从 checkpoint 恢复或加载 GPT-2 预训练权重。

```python
if load_meta:
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={""})
    decode = lambda l: enc.decode(l)
```
加载编码器：如果有 `meta.pkl` 就用字符级编码器，否则用 GPT-2 的 BPE 编码器。

```python
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
```
编码起始文本，转成 tensor，调用 `model.generate` 生成 10 个样本，每个最多 500 个 token。`generate` 方法在 model.py 中定义：每次取最后一个位置的 logits，除以 temperature，top_k 过滤，softmax 采样，拼接到序列末尾，重复 max_new_tokens 次。

# configurator.py + config/——参数配置系统
`config/*.py` 文件定义了预设的超参数配置。以 `config/train_shakespeare_char.py` 为例：
```python
out_dir = 'out-shakespeare-char'
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
```
这是一个针对 Shakespeare 小数据集的配置：6 层、6 头、384 维的小模型，梯度累积 1 步（不需要累积），batch_size 64，学习率更高（$10^{-3}$），训练 5000 步。

使用方式：
```bash
python train.py config/train_shakespeare_char.py --device=mps --compile=False
```
先执行 config 文件里的代码（覆盖默认超参数），再用命令行参数覆盖（`--device=mps`）。

`configurator.py` 实现了这个覆盖机制。核心逻辑：遍历命令行参数，如果有 `=`（如 `--batch_size=64`），就解析出 key 和 value，用 `globals()[key] = value` 直接修改全局变量。如果没有 `=`（如 `config/train_shakespeare_char.py`），就当成配置文件执行。这样实现了"配置文件 → 命令行参数"的优先级覆盖。




# 预告
本次博客介绍了一个大模型的简要结构(以nanoGPT为例)，后面我们将丰富这个大模型以便更接近更新的大模型，下一篇博客将从数据处理讲起，预计5🈷24日更新。

