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
        assert config.n_embd % config.n_head == 0  # 确保维度能被 head 数整除

        # 一次性计算 Q, K, V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        """
        前向传播：计算输入 x 的注意力输出。
        输入: (B, T, C) - 批次大小, 序列长度, embedding维度
        输出: (B, T, C) - 同形状
        """
        B, T, C = x.size()
        # 一次性计算 Q, K, V然后再拆分
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  
        # 计算注意力
        if self.flash:
            # 使用 Flash Attention（高效）
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # 手动实现
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # 注意力分数
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))  # 因果掩码
            att = F.softmax(att, dim=-1)  
            att = self.attn_dropout(att)  
            y = att @ v  
        合并多头
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
class MLP(nn.Module):#将输入进行维数转化。
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)  # 低维映射到高维
        self.gelu = nn.GELU()  # 激活函数
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)  #高维压缩到低维函数
        self.dropout = nn.Dropout(config.dropout)#随机丢弃一些特征

    def forward(self, x):
        x = self.c_fc(x)      # 线性变换 1，将768变成768*4维度
        x = self.gelu(x)      # 激活
        x = self.c_proj(x)    # 线性变换 2，同理
        x = self.dropout(x)   # Dropout
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

### 4. `get_num_params`：参数数量
```python
def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding:
        n_params -= self.transformer.wpe.weight.numel()
    return n_params
```

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


# train.py——训练细节
# sample.py——推理生成

# configurator.py + config/ ——参数配置系统