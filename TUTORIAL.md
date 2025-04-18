# GPT-2预训练教程

本教程将详细介绍GPT-2模型的实现原理和预训练过程。

## 1. GPT-2模型架构

GPT-2是基于Transformer的解码器部分构建的，主要包含以下组件：

### 1.1 多头注意力机制（Multi-Head Attention）

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False) # config.n_embd为词向量维度
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.register_buffer('attention_mask', torch.tril(torch.ones(config.block_size, config.block_size))) # register_buffer会自动成为模型中的参数，随着模型移动（gpu/cpu）而移动，但是不会随着梯度进行更新
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()
        # transpose(1, 2)是为了能够并行计算多头注意力机制，(B,n_head,T,head_size)在矩阵运算时在（T,head_size）维度上进行，从而提高计算效率
        k = self.key(x).view(B, T, self.config.n_head, self.config.head_size).transpose(1, 2).contiguous()  # (B,n_head,T,head_size)
        q = self.query(x).view(B, T, self.config.n_head, self.config.head_size).transpose(1, 2).contiguous()  # (B,n_head,T,head_size)
        v = self.value(x).view(B, T, self.config.n_head, self.config.head_size).transpose(1, 2).contiguous()  # (B,n_head,T,head_size)
        # 计算注意力分数
        wei = q @ k.transpose(-2,-1) * (self.config.head_size ** -0.5)  # (B,n_head,T,T)
        wei = wei.masked_fill(self.attention_mask[:T,:T] == 0, float('-inf'))  # (B,n_head,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,n_head,T,T)
        wei = self.dropout(wei)
        
        # 加权聚合
        out = wei @ v  # (B,n_head,T,head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B,T,n_head*head_size)
        
        return self.dropout(self.out_proj(out))
```

初学时很多人不太喜欢这种多头注意力代码实现方式，更喜欢通过SingleHead来实现MultiHead的实现方式，但面试是选拔性的，所以建议初学时还是努力去理解这种实现方式吧。

### 1.2 前馈神经网络（Feed Forward）

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd), # 注意升维4倍
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )
```

前馈网络对每个位置的表示进行非线性变换。

### 1.3 GPT2块（Block）
一个完整的Block块
```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(config.n_embd,eps=config.layer_norm_epsilon)
        self.ffwd = FeedForward(config)
        self.ln2 = nn.LayerNorm(config.n_embd,eps=config.layer_norm_epsilon)

    def forward(self, x, attention_mask=None):
        x = x + self.att(self.ln1(x), attention_mask)
        x = x + self.ffwd(self.ln2(x))
        return x
```

每个GPT2块包含一个多头注意力层和一个前馈网络，都带有残差连接和层归一化。

### 1.4 GPT-2
```python
class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_final = nn.LayerNorm(config.n_embd,eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # 线性层的weight为(output_dim,input_dim）与Embedding的weight形状相同
        self.token_embedding_table.weight = self.lm_head.weight # tie weights
        self.apply(self._init_weights)

    def forward(self, idx, attention_mask=None, targets=None):
        _,T = idx.size()
        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T).to(idx.device))
        x = tok_embd + pos_embd
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        if targets is not None:
            logits = logits[:,:-1,:] # 预测的时候不包含最后一个token（因为最后一个token后面没有token，即没有预测目标）
            targets = targets[:,1:] # logits的预测目标（不包含第一个token）
            loss = F.cross_entropy(logits.contiguous().view(-1,self.config.vocab_size),targets.contiguous().view(-1))
            return logits,loss
        else:
            return logits
```

## 2. 预训练过程

### 2.1 数据准备

```python
class PreTrainDataset(Dataset):
    def __init__(self, config):
        self.data_path = config.data_path
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        self.eot = '<|endoftext|>'
        self.encoded_data = []
```

数据预处理步骤：
1. 加载原始文本数据
2. 使用tokenizer将文本转换为token
3. 将token序列分割成固定长度的块

### 2.2 训练循环

```python
def train(model,optimizer,scheduler,train_dataloader,device,epoch):
    model.train()
    total_loss = 0
    for idx,(x,y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        _,loss = model(x,targets=y)
        loss.backward()
        optimizer.step()
        scheduler.step()
```

训练过程：
1. 前向传播计算损失
2. 反向传播更新参数
3. 使用学习率调度器调整学习率

## 3. 关键概念解释

### 3.1 自回归生成

GPT-2是一个自回归模型，它通过预测下一个token来生成文本。在训练时，模型的目标是最大化下一个token的预测概率。

### 3.2 注意力掩码

```python
self.register_buffer('attention_mask', torch.tril(torch.ones(config.block_size, config.block_size)))
```

```
# 示例 config.block_size = 3
attention_mask = [[1,0,0],
				  [1,1,0],
				  [1,1,1]]
```

注意力掩码确保模型只能看到当前位置之前的token，这是自回归生成的关键。

### 3.3 权重绑定（tie weight）

```python
self.token_embedding_table.weight = self.lm_head.weight
```

将输入嵌入和输出投影层的权重绑定，可以减少参数数量并提高模型性能(小模型时，学习时重要的地方不是position_embedding，但这里的参数比较多，因而可以使用这种方式减少占比提高学习效果)。

### 预训练目标

GPT-2的预训练目标是通过自回归语言建模（Autoregressive Language Modeling）来学习文本的生成。具体来说：

1. **自回归预测**：
   - 给定一个文本序列，模型需要预测下一个token
   - 例如：对于输入序列 "今天天气真"，模型需要预测下一个token可能是"好"
   - 这种预测是单向的，只能看到当前位置之前的token

2. **损失函数**：
```python
if targets is not None:
    logits = logits[:,:-1,:]  # 预测的时候不包含最后一个token
    targets = targets[:,1:]   # 目标不包含第一个token
    loss = F.cross_entropy(logits.contiguous().view(-1,self.config.vocab_size),
                          targets.contiguous().view(-1))
```
   - 使用交叉熵损失函数
   - 对于每个位置，计算预测token与真实token之间的损失
   - 最终损失是所有位置损失的平均值

3. **训练过程**：
   - 将输入序列和目标序列错开一个位置，代码实现方式是针对一个sequence，取前N-1个作为输入序列，后N-1个作为目标序列
   - 模型输入: `[token1, token2, token3, ..., tokenN-1, tokenN]`
   - 输入序列：`[token1, token2, token3, ..., tokenN-1]`
   - 目标序列：`[token2, token3, token4, ..., tokenN]`
   - 这样模型就能学习到预测下一个token的能力

4. **示例说明**：
   假设我们有一个句子："今天天气真好"
   - 模型输入: `[今天, 天气, 真, 好, <eos>]`
   - 输入序列：`[今天, 天气, 真, 好]`
   - 目标序列：`[天气, 真, 好, <eos>]`
   - 模型需要预测：
     - 看到"今天"时，预测"天气"
     - 看到"今天天气"时，预测"真"
     - 看到"今天天气真"时，预测"好"
     - 看到"今天天气真好"时，预测结束符`<eos>`