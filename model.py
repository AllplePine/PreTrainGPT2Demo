import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 512 # 原medium = 1024
    batch_size: int = 12
    epoches: int = 10
    n_layer: int = 12 # 原medium = 24
    n_head: int = 12 # 原medium = 16   
    n_embd: int = 768 # 原medium = 1024
    dropout: float = 0.1
    head_size: int  = n_embd // n_head
    vocab_size: int = 50257 # gpt2的vocab_size
    layer_norm_epsilon:float = 1e-5
    data_path: str = 'c4_demo.json'
    tokenizer_path: str = 'gpt2'

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 定义三个线性层用于计算注意力
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # 注册一个三角形掩码缓冲区
        self.register_buffer('attention_mask', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()
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

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)

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

    # 注意在init方法中调用self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)