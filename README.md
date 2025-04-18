# GPT-2预训练示例项目

这是一个基于PyTorch实现的GPT-2预训练示例项目，旨在帮助学习者理解GPT-2模型的实现原理和预训练过程。

## 项目特点

- 完整的GPT-2模型实现
- 可以不使用GPU，因为这只是一个示例，主要学习模型结构和预训练处理，训练集很少
- 包含详细的代码注释

## 环境要求

- 使用miniconda创建环境
```bash
conda create -n gpt2 python=3.10
conda activate gpt2
```
- 安装pytorch。如果使用GPU，这里推荐使用阿里源来安装,将`index-url`按照下面的替换方式进行替换即可（注意是pip的安装方式）
```bash
--index-url https://mirrors.aliyun.com/pytorch-wheels/原链接末尾版本号
```

- 安装transformers
```bash
pip install transformers -i https://mirrors.aliyun.com/pypi/simple
```

## 项目结构

```
PreTrainGPT2Demo/
├── model.py          # GPT-2模型实现
├── train.py          # 训练脚本
├── dataset.py        # 数据集处理
├── c4_demo.json      # 示例数据集
└── gpt2/             # tokenizer目录
```

## 使用方法

1. 准备数据：
   - 将训练数据放在`c4_demo.json`中
   - 数据格式应为JSON，每个条目包含"text"字段

2. 开始训练：
```bash
python train.py
```

3. 训练参数配置：
   - 在`model.py`中的`GPTConfig`类中修改配置参数
   - 主要参数包括：
     - block_size: 序列长度
     - batch_size: 批次大小
     - n_layer: 层数
     - n_head: 注意力头数
     - n_embd: 嵌入维度
