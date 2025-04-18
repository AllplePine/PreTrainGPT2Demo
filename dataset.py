from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
import json

class PreTrainDataset(Dataset):
    def __init__(self, config):
        self.data_path = config.data_path
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        self.eot = '<|endoftext|>'
        self.encoded_data = []
        full_enconded_data = []
        with open(self.data_path, 'r',encoding='utf-8') as f:
            text = json.load(f)
            for item in text:
                _text =item['text'].strip().replace('<s>',"").replace('<\s>',"") # 针对使用的数据集进行的特殊处理，正常情况下应该另外单独写一个脚本处理
                _text = _text+self.eot
                full_enconded_data.extend(self.tokenizer.encode(_text))
        for i in range(0,len(full_enconded_data),config.block_size):
            if i+config.block_size <= len(full_enconded_data): # 最后不足一个block_size的直接舍弃
                # 按照最大的block_size进行截断，生成一个一个的block
                self.encoded_data.append(full_enconded_data[i:i+config.block_size])
        self.encoded_data = torch.tensor(self.encoded_data)

    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        return self.encoded_data[idx],self.encoded_data[idx]