from model import GPT2,GPTConfig
from dataset import PreTrainDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os

torch.manual_seed(1024)
config = GPTConfig()
model = GPT2(config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# 打印模型参数
total_params = sum(p.numel()for p in model.parameters())
print(f'Total parameters: {total_params / 1e6}M')

optimizer = optim.AdamW(model.parameters(), lr=3e-4)
# cosine学习率
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

train_dataset = PreTrainDataset(config)
train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[0.9,0.1])

train_dataloader = DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=config.batch_size,shuffle=False)

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
        total_loss += loss.item()
        if idx % 100 == 0:
            print(f'Epoch {epoch}, batch % 100 : {idx}, Loss: {loss.item()}')
    return total_loss

def evaluate(model,val_dataloader,device):
    model.eval() # dropout不起作用
    total_loss = 0
    with torch.no_grad(): # 不计算梯度
        for x,y in val_dataloader:
            x = x.to(device)
            y = y.to(device)
            _,loss = model(x,targets=y)
            total_loss += loss.item()
    return total_loss
epoches = config.epoches
for epoch in range(epoches):
    train_loss = train(model,optimizer,scheduler,train_dataloader,device,epoch)
    val_loss = evaluate(model,val_dataloader,device)
    print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')

    avg_val_loss = val_loss / len(val_dataloader)
    # 保存模型
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_val_loss
    }

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    torch.save(checkpoint,f'checkpoints/model_epoch_{epoch}.pt')
