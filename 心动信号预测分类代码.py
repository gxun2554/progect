#!/usr/bin/env python
# coding: utf-8

# In[3]:


# 心动信号分类 + abs-sum 评价指标 完整代码
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 解决Matplotlib中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ===================== 1. 数据读取与解析 =====================
df = pd.read_csv("train.csv")

def parse_signal(s):
    return np.array(s.split(','), dtype=np.float32)

signals = np.array([parse_signal(s) for s in df['heartbeat_signals']])
labels = df['label'].values

# 划分训练集、测试集 8:2
X_train, X_test, y_train, y_test = train_test_split(
    signals, labels, test_size=0.2, random_state=42, stratify=labels
)

print("===== 数据集大小 =====")
print(f"训练集：{X_train.shape}")
print(f"测试集：{X_test.shape}")

# ===================== 2. 绘制四类心跳信号波形图 =====================
colors = ["blue", "orange", "green", "red"]
class_names = ["正常心跳", "室性早搏", "右束支阻滞", "房性早搏"]

plt.figure(figsize=(12, 5))
for c in range(4):
    idx = np.where(labels == c)[0][0]
    plt.plot(signals[idx], color=colors[c], label=f"类别{c}：{class_names[c]}")
plt.title("四类心跳信号波形图")
plt.legend()
plt.show()

# ===================== 3. 数据集类 =====================
class HeartDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

train_loader = DataLoader(HeartDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(HeartDataset(X_test, y_test), batch_size=64, shuffle=False)

# ===================== 4. 1D-CNN 模型 =====================
class HeartCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 51, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        self.softmax = nn.Softmax(dim=1)  # 输出转为概率

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        out = self.fc(x)
        prob = self.softmax(out)
        return out, prob

# ===================== 5. 模型训练 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HeartCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_history = []
print("\n===== 训练过程损失 =====")
for epoch in range(10):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"第 {epoch+1} 轮 训练损失：{avg_loss:.4f}")

# 绘制损失曲线
plt.figure(figsize=(10, 4))
plt.plot(loss_history)
plt.title("训练损失曲线")
plt.xlabel("训练轮数")
plt.ylabel("损失值")
plt.show()

# ===================== 6. 测试集评估：准确率 + abs-sum指标 =====================
model.eval()
all_prob = []
all_label = []
abs_sum_total = 0   # 全局abs-sum总和

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        _, prob = model(x)
        all_prob.extend(prob.cpu().numpy())
        all_label.extend(y.numpy())
        
        # 1. 标签转独热编码 [y1,y2,y3,y4]
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=4).float().cpu().numpy()
        prob_np = prob.cpu().numpy()
        
        # 2. 计算单批次绝对误差和
        batch_abs = np.sum(np.abs(y_one_hot - prob_np))
        abs_sum_total += batch_abs

# 计算分类准确率
pred_cls = np.argmax(all_prob, axis=1)
acc = accuracy_score(all_label, pred_cls)

print("\n===== 测试集评价结果 =====")
print(f"分类准确率：{acc:.4f}")
print(f"整体 abs-sum 值：{abs_sum_total:.4f}")
print("说明：abs-sum 数值越小，模型概率预测结果越接近真实分布")


# In[5]:


pip install graphviz


# In[ ]:




