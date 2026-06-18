#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple')
    


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

# ===================== 1. 配置与路径 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-chinese"  # 中文BERT预训练模型
max_len = 64
batch_size = 32
epochs = 3
lr = 2e-5

# 读取数据集
df = pd.read_csv("train_set.csv")
print("数据集总样本数：", len(df))

# 8:2 划分训练集、测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
print(f"训练集：{len(train_df)} 条，测试集：{len(test_df)} 条")

# ===================== 2. 数据可视化分析 =====================
# 类别分布
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.figure(figsize=(10,4))
train_df["label"].value_counts().sort_index().plot(kind="bar")
plt.title("新闻类别分布")
plt.xlabel("类别标签")
plt.ylabel("样本数量")
plt.show()

# 文本长度分布
train_df["text_len"] = train_df["text"].apply(lambda x: len(str(x)))
plt.figure(figsize=(10,4))
plt.hist(train_df["text_len"], bins=30)
plt.title("新闻文本长度分布")
plt.xlabel("文本长度")
plt.ylabel("数量")
plt.show()

# ===================== 3. 数据集类与分词器 =====================
tokenizer = BertTokenizer.from_pretrained(model_name)

class NewsDataset(Dataset):
    def __init__(self, df):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encode = tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encode["input_ids"].flatten()
        attention_mask = encode["attention_mask"].flatten()
        return input_ids, attention_mask, torch.tensor(label)

# 构建DataLoader
train_dataset = NewsDataset(train_df)
test_dataset = NewsDataset(test_df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ===================== 4. BERT分类模型 =====================
class BertNewsModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_feat = out.pooler_output
        cls_feat = self.dropout(cls_feat)
        logits = self.fc(cls_feat)
        return logits

# 获取类别总数
num_classes = df["label"].nunique()
model = BertNewsModel(num_classes).to(device)

# 损失函数、优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# ===================== 5. 模型训练 =====================
print("\n===== 开始训练 =====")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for input_ids, attn_mask, label in train_loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attn_mask)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"第{epoch+1}轮 训练损失：{avg_loss:.4f}")

# ===================== 6. 模型测试评估 =====================
model.eval()
all_pred = []
all_true = []
with torch.no_grad():
    for input_ids, attn_mask, label in test_loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        logits = model(input_ids, attn_mask)
        pred = torch.argmax(logits, dim=1)
        all_pred.extend(pred.cpu().numpy())
        all_true.extend(label.numpy())

acc = accuracy_score(all_true, all_pred)
print(f"\n测试集分类准确率：{acc:.4f}")


# In[4]:


pip show transformers


# In[ ]:


#!/usr/bin/env python
# coding: utf-8
"""
实验一：基于一维CNN的心动信号分类 — 图片生成脚本
"""
import os
os.chdir(r"c:\Users\27206\Desktop\机器学习作业\test")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 解决Matplotlib中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                              confusion_matrix, classification_report)

# ===================== 1. 数据读取与解析 =====================
print("1. 读取数据...")
df = pd.read_csv("train.csv")

def parse_signal(s):
    return np.array(s.split(','), dtype=np.float32)

signals = np.array([parse_signal(s) for s in df['heartbeat_signals']])
labels = df['label'].values.astype(int)

class_names = ["正常心跳", "室性早搏", "右束支阻滞", "房性早搏"]

print(f"总样本数: {len(signals)}, 信号长度: {signals.shape[1]}")
print(f"各类别分布:")
for c in range(4):
    print(f"  类别{c}({class_names[c]}): {np.sum(labels==c)}")

# 划分训练集、测试集 8:2
X_train, X_test, y_train, y_test = train_test_split(
    signals, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"\n训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

# ===================== 2. 图1: 四类心跳信号波形图 =====================
print("\n2. 绘制波形图...")
colors_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for c in range(4):
    ax = axes[c // 2, c % 2]
    # 绘制该类别的前5个样本
    idxs = np.where(y_train == c)[0][:5]
    for j, idx in enumerate(idxs):
        ax.plot(X_train[idx], alpha=0.6, linewidth=0.8,
                color=colors_palette[c], label=f"样本{j+1}" if c==0 or j>0 else "")
    ax.set_title(f"类别{c}: {class_names[c]} (n={np.sum(labels==c)})", fontsize=12)
    ax.set_xlabel("时间步")
    ax.set_ylabel("信号幅值")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
plt.suptitle("四类心跳信号波形图", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("实验一_01_波形图.png", dpi=150, bbox_inches='tight')
plt.close()
print("   -> 实验一_01_波形图.png")

# ===================== 3. 图2: 标签分布饼图 =====================
print("\n3. 绘制标签分布...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
counts = [np.sum(labels == i) for i in range(4)]
colors_bar = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
# 饼图
wedges, texts, autotexts = ax1.pie(counts, labels=class_names, colors=colors_bar,
                                     autopct='%1.1f%%', startangle=90, explode=[0.02]*4)
for at in autotexts:
    at.set_fontsize(10)
ax1.set_title("各类别样本占比", fontsize=12, fontweight='bold')
# 条形图
bars = ax2.bar(range(4), counts, color=colors_bar, edgecolor='white', linewidth=1.2)
for bar, count in zip(bars, counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             str(count), ha='center', fontsize=10, fontweight='bold')
ax2.set_xticks(range(4))
ax2.set_xticklabels(class_names)
ax2.set_ylabel("样本数量")
ax2.set_title("各类别样本数量", fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
plt.suptitle("数据集标签分布分析", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("实验一_02_标签分布.png", dpi=150, bbox_inches='tight')
plt.close()
print("   -> 实验一_02_标签分布.png")

# ===================== 4. 数据集类 =====================
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

# ===================== 5. 1D-CNN 模型 =====================
class HeartCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),           # 205 -> 102

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),           # 102 -> 51

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(25),  # 51 -> 25
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 25, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        logits = self.fc(x)
        prob = self.softmax(logits)
        return logits, prob

# ===================== 6. 模型训练 =====================
print("\n4. 开始训练模型...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   使用设备: {device}")
model = HeartCNN(num_classes=4).to(device)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses = []
test_losses = []
train_accs = []
test_accs = []

NUM_EPOCHS = 15

for epoch in range(NUM_EPOCHS):
    # 训练阶段
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    avg_loss = total_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(avg_loss)
    train_accs.append(train_acc)

    # 评估阶段
    model.eval()
    total_test_loss = 0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            total_test_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct_test += (pred == y).sum().item()
            total_test += y.size(0)
    avg_test_loss = total_test_loss / len(test_loader)
    test_acc = correct_test / total_test
    test_losses.append(avg_test_loss)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS}: Train Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, "
          f"Test Loss={avg_test_loss:.4f}, Test Acc={test_acc:.4f}")

# ===================== 7. 图3: 训练曲线 =====================
print("\n5. 绘制训练曲线...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 损失曲线
ax1.plot(range(1, NUM_EPOCHS+1), train_losses, 'o-', color='#1f77b4', linewidth=2,
         markersize=5, label='训练损失')
ax1.plot(range(1, NUM_EPOCHS+1), test_losses, 's-', color='#d62728', linewidth=2,
         markersize=5, label='测试损失')
ax1.set_xlabel('训练轮数 (Epoch)')
ax1.set_ylabel('损失值')
ax1.set_title('训练与测试损失曲线')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 准确率曲线
ax2.plot(range(1, NUM_EPOCHS+1), train_accs, 'o-', color='#1f77b4', linewidth=2,
         markersize=5, label='训练准确率')
ax2.plot(range(1, NUM_EPOCHS+1), test_accs, 's-', color='#d62728', linewidth=2,
         markersize=5, label='测试准确率')
ax2.set_xlabel('训练轮数 (Epoch)')
ax2.set_ylabel('准确率')
ax2.set_title('训练与测试准确率曲线')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('模型训练过程', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("实验一_03_训练曲线.png", dpi=150, bbox_inches='tight')
plt.close()
print("   -> 实验一_03_训练曲线.png")

# ===================== 8. 测试集评估 =====================
print("\n6. 评估模型...")
model.eval()
all_prob = []
all_label = []
all_pred = []
abs_sum_total = 0

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits, prob = model(x)
        all_prob.extend(prob.cpu().numpy())
        all_label.extend(y.numpy())
        all_pred.extend(logits.argmax(dim=1).cpu().numpy())

        y_one_hot = torch.nn.functional.one_hot(y, num_classes=4).float().cpu().numpy()
        prob_np = prob.cpu().numpy()
        abs_sum_total += np.sum(np.abs(y_one_hot - prob_np))

all_prob = np.array(all_prob)
all_label = np.array(all_label)
all_pred = np.array(all_pred)

acc = accuracy_score(all_label, all_pred)
macro_f1 = f1_score(all_label, all_pred, average='macro')
micro_f1 = f1_score(all_label, all_pred, average='micro')
weighted_f1 = f1_score(all_label, all_pred, average='weighted')
per_class_f1 = f1_score(all_label, all_pred, average=None)
per_class_precision = precision_score(all_label, all_pred, average=None)
per_class_recall = recall_score(all_label, all_pred, average=None)

print(f"\n===== 测试集评价结果 =====")
print(f"分类准确率 (Accuracy): {acc:.4f}")
print(f"Macro-F1:  {macro_f1:.4f}")
print(f"Micro-F1:  {micro_f1:.4f}")
print(f"Weighted-F1: {weighted_f1:.4f}")
print(f"整体 abs-sum 值: {abs_sum_total:.4f}")
print(f"\n各类别详细指标:")
for c in range(4):
    print(f"  类别{c}({class_names[c]}): Precision={per_class_precision[c]:.4f}, "
          f"Recall={per_class_recall[c]:.4f}, F1={per_class_f1[c]:.4f}")

# ===================== 9. 图4: 各类别P/R/F1对比 =====================
print("\n7. 绘制各类别指标对比图...")
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(4)
width = 0.25
bars1 = ax.bar(x - width, per_class_precision, width, label='精确率 (Precision)',
               color='#1f77b4', edgecolor='white')
bars2 = ax.bar(x, per_class_recall, width, label='召回率 (Recall)',
               color='#ff7f0e', edgecolor='white')
bars3 = ax.bar(x + width, per_class_f1, width, label='F1分数',
               color='#2ca02c', edgecolor='white')
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                ha='center', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([f"类别{c}\n{class_names[c]}" for c in range(4)])
ax.set_ylabel("分数")
ax.set_title("各类别 Precision / Recall / F1 对比")
ax.legend(loc='lower right')
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("实验一_04_PRF1对比.png", dpi=150, bbox_inches='tight')
plt.close()
print("   -> 实验一_04_PRF1对比.png")

# ===================== 10. 图5: 混淆矩阵 =====================
print("\n8. 绘制混淆矩阵...")
cm = confusion_matrix(all_label, all_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
# 原始计数混淆矩阵
im1 = ax1.imshow(cm, cmap='Blues', aspect='auto')
for i in range(4):
    for j in range(4):
        ax1.text(j, i, str(cm[i, j]), ha='center', va='center',
                 fontsize=12, fontweight='bold',
                 color='white' if cm[i, j] > cm.max()/2 else 'black')
ax1.set_xticks(range(4))
ax1.set_yticks(range(4))
ax1.set_xticklabels(class_names)
ax1.set_yticklabels(class_names)
ax1.set_xlabel('预测类别')
ax1.set_ylabel('真实类别')
ax1.set_title('混淆矩阵（样本数）', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=ax1, shrink=0.8)

# 归一化混淆矩阵
im2 = ax2.imshow(cm_norm, cmap='Greens', aspect='auto', vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax2.text(j, i, f'{cm_norm[i, j]:.3f}', ha='center', va='center',
                 fontsize=11, fontweight='bold',
                 color='white' if cm_norm[i, j] > 0.5 else 'black')
ax2.set_xticks(range(4))
ax2.set_yticks(range(4))
ax2.set_xticklabels(class_names)
ax2.set_yticklabels(class_names)
ax2.set_xlabel('预测类别')
ax2.set_ylabel('真实类别')
ax2.set_title('归一化混淆矩阵（召回率）', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=ax2, shrink=0.8)

plt.suptitle('测试集混淆矩阵', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("实验一_05_混淆矩阵.png", dpi=150, bbox_inches='tight')
plt.close()
print("   -> 实验一_05_混淆矩阵.png")

# ===================== 11. 图6: F1与样本数量关系 =====================
print("\n9. 绘制F1-样本数关系图...")
fig, ax1 = plt.subplots(figsize=(10, 6))
counts_arr = np.array(counts)
ax2 = ax1.twinx()
bars = ax1.bar(range(4), counts_arr, color=colors_bar, alpha=0.6, edgecolor='white', label='样本数量')
ax2.plot(range(4), per_class_f1, 'o-', color='red', linewidth=3, markersize=12, label='F1分数')
for c in range(4):
    ax2.annotate(f'F1={per_class_f1[c]:.3f}', (c, per_class_f1[c]),
                 textcoords="offset points", xytext=(0, 18),
                 ha='center', fontsize=10, fontweight='bold', color='red')
ax1.set_xticks(range(4))
ax1.set_xticklabels([f"类别{c}\n{class_names[c]}" for c in range(4)])
ax1.set_ylabel('样本数量', color='steelblue')
ax2.set_ylabel('F1分数', color='red')
ax1.set_title('各类别训练样本数量与分类F1分数的关系')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("实验一_06_F1与样本关系.png", dpi=150, bbox_inches='tight')
plt.close()
print("   -> 实验一_06_F1与样本关系.png")

# ===================== 12. 图7: 模型预测概率分布（Abs-Sum可视化）=====================
print("\n10. 绘制Abs-Sum分析图...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for c in range(4):
    ax = axes[c // 2, c % 2]
    class_mask = all_label == c
    class_probs = all_prob[class_mask]
    if len(class_probs) > 20:
        class_probs = class_probs[:20]
    x_pos = np.arange(len(class_probs))
    bottom = np.zeros(len(class_probs))
    for k in range(4):
        ax.bar(x_pos, class_probs[:, k], bottom=bottom,
               color=colors_palette[k], alpha=0.85,
               label=f'{class_names[k]}' if k == 0 else (f'{class_names[k]}' if True else ''))
        bottom += class_probs[:, k]
    ax.set_title(f'真实类别{c}: {class_names[c]} (前20个样本)')
    ax.set_xlabel('样本序号')
    ax.set_ylabel('预测概率')
    ax.set_ylim(0, 1.05)
    if c == 0:
        ax.legend(loc='upper right', fontsize=8)

plt.suptitle('模型预测概率分布可视化', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("实验一_07_概率分布.png", dpi=150, bbox_inches='tight')
plt.close()
print("   -> 实验一_07_概率分布.png")

# ===================== 13. 图8: Abs-Sum逐类分析 =====================
print("\n11. 绘制Abs-Sum逐类分析...")
per_class_abs_sum = []
for c in range(4):
    class_mask = all_label == c
    y_one_hot = np.zeros((class_mask.sum(), 4))
    y_one_hot[np.arange(class_mask.sum()), np.full(class_mask.sum(), c)] = 1
    class_abs = np.sum(np.abs(y_one_hot - all_prob[class_mask]))
    per_class_abs_sum.append(class_abs)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(4), per_class_abs_sum, color=colors_bar, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, per_class_abs_sum):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')
ax.set_xticks(range(4))
ax.set_xticklabels(class_names)
ax.set_ylabel('Abs-Sum值')
ax.set_title('各类别 Abs-Sum 指标对比')
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=abs_sum_total/4, color='red', linestyle='--',
           label=f'类别平均: {abs_sum_total/4:.1f}')
ax.legend()
plt.tight_layout()
plt.savefig("实验一_08_AbsSum分析.png", dpi=150, bbox_inches='tight')
plt.close()
print("   -> 实验一_08_AbsSum分析.png")

print("\n===== 所有图片生成完毕! =====")
print(f"最终测试准确率: {acc:.4f}")
print(f"最终Macro-F1: {macro_f1:.4f}")
print(f"最终Abs-Sum: {abs_sum_total:.2f}")


# In[ ]:




