# progect
新闻文本分类实验报告
新闻文本分类实验报
#!/usr/bin/env python
# coding: utf-8
# ==============================================================================
# 实验二：新闻文本分类 —— 基于 Transformer 编码器的深度学习方法
# 机器学习实验指导书 2026 | 环境：Anaconda3 + Jupyter Notebook | 框架：PyTorch
# 包含：文本分析 + 交叉熵损失分析 + F1/精确率/召回率评估 + 全部结果图
# ==============================================================================

# In[1]: 导入所有依赖库
import pandas as pd
import numpy as np
import csv, sys
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 增大 CSV 字段上限（文本序列较长）
csv.field_size_limit(2147483647)

import matplotlib
matplotlib.use('Agg')   # 非交互后端，无需 GUI
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
_HAS_PLT = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix,
                              precision_recall_curve, average_precision_score)
from scipy.stats import pearsonr

# 固定随机种子（保证可复现）
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"计算设备: {DEVICE}")

# ==============================================================================
# 第 1 部分：文本数据读取与基础分析
# ==============================================================================
print("\n" + "=" * 65)
print(" 第 1 部分：文本数据读取与基础分析")
print("=" * 65)

# ★ 小批量测试模式：设置 SAMPLE_SIZE 以限制数据量（None = 全量）
SAMPLE_SIZE = 5000  # 设为 None 使用全量 200,000 条；设数字做快速测试
df = pd.read_csv("train_set.csv", sep="\t", nrows=SAMPLE_SIZE)
if SAMPLE_SIZE:
    print(f"[FAST TEST MODE] Using first {SAMPLE_SIZE:,} samples")
print(f"数据集大小: {df.shape[0]} 条 × {df.shape[1]} 列")
print(f"列名: label (标签), text (空格分隔的词索引)")

# ---- 1.1 标签分布分析 ----
label_counts = df['label'].value_counts().sort_index()
NUM_CLASSES = len(label_counts)
print(f"\n类别总数: {NUM_CLASSES}")
print(f"各类别样本数:")
for k, v in label_counts.items():
    print(f"  类别 {k:>2}:  {v:>8} 条  ({v / len(df) * 100:5.1f}%)")

# 标签分布双图（柱状 + 饼图）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
colors_14 = plt.cm.tab20(np.linspace(0, 1, NUM_CLASSES))

bars = ax1.bar(label_counts.index, label_counts.values, color=colors_14, edgecolor='white')
ax1.set_title("各类别样本数量分布", fontsize=13, fontweight='bold')
ax1.set_xlabel("类别标签"); ax1.set_ylabel("样本数量")
for bar, val in zip(bars, label_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
             str(val), ha='center', fontsize=7)

wedges, texts, autotexts = ax2.pie(label_counts.values, labels=label_counts.index,
                                     autopct='%1.1f%%', colors=colors_14,
                                     pctdistance=0.82, textprops={'fontsize': 8})
ax2.set_title("各类别占比", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("实验二_01_标签分布.png", dpi=150, bbox_inches='tight')
plt.show()

# ---- 1.2 文本长度分析 ----
df['text_len'] = df['text'].apply(lambda x: len(str(x).split()))
print(f"\n文本长度统计:")
print(f"  最大: {df['text_len'].max():,}  最小: {df['text_len'].min()}  "
      f"均值: {df['text_len'].mean():.1f}  中位数: {df['text_len'].median():.1f}")
print("分位数:", {p: f"{np.percentile(df['text_len'], p):.0f}" for p in [50, 75, 90, 95, 99]})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
ax1.hist(df['text_len'], bins=120, color='steelblue', edgecolor='white', alpha=0.85)
ax1.axvline(df['text_len'].mean(), color='red', linestyle='--', linewidth=2,
            label=f"均值={df['text_len'].mean():.0f}")
ax1.axvline(df['text_len'].median(), color='orange', linestyle='--', linewidth=2,
            label=f"中位数={df['text_len'].median():.0f}")
ax1.set_title("文本长度分布直方图"); ax1.set_xlabel("文本长度 (词数)"); ax1.set_ylabel("样本数")
ax1.legend()

ax2.hist(df['text_len'], bins=120, color='coral', edgecolor='white', alpha=0.85, range=(0, 3000))
ax2.axvline(df['text_len'].mean(), color='red', linestyle='--', linewidth=2)
ax2.axvline(df['text_len'].median(), color='orange', linestyle='--', linewidth=2)
ax2.set_title("文本长度分布 (0-3000 词)"); ax2.set_xlabel("文本长度 (词数)"); ax2.set_ylabel("样本数")
plt.tight_layout()
plt.savefig("实验二_02_文本长度分布.png", dpi=150, bbox_inches='tight')
plt.show()

# ---- 1.3 词汇表分析 ----
all_words = []
for text in df['text']:
    all_words.extend(str(text).split())
vocab_counter = Counter(all_words)
VOCAB_SIZE = len(vocab_counter)
print(f"\n词汇表大小: {VOCAB_SIZE:,}  总词数: {len(all_words):,}")
print(f"Top-20 高频词: {vocab_counter.most_common(20)}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
freqs = sorted(vocab_counter.values(), reverse=True)
ax1.loglog(range(1, len(freqs) + 1), freqs, linewidth=1, color='steelblue')
ax1.set_title("词频-排名分布 (双对数)", fontweight='bold')
ax1.set_xlabel("排名"); ax1.set_ylabel("词频"); ax1.grid(True, alpha=0.3)

top_n = 30
words_top, freqs_top = zip(*vocab_counter.most_common(top_n))
ax2.barh(range(top_n), freqs_top, color=plt.cm.viridis(np.linspace(0.2, 0.9, top_n)), edgecolor='white')
ax2.set_yticks(range(top_n))
ax2.set_yticklabels(words_top, fontsize=7)
ax2.invert_yaxis()
ax2.set_title(f"Top-{top_n} 高频词索引", fontweight='bold')
ax2.set_xlabel("出现次数")
plt.tight_layout()
plt.savefig("实验二_03_词频分析.png", dpi=150, bbox_inches='tight')
plt.show()

# ==============================================================================
# 第 2 部分：数据预处理
# ==============================================================================
print("\n" + "=" * 65)
print(" 第 2 部分：数据预处理")
print("=" * 65)

# ---- 超参数（根据 SAMPLE_SIZE 自动切换全量/测试模式） ----
if SAMPLE_SIZE:
    MAX_SEQ_LEN  = 256    # 测试模式：序列截断长度
    EMB_DIM      = 64     # 测试模式：词嵌入维度
    NUM_HEADS    = 4      # 测试模式：注意力头数
    NUM_LAYERS   = 2      # 测试模式：Transformer 层数
    FFN_DIM      = 128    # 测试模式：前馈网络维度
    DROPOUT      = 0.2    # 测试模式：Dropout
    BATCH_SIZE   = 32     # 测试模式：批量大小
    LEARNING_RATE = 1e-3  # 测试模式：学习率
    NUM_EPOCHS   = 3      # 测试模式：训练轮数
else:
    MAX_SEQ_LEN  = 512    # 序列截断/填充长度
    EMB_DIM      = 256    # 词嵌入维度
    NUM_HEADS    = 8      # 多头注意力头数
    NUM_LAYERS   = 4      # Transformer 层数
    FFN_DIM      = 512    # 前馈网络隐藏层维度
    DROPOUT      = 0.3    # Dropout 比率
    BATCH_SIZE   = 64     # 批量大小
    LEARNING_RATE = 1e-4  # 学习率
    NUM_EPOCHS   = 10     # 训练轮数
VOCAB_SIZE   = 7000   # 词汇表大小（保留余量）

# ---- 文本→固定长度序列 ----
def parse_and_pad(text_str, max_len=MAX_SEQ_LEN):
    ids = [int(x) for x in str(text_str).split()]
    # 裁剪词索引到 [0, VOCAB_SIZE-1] 防止越界
    ids = [min(i, VOCAB_SIZE - 1) for i in ids]
    if len(ids) > max_len:
        return ids[:max_len]
    return ids + [0] * (max_len - len(ids))

texts_padded = np.array([parse_and_pad(t) for t in df['text']])
labels       = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    texts_padded, labels, test_size=0.2, random_state=42, stratify=labels)

class_counts_train = np.bincount(y_train)
print(f"训练集: {X_train.shape[0]:,} 样本   测试集: {X_test.shape[0]:,} 样本")

# ---- DataLoader ----
class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i], self.labels[i]

train_loader = DataLoader(NewsDataset(X_train, y_train), batch_size=BATCH_SIZE,
                           shuffle=True,  num_workers=0)
test_loader  = DataLoader(NewsDataset(X_test,  y_test),  batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=0)

# ==============================================================================
# 第 3 部分：Transformer 文本分类模型
# ==============================================================================
print("\n" + "=" * 65)
print(" 第 3 部分：模型构建 —— Transformer 文本分类器")
print("=" * 65)

class PositionalEncoding(nn.Module):
    """正弦-余弦位置编码"""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class TransformerTextClassifier(nn.Module):
    """基于 Transformer 编码器的文本分类模型"""
    def __init__(self, vocab_size, emb_dim, num_heads, num_layers,
                 ffn_dim, num_classes, max_seq_len, dropout=0.3):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(emb_dim, max_seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(emb_dim // 2, num_classes))

    def forward(self, x):
        pad_mask = (x == 0)
        x = self.embedding(x) * np.sqrt(self.emb_dim)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        # 带掩码的全局平均池化
        mask_f = (~pad_mask).unsqueeze(-1).float()
        x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        x = self.ln(x)
        x = self.dropout(x)
        return self.classifier(x)

model = TransformerTextClassifier(VOCAB_SIZE, EMB_DIM, NUM_HEADS, NUM_LAYERS,
                                   FFN_DIM, NUM_CLASSES, MAX_SEQ_LEN, DROPOUT)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")
print(f"模型结构:\n{model}")

# ==============================================================================
# 第 4 部分：交叉熵损失函数 —— 原理分析 + 可视化 + 手动验证
# ==============================================================================
print("\n" + "=" * 65)
print(" 第 4 部分：交叉熵损失函数详细分析")
print("=" * 65)

# ---- 4.1 类别权重计算 ----
class_weights_raw = 1.0 / (class_counts_train + 1e-6)
class_weights_tensor = torch.tensor(
    class_weights_raw / class_weights_raw.sum() * NUM_CLASSES,
    dtype=torch.float).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

print("损失函数: 加权交叉熵 CrossEntropyLoss")
print("各类别权重 (w_c):")
for c in range(NUM_CLASSES):
    print(f"  类别 {c:>2}: 样本 {class_counts_train[c]:>6}  →  权重 {class_weights_tensor[c].item():.4f}")

# ---- 4.2 权重 vs 样本数可视化 ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(NUM_CLASSES)
w = 0.35

b1 = ax1.bar(x - w/2, class_counts_train, w, color='steelblue', label='训练样本数', edgecolor='white')
ax1b = ax1.twinx()
b2 = ax1b.bar(x + w/2, class_weights_tensor.cpu().numpy(), w, color='coral',
              label='损失权重', edgecolor='white')
ax1.set_title("样本数 vs 损失权重", fontweight='bold')
ax1.set_xlabel("类别"); ax1.set_ylabel("样本数", color='steelblue')
ax1b.set_ylabel("权重", color='coral')
ax1.set_xticks(x)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

ax2.bar(x, class_weights_tensor.cpu().numpy(), color='coral', edgecolor='darkred')
for i, wv in enumerate(class_weights_tensor.cpu().numpy()):
    ax2.text(i, wv + 0.02, f'{wv:.3f}', ha='center', fontsize=7)
ax2.axhline(1.0, color='blue', linestyle='--', linewidth=1.5, label='无加权 (w=1)')
ax2.set_title("各类别损失权重", fontweight='bold'); ax2.set_xlabel("类别"); ax2.set_ylabel("权重")
ax2.set_xticks(x); ax2.legend()
plt.tight_layout()
plt.savefig("实验二_04_交叉熵权重分析.png", dpi=150, bbox_inches='tight')
plt.show()

# ---- 4.3 手动计算交叉熵 (验证 PyTorch 实现) ----
print("\n--- 手动交叉熵计算验证 ---")
model.eval()
demo_x = torch.tensor(X_train[:8], dtype=torch.long).to(DEVICE)
demo_y = torch.tensor(y_train[:8], dtype=torch.long).to(DEVICE)
with torch.no_grad():
    demo_logits = model(demo_x)
    demo_probs  = F.softmax(demo_logits, dim=1)

# PyTorch 内置
loss_builtin = criterion(demo_logits, demo_y)
# 手动计算
nll = -torch.log(demo_probs[range(8), demo_y] + 1e-10)
loss_manual_unweighted = nll.mean()
loss_manual_weighted   = (nll * class_weights_tensor[demo_y]).mean()

print(f"  PyTorch CE (加权):       {loss_builtin.item():.6f}")
print(f"  手动计算 (加权):         {loss_manual_weighted.item():.6f}")
print(f"  手动计算 (无加权):       {loss_manual_unweighted.item():.6f}")
print(f"  [OK] Both match" if abs(loss_builtin.item() - loss_manual_weighted.item()) < 1e-5
      else "  [FAIL] Mismatch - check!")

# ---- 4.4 Softmax 概率与 NLL 可视化 ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# 左：第一个样本的预测概率分布
prob0 = demo_probs[0].cpu().numpy()
bar_colors = []
for i in range(NUM_CLASSES):
    if i == demo_y[0].item():
        bar_colors.append('#2ecc71')       # 真实类别 → 绿色
    elif i == prob0.argmax() and i != demo_y[0].item():
        bar_colors.append('#e74c3c')       # 错误 top-1 → 红色
    else:
        bar_colors.append('#bdc3c7')       # 其他 → 灰色

ax1.bar(range(NUM_CLASSES), prob0, color=bar_colors, edgecolor='white')
for i, p in enumerate(prob0):
    if p > 0.03:
        ax1.text(i, p + 0.01, f'{p:.3f}', ha='center', fontsize=6)
ax1.set_title(f"样本1 预测概率分布 (真实类别={demo_y[0].item()})", fontweight='bold')
ax1.set_xlabel("类别"); ax1.set_ylabel("Softmax 概率"); ax1.set_xticks(range(NUM_CLASSES))
try:
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(color='#2ecc71', label='真实类别'),
        Patch(color='#e74c3c', label='错误Top-1预测'),
        Patch(color='#bdc3c7', label='其他')], fontsize=8)
except ImportError:
    pass

# 右：负对数似然曲线
p_range = np.linspace(0.01, 0.99, 200)
ax2.plot(p_range, -np.log(p_range), 'b-', linewidth=2.5)
ax2.fill_between(p_range, 0, -np.log(p_range), alpha=0.15)
nll_vals = nll.cpu().numpy()
correct_probs = demo_probs[range(8), demo_y].cpu().numpy()
ax2.scatter(correct_probs, nll_vals, c='red', s=120, zorder=5, edgecolors='darkred', linewidth=1.5)
for i, (px, py) in enumerate(zip(correct_probs, nll_vals)):
    ax2.annotate(f'样{i+1}', (px, py), textcoords="offset points", xytext=(10, -10),
                 fontsize=8, color='darkred', fontweight='bold')
ax2.set_title("负对数似然函数 L = -log(p_correct)", fontweight='bold')
ax2.set_xlabel("正确类别的预测概率 p"); ax2.set_ylabel("损失值"); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("实验二_05_交叉熵损失原理.png", dpi=150, bbox_inches='tight')
plt.show()

# ==============================================================================
# 第 5 部分：训练 + 逐batch损失追踪
# ==============================================================================
print("\n" + "=" * 65)
print(" 第 5 部分：模型训练")
print("=" * 65)

# 评估函数（返回精度/召回/F1 的各类别值）
@torch.no_grad()
def evaluate_full(model, loader):
    model.eval()
    all_loss, all_preds, all_labels, all_probs = 0, [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        all_loss += criterion(logits, yb).item() * xb.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(yb.cpu().numpy())
        all_probs.extend(F.softmax(logits, dim=1).cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    return {
        'loss':     all_loss / len(all_labels),
        'acc':      accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_micro': f1_score(all_labels, all_preds, average='micro'),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'precision_per_class': precision_score(all_labels, all_preds, average=None,
                                                labels=list(range(NUM_CLASSES))),
        'recall_per_class':    recall_score(all_labels, all_preds, average=None,
                                             labels=list(range(NUM_CLASSES))),
        'f1_per_class':        f1_score(all_labels, all_preds, average=None,
                                         labels=list(range(NUM_CLASSES))),
        'preds': all_preds, 'labels': all_labels, 'probs': all_probs
    }

# 训练循环
history = {
    'train_loss': [], 'test_loss': [], 'test_acc': [],
    'test_f1_macro': [], 'test_f1_micro': [], 'test_f1_weighted': [],
    'test_f1_per_class': [[] for _ in range(NUM_CLASSES)],
    'test_precision_per_class': [[] for _ in range(NUM_CLASSES)],
    'test_recall_per_class':    [[] for _ in range(NUM_CLASSES)]
}
batch_losses_first, batch_losses_last = [], []
best_f1, best_state = 0, None

print(f"\n开始训练 ({NUM_EPOCHS} 轮) ... 请耐心等待 ...\n")

for epoch in range(NUM_EPOCHS):
    # ---- 训练阶段 ----
    model.train()
    epoch_loss, epoch_correct, epoch_total = 0, 0, 0
    batch_losses_curr = []

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        lv = loss.item()
        epoch_loss += lv
        batch_losses_curr.append(lv)
        epoch_correct += (logits.argmax(1) == yb).sum().item()
        epoch_total   += yb.size(0)

    train_loss = epoch_loss / len(train_loader)
    train_acc  = epoch_correct / epoch_total
    scheduler.step()

    if epoch == 0:
        batch_losses_first = batch_losses_curr.copy()
    if epoch == NUM_EPOCHS - 1:
        batch_losses_last = batch_losses_curr.copy()

    # ---- 测试阶段 ----
    result = evaluate_full(model, test_loader)
    history['train_loss'].append(train_loss)
    history['test_loss'].append(result['loss'])
    history['test_acc'].append(result['acc'])
    history['test_f1_macro'].append(result['f1_macro'])
    history['test_f1_micro'].append(result['f1_micro'])
    history['test_f1_weighted'].append(result['f1_weighted'])
    for c in range(NUM_CLASSES):
        history['test_f1_per_class'][c].append(result['f1_per_class'][c])
        history['test_precision_per_class'][c].append(result['precision_per_class'][c])
        history['test_recall_per_class'][c].append(result['recall_per_class'][c])

    print(f"Epoch [{epoch+1:>2}/{NUM_EPOCHS}]  "
          f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
          f"Test Loss: {result['loss']:.4f}  Test Acc: {result['acc']:.4f}  "
          f"F1-Macro: {result['f1_macro']:.4f}")

    if result['f1_macro'] > best_f1:
        best_f1 = result['f1_macro']
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  >>> ** NEW BEST MODEL ** F1-Macro = {best_f1:.4f}")

model.load_state_dict(best_state)
print(f"\n训练完成!  最佳 Macro-F1: {best_f1:.4f}")

# ---- 训练曲线 ----
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
epochs_r = range(1, NUM_EPOCHS + 1)

axes[0].plot(epochs_r, history['train_loss'], 'b-o', ms=6, label='Train', linewidth=2)
axes[0].plot(epochs_r, history['test_loss'],  'r-s', ms=6, label='Test',  linewidth=2)
axes[0].set_title("损失曲线", fontweight='bold'); axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_r, history['test_acc'], 'g-D', ms=6, linewidth=2)
axes[1].set_title("测试准确率", fontweight='bold'); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs_r, history['test_f1_macro'],   'o-', ms=6, label='Macro-F1',   linewidth=2)
axes[2].plot(epochs_r, history['test_f1_micro'],   's-', ms=6, label='Micro-F1',   linewidth=2)
axes[2].plot(epochs_r, history['test_f1_weighted'],'D-', ms=6, label='Weighted-F1', linewidth=2)
axes[2].set_title("F1 曲线", fontweight='bold'); axes[2].set_xlabel("Epoch")
axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

# 泛化差距
gap = np.array(history['test_loss']) - np.array(history['train_loss'])
axes[3].fill_between(epochs_r, 0, gap, alpha=0.3, color='red' if np.mean(gap) > 0 else 'green')
axes[3].plot(epochs_r, gap, 'o-', color='darkred', ms=8, linewidth=2)
axes[3].axhline(0, color='black', linewidth=1)
axes[3].set_title("泛化差距 (Test-Train Loss)", fontweight='bold'); axes[3].set_xlabel("Epoch")
for ep, g in zip(epochs_r, gap):
    axes[3].annotate(f'{g:.4f}', (ep, g), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=7)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("实验二_06_训练曲线.png", dpi=150, bbox_inches='tight')
plt.show()

# ---- 逐Batch损失收敛 ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
ax1.plot(batch_losses_first, alpha=0.7, linewidth=0.6, color='steelblue',
         label=f'Epoch 1 (均值={np.mean(batch_losses_first):.4f})')
ax1.plot(batch_losses_last,  alpha=0.7, linewidth=0.6, color='coral',
         label=f'Epoch {NUM_EPOCHS} (均值={np.mean(batch_losses_last):.4f})')
ax1.set_title("首轮 vs 末轮 逐Batch损失", fontweight='bold')
ax1.set_xlabel("Batch序号"); ax1.set_ylabel("损失值"); ax1.legend(); ax1.grid(True, alpha=0.3)

win = max(len(batch_losses_first) // 30, 1)
sma = np.convolve(batch_losses_first, np.ones(win)/win, mode='valid')
ax2.plot(sma, 'b-', linewidth=1.8)
ax2.axhline(np.mean(batch_losses_first), color='red', linestyle='--', linewidth=2,
            label=f'均值={np.mean(batch_losses_first):.4f}')
ax2.axhline(np.median(batch_losses_first), color='green', linestyle='--', linewidth=2,
            label=f'中位数={np.median(batch_losses_first):.4f}')
ax2.set_title(f"Epoch 1 损失平滑 (窗口={win})", fontweight='bold')
ax2.set_xlabel("Batch序号 (滑动后)"); ax2.set_ylabel("损失值"); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("实验二_07_逐批损失收敛.png", dpi=150, bbox_inches='tight')
plt.show()

# ==============================================================================
# 第 6 部分：F1 / 精确率 / 召回率 详细评估
# ==============================================================================
print("\n" + "=" * 65)
print(" 第 6 部分：F1 / 精确率 (Precision) / 召回率 (Recall) 详细评估")
print("=" * 65)

# 最佳模型最终评估
final = evaluate_full(model, test_loader)
all_labels = final['labels']
all_preds  = final['preds']
all_probs  = final['probs']

prec_per_class = final['precision_per_class']
rec_per_class  = final['recall_per_class']
f1_per_class   = final['f1_per_class']

# ---- 6.1 打印完整的 P/R/F1 表格 ----
print(f"""
{'='*75}
            精确率 (Precision) / 召回率 (Recall) / F1 值  完整报告
{'='*75}
{'类别':<6} {'样本数':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}
{'-'*75}""")
for c in range(NUM_CLASSES):
    print(f"类别 {c:<3}  {class_counts_train[c]:<8}  {prec_per_class[c]:<12.4f}  "
          f"{rec_per_class[c]:<12.4f}  {f1_per_class[c]:<12.4f}")
print(f"{'-'*75}")
print(f"  Macro-F1    = {final['f1_macro']:.4f}  (各类别F1算术平均)")
print(f"  Micro-F1    = {final['f1_micro']:.4f}  (总体TP/FP/FN汇总)")
print(f"  Weighted-F1 = {final['f1_weighted']:.4f}  (按样本数加权)")
print(f"  Accuracy    = {final['acc']:.4f}")
print(f"{'='*75}")

#  sklearn 分类报告
print("\n--- sklearn classification_report ---")
print(classification_report(all_labels, all_preds,
      target_names=[f"类别{i}" for i in range(NUM_CLASSES)], digits=4))

# ---- 6.2 P/R/F1 柱状图 ----
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, vals, title, cmap_name in [
    (axes[0], prec_per_class, '各类别 精确率 (Precision)', 'Blues'),
    (axes[1], rec_per_class,  '各类别 召回率 (Recall)',    'Oranges'),
    (axes[2], f1_per_class,   '各类别 F1 分数',           'Greens')]:
    colors_bar = plt.cm.get_cmap(cmap_name)(np.linspace(0.4, 0.95, NUM_CLASSES))
    bars = ax.bar(range(NUM_CLASSES), vals, color=colors_bar, edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.3f}', ha='center', fontsize=7)
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.set_xlabel("类别标签"); ax.set_ylim(0, 1.08)
    ax.set_xticks(range(NUM_CLASSES)); ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("实验二_08_Precision_Recall_F1_各类别.png", dpi=150, bbox_inches='tight')
plt.show()

# ---- 6.3 F1 三种聚合方式对比 ----
fig, ax = plt.subplots(figsize=(6, 5))
f1_agg = {'Macro': final['f1_macro'], 'Micro': final['f1_micro'], 'Weighted': final['f1_weighted']}
bars = ax.bar(f1_agg.keys(), f1_agg.values(), color=['#e74c3c', '#2ecc71', '#3498db'], edgecolor='white', width=0.5)
for bar, (k, v) in zip(bars, f1_agg.items()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{v:.4f}', ha='center', fontsize=14, fontweight='bold')
ax.set_title("F1 三种聚合方式对比", fontweight='bold', fontsize=13)
ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.08); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("实验二_09_F1聚合对比.png", dpi=150, bbox_inches='tight')
plt.show()

# ---- 6.4 Precision/Recall 散点图 ----
fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(prec_per_class, rec_per_class, c=range(NUM_CLASSES),
                cmap='tab10', s=200, edgecolors='black', linewidth=1.5, zorder=5)
for i in range(NUM_CLASSES):
    ax.annotate(f'类别{i}', (prec_per_class[i], rec_per_class[i]),
                textcoords="offset points", xytext=(8, 5), fontsize=9, fontweight='bold')
# F1 等值线
for f1_val in np.arange(0.1, 1.0, 0.1):
    p_vals = np.linspace(f1_val / (2 - f1_val) + 0.001, 1, 200)
    r_vals = f1_val * p_vals / (2 * p_vals - f1_val + 1e-10)
    valid = (r_vals > 0) & (r_vals <= 1)
    ax.plot(p_vals[valid], r_vals[valid], 'gray', alpha=0.2, linewidth=0.5)
    # 标注
    mid_idx = len(p_vals[valid]) // 2
    if mid_idx < len(p_vals[valid]):
        ax.annotate(f'F1={f1_val:.1f}', (p_vals[valid][mid_idx], r_vals[valid][mid_idx]),
                    alpha=0.35, fontsize=6, rotation=45)

ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("精确率 (Precision)", fontsize=12)
ax.set_ylabel("召回率 (Recall)", fontsize=12)
ax.set_title("各类别 Precision-Recall 分布 (含F1等值线)", fontweight='bold', fontsize=13)
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("类别标签", fontsize=10)
plt.tight_layout()
plt.savefig("实验二_10_Precision_Recall_散点.png", dpi=150, bbox_inches='tight')
plt.show()

# ---- 6.5 各类别 Precision-Recall 曲线 ----
fig, axes = plt.subplots(2, 7, figsize=(24, 9))
axes = axes.flatten()
y_onehot = np.eye(NUM_CLASSES)[all_labels]
for c in range(NUM_CLASSES):
    ax = axes[c]
    p, r, _ = precision_recall_curve(y_onehot[:, c], all_probs[:, c])
    ap = average_precision_score(y_onehot[:, c], all_probs[:, c])
    ax.plot(r, p, 'b-', linewidth=1.8, label=f'AP={ap:.3f}')
    ax.fill_between(r, p, alpha=0.12, color='blue')
    ax.set_title(f'类别 {c}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Recall', fontsize=8); ax.set_ylabel('Precision', fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc='lower left'); ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
for c in range(NUM_CLASSES, len(axes)):
    axes[c].set_visible(False)
plt.suptitle("各类别 Precision-Recall 曲线", fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("实验二_11_Precision_Recall_曲线.png", dpi=150, bbox_inches='tight')
plt.show()

# ---- 6.6 P/R/F1 热力图 ----
fig, ax = plt.subplots(figsize=(10, 8))
heatmap_data = np.column_stack([prec_per_class, rec_per_class, f1_per_class])
sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd',
            xticklabels=['Precision', 'Recall', 'F1-Score'],
            yticklabels=[f'类别{i}' for i in range(NUM_CLASSES)],
            cbar_kws={'label': '分数'}, ax=ax, vmin=0, vmax=1, linewidths=1,
            annot_kws={'fontsize': 9})
ax.set_title("各类别 Precision / Recall / F1 热力图", fontweight='bold', fontsize=13)
ax.set_xlabel("评价指标", fontsize=12); ax.set_ylabel("类别", fontsize=12)
plt.tight_layout()
plt.savefig("实验二_12_PRF1_热力图.png", dpi=150, bbox_inches='tight')
plt.show()

# ---- 6.7 F1 vs 样本数量关系 ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

ax1.scatter(class_counts_train, f1_per_class, s=180, c=range(NUM_CLASSES),
            cmap='tab10', edgecolors='black', linewidth=1.5, zorder=5)
for i in range(NUM_CLASSES):
    ax1.annotate(str(i), (class_counts_train[i], f1_per_class[i]),
                textcoords="offset points", xytext=(6, 4), fontsize=10, fontweight='bold')
z_fit = np.polyfit(np.log10(class_counts_train + 1), f1_per_class, 1)
p_fit = np.poly1d(z_fit)
x_fit = np.logspace(2, np.log10(max(class_counts_train) + 1000), 100)
ax1.plot(x_fit, p_fit(np.log10(x_fit)), 'r--', linewidth=2, label=f'log拟合 slope={z_fit[0]:.4f}')
ax1.set_xscale('log'); ax1.set_xlabel("训练样本数 (对数)"); ax1.set_ylabel("F1 Score")
ax1.set_title("F1 vs 训练样本数", fontweight='bold'); ax1.legend(); ax1.grid(True, alpha=0.3)

corr, pval = pearsonr(class_counts_train, f1_per_class)
ax2.bar(range(NUM_CLASSES), f1_per_class, color=plt.cm.RdYlGn(
    (f1_per_class - f1_per_class.min()) / (f1_per_class.max() - f1_per_class.min() + 1e-6)),
        edgecolor='black', linewidth=0.5)
ax2.axhline(final['f1_macro'], color='red', linestyle='--', linewidth=2,
            label=f'Macro-F1 = {final["f1_macro"]:.4f}')
for i, v in enumerate(f1_per_class):
    ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)
ax2.set_title(f"各类别F1 (Pearson r={corr:.4f}, p={pval:.4f})", fontweight='bold')
ax2.set_xlabel("类别"); ax2.set_ylabel("F1 Score"); ax2.set_ylim(0, 1.08)
ax2.set_xticks(range(NUM_CLASSES)); ax2.legend(); ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("实验二_13_F1与样本关系.png", dpi=150, bbox_inches='tight')
plt.show()

# ---- 6.8 每轮各类别 F1 演化 ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
f1_history_arr = np.array(history['test_f1_per_class'])
for c in range(NUM_CLASSES):
    alpha_val = 1.0 if c < 7 else 0.4
    lw_val = 2 if c < 7 else 1
    ax1.plot(epochs_r, f1_history_arr[c, :], 'o-', linewidth=lw_val,
             markersize=4, alpha=alpha_val, label=f'类{c}')
ax1.set_title("各类别 F1 随训练变化", fontweight='bold')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("F1 Score"); ax1.set_ylim(0, 1.05)
ax1.legend(ncol=2, fontsize=7, loc='lower right'); ax1.grid(True, alpha=0.3)

f1_gain = f1_history_arr[-1] - f1_history_arr[0]
colors_gain = ['#2ecc71' if g >= 0 else '#e74c3c' for g in f1_gain]
ax2.bar(range(NUM_CLASSES), f1_gain, color=colors_gain, edgecolor='black', linewidth=0.5)
ax2.axhline(0, color='black', linewidth=1)
for i, (g, f0, fl) in enumerate(zip(f1_gain, f1_history_arr[0], f1_history_arr[-1])):
    ax2.text(i, g + (0.03 if g >= 0 else -0.08),
            f'{f0:.2f}→{fl:.2f}', ha='center', fontsize=6, rotation=90)
ax2.set_title(f"F1 提升幅度 (Epoch 1 → {NUM_EPOCHS})", fontweight='bold')
ax2.set_xlabel("类别"); ax2.set_ylabel("F1 变化量"); ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("实验二_14_F1演化与提升.png", dpi=150, bbox_inches='tight')
plt.show()

# ==============================================================================
# 第 7 部分：混淆矩阵 + 各类别准确率
# ==============================================================================
print("\n" + "=" * 65)
print(" 第 7 部分：混淆矩阵与各类别准确率")
print("=" * 65)

cm = confusion_matrix(all_labels, all_preds)
class_acc = cm.diagonal() / cm.sum(axis=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=[f'类{i}' for i in range(NUM_CLASSES)],
            yticklabels=[f'类{i}' for i in range(NUM_CLASSES)],
            vmin=0, vmax=1, cbar_kws={'label': '归一化比例'}, ax=ax1,
            linewidths=0.5)
ax1.set_title("归一化混淆矩阵 (行归一化)", fontweight='bold', fontsize=13)
ax1.set_xlabel("预测标签"); ax1.set_ylabel("真实标签")

bars = ax2.bar(range(NUM_CLASSES), class_acc, color=colors_14, edgecolor='black', linewidth=0.5)
ax2.axhline(final['acc'], color='red', linestyle='--', linewidth=2,
            label=f'总体 Acc = {final["acc"]:.4f}')
for bar, acc in zip(bars, class_acc):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{acc:.3f}', ha='center', fontsize=8)
ax2.set_title("各类别分类准确率", fontweight='bold', fontsize=13)
ax2.set_xlabel("类别"); ax2.set_ylabel("准确率"); ax2.set_ylim(0, 1.08)
ax2.set_xticks(range(NUM_CLASSES)); ax2.legend(); ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("实验二_15_混淆矩阵与各类别准确率.png", dpi=150, bbox_inches='tight')
plt.show()

# ==============================================================================
# 第 8 部分：Abs-Sum 评价指标
# ==============================================================================
print("\n" + "=" * 65)
print(" 第 8 部分：Abs-Sum 评价指标")
print("=" * 65)

y_onehot = np.eye(NUM_CLASSES)[all_labels]
abs_sum = np.sum(np.abs(y_onehot - all_probs))
print(f"  Abs-Sum = {abs_sum:.4f}")
print("  Abs-Sum = Σ|y_true_onehot - y_pred_prob|  (越小越好)")

# ==============================================================================
# 第 9 部分：最终汇总
# ==============================================================================
print("\n" + "=" * 65)
print(" 第 9 部分：最终实验汇总")
print("=" * 65)

print(f"""
{'='*60}
      Experiment 2: News Text Classification — Final Results
{'='*60}
  Dataset        : train_set.csv ({200000 if SAMPLE_SIZE is None else SAMPLE_SIZE} samples, 14 classes)
  Vocabulary     : {VOCAB_SIZE} word indices
  Model          : Transformer Encoder ({NUM_LAYERS} layers, {NUM_HEADS} heads, {EMB_DIM} dim)
  Params         : {total_params:,}
  Loss           : Weighted CrossEntropyLoss
  Optimizer      : AdamW (lr={LEARNING_RATE})
  Epochs         : {NUM_EPOCHS}
{'='*60}
  Test Accuracy  : {final['acc']:.4f}
  Test Macro-F1  : {final['f1_macro']:.4f}
  Test Micro-F1  : {final['f1_micro']:.4f}
  Test Weighted-F1: {final['f1_weighted']:.4f}
  Abs-Sum        : {abs_sum:.4f}
{'='*60}
  Best Class     : Class {np.argmax(class_acc)} (Acc={class_acc.max():.4f})
  Worst Class    : Class {np.argmin(class_acc)} (Acc={class_acc.min():.4f})
  Best F1 Class  : Class {np.argmax(f1_per_class)} (F1={f1_per_class.max():.4f})
  Worst F1 Class : Class {np.argmin(f1_per_class)} (F1={f1_per_class.min():.4f})
{'='*60}
""")

print("[DONE] Program completed!")
print("[PLOTS] Generated charts:")
import glob
png_files = sorted(glob.glob("实验二_*.png"))
for f in png_files:
    print(f"    {f}")
