from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
import seaborn as sns
sns.set_style("whitegrid")



tokenizer_folder = 'model'
tokenizer = BertTokenizer.from_pretrained(tokenizer_folder)
model = BertForSequenceClassification.from_pretrained(tokenizer_folder, num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def print_data_info(train, val, test, n=3):
    for name, data in zip(['训练集', '验证集', '测试集'], [train, val, test]):
        print(f"{name}大小: {len(data)}\n列名: {data.columns.tolist()}\n示例:\n{data.sample(n)}\n")



class SentimentDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=128):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx): 
        sentense, label = self.data.iloc[idx]['sentence'], self.data.iloc[idx]['label']
        inputs = self.tokenizer(sentense, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }



train_data = pd.read_parquet("data/sst2/data/train-00000-of-00001.parquet")
val_data = pd.read_parquet("data/sst2/data/validation-00000-of-00001.parquet")
test_data_r  = pd.read_parquet("data/sst2/data/test-00000-of-00001.parquet")
print('改进前的test_data\n')
print_data_info(train_data,val_data,test_data_r)
'''
由于sst-2数据集的test集没有label，所以考虑对数据集进⾏处理，将train_data做split
error:test_data = pd.read_parquet("data/sst2/data/test-00000-of-00001.parquet")
# 测试集示例:
        idx                                           sentence  label
541    541  warmed-over tarantino by way of wannabe elmore...     -1
1577  1577  lacking gravitas , macdowell is a placeholder ...     -1
531    531  i regret to report that these ops are just not...     -1
481    481  when your leading ladies are a couple of scree...     -1
40      40  it is a kickass , dense sci-fi action thriller...     -1
'''

# 划分训练集为新的训练集和测试集
from sklearn.model_selection import train_test_split
train_data_new, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

print(train_data_new.columns)  # 确认列名是 'sentence' 还是 'text'

train_data_new = train_data_new.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)

# 创建数据集
train_dataset = SentimentDataset(train_data_new, tokenizer)
val_dataset = SentimentDataset(val_data, tokenizer)
test_dataset = SentimentDataset(test_data, tokenizer)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


print("改进后的test_data\n")
print_data_info(train_data_new,val_data,test_data)

'''
学习率调度器，分层学习率
'''

no_decay = ['bias', 'LayerNorm.weight']
optimizer_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_parameters, lr=3e-5)

num_epochs = 3

total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0.1*total_steps,
    num_training_steps=total_steps
)


best_val_accuracy = 0.0
train_losses = []
val_losses = []
val_accuracies = []
global_step = 0
OUTPUT_DIR = "./results" # 模型和训练结果输出目录
PLOTS_DIR = "./plots"   # 图表保存目录

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

'''
BertForSequenceClassification 模型默认使用的是交叉熵损失函数（CrossEntropyLoss）

'''

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in progress_bar:
        optimizer.zero_grad() 
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch['labels']
        outputs = model(**batch)
        loss = outputs.loss 
        loss.backward() 
        total_loss += loss.item() 
        optimizer.step() 
        scheduler.step() 
        progress_bar.set_postfix({'training_loss': f'{loss.item():.3f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

    avg_loss = total_loss / len(train_loader) # 计算平均损失
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}") 

    train_losses.append(avg_loss) # 记录训练损失

    # 验证阶段
    model.eval()
    total_val_loss = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            val_loss = outputs.loss
            total_val_loss += val_loss.item()

            val_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            val_labels = batch['labels'].cpu().numpy()
            all_val_preds.extend(val_preds)
            all_val_labels.extend(val_labels)

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = accuracy_score(all_val_labels, all_val_preds)
    val_accuracies.append(val_accuracy)
    val_precision = precision_score(all_val_labels, all_val_preds, average='weighted')
    val_recall = recall_score(all_val_labels, all_val_preds, average='weighted')
    val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')

    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
          f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
    
    if val_accuracy > best_val_accuracy:
        print(f" best accuracy {val_accuracy:.4f})")
        best_val_accuracy = val_accuracy
        model.save_pretrained(OUTPUT_DIR, safe_serialization=False)


model = BertForSequenceClassification.from_pretrained(OUTPUT_DIR)
model.to(device)
model.eval()  # 切换到评估模式

import matplotlib.pyplot as plt
import numpy as np

def evaluate(model, data_loader):
    all_preds = []
    all_labels = []
    incorrect_samples = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Testing"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            
            incorrect_mask = (preds != batch['labels']).cpu().numpy()
            incorrect_indices = np.where(incorrect_mask)[0]
            
            # 解码并保存部分错误样本（最多5个）
            for i in incorrect_indices:
                if len(incorrect_samples) >= 5:
                    break
                incorrect_samples.append({
                    'sentence': tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True),
                    'true_label': batch['labels'][i].item(),
                    'pred_label': preds[i].item()
                })
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print("随机输出5个错误案例:")
    for case in incorrect_samples[:5]:
        print(f"Sentence: {case['sentence']}")
        print(f"True Label: {case['true_label']}, Predicted Label: {case['pred_label']}\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'preds': all_preds,
        'labels': all_labels,
        # 'incorrect_samples' : incorrect_samples
    }

def plot_metrics(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 5))
    
    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    
    # Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/training_metrics.png")  # 保存图表
    plt.show()


plot_metrics(train_losses, val_losses, val_accuracies)



# 运行测试
test_results = evaluate(model, test_loader)
print(f"""
测试集结果:
准确率: {test_results['accuracy']:.4f}
精确率: {test_results['precision']:.4f}
召回率: {test_results['recall']:.4f}
F1分数: {test_results['f1']:.4f}
""")


        

    