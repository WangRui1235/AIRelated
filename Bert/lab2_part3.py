import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datasets import load_dataset

device = torch.device('cuda:6')
dataset = load_dataset(path="data/imdb")["train"]

df = pd.DataFrame(dataset[:5])
print(df[['text']])

tokenizer = BertTokenizer.from_pretrained('model')

classifier_model = BertForSequenceClassification.from_pretrained(
    "./mlm_results",
    num_labels=2
).to(device)


class MLMDataset(Dataset):
    def __init__(self, data):
        # 直接将数据集中的 'text' 和 'label' 提取出来
        self.sentences = [item['text'] for item in data]
        self.labels = [item['label'] for item in data]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        inputs = prepare_classification_data([sentence], [label])
        return {k: v.squeeze(0) for k, v in inputs.items()}


def prepare_classification_data(examples, labels):
    tokenized_inputs = tokenizer(examples, truncation=True, max_length=128, padding='max_length', return_tensors='pt')
    tokenized_inputs["labels"] = torch.tensor(labels)
    return tokenized_inputs


# 划分训练集和验证集
train_data, val_data = dataset.train_test_split(test_size=0.1).values()

# 创建数据集实例
train_dataset = MLMDataset(train_data)
val_dataset = MLMDataset(val_data)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

optimizer = AdamW([
    {'params': classifier_model.bert.parameters(), 'lr': 2e-5},  # 底层较小学习率
    {'params': classifier_model.classifier.parameters(), 'lr': 5e-5}  # 分类头较大学习率
])


def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted')
    }

train_losses = []
val_metrics = []
best_f1 = 0

for epoch in range(3):  # 微调 3 个 epoch
    print(f"\nEpoch {epoch + 1}")
    loss = train_epoch(classifier_model, train_loader, optimizer)
    metrics = evaluate(classifier_model, val_loader)

    train_losses.append(loss)
    val_metrics.append(metrics)

    print(f"Train Loss: {loss:.4f}")
    print(f"Val Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

    if metrics['f1'] > best_f1:
        best_f1 = metrics['f1']
        classifier_model.save_pretrained("./best_classifier",safe_serialization=False)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.title("Training Loss")
plt.xlabel("Epoch")

plt.subplot(1, 2, 2)
plt.plot([m['accuracy'] for m in val_metrics], label='Accuracy')
plt.plot([m['f1'] for m in val_metrics], label='F1 Score')
plt.title("Validation Metrics")
plt.xlabel("Epoch")
plt.legend()
plt.savefig("./training_curves.png")
plt.show()

test_data = load_dataset(path="data/imdb")["test"]
test_dataset = MLMDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=16)

final_metrics = evaluate(classifier_model, test_loader)
print(f"\nFinal Test Performance:")
print(f"Accuracy: {final_metrics['accuracy']:.4f}")
print(f"F1 Score: {final_metrics['f1']:.4f}")
    