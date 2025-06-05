import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForMaskedLM, BertModel, get_linear_schedule_with_warmup, DataCollatorForLanguageModeling
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
from datasets import load_dataset
import pandas as pd
import pynvml

tokenizer_folder = 'model'
tokenizer = BertTokenizer.from_pretrained(tokenizer_folder)
# 这个模型在预训练时使用
mlm_model = BertForMaskedLM.from_pretrained(tokenizer_folder)
device = torch.device('cuda:6')
mlm_model.to(device)

'''
BertForMaskedLM主要由BERT编码器+MaskedLM预测头组成:
可以使用Trainer进行微调,让BERT适应特定领域的词汇预测
'''

dataset = load_dataset(path="data/imdb")["unsupervised"]
data_unsupervised = dataset.train_test_split(test_size=0.2)

# 转换为DataFrame并显示前5条
df = pd.DataFrame(dataset[:5])
print(df[['text']])

# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

# 检查是否已经保存了处理后的数据

tokenized_data = data_unsupervised.map(
    preprocess_function,
    batched=True,  # batched=True将预处理函数一次应用于多个元素。
    remove_columns=["text", "label"]
)


print(tokenized_data)

# 创建MLM数据收集器
'''
进行MLM任务时需要使用的数据收集器，该数据收集器会以一定概率（由参数mlm_probability控制）将序列中的Token替换成Mask标签。
不同于DataCollatorWithPadding、DataCollatorForTokenClassification
和DataCollatorForTokenClassification，该数据收集器只会将序列填充到最长序列长度。
'''
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=0.15
)

pretrain_loader = DataLoader(
    tokenized_data["train"],  # 这里需要指定是训练集
    collate_fn=data_collator,
    batch_size=48,
    shuffle=True
)

pretrain_epoch = 1
pretrain_steps = pretrain_epoch * len(pretrain_loader)
pretrain_optimizer = AdamW(mlm_model.parameters(), lr=3e-5)
pretrain_scheduler = get_linear_schedule_with_warmup(
    optimizer=pretrain_optimizer,
    num_warmup_steps=0,
    num_training_steps=pretrain_steps
)

mlm_model.train()
total_pretrain_loss = 0
progress_bar = tqdm(pretrain_loader)
for batch in progress_bar:
    pretrain_optimizer.zero_grad()
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = mlm_model(**batch)
    loss = outputs.loss
    total_pretrain_loss += loss.item()
    loss.backward()
    pretrain_optimizer.step()
    pretrain_scheduler.step()
    progress_bar.set_postfix({'pretraining_loss': f'{loss.item():.3f}', 'lr': f'{pretrain_scheduler.get_last_lr()[0]:.2e}'})

print("预训练完成!模型文件已保存在mlm_results")
mlm_model.save_pretrained('mlm_results', safe_serialization=False)