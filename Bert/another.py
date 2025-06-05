# %% [markdown]
# 加载相关库文件

# %%
!pip install datasets

# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer,BertForMaskedLM,BertModel,get_linear_schedule_with_warmup,DataCollatorForLanguageModeling
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import torch.nn.functional as F

NORMAL_OUTPUT="/normal"
DISTIL_OUTPUT="/distillation"

os.makedirs(NORMAL_OUTPUT, exist_ok=True)
os.makedirs(DISTIL_OUTPUT,exist_ok=True)

if torch.cuda.is_available():
  device =torch.device("cuda")
  print("use cuda device",torch.cuda.get_device_name(0))
else:
  device = torch.device("cpu")
  print("no GPU found,use cpu instead")

# %% [markdown]
# 定义之前训练的模型结构

# %%
class Bert_Classification(nn.Module):
  def __init__(self) -> None:
    super(Bert_Classification,self).__init__()
    self.bertmlm=BertForMaskedLM.from_pretrained("bert-base-uncased")
    self.bert=self.bertmlm.bert
    self.cls=nn.Linear(self.bert.config.hidden_size,2)
    self.criterion = nn.CrossEntropyLoss()

  def forward(self,input_ids,attention_mask,token_type_ids,labels):
    outputs=self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,return_dict=True)
    last_hidden_state=outputs.last_hidden_state
    CLS_output=last_hidden_state[:, 0, :]
    logits=self.cls(CLS_output)
    output_dict = {"logits": logits}
    loss = self.criterion(logits, labels)
    output_dict["loss"] = loss
    return output_dict

# %% [markdown]
# 加载teacher模型，使用之前预训练后微调的模型

# %%
teacher_model=Bert_Classification()
teacher_model.load_state_dict(torch.load("drive/MyDrive/part2_/results.pth"))
teacher_model.to(device)
teacher_model.eval()

# %% [markdown]
# 加载student模型，使用distilbert-base-uncased，加载tokenizer

# %%
tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
student_model=AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
student_model.to(device)

# %% [markdown]
# 加载IMDB数据集并处理

# %%
raw_dataset=load_dataset("imdb")
print(raw_dataset)
del raw_dataset["unsupervised"]

def preprocess(examples):
  examples["labels"]=examples.pop("label")
  return tokenizer(
      examples["text"],
      truncation=True,
      padding=True,
      return_token_type_ids=True
  )

tokenized_dataset=raw_dataset.map(preprocess,batched=True)
tokenized_dataset=tokenized_dataset.remove_columns(["text"])
print(tokenized_dataset)

# %%
tokenized_dataset.set_format("torch",columns=["input_ids", "attention_mask", "token_type_ids", "labels"])

# %% [markdown]
# 创建DataLoader

# %%
split=tokenized_dataset["train"].train_test_split(test_size=0.3)
train_loader=DataLoader(split["train"],shuffle=True,batch_size=128)
val_loader=DataLoader(split["test"],batch_size=128)
test_loader=DataLoader(tokenized_dataset["test"],batch_size=128)

print(train_loader.dataset)

# %% [markdown]
# 创建损失函数，优化器和调度器

# %%
NUM_EPOCHS=3

optimizer = AdamW(student_model.parameters(), lr=3e-5)
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1,
    num_training_steps=total_steps
)


kl_loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=False)

# %% [markdown]
# 评估函数定义

# %%
def evaluate(model, data_loader, device, description="Evaluating"):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    error_result={}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=description):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop('token_type_ids', None)
            labels = batch['labels']
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_np)

    for i,pred in enumerate(all_preds):
      if(pred != all_labels[i]):
         error_result["idx"]=i
         error_result["true_label"]=all_labels[i]
         error_result["pred_label"]=pred
         error_result["sentence"]=tokenizer.decode(test_loader.dataset[i]["input_ids"])
         break

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        "error_result":error_result
    }

# %% [markdown]
# 开始模型蒸馏的训练

# %%
best_val_accuracy = 0.0
train_losses = []
val_losses = []
val_accuracies = []
global_step = 0

for epoch in range(NUM_EPOCHS):
    student_model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for batch in progress_bar:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}

        # 得到teacher模型的输出logits
        with torch.no_grad():
          teacher_outputs = teacher_model(**batch)
          teacher_logits = teacher_outputs["logits"]

        # 得到student模型的输出logits
        batch.pop('token_type_ids', None)
        outputs = student_model(**batch)
        loss_ce = outputs.loss
        logits=outputs.logits

        # 计算student和teacher模型在蒸馏温度为2的情况下的概率输出分布
        soft_teacher_probs = F.softmax(teacher_logits / 2, dim=-1)
        log_soft_student_probs = F.log_softmax(logits / 2, dim=-1)

        # 计算KL散度下的损失
        loss_kd = kl_loss_fn(log_soft_student_probs, soft_teacher_probs) * 4

        # 计算总损失
        loss = 0.3 * loss_ce + 0.7 * loss_kd

        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        global_step += 1
        progress_bar.set_postfix({'training_loss': f'{loss.item():.3f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"\nepoch {epoch+1} average training Loss: {avg_train_loss:.4f}")
    train_losses.append(avg_train_loss)

    print("validation：")
    val_metrics = evaluate(student_model, val_loader, device, description="Validating")
    val_loss = val_metrics['loss']
    val_accuracy = val_metrics['accuracy']
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch+1} validation results: loss={val_loss:.4f}, accuracy={val_accuracy:.4f}, precision={val_metrics['precision']:.4f}, recall={val_metrics['recall']:.4f}, f1={val_metrics['f1']:.4f}")

    if val_accuracy > best_val_accuracy:
        print(f"accuracy improved from ({best_val_accuracy:.4f} to {val_accuracy:.4f}). saving model to {DISTIL_OUTPUT}...")
        best_val_accuracy = val_accuracy
        student_model.save_pretrained(DISTIL_OUTPUT)

print("\ntraining finished")
print(f"best validation accuracy: {best_val_accuracy:.4f}")

# %% [markdown]
# 测试集上评估模型

# %%
print("loaded successfully.")
test_metrics = evaluate(student_model, test_loader, device, description="Testing")

print("\ntest results")
print(f"loss:      {test_metrics['loss']:.4f}")
print(f"accuracy:  {test_metrics['accuracy']:.4f}")
print(f"precision: {test_metrics['precision']:.4f}")
print(f"recall:    {test_metrics['recall']:.4f}")
print(f"f1 Score:  {test_metrics['f1']:.4f}")
print(f"confusion matrix:")
print(test_metrics['confusion_matrix'])
print(f"error_result:  {test_metrics['error_result']['sentence']}")
print(f"true_label:  {test_metrics['error_result']['true_label']}")
print(f"error_label:  {test_metrics['error_result']['pred_label']}")

# %% [markdown]
# 绘制训练曲线

# %%
del range
range_ = range(1, 4)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range_, train_losses, label='training Loss', marker='o')
plt.plot(range_, val_losses, label='validation Loss', marker='o')
plt.title('training and validation Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.xticks(range_)
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(range_, val_accuracies, label='validation accuracy', marker='o', color='orange')
plt.title('validation Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.xticks(range_)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


