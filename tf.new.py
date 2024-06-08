#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# 设置随机种子以保证结果的可重复性
# input_size = 107
# hidden_size = 32
# num_heads = 4
# num_layers = 1
# output_size = 239
# batch_size = 128
# num_epochs = 50
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
class SurvivalAnalysisModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, output_size):
        super(SurvivalAnalysisModel, self).__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads), num_layers
        )
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.dense(x)
        x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


class SurvivalAnalysisDataset(Dataset):
    def __init__(self, n_patients, n_features):
        self.features = torch.randn(n_patients, n_features)
        self.times = torch.randint(1, 240, (n_patients,)).float()
        self.events = torch.randint(0, 2, (n_patients,)).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.times[idx], self.events[idx]


class SurvivalDataset(Dataset):
    def __init__(self, features, times, events):
        self.features = features
        self.times = times
        self.events = events

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.times[index], self.events[index]


def auc_trapezoidal(y):
    area = 0
    area = np.sum(y[:, 1:-1], axis=1) + (y[:, 0] + y[:, 1])/2
    return area





def monotonicity_loss(predictions):
    diff = predictions[:, 1:] - predictions[:, :-1]
    zero_tensor = torch.zeros_like(diff)
    # return torch.sum(torch.square(torch.square(torch.maximum(zero_tensor, diff)))) 
    return torch.sum(torch.square(torch.square(torch.maximum(zero_tensor, diff)))) / (predictions.size(0) * (predictions.size(1) - 1))


def c_index(pred_times, true_times, events):
    n = len(pred_times)
    concordant = 0
    permissible = 0
    tied_risk = 0

    for i in range(n):
        for j in range(i + 1, n):
            if events[i] == 1 and events[j] == 1:
                if true_times[i] < true_times[j]:
                    if pred_times[i] < pred_times[j]:
                        concordant += 1
                    elif pred_times[i] == pred_times[j]:
                        tied_risk += 1
                    permissible += 1
                elif true_times[i] > true_times[j]:
                    if pred_times[i] > pred_times[j]:
                        concordant += 1
                    elif pred_times[i] == pred_times[j]:
                        tied_risk += 1
                    permissible += 1
            elif (events[i] == 1 and events[j] == 0) and true_times[i] < true_times[j]:
                permissible += 1
                if  pred_times[i] < pred_times[j]:
                    concordant += 1
                elif pred_times[i] == pred_times[j]:
                    tied_risk += 1
            elif (events[i] == 0 and events[j] == 1) and true_times[i] > true_times[j]:
                permissible += 1
                if  pred_times[i] > pred_times[j]:
                    concordant += 1
                elif pred_times[i] == pred_times[j]:
                    tied_risk += 1
    return (concordant + 0.5 * tied_risk) / permissible if permissible != 0 else 0

input_size = 107
hidden_size = 64
num_heads = 4
num_layers = 1
output_size = 239
batch_size = 64
num_epochs = 50
learning_rate = 1e-3
import pandas as pd
model = SurvivalAnalysisModel(input_size, hidden_size, num_heads, num_layers, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = pd.read_csv("trainset.csv")
val_dataset = pd.read_csv("valset.csv")
test_dataset = pd.read_csv("testset.csv")

train_features = torch.tensor(train_dataset.iloc[:,:107].values, dtype=torch.float32)
train_times = torch.tensor(train_dataset.iloc[:,-2].values, dtype=torch.float32)
train_events = torch.tensor(train_dataset.iloc[:,-1].values, dtype=torch.float32)

val_features = torch.tensor(val_dataset.iloc[:,:107].values, dtype=torch.float32)
val_times = torch.tensor(val_dataset.iloc[:,-2].values, dtype=torch.float32)
val_events = torch.tensor(val_dataset.iloc[:,-1].values, dtype=torch.float32)

test_features = torch.tensor(test_dataset.iloc[:,:107].values, dtype=torch.float32)
test_times = torch.tensor(test_dataset.iloc[:,-2].values, dtype=torch.float32)
test_events = torch.tensor(test_dataset.iloc[:,-1].values, dtype=torch.float32)

train_dataset = SurvivalDataset(train_features, train_times, train_events)
val_dataset = SurvivalDataset(val_features, val_times, val_events)
test_dataset = SurvivalDataset(test_features, test_times, test_events)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


num_epochs = 50
for epoch in range(num_epochs):

    epoch_loss = 0
    epoch_bce_loss = 0
    epoch_monotonicity_loss = 0
    epoch_cindex_loss = 0
    all_preds = []
    all_times = []
    all_events = []
    auc = []
    x_values = np.arange(output_size)
    for batch_features, batch_times, batch_events in train_dataloader:

        optimizer.zero_grad()
        predictions = model(batch_features)
        targets = torch.ones_like(predictions)
        for i, t in enumerate(batch_times):
            t = int(t.item())
            event = int(batch_events[i].item())
            if event == 1:
                targets[i, t:] = 0

        #cross entropy
        bce_loss = criterion(predictions, targets)
        #单调
        mono_loss = 10*monotonicity_loss(predictions)
        #cindex_loss
        # batch_cindex_loss = c_index_loss(batch_times, predictions, batch_events)
        total_loss = bce_loss + 1000* mono_loss
        total_loss.backward()

        optimizer.step()

        epoch_loss += total_loss.item()
        epoch_bce_loss += bce_loss.item()
        epoch_monotonicity_loss += mono_loss.item()
        all_preds.extend(predictions.detach().numpy().tolist())
        all_times.extend(batch_times.numpy().tolist())
        all_events.extend(batch_events.numpy().tolist())
        # epoch_cindex_loss += batch_cindex_loss.item()
    # all_preds = np.array(all_preds)
    # all_times = np.array(all_times)
    # all_events = np.array(all_events)
    # auc = auc_trapezoidal(all_preds)
    # c_index_score = c_index(auc, all_times, all_events)

    with torch.no_grad():
      val_bceloss = 0
      val_monoloss = 0
      epoch_val_loss = 0.0
      valall_preds = []
      valall_times = []
      valall_events = []
      valauc = []
      val_cindex = 0
      for val_batch_features, val_batch_times, val_batch_events in val_dataloader:

          val_predictions = model(val_batch_features)
          val_targets = torch.ones_like(val_predictions)
          for i, t in enumerate(val_batch_times):
              t = int(t.item())
              event = int(val_batch_events[i].item())
              if event == 1:
                  val_targets[i, t:] = 0

          val_bceloss = criterion(val_predictions, val_targets)
          val_monoloss = 1000*monotonicity_loss(val_predictions)
          val_totalloss = val_bceloss + val_monoloss
          epoch_val_loss += val_totalloss.item()
          valall_preds.extend(val_predictions.detach().numpy().tolist())
          valall_times.extend(val_batch_times.numpy().tolist())
          valall_events.extend(val_batch_events.numpy().tolist())
          # epoch_val_bce_loss += val_bceloss.item()
          # epoch_val_monotonicity_loss += val_monoloss.item()
      valall_preds = np.array(valall_preds)
      valall_times = np.array(valall_times)
      valall_events = np.array(valall_events)
      valauc = auc_trapezoidal(valall_preds)
      val_cindex_score = c_index(valauc, valall_times, valall_events)

    with torch.no_grad():
          test_bceloss = 0
          test_monoloss = 0
          epoch_test_loss = 0.0
          testall_preds = []
          testall_times = []
          testall_events = []
          testauc = []
          test_cindex = 0
          for test_batch_features, test_batch_times, test_batch_events in test_dataloader:

              test_predictions = model(test_batch_features)
              test_targets = torch.ones_like(test_predictions)
              for i, t in enumerate(test_batch_times):
                  t = int(t.item())
                  event = int(test_batch_events[i].item())
                  if event == 1:
                      test_targets[i, t:] = 0

              test_bceloss = criterion(test_predictions, test_targets)
              test_monoloss = 1000*monotonicity_loss(test_predictions)
              test_totalloss = test_bceloss + test_monoloss
              epoch_test_loss += test_totalloss.item()
              testall_preds.extend(test_predictions.detach().numpy().tolist())
              testall_times.extend(test_batch_times.numpy().tolist())
              testall_events.extend(test_batch_events.numpy().tolist())
              # epoch_val_bce_loss += val_bceloss.item()
              # epoch_val_monotonicity_loss += val_monoloss.item()
          testall_preds = np.array(testall_preds)
          testall_times = np.array(testall_times)
          testall_events = np.array(testall_events)
          testauc = auc_trapezoidal(testall_preds)
          test_cindex_score = c_index(testauc, testall_times, testall_events)
    print(f"Epoch {epoch + 1}/{num_epochs} - Val c-index: {val_cindex_score :.4f},test c-index: {test_cindex_score :.4f},BCE Loss: {epoch_bce_loss:.4f}, Monotonicity Loss: {epoch_monotonicity_loss:.8f}, Train Total Loss: {epoch_bce_loss + epoch_monotonicity_loss:.4f},Val Total Loss: {epoch_val_loss:.4f}")
    
    
import matplotlib.pyplot as plt

# 获取预测的生存概率
pred_probs = np.array(testall_preds)

# 获取死亡和存活病人的索引
dead_indices = np.where(testall_events == 1)
alive_indices = np.where(testall_events == 0)

# 绘制死亡病人的生存概率曲线（红色）
for i in dead_indices[0]:
    plt.plot(pred_probs[i], color='red', alpha=0.3)

# 绘制存活病人的生存概率曲线（绿色）
for i in alive_indices[0]:
    plt.plot(pred_probs[i], color='green', alpha=0.3)

# 添加图例
plt.legend(['Dead', 'Alive'])

# 添加轴标签
plt.xlabel('Time')
plt.ylabel('Survival Probability')

# 显示图形
plt.show()
