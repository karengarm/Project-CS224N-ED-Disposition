import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
import sklearn as sk
from tqdm.auto import tqdm

import sys  
sys.path.insert(0, '/root/Project-CS224N-ED-Disposition/')
from scripts import utils
import importlib
importlib.reload(utils)

index_col = 'Unnamed: 0'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

df = pd.read_csv('/root/data/zero_shot_predictions/scores.csv').set_index(index_col)
raw_train = pd.read_csv('/root/data/ed_train.csv').set_index(index_col)
raw_val = pd.read_csv('/root/data/ed_val.csv').set_index(index_col)
raw_test = pd.read_csv('/root/data/ed_test.csv').set_index(index_col)
print(len(raw_train), len(raw_val), len(raw_test))

total_rows = sum([len(raw_train), len(raw_val), len(raw_test)])
print(total_rows)
assert total_rows == len(df)


train_df = df.join(raw_train, how='right')
val_df = df.join(raw_val, how='right')
test_df = df.join(raw_test, how='right')
print(len(train_df), len(val_df), len(test_df))
assert len(train_df) == len(raw_train)
assert len(val_df) == len(raw_val)
assert len(test_df) == len(raw_test)

X_train = train_df[utils.labels]
X_val = val_df[utils.labels]
X_test = test_df[utils.labels]

y_col = "ED_dispo"
y_train = train_df[[y_col]]
y_val = val_df[[y_col]]
y_test = test_df[[y_col]]

input_dim = len(X_train.columns)
output_dim = len(y_train[y_col].unique())
print(input_dim, output_dim)

label_enc = OneHotEncoder(handle_unknown='ignore')
label_enc.fit(y_train)
print(y_train[y_col].unique())
print(label_enc.categories_)
print(label_enc.transform(y_train).toarray())

class Data(Dataset):
  def __init__(self, X_train, y_train):
    # need to convert float64 to float32 else 
    # will get the following error
    # RuntimeError: expected scalar type Double but found Float
    self.X = torch.from_numpy(X_train.astype(np.float32).values)
    self.y = torch.from_numpy(label_enc.transform(y_train).toarray())
    self.len = self.X.shape[0]
  
  def __getitem__(self, index):
    return self.X[index], self.y[index]

  def __len__(self):
    return self.len

train_data = Data(X_train, y_train)
val_data = Data(X_val, y_val)
test_data = Data(X_test, y_test)
print(train_data[0])

batch_size = 4096
num_workers = 2
train_loader = DataLoader(
    train_data,
    batch_size=batch_size, 
    shuffle=True,
#     num_workers=num_workers,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
  def __init__(self, input_dim=31, output_dim=4, hidden_dim=64, dropout_prob=0.2, hidden_layers=1):
    super(Network, self).__init__()
    self.hidden_layers = hidden_layers
    self.dropout_in = nn.Dropout(dropout_prob)
    self.fc_in = nn.Linear(input_dim, hidden_dim)
    self.fc_out = nn.Linear(hidden_dim, output_dim)
    if hidden_layers > 1:
        self.dropout_mid = nn.Dropout(dropout_prob)
        self.fc_mid = nn.Linear(hidden_dim, hidden_dim)
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x):
    x = self.dropout_in(x)
    x = self.fc_in(x)
    x = F.gelu(x)
    
    if self.hidden_layers > 1:
        x = self.dropout_mid(x)
        x = self.fc_mid(x)
        x = F.gelu(x)

    x = self.fc_out(x)
    x = self.softmax(x)
    return x

Network(input_dim=5, output_dim=3, hidden_dim=32, dropout_prob=0.5, hidden_layers=2)


def train(log_file, n, total, epochs, lr, hidden_dim, dropout_prob, hidden_layers):
    classifier = Network(
        hidden_dim=hidden_dim,
        dropout_prob=dropout_prob,
        hidden_layers=hidden_layers,
    ).to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr)

    for epoch in range(epochs):
#         print(f"epoch {epoch}")
        classifier.train() # training mode, to apply dropout
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
    #         inputs = train_data.X
    #         labels = train_data.y
    #         print(inputs.shape)

            optimizer.zero_grad() # set optimizer to zero grad to remove previous epoch gradients
            outputs = classifier(inputs) # forward propagation
            loss = loss_func(outputs, labels)
            loss.backward() # backward propagation
            optimizer.step()

    classifier.eval()
    outputs = classifier(val_data.X.to(device))
    val_data_y = val_data.y.to(device)
    loss = loss_func(outputs, val_data_y)

    y_true = torch.argmax(val_data.y, dim=1).cpu()
    y_pred = torch.argmax(outputs, dim=1).cpu()
    f1 = sk.metrics.f1_score(y_true, y_pred, average='micro')
    acc = sk.metrics.accuracy_score(y_true, y_pred)
    log = f'{n}/{total} params epochs={epochs} lr={lr} hidden_dim={hidden_dim} dropout_prob={dropout_prob} hidden_layers={hidden_layers} loss={loss.item():.5f} acc={acc:.5f} f1={f1:.5f}'
    print(log)
    log_file.write(log)
    log_file.write('\n')
    log_file.flush()
    return classifier, acc, f1


from sklearn.model_selection import ParameterGrid
param_grid = ParameterGrid({
    'epochs': [1, 10, 100, 500],
    'lr': [1e-2, 1e-3, 1e-4, 1e-5],
    'hidden_dim': [16, 64, 256],
    'dropout_prob': [0, 0.2, 0.4],
    'hidden_layers': [1, 2],
})
trials = []
best_acc = 0
best_model = None
best_params = None
with open('/root/models/2/logs.txt', 'w') as log_file:
    for i, params in enumerate(param_grid):
        classifier, acc, f1 = train(log_file, i, len(param_grid), **params)
        trials.append([classifier, acc, f1, params])
        if acc > best_acc:
            best_acc = acc
            best_model = classifier
            best_params = params
#         break
    #     print(params)

def wpickle(obj, filename, base_path='/root/models/2/'):
    import pickle
    # open a file, where you ant to store the data
    file = open(f"{base_path}{filename}", 'wb')

    # dump information to that file
    pickle.dump(obj, file)

    # close the file
    file.close()

wpickle({"best_acc": best_acc, "best_model": best_model, "best_params": best_params}, "best.pickle")
wpickle(trials, "trials.pickle")