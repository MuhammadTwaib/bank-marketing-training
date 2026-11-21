import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

data = pd.read_csv("bank-full.csv", sep=";")
print("Shape:", data.shape)

X = data.drop("y", axis=1)
y = data["y"].map({"no": 0, "yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train:", X_train.shape)
print("Test: ", X_test.shape)

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
print("Columns after one hot:", X_train.shape[1])

X_train_np = X_train.values.astype("float32")
X_test_np = X_test.values.astype("float32")
y_train_np = y_train.values.astype("float32").reshape(-1, 1)
y_test_np = y_test.values.astype("float32").reshape(-1, 1)

scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train_np).astype("float32")
X_test_np = scaler.transform(X_test_np).astype("float32")

inputs = torch.from_numpy(X_train_np)
targets = torch.from_numpy(y_train_np)
test_inputs = torch.from_numpy(X_test_np)
test_targets = torch.from_numpy(y_test_np)
print("Inputs tensor:", inputs.shape)
print("Targets tensor:", targets.shape)

train_ds = TensorDataset(inputs, targets)
test_ds = TensorDataset(test_inputs, test_targets)

batch_size = 32
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

for xb, yb in train_dl:
    print("Batch X:", xb.shape)
    print("Batch y:", yb.shape)
    break

model = nn.Linear(51, 1)
print(model.weight.shape)

loss_fn = F.mse_loss
loss = loss_fn(model(inputs), targets)
print("Start loss:", loss.item())

opt = torch.optim.SGD(model.parameters(), lr=0.01)


def fit(num_epochs, model, loss_fn, opt, train_dl):
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        if (epoch + 1) % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))


fit(20, model, loss_fn, opt, train_dl)

preds = model(test_inputs)
print("Pehli 5 predictions:")
print(preds[:5].flatten())
print("Asli jawab (y test):")
print(test_targets[:5].flatten())
