# Bank Marketing Training

PyTorch training on the Bank Marketing dataset (bank-full.csv, 45,211 rows).

What the code does:
1. Load the dataset and split it into train and test
2. One hot encoding (16 columns become 51)
3. Scaling and conversion into tensors
4. TensorDataset and DataLoader (batch size 32)
5. nn.Linear model, MSE loss, SGD optimizer
6. Training for 20 epochs and predictions on the test set

How to run:

    pip install pandas scikit-learn torch
    python bank_training.py

Files:
- bank_training.py - the full code (VS Code / terminal)
- Bank_Dataset_Full_Training.ipynb - the same work as a notebook (Colab / Jupyter)
- bank-full.csv - the dataset
