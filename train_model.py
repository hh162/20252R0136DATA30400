import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer

#------------------------
# Reproducibility
#-------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "Amazon_products", "train", "train_corpus.txt")
SILVER_PATH = os.path.join(BASE_DIR, "silver_labels.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "clf.pkl")

texts = []
with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    for line in f:
        _, text = line.split("\t", 1)
        texts.append(text.strip())

with open(SILVER_PATH, "rb") as f:
    silver_labels = pickle.load(f)

mlb = MultiLabelBinarizer(classes=list(range(531)))
Y = mlb.fit_transform(silver_labels)

encoder = SentenceTransformer("all-MiniLM-L6-v2")
X = encoder.encode(texts, batch_size=32, show_progress_bar=True)
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

class MLP(nn.Module):
    def __init__(self, dim, num_labels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(512, num_labels)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(X.shape[1], 531)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
criterion = nn.BCEWithLogitsLoss()

print("Training...")
for epoch in range(5):
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

with open(MODEL_PATH, "wb") as f:
    pickle.dump((model.state_dict(), mlb), f)

print("Model saved.")
