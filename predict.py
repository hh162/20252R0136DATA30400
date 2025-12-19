import os
import random
import pickle
import numpy as np
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from utils_taxonomy import load_hierarchy, get_all_ancestors

# -----------------------------------
# Reproducibility
# ----------------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# -----------------------------------
# Paths
# ----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(BASE_DIR, "Amazon_products", "test", "test_corpus.txt")
MODEL_PATH = os.path.join(BASE_DIR, "clf.pkl")
HIERARCHY_PATH = os.path.join(BASE_DIR, "Amazon_products", "class_hierarchy.txt")
SUBMISSION_PATH = os.path.join(BASE_DIR, "submission.csv")

# --------------------------
# Load hierarchy
# --------------------------
parents, _ = load_hierarchy(HIERARCHY_PATH)

# --------------------------------
# Load model
# ---------------------------------
with open(MODEL_PATH, "rb") as f:
    state_dict, mlb = pickle.load(f)

# -----------------------------------
# Encoder
# ----------------------------------
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------
# MLP model definition
# -----------------------------------
class MLP(torch.nn.Module):
    def __init__(self, dim, num_labels):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(512, num_labels)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(384, 531)
model.load_state_dict(state_dict)
model.eval()

# ------------------------------------
# Load test corpus
# ----------------------------------
ids, texts = [], []
with open(TEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        pid, text = line.split("\t", 1)
        ids.append(pid)
        texts.append(text.strip())

# -----------------------------------
# Encode embeddings
# ----------------------------------
X = encoder.encode(texts, batch_size=32, show_progress_bar=True)
X = torch.tensor(X, dtype=torch.float32)

# --------------------------------
# Predict probabilities
# ------------------------------------
with torch.no_grad():
    probs_model = torch.sigmoid(model(X)).numpy()

# --------------------------------
# TF-IDF similarity
# ---------------------------------
# Build class keyword docs
from utils_taxonomy import load_keywords, load_classes

CLASSES_PATH = os.path.join(BASE_DIR, "Amazon_products", "classes.txt")
KEYWORDS_PATH = os.path.join(BASE_DIR, "Amazon_products", "class_related_keywords.txt")

id2name = load_classes(CLASSES_PATH)
name2id = {name: cid for cid, name in id2name.items()}
keywords = load_keywords(KEYWORDS_PATH)

class_docs = []
class_ids = []
for cname, kws in keywords.items():
    if cname in name2id:
        class_docs.append(" ".join(kws))
        class_ids.append(name2id[cname])

vectorizer = TfidfVectorizer(stop_words="english", max_features=30000, ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(texts + class_docs)
doc_vecs = X_tfidf[:len(texts)]
class_vecs = X_tfidf[len(texts):]
probs_tfidf = cosine_similarity(doc_vecs, class_vecs)

# -------------------------------------
# Ensemble: weighted average
# -----------------------------------
ALPHA = 0.7  # model weight
probs = ALPHA * probs_model + (1 - ALPHA) * probs_tfidf

#------------------------------------
# Hierarchy-aware smoothing
#-----------------------------------
for i in range(len(probs)):
    for child in range(probs.shape[1]):
        for parent in parents.get(child, []):
            probs[i, parent] = max(probs[i, parent], probs[i, child] * 0.8)

#--------------------------------------
# Adaptive top-k selection
#---------------------------------
predictions = []
for p in probs:
    # adaptive k
    if np.max(p) > 0.6:
        top_k = 2
    else:
        top_k = 3

    top_idx = np.argsort(p)[-top_k:]
    labels = set(top_idx.tolist())

    # propagate hierarchy
    for l in list(labels):
        labels.update(get_all_ancestors(l, parents))

    predictions.append(",".join(map(str, sorted(labels))))

# --------------------------------
# Save submission
#----------------------------------
df = pd.DataFrame({"id": ids, "labels": predictions})
df.to_csv(SUBMISSION_PATH, index=False)

print(f"Submission saved: {SUBMISSION_PATH}")
