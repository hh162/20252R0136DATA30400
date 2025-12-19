import os
import random
import pickle
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils_taxonomy import load_classes, load_keywords, load_hierarchy, get_all_ancestors

#------------------------
# Reproducibility
#-------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "Amazon_products", "train", "train_corpus.txt")
CLASSES_PATH = os.path.join(BASE_DIR, "Amazon_products", "classes.txt")
KEYWORDS_PATH = os.path.join(BASE_DIR, "Amazon_products", "class_related_keywords.txt")
HIERARCHY_PATH = os.path.join(BASE_DIR, "Amazon_products", "class_hierarchy.txt")
OUTPUT_PATH = os.path.join(BASE_DIR, "silver_labels.pkl")

def load_corpus(path):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            _, text = line.split("\t", 1)
            texts.append(text.strip())
    return texts

print("Loading data...")
train_texts = load_corpus(TRAIN_PATH)
id2name = load_classes(CLASSES_PATH)
name2id = {v: k for k, v in id2name.items()}
keywords = load_keywords(KEYWORDS_PATH)
parents, _ = load_hierarchy(HIERARCHY_PATH)

print("Building class descriptions...")
class_texts = []
class_ids = []
for name, kws in keywords.items():
    if name in name2id:
        text = name.replace("_", " ") + " " + " ".join(kws)
        class_texts.append(text)
        class_ids.append(name2id[name])

encoder = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding texts...")
doc_emb = encoder.encode(train_texts, batch_size=32, show_progress_bar=True)
class_emb = encoder.encode(class_texts, batch_size=32, show_progress_bar=True)

print("Computing similarities...")
sim = cosine_similarity(doc_emb, class_emb)

silver_labels = []

for i in range(len(train_texts)):
    sims = sim[i]
    top_idx = np.argsort(sims)[-5:]
    selected = [class_ids[j] for j in top_idx if sims[j] >= 0.35]

    #fallback
    if len(selected) < 2:
        top_idx = np.argsort(sims)[-3:]
        selected = [class_ids[j] for j in top_idx]

    final = set(selected)
    for l in selected:
        final.update(get_all_ancestors(l, parents))

    silver_labels.append(sorted(final))

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(silver_labels, f)

print(f"Saved silver labels: {OUTPUT_PATH}")
