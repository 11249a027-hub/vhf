import os
import sys
import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("ml"))

from ml.preprocess_v3 import build_signature_cache
from ml.models_v3 import SiameseDeepV3
from ml.eval_utils import compute_pairwise_metrics, find_optimal_threshold
from ml.dataset_cached import CachedPairDataset

FULL_ORG = "signatures/full_org"
FULL_FORG = "signatures/full_forg"
SPLIT_MAPPING = "ml/split_mapping.json"
DEVICE = torch.device("cpu")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class BatchMetricDataset(Dataset):
    """
    Samples individual images with their metadata:
    (image_tensor, writer_id, is_forgery_flag)
    """
    def __init__(self, cache, full_org_path, full_forg_path, split_mapping_path="ml/split_mapping.json"):
        self.cache = cache
        with open(split_mapping_path, "r") as f:
            split_map = json.load(f)
        allowed_writers = set(split_map["train"])

        org_pat = re.compile(r"original_(\d+)_(\d+)\.(png|jpg|jpeg)", re.IGNORECASE)
        forg_pat = re.compile(r"forgeries_(\d+)_(\d+)\.(png|jpg|jpeg)", re.IGNORECASE)

        self.samples = []
        for f in os.listdir(full_org_path):
            m = org_pat.match(f)
            if m and int(m.group(1)) in allowed_writers:
                w = int(m.group(1))
                p = os.path.normpath(os.path.join(full_org_path, f)).replace("\\", "/")
                self.samples.append((p, w, 0)) # 0 = genuine

        for f in os.listdir(full_forg_path):
            m = forg_pat.match(f)
            if m and int(m.group(1)) in allowed_writers:
                w = int(m.group(1))
                p = os.path.normpath(os.path.join(full_forg_path, f)).replace("\\", "/")
                self.samples.append((p, w, 1)) # 1 = forgery

        self.writer_to_idx = {w: i for i, w in enumerate(sorted(allowed_writers))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, w, is_forg = self.samples[idx]
        img = self.cache[p]
        return torch.from_numpy(img), self.writer_to_idx[w], is_forg

class PKBatchSampler:
    """
    Samples P writers, and for each writer samples K images
    (e.g., P=8 writers, K=4 images = 32 images per batch).
    Ensures every batch has multiple genuine pairs and both skilled and cross-writer negatives.
    """
    def __init__(self, dataset, p_writers=8, k_images=4, num_batches=60):
        self.dataset = dataset
        self.p = p_writers
        self.k = k_images
        self.num_batches = num_batches

        self.writer_indices = {}
        for idx, (_, w_idx, is_forg) in enumerate(dataset.samples):
            self.writer_indices.setdefault(w_idx, []).append(idx)
        self.writers = list(self.writer_indices.keys())

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random()
        for _ in range(self.num_batches):
            selected_writers = rng.sample(self.writers, self.p)
            batch = []
            for w in selected_writers:
                idxs = self.writer_indices[w]
                # sample k images for this writer
                chosen = rng.sample(idxs, min(self.k, len(idxs)))
                batch.extend(chosen)
            yield batch

def compute_batch_metric_loss(embeddings, writer_ids, is_forg_flags, margin=1.0):
    """
    Vectorized Supervised Metric Loss:
    - Positive: Same writer, both genuine -> minimize distance
    - Negative: Different writers OR same writer with forgery -> push distance >= margin
    """
    same_writer = (writer_ids.unsqueeze(0) == writer_ids.unsqueeze(1))
    both_genuine = ((is_forg_flags.unsqueeze(0) == 0) & (is_forg_flags.unsqueeze(1) == 0))
    pos_mask = same_writer & both_genuine
    pos_mask.fill_diagonal_(False)

    neg_mask = (~same_writer) | (same_writer & (is_forg_flags.unsqueeze(0) != is_forg_flags.unsqueeze(1)))

    dist = torch.cdist(embeddings, embeddings, p=2)

    if pos_mask.sum() > 0:
        pos_loss = torch.pow(dist[pos_mask], 2).mean()
    else:
        pos_loss = torch.tensor(0.0, device=embeddings.device)

    if neg_mask.sum() > 0:
        neg_loss = torch.pow(torch.clamp(margin - dist[neg_mask], min=0.0), 2).mean()
    else:
        neg_loss = torch.tensor(0.0, device=embeddings.device)

    return 0.5 * (pos_loss + neg_loss)

def evaluate_loader(model, dataloader):
    model.eval()
    dists, labels = [], []
    with torch.no_grad():
        for x1, x2, y in dataloader:
            o1, o2 = model(x1, x2)
            d = torch.norm(o1 - o2, p=2, dim=1)
            dists.extend(d.numpy())
            labels.extend(y.numpy())
    return compute_pairwise_metrics(np.array(labels), np.array(dists))

def main():
    print("Building signature cache in memory...", flush=True)
    t0 = time.time()
    cache = build_signature_cache(FULL_ORG, FULL_FORG)
    print(f"Cached {len(cache)} images in {time.time()-t0:.2f}s.", flush=True)

    val_set = CachedPairDataset(cache, FULL_ORG, FULL_FORG, split="val", split_mapping_path=SPLIT_MAPPING)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    train_ds = BatchMetricDataset(cache, FULL_ORG, FULL_FORG, split_mapping_path=SPLIT_MAPPING)
    sampler = PKBatchSampler(train_ds, p_writers=8, k_images=4, num_batches=60)
    train_loader = DataLoader(train_ds, batch_sampler=sampler)

    model = SiameseDeepV3(emb_dim=256, dropout_rate=0.2).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    print("Starting 5-epoch test with PK-sampling metric learning...", flush=True)
    for epoch in range(5):
        model.train()
        total_loss = 0.0
        t_start = time.time()
        for step, (imgs, w_ids, is_forg) in enumerate(train_loader):
            optimizer.zero_grad()
            embs = model.forward_once(imgs)
            loss = compute_batch_metric_loss(embs, w_ids, is_forg, margin=1.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        ep_time = time.time() - t_start
        val_res = evaluate_loader(model, val_loader)
        print(f"Epoch [{epoch+1}/5] Loss: {total_loss/len(train_loader):.4f} | Val ROC-AUC: {val_res['roc_auc']:.4f} | Val EER: {val_res['eer']*100:.2f}% | Time: {ep_time:.1f}s", flush=True)

if __name__ == "__main__":
    main()
