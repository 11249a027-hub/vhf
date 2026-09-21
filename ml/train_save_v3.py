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
from sklearn.metrics import confusion_matrix, roc_auc_score
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
V3_MODEL_PATH = "ml_models/siamese_signature_v3.pth"
V3_CONFIG_PATH = "ml_models/model_v3_config.json"
DEVICE = torch.device("cpu")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class BatchMetricDataset(Dataset):
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
    def __init__(self, dataset, p_writers=8, k_images=4, num_batches=60):
        self.dataset = dataset
        self.p = p_writers
        self.k = k_images
        self.num_batches = num_batches

        self.writer_indices = {}
        for idx, (_, w_idx, _) in enumerate(dataset.samples):
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
                chosen = rng.sample(idxs, min(self.k, len(idxs)))
                batch.extend(chosen)
            yield batch

def compute_batch_metric_loss(embeddings, writer_ids, is_forg_flags, margin=1.0):
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

def evaluate_loader(model, dataloader, fixed_threshold=None):
    model.eval()
    dists, labels = [], []
    with torch.no_grad():
        for x1, x2, y in dataloader:
            o1, o2 = model(x1, x2)
            d = torch.norm(o1 - o2, p=2, dim=1)
            dists.extend(d.numpy())
            labels.extend(y.numpy())
    return compute_pairwise_metrics(np.array(labels), np.array(dists), fixed_threshold=fixed_threshold)

def evaluate_multireference_scenario(model, cache, full_org_path, full_forg_path, split_mapping_path, ref_counts=[1, 3, 5, 10], locked_thr=1.0):
    """
    Simulates real-world enterprise verification:
    An applicant enrolls K authentic specimen signatures.
    A new test signature is verified against the enrolled reference pool.
    """
    model.eval()
    with open(split_mapping_path, "r") as f:
        split_map = json.load(f)
    test_writers = sorted(split_map["test"])

    org_pat = re.compile(r"original_(\d+)_(\d+)\.(png|jpg|jpeg)", re.IGNORECASE)
    forg_pat = re.compile(r"forgeries_(\d+)_(\d+)\.(png|jpg|jpeg)", re.IGNORECASE)

    test_org = {}
    for f in os.listdir(full_org_path):
        m = org_pat.match(f)
        if m and int(m.group(1)) in test_writers:
            w = int(m.group(1))
            p = os.path.normpath(os.path.join(full_org_path, f)).replace("\\", "/")
            test_org.setdefault(w, []).append(p)

    test_forg = {}
    for f in os.listdir(full_forg_path):
        m = forg_pat.match(f)
        if m and int(m.group(1)) in test_writers:
            w = int(m.group(1))
            p = os.path.normpath(os.path.join(full_forg_path, f)).replace("\\", "/")
            test_forg.setdefault(w, []).append(p)

    results_by_ref = {}

    for k in ref_counts:
        y_true = []
        scores_min = []
        scores_mean = []
        scores_median = []

        for w in test_writers:
            orgs = test_org[w]
            forgs = test_forg.get(w, [])
            other_writers = [ow for ow in test_writers if ow != w]

            # Fixed enrollment references: first k authentic signatures
            refs = orgs[:k]
            # Queries:
            # 1. Authentic queries (remaining authentic signatures) -> label 0
            genuine_queries = orgs[k:]
            # 2. Skilled forgeries -> label 1
            forged_queries = forgs
            # 3. Cross-writer queries (1 from each other writer) -> label 1
            cross_queries = [test_org[ow][0] for ow in other_writers]

            # Extract embeddings for references
            with torch.no_grad():
                ref_tensors = torch.stack([torch.from_numpy(cache[r]) for r in refs]).to(DEVICE)
                ref_embs = model.forward_once(ref_tensors) # (K, 256)

                all_queries = (
                    [(q, 0) for q in genuine_queries] +
                    [(q, 1) for q in forged_queries] +
                    [(q, 1) for q in cross_queries]
                )

                for q_path, lbl in all_queries:
                    q_tensor = torch.from_numpy(cache[q_path]).unsqueeze(0).to(DEVICE)
                    q_emb = model.forward_once(q_tensor) # (1, 256)

                    # Distances to all enrolled references
                    dists = torch.norm(ref_embs - q_emb, p=2, dim=1).cpu().numpy()

                    y_true.append(lbl)
                    scores_min.append(float(np.min(dists)))
                    scores_mean.append(float(np.mean(dists)))
                    scores_median.append(float(np.median(dists)))

        y_true = np.array(y_true)
        # Compute metrics for each aggregation strategy using locked_threshold
        def get_strat_metrics(dists_arr):
            preds = (dists_arr >= locked_thr).astype(int)
            acc = float(np.mean(preds == y_true))
            cm = confusion_matrix(y_true, preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            far = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            frr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
            auc = float(roc_auc_score(y_true, dists_arr))
            return {"accuracy": acc, "far": far, "frr": frr, "roc_auc": auc}

        results_by_ref[f"{k}_references"] = {
            "min_distance_best_match": get_strat_metrics(np.array(scores_min)),
            "mean_distance": get_strat_metrics(np.array(scores_mean)),
            "median_distance": get_strat_metrics(np.array(scores_median))
        }

    return results_by_ref

def main():
    print("=================================================================", flush=True)
    print("VHF SIGNATURE FRAUD DETECTION — CONSOLIDATED V3 MODEL TRAINING", flush=True)
    print("=================================================================", flush=True)

    # 1. Build In-Memory Preprocessing Cache
    print("Phase 1: Preprocessing and caching CEDAR dataset in memory...", flush=True)
    t0 = time.time()
    cache = build_signature_cache(FULL_ORG, FULL_FORG)
    print(f"Cached {len(cache)} images in {time.time()-t0:.2f}s.\n", flush=True)

    # 2. Setup Validation and Training Sets
    val_set = CachedPairDataset(cache, FULL_ORG, FULL_FORG, split="val", split_mapping_path=SPLIT_MAPPING)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    print(f"Validation Set: {len(val_set)} pairs across 8 unseen validation writers.", flush=True)

    train_ds = BatchMetricDataset(cache, FULL_ORG, FULL_FORG, split_mapping_path=SPLIT_MAPPING)
    sampler = PKBatchSampler(train_ds, p_writers=8, k_images=4, num_batches=60)
    train_loader = DataLoader(train_ds, batch_sampler=sampler)
    print(f"Training Set: 38 writers with PK-Batch Sampler (P=8, K=4, 60 batches/epoch).\n", flush=True)

    # 3. Model, Optimizer, Scheduler
    model = SiameseDeepV3(emb_dim=256, dropout_rate=0.2).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    EPOCHS = 5
    best_val_auc = -1.0
    best_val_eer = 1.0
    best_epoch = -1
    best_state_dict = None
    epoch_hist = []

    print("Phase 2: Training SiameseDeepV3 (5 Epochs)...", flush=True)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        t_ep = time.time()

        for step, (imgs, w_ids, is_forg) in enumerate(train_loader):
            optimizer.zero_grad()
            embs = model.forward_once(imgs)
            loss = compute_batch_metric_loss(embs, w_ids, is_forg, margin=1.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        ep_duration = time.time() - t_ep
        avg_loss = total_loss / len(train_loader)

        # Validation after each epoch
        val_res = evaluate_loader(model, val_loader)
        val_auc = val_res["roc_auc"]
        val_eer = val_res["eer"]

        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            best_val_eer = val_eer
            best_epoch = epoch + 1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        epoch_hist.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "val_roc_auc": val_auc,
            "val_eer": val_eer,
            "time_sec": ep_duration
        })

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Val ROC-AUC: {val_auc:.4f} | Val EER: {val_eer*100:.2f}% | Duration: {ep_duration:.1f}s {'[BEST]' if is_best else ''}", flush=True)

    print(f"\n--- Training Complete ---", flush=True)
    print(f"Best Validation Epoch: {best_epoch} (Val ROC-AUC: {best_val_auc:.4f}, Val EER: {best_val_eer*100:.2f}%)\n", flush=True)

    # 4. Save Champion Model to V3 checkpoint (NEVER touching V1 or V2)
    print(f"Phase 3: Saving checkpoint to {V3_MODEL_PATH}...", flush=True)
    os.makedirs("ml_models", exist_ok=True)
    torch.save(best_state_dict, V3_MODEL_PATH)
    print("V3 checkpoint saved successfully. Original V1 and V2 models left untouched.\n", flush=True)

    # 5. Lock Optimal Validation Threshold
    model.load_state_dict(best_state_dict)
    model.eval()

    val_res = evaluate_loader(model, val_loader)
    locked_val_thr = val_res["eer_threshold"]
    print(f"Phase 4: Calibrating & Locking Threshold on Validation Data...", flush=True)
    print(f"Locked Validation Threshold: {locked_val_thr:.4f} (EER: {val_res['eer']*100:.2f}%, Accuracy: {val_res['accuracy']*100:.2f}%)\n", flush=True)

    # 6. Evaluate on Held-Out Test Set (9 Unseen Writers, 7,704 Pairs)
    print("Phase 5: Evaluating Held-Out TEST Set with LOCKED Validation Threshold...", flush=True)
    test_set = CachedPairDataset(cache, FULL_ORG, FULL_FORG, split="test", split_mapping_path=SPLIT_MAPPING)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    test_metrics_locked = evaluate_loader(model, test_loader, fixed_threshold=locked_val_thr)
    test_metrics_eer = evaluate_loader(model, test_loader, fixed_threshold=None)

    print("--- HELD-OUT TEST BENCHMARK (9 Unseen Writers, 7,704 Pairs) ---", flush=True)
    print(f"ROC-AUC:                  {test_metrics_locked['roc_auc']:.4f}", flush=True)
    print(f"Test EER (Intrinsic):     {test_metrics_eer['eer']*100:.2f}% (at distance {test_metrics_eer['eer_threshold']:.4f})", flush=True)
    print(f"Locked Val Threshold:     {locked_val_thr:.4f}", flush=True)
    print(f"Accuracy (Locked Thr):    {test_metrics_locked['accuracy']*100:.2f}%", flush=True)
    print(f"Precision:                {test_metrics_locked['precision']:.4f}", flush=True)
    print(f"Recall:                   {test_metrics_locked['recall']:.4f}", flush=True)
    print(f"F1 Score:                 {test_metrics_locked['f1']:.4f}", flush=True)
    print(f"FAR (Forged Accepted):    {test_metrics_locked['far']*100:.2f}%", flush=True)
    print(f"FRR (Genuine Rejected):   {test_metrics_locked['frr']*100:.2f}%", flush=True)
    print(f"Confusion Matrix:         {test_metrics_locked['confusion_matrix']}\n", flush=True)

    # 7. Evaluate Multi-Reference Applicant Verification Scenario
    print("Phase 6: Evaluating Multi-Reference Applicant Verification Scenario...", flush=True)
    multiref_results = evaluate_multireference_scenario(
        model, cache, FULL_ORG, FULL_FORG, SPLIT_MAPPING,
        ref_counts=[1, 3, 5, 10], locked_thr=locked_val_thr
    )

    for k_str, strats in multiref_results.items():
        print(f"\n--- Enrollment: {k_str.replace('_', ' ').upper()} ---", flush=True)
        for strat_name, s_met in strats.items():
            print(f"  [{strat_name}]: Accuracy = {s_met['accuracy']*100:.2f}% | FAR = {s_met['far']*100:.2f}% | FRR = {s_met['frr']*100:.2f}% | AUC = {s_met['roc_auc']:.4f}", flush=True)

    # 8. Save Complete Metadata & Configuration
    # Cosine similarity equivalent for distance threshold d:
    # d^2 = 2 - 2*cos => cos = 1 - (d^2)/2
    # In 0-100 scale: ((cos + 1)/2)*100
    cos_val = 1.0 - (locked_val_thr ** 2) / 2.0
    similarity_score_100 = max(0.0, min(100.0, ((cos_val + 1.0) / 2.0) * 100.0))

    config_data = {
        "model_version": "V3",
        "model_file": "siamese_signature_v3.pth",
        "checkpoint_path": V3_MODEL_PATH,
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": 42,
        "architecture": {
            "name": "SiameseDeepV3",
            "backbone": "4-Stage Conv2d with BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d(3,4)",
            "head": "Linear(3072, 512) -> BatchNorm1d -> ReLU -> Dropout(0.2) -> Linear(512, 256)",
            "embedding_dimension": 256,
            "normalization": "L2 unit hypersphere (||x||_2 = 1.0)",
            "total_parameters": sum(p.numel() for p in model.parameters())
        },
        "training": {
            "epochs": EPOCHS,
            "best_epoch": best_epoch,
            "batch_strategy": "PK-Batch Sampler (P=8 writers, K=4 images, 60 batches/epoch)",
            "loss": "Vectorized Supervised Metric Loss (Contrastive Margin = 1.0)",
            "optimizer": "AdamW (lr=5e-4, weight_decay=1e-4) with CosineAnnealingLR",
            "epoch_history": epoch_hist
        },
        "threshold": {
            "locked_validation_distance_threshold": locked_val_thr,
            "cosine_similarity_equivalent": float(cos_val),
            "similarity_percentage_threshold": round(float(similarity_score_100), 2)
        },
        "validation_metrics": val_res,
        "test_metrics_pairwise_locked": test_metrics_locked,
        "test_metrics_pairwise_intrinsic_eer": test_metrics_eer,
        "multireference_applicant_scenario": multiref_results
    }

    with open(V3_CONFIG_PATH, "w") as jf:
        json.dump(config_data, jf, indent=2)
    print(f"\nMetadata and configuration saved to {V3_CONFIG_PATH}", flush=True)
    print("=================================================================", flush=True)

if __name__ == "__main__":
    main()
