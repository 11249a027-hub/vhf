import os
import sys
import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support
import re

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("ml"))

from ml.preprocess_v3 import build_signature_cache
from ml.models_v3 import SiameseDeepV3

FULL_ORG = "signatures/full_org"
FULL_FORG = "signatures/full_forg"
SPLIT_MAPPING = "ml/split_mapping.json"
V4_MODEL_PATH = "ml_models/siamese_signature_v4.pth"
V4_CONFIG_PATH = "ml_models/model_v4_config.json"
DEVICE = torch.device("cpu")

# Set random seeds for deterministic reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print("=" * 70, flush=True)
print("VHF SIGNATURE FRAUD DETECTION — V4 WRITER-IDENTITY TRAINING", flush=True)
print("=" * 70, flush=True)

# ----------------------------------------------------------------------
# 1. Preprocessing and Caching
# ----------------------------------------------------------------------
print("\n[Step 1] Loading and preprocessing CEDAR dataset into memory...", flush=True)
t_cache = time.time()
cache = build_signature_cache(FULL_ORG, FULL_FORG)
print(f"Cached {len(cache)} signature images in {time.time() - t_cache:.2f}s.", flush=True)

# ----------------------------------------------------------------------
# 2. Split Setup
# ----------------------------------------------------------------------
with open(SPLIT_MAPPING, "r") as f:
    split_map = json.load(f)

train_writers = sorted(split_map["train"])
val_writers = sorted(split_map["val"])
test_writers = sorted(split_map["test"])

print(f"Dataset split: {len(train_writers)} Train | {len(val_writers)} Val | {len(test_writers)} Test writers.", flush=True)

org_pat = re.compile(r"original_(\d+)_(\d+)\.(png|jpg|jpeg)", re.IGNORECASE)
forg_pat = re.compile(r"forgeries_(\d+)_(\d+)\.(png|jpg|jpeg)", re.IGNORECASE)

def index_signatures(writer_list):
    org_dict = {w: [] for w in writer_list}
    forg_dict = {w: [] for w in writer_list}
    for f in os.listdir(FULL_ORG):
        m = org_pat.match(f)
        if m and int(m.group(1)) in org_dict:
            p = os.path.normpath(os.path.join(FULL_ORG, f)).replace("\\", "/")
            org_dict[int(m.group(1))].append(p)
    for f in os.listdir(FULL_FORG):
        m = forg_pat.match(f)
        if m and int(m.group(1)) in forg_dict:
            p = os.path.normpath(os.path.join(FULL_FORG, f)).replace("\\", "/")
            forg_dict[int(m.group(1))].append(p)
    return org_dict, forg_dict

train_org, train_forg = index_signatures(train_writers)
val_org, val_forg = index_signatures(val_writers)
test_org, test_forg = index_signatures(test_writers)

# ----------------------------------------------------------------------
# 3. Balanced Batch Sampler
# ----------------------------------------------------------------------
class BalancedPKBatchSampler:
    """
    Constructs batches with P writers.
    For each writer: 2 genuine specimens and 2 skilled forgeries.
    Batch size = P * 4 = 32 images.
    Guarantees:
    - 8 Same-Writer Genuine Positive pairs
    - 32 Same-Writer Skilled Forgery Negative pairs
    - 112 Cross-Writer Genuine Impostor Negative pairs
    """
    def __init__(self, org_dict, forg_dict, p_writers=8, num_batches=50, seed=42):
        self.org_dict = org_dict
        self.forg_dict = forg_dict
        self.writers = sorted(org_dict.keys())
        self.p = p_writers
        self.num_batches = num_batches
        self.rng = random.Random(seed)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            selected_writers = self.rng.sample(self.writers, self.p)
            batch = []
            for w in selected_writers:
                # 2 genuine
                for g in self.rng.sample(self.org_dict[w], 2):
                    batch.append((g, w, 0)) # 0 = genuine
                # 2 forgeries
                for f in self.rng.sample(self.forg_dict[w], 2):
                    batch.append((f, w, 1)) # 1 = forgery
            yield batch

# ----------------------------------------------------------------------
# 4. Vectorized 50/50 Balanced Supervised Metric & Triplet Loss
# ----------------------------------------------------------------------
def compute_v4_balanced_loss(embeddings, writer_ids, is_forg_flags, margin_forg=1.0, margin_cross=1.0, margin_triplet=0.5):
    N = embeddings.size(0)
    w_same = (writer_ids.unsqueeze(0) == writer_ids.unsqueeze(1))
    both_gen = ((is_forg_flags.unsqueeze(0) == 0) & (is_forg_flags.unsqueeze(1) == 0))

    # Masks
    pos_mask = w_same & both_gen
    pos_mask.fill_diagonal_(False)

    forg_mask = w_same & (is_forg_flags.unsqueeze(0) != is_forg_flags.unsqueeze(1))
    cross_mask = (~w_same) & both_gen

    dist = torch.cdist(embeddings, embeddings, p=2)

    # 1. Compactness of same-writer genuine signatures
    pos_loss = torch.pow(dist[pos_mask], 2).mean() if pos_mask.sum() > 0 else torch.tensor(0.0)

    # 2. Balanced push: 50% skilled forgery + 50% cross-writer genuine
    forg_loss = torch.pow(torch.clamp(margin_forg - dist[forg_mask], min=0.0), 2).mean() if forg_mask.sum() > 0 else torch.tensor(0.0)
    cross_loss = torch.pow(torch.clamp(margin_cross - dist[cross_mask], min=0.0), 2).mean() if cross_mask.sum() > 0 else torch.tensor(0.0)

    # 3. Vectorized Triplet Ranking:
    # For genuine anchors: ensure positive distance + margin <= negative distance
    gen_mask = (is_forg_flags == 0)

    # Positive distance for each sample
    pos_dists = dist.clone()
    pos_dists[~pos_mask] = 1e6
    min_pos = pos_dists.min(dim=1).values[gen_mask]

    # Nearest same-writer skilled forgery
    forg_dists = dist.clone()
    forg_dists[~forg_mask] = 1e6
    min_forg = forg_dists.min(dim=1).values[gen_mask]

    # Nearest cross-writer genuine impostor
    cross_dists = dist.clone()
    cross_dists[~cross_mask] = 1e6
    min_cross = cross_dists.min(dim=1).values[gen_mask]

    trip_forg = torch.clamp(min_pos - min_forg + margin_triplet, min=0.0).mean()
    trip_cross = torch.clamp(min_pos - min_cross + margin_triplet, min=0.0).mean()

    # Total loss with equal 50/50 balance
    total_loss = pos_loss + 0.5 * forg_loss + 0.5 * cross_loss + 0.5 * trip_forg + 0.5 * trip_cross
    return total_loss, pos_loss.item(), forg_loss.item(), cross_loss.item()

# ----------------------------------------------------------------------
# 5. Diagnostic Evaluation Across All 4 Categories
# ----------------------------------------------------------------------
def compute_split_embeddings(model, org_dict, forg_dict):
    model.eval()
    embs = {}
    with torch.no_grad():
        all_paths = []
        for w in org_dict:
            all_paths.extend(org_dict[w])
            all_paths.extend(forg_dict[w])
        # Batch in chunks of 32
        for i in range(0, len(all_paths), 32):
            chunk = all_paths[i:i+32]
            tensors = torch.stack([torch.from_numpy(cache[p]) for p in chunk]).to(DEVICE)
            e = model.forward_once(tensors)
            for p, vec in zip(chunk, e):
                embs[p] = vec.cpu()
    return embs

def evaluate_4_categories(embs, org_dict, forg_dict, max_cross_pairs=2500, seed=42):
    rng = random.Random(seed)
    writers = sorted(org_dict.keys())

    # 1. Same-Writer Genuine vs Genuine
    d1 = []
    for w in writers:
        orgs = org_dict[w]
        for i in range(len(orgs)):
            for j in range(i + 1, len(orgs)):
                d1.append(float(torch.norm(embs[orgs[i]] - embs[orgs[j]], p=2).item()))

    # 2. Same-Writer Genuine vs Forgery
    d2 = []
    for w in writers:
        orgs = org_dict[w]
        forgs = forg_dict[w]
        for o in orgs:
            for f in forgs:
                d2.append(float(torch.norm(embs[o] - embs[f], p=2).item()))

    # 3. Different-Writer Genuine vs Genuine
    d3 = []
    all_cross_gen = []
    for i in range(len(writers)):
        for j in range(i + 1, len(writers)):
            w1, w2 = writers[i], writers[j]
            for o1 in org_dict[w1]:
                for o2 in org_dict[w2]:
                    all_cross_gen.append((o1, o2))
    if len(all_cross_gen) > max_cross_pairs:
        all_cross_gen = rng.sample(all_cross_gen, max_cross_pairs)
    for o1, o2 in all_cross_gen:
        d3.append(float(torch.norm(embs[o1] - embs[o2], p=2).item()))

    # 4. Different-Writer Genuine vs Forgery
    d4 = []
    all_cross_forg = []
    for i in range(len(writers)):
        for j in range(len(writers)):
            if i != j:
                w1, w2 = writers[i], writers[j]
                for o1 in org_dict[w1]:
                    for f2 in forg_dict[w2]:
                        all_cross_forg.append((o1, f2))
    if len(all_cross_forg) > max_cross_pairs:
        all_cross_forg = rng.sample(all_cross_forg, max_cross_pairs)
    for o1, f2 in all_cross_forg:
        d4.append(float(torch.norm(embs[o1] - embs[f2], p=2).item()))

    return np.array(d1), np.array(d2), np.array(d3), np.array(d4)

def calibrate_joint_threshold(d1_val, d2_val, d3_val):
    """
    Calibrate optimal decision threshold on validation data ONLY.
    Jointly controls:
    - Same-writer genuine FRR (d1 > threshold)
    - Same-writer skilled forgery FAR (d2 <= threshold)
    - Cross-writer genuine impostor FAR (d3 <= threshold)
    """
    candidates = np.linspace(0.1, 1.4, 1301)
    best_thr = 0.5
    best_cost = float("inf")
    best_metrics = {}

    for t in candidates:
        frr = float(np.mean(d1_val > t))
        far_forg = float(np.mean(d2_val <= t))
        far_cross = float(np.mean(d3_val <= t))

        # Joint cost: balances genuine acceptance with protection against BOTH forgery types
        # Combined impostor FAR is 50% skilled forgery FAR + 50% cross-writer FAR
        combined_far = 0.5 * far_forg + 0.5 * far_cross
        cost = abs(frr - combined_far) + 0.1 * (frr + combined_far)

        if cost < best_cost:
            best_cost = cost
            best_thr = float(t)
            best_metrics = {
                "threshold": float(t),
                "frr": frr,
                "far_forgery": far_forg,
                "far_cross": far_cross,
                "combined_far": combined_far
            }

    return best_thr, best_metrics

# ----------------------------------------------------------------------
# 6. Training Execution
# ----------------------------------------------------------------------
print("\n[Step 2] Initializing SiameseDeepV3 and Training (5 Epochs)...", flush=True)
model = SiameseDeepV3(emb_dim=256, dropout_rate=0.2).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

EPOCHS = 5
NUM_BATCHES = 50 # 50 batches * 32 images = 1600 images/epoch

best_val_combined_far = float("inf")
best_state_dict = None
best_epoch = -1

training_start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    t_ep = time.time()
    total_loss = 0.0
    sampler = BalancedPKBatchSampler(train_org, train_forg, p_writers=8, num_batches=NUM_BATCHES, seed=SEED + epoch)

    for batch_items in sampler:
        imgs = torch.stack([torch.from_numpy(cache[p]) for p, _, _ in batch_items]).to(DEVICE)
        w_ids = torch.tensor([w for _, w, _ in batch_items], device=DEVICE)
        is_f = torch.tensor([f for _, _, f in batch_items], device=DEVICE)

        optimizer.zero_grad()
        embs = model.forward_once(imgs)
        loss, p_l, f_l, c_l = compute_v4_balanced_loss(embs, w_ids, is_f)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    ep_duration = time.time() - t_ep
    avg_loss = total_loss / NUM_BATCHES

    # Evaluate validation categories
    val_embs = compute_split_embeddings(model, val_org, val_forg)
    d1_v, d2_v, d3_v, d4_v = evaluate_4_categories(val_embs, val_org, val_forg)

    # Check validation distance means
    d1_m, d2_m, d3_m = np.mean(d1_v), np.mean(d2_v), np.mean(d3_v)

    # Find candidate threshold for this epoch
    cand_thr, cand_met = calibrate_joint_threshold(d1_v, d2_v, d3_v)

    is_best = (cand_met["frr"] + cand_met["combined_far"]) < best_val_combined_far
    if is_best:
        best_val_combined_far = cand_met["frr"] + cand_met["combined_far"]
        best_epoch = epoch + 1
        best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"Epoch [{epoch+1}/{EPOCHS}] ({ep_duration:.1f}s) Loss: {avg_loss:.4f} | Val Means: Gen-Gen={d1_m:.3f}, Gen-Forg={d2_m:.3f}, Cross-Gen={d3_m:.3f} | Best Thr={cand_thr:.3f} (FRR={cand_met['frr']*100:.1f}%, Forg-FAR={cand_met['far_forgery']*100:.1f}%, Cross-FAR={cand_met['far_cross']*100:.1f}%) {'[BEST]' if is_best else ''}", flush=True)

total_train_time = time.time() - training_start_time
print(f"\nTraining Complete in {total_train_time:.1f}s. Best Epoch: {best_epoch}", flush=True)

# ----------------------------------------------------------------------
# 7. Save Champion V4 Model
# ----------------------------------------------------------------------
print(f"\n[Step 3] Saving V4 model checkpoint to {V4_MODEL_PATH}...", flush=True)
os.makedirs("ml_models", exist_ok=True)
torch.save(best_state_dict, V4_MODEL_PATH)
print("Saved successfully. Existing V1, V2, and V3 models remain untouched.", flush=True)

# Load best checkpoint
model.load_state_dict(best_state_dict)
model.eval()

# ----------------------------------------------------------------------
# 8. Final Threshold Calibration on Validation Data
# ----------------------------------------------------------------------
print("\n[Step 4] Final Calibration on Validation Split...", flush=True)
val_embs = compute_split_embeddings(model, val_org, val_forg)
d1_val, d2_val, d3_val, d4_val = evaluate_4_categories(val_embs, val_org, val_forg)

calibrated_thr, val_metrics = calibrate_joint_threshold(d1_val, d2_val, d3_val)
sim_thr = max(0.0, min(100.0, ((1.0 - (calibrated_thr**2)/2.0 + 1.0)/2.0) * 100.0))

print(f"  Calibrated Distance Threshold: {calibrated_thr:.4f}")
print(f"  Corresponding Similarity Threshold: {sim_thr:.2f}%")
print(f"  Validation FRR (Same-Writer Genuine Rejected): {val_metrics['frr']*100:.2f}%")
print(f"  Validation Skilled Forgery FAR: {val_metrics['far_forgery']*100:.2f}%")
print(f"  Validation Cross-Writer Genuine FAR: {val_metrics['far_cross']*100:.2f}%")
print(f"  Validation Combined Impostor FAR: {val_metrics['combined_far']*100:.2f}%")

# Save V4 config metadata
config_v4 = {
    "model_name": "Siamese-V4-WriterIdentity",
    "architecture": {
        "model_class": "SiameseDeepV3",
        "embedding_dimension": 256,
        "input_height": 155,
        "input_width": 220
    },
    "threshold": {
        "locked_validation_distance_threshold": round(calibrated_thr, 4),
        "similarity_percentage_threshold": round(sim_thr, 2)
    },
    "training": {
        "epochs": EPOCHS,
        "train_time_sec": round(total_train_time, 1),
        "best_epoch": best_epoch,
        "batch_size": 32,
        "seed": SEED,
        "loss_semantics": "Balanced 50/50 Skilled Forgery & Cross-Writer Genuine Negatives"
    },
    "validation_metrics": val_metrics
}
with open(V4_CONFIG_PATH, "w") as f:
    json.dump(config_v4, f, indent=2)
print(f"V4 metadata saved to {V4_CONFIG_PATH}.", flush=True)

# ----------------------------------------------------------------------
# 9. Held-Out Test Evaluation (9 Unseen Writers)
# ----------------------------------------------------------------------
print("\n[Step 5] Evaluating Held-Out TEST Set (9 Unseen Writers)...", flush=True)
test_embs = compute_split_embeddings(model, test_org, test_forg)
d1_test, d2_test, d3_test, d4_test = evaluate_4_categories(test_embs, test_org, test_forg)

# Distance statistics
def get_stats(arr):
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }

stats_d1 = get_stats(d1_test)
stats_d2 = get_stats(d2_test)
stats_d3 = get_stats(d3_test)
stats_d4 = get_stats(d4_test)

# Test rates using locked threshold
test_gen_accept = float(np.mean(d1_test <= calibrated_thr))
test_frr = float(np.mean(d1_test > calibrated_thr))

test_forg_reject = float(np.mean(d2_test > calibrated_thr))
test_forg_far = float(np.mean(d2_test <= calibrated_thr))

test_cross_gen_reject = float(np.mean(d3_test > calibrated_thr))
test_cross_gen_far = float(np.mean(d3_test <= calibrated_thr))

# Balanced Test Classification Metrics:
# Combine Category 1 (Positive, label 0) with equal parts Category 2 & 3 (Negative, label 1)
n_eval = min(len(d1_test), len(d2_test), len(d3_test))
rng_test = random.Random(SEED)

eval_pos = d1_test[:n_eval]
# Half skilled forgeries, half cross-writer genuine
eval_neg_forg = rng_test.sample(list(d2_test), n_eval // 2)
eval_neg_cross = rng_test.sample(list(d3_test), n_eval // 2)
eval_neg = np.array(eval_neg_forg + eval_neg_cross)

y_true = np.concatenate([np.zeros(len(eval_pos)), np.ones(len(eval_neg))])
y_scores = np.concatenate([eval_pos, eval_neg]) # distance: lower is genuine (0)

# Binary classification: pred 0 if d <= thr, else 1
y_pred = (y_scores > calibrated_thr).astype(int)

auc = float(roc_auc_score(y_true, y_scores))
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
acc = float(np.mean(y_pred == y_true))
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=0)

# Calculate EER
sort_idx = np.argsort(y_scores)
y_sorted = y_true[sort_idx]
scores_sorted = y_scores[sort_idx]
fnr_arr = np.cumsum(y_sorted == 0) / np.sum(y_sorted == 0) # FRR
fpr_arr = 1.0 - (np.cumsum(y_sorted == 1) / np.sum(y_sorted == 1)) # FAR
eer_idx = np.argmin(np.abs(fnr_arr - fpr_arr))
eer = float((fnr_arr[eer_idx] + fpr_arr[eer_idx]) / 2.0)

print("=" * 70, flush=True)
print("HELD-OUT TEST SET RESULTS (9 UNSEEN WRITERS):", flush=True)
print("=" * 70, flush=True)
print(f"1. Same-Writer Genuine vs Genuine:   Mean d={stats_d1['mean']:.4f} +/- {stats_d1['std']:.4f} | Median={stats_d1['median']:.4f}")
print(f"   -> Genuine Acceptance: {test_gen_accept*100:.2f}% | FRR: {test_frr*100:.2f}%")
print(f"2. Same-Writer Genuine vs Forgery:   Mean d={stats_d2['mean']:.4f} +/- {stats_d2['std']:.4f} | Median={stats_d2['median']:.4f}")
print(f"   -> Forgery Rejection:  {test_forg_reject*100:.2f}% | FAR: {test_forg_far*100:.2f}%")
print(f"3. Different-Writer Genuine vs Genuine: Mean d={stats_d3['mean']:.4f} +/- {stats_d3['std']:.4f} | Median={stats_d3['median']:.4f}")
print(f"   -> Cross-Writer Rejection: {test_cross_gen_reject*100:.2f}% | Cross-Writer FAR: {test_cross_gen_far*100:.2f}%")
print(f"4. Different-Writer Genuine vs Forgery: Mean d={stats_d4['mean']:.4f} +/- {stats_d4['std']:.4f} | Median={stats_d4['median']:.4f}")
print("-" * 70, flush=True)
print(f"Balanced ROC-AUC:   {auc:.4f}")
print(f"Balanced EER:       {eer*100:.2f}%")
print(f"Accuracy:           {acc*100:.2f}%")
print(f"Precision:          {prec*100:.2f}%")
print(f"Recall:             {rec*100:.2f}%")
print(f"F1-Score:           {f1*100:.2f}%")
print(f"False Accept Rate:  {(fp / (tn + fp))*100:.2f}%")
print(f"False Reject Rate:  {(fn / (fn + tp))*100:.2f}%")
print("=" * 70, flush=True)

# ----------------------------------------------------------------------
# 10. Multi-Reference Runtime Simulation (1, 3, 5, 10 Enrolled Specimen)
# ----------------------------------------------------------------------
print("\n[Step 6] Running Enterprise Multi-Reference Simulation...", flush=True)
print("(Simulating applicant enrollment and min-distance verification)", flush=True)

def run_multi_reference_sim(ref_counts=[1, 3, 5, 10]):
    res = {}
    for k in ref_counts:
        unseen_gen_verdicts = []
        skilled_forg_verdicts = []
        cross_gen_verdicts = []

        for w in test_writers:
            orgs = test_org[w]
            forgs = test_forg[w]
            other_writers = [ow for ow in test_writers if ow != w]

            # K enrollment specimens
            refs = orgs[:k]
            ref_vecs = torch.stack([test_embs[r] for r in refs]) # (K, 256)

            # Query 1: Unseen genuine of this applicant (remaining orgs)
            for q in orgs[k:]:
                q_vec = test_embs[q].unsqueeze(0)
                min_d = float(torch.norm(ref_vecs - q_vec, p=2, dim=1).min().item())
                unseen_gen_verdicts.append(min_d <= calibrated_thr)

            # Query 2: Skilled forgeries of this applicant
            for q in forgs:
                q_vec = test_embs[q].unsqueeze(0)
                min_d = float(torch.norm(ref_vecs - q_vec, p=2, dim=1).min().item())
                skilled_forg_verdicts.append(min_d > calibrated_thr) # True if correctly rejected

            # Query 3: Cross-writer genuine signatures (1 from each other test writer)
            for ow in other_writers:
                for q in test_org[ow][:2]: # 2 samples from each other writer
                    q_vec = test_embs[q].unsqueeze(0)
                    min_d = float(torch.norm(ref_vecs - q_vec, p=2, dim=1).min().item())
                    cross_gen_verdicts.append(min_d > calibrated_thr) # True if correctly rejected

        res[k] = {
            "unseen_genuine_pass_rate": float(np.mean(unseen_gen_verdicts)),
            "skilled_forgery_reject_rate": float(np.mean(skilled_forg_verdicts)),
            "cross_writer_reject_rate": float(np.mean(cross_gen_verdicts))
        }

    return res

multi_res = run_multi_reference_sim([1, 3, 5, 10])
for k, m in multi_res.items():
    print(f"Enrollment Pool: {k} Specimen Signatures:")
    print(f"  Unseen Genuine Accepted:      {m['unseen_genuine_pass_rate']*100:.2f}% (Target: HIGH)")
    print(f"  Skilled Forgery Rejected:     {m['skilled_forgery_reject_rate']*100:.2f}% (Target: HIGH)")
    print(f"  Cross-Writer Genuine Rejected: {m['cross_writer_reject_rate']*100:.2f}% (Target: HIGH)")

# ----------------------------------------------------------------------
# 11. Test the Problematic UI Screenshot Pair (EMP027 / Writer 2)
# ----------------------------------------------------------------------
print("\n[Step 7] Testing the Exact Problematic UI Pair (EMP027 / Writer 2 vs Unrelated Writers)...", flush=True)
writer2_refs = test_org[2][:10] # Enrolled signatures of EMP027
writer2_ref_vecs = torch.stack([test_embs[r] for r in writer2_refs])

# Test with various unrelated writers
unrelated_samples = [
    ("Writer 20 (original_20_11.png)", "signatures/full_org/original_20_11.png"),
    ("Writer 41 (original_41_19.png)", "signatures/full_org/original_41_19.png"),
    ("Writer 28 (original_28_17.png)", "signatures/full_org/original_28_17.png"),
    ("Writer 8  (original_8_5.png)",   "signatures/full_org/original_8_5.png"),
    ("Writer 7  (original_7_1.png)",   "signatures/full_org/original_7_1.png"),
]

print(f"Decision Threshold: {calibrated_thr:.4f} (<= {calibrated_thr:.4f} is GENUINE, > {calibrated_thr:.4f} is FORGED)")
print("-" * 70)
for label, path in unrelated_samples:
    p_norm = os.path.normpath(path).replace("\\", "/")
    # compute embedding
    with torch.no_grad():
        t = torch.from_numpy(cache[p_norm]).unsqueeze(0).to(DEVICE)
        e = model.forward_once(t).cpu()
    dists = torch.norm(writer2_ref_vecs - e, p=2, dim=1).numpy()
    min_d = float(np.min(dists))
    sim = max(0.0, min(100.0, ((1.0 - (min_d**2)/2.0 + 1.0)/2.0) * 100.0))
    verdict = "GENUINE" if min_d <= calibrated_thr else "FORGED"
    status_tag = "[STILL ACCEPTED]" if verdict == "GENUINE" else "[REJECTED AS FORGED]"
    print(f"Unrelated Query: {label}")
    print(f"  Min Distance to EMP027 references: {min_d:.4f} (Threshold: {calibrated_thr:.4f})")
    print(f"  Similarity Score:                  {sim:.2f}%")
    print(f"  System Verdict:                    {verdict} {status_tag}")
    print("-" * 70)

# Also test genuine unseen signature of Writer 2
w2_unseen = test_org[2][10]
with torch.no_grad():
    t = torch.from_numpy(cache[w2_unseen]).unsqueeze(0).to(DEVICE)
    e = model.forward_once(t).cpu()
dists = torch.norm(writer2_ref_vecs - e, p=2, dim=1).numpy()
min_d = float(np.min(dists))
sim = max(0.0, min(100.0, ((1.0 - (min_d**2)/2.0 + 1.0)/2.0) * 100.0))
verdict = "GENUINE" if min_d <= calibrated_thr else "FORGED"
auth_tag = "[CORRECT]" if verdict == "GENUINE" else "[REJECTED]"
print(f"Authentic Query: Writer 2 Unseen Genuine ({os.path.basename(w2_unseen)})")
print(f"  Min Distance: {min_d:.4f} | Similarity: {sim:.2f}% | Verdict: {verdict} {auth_tag}")
print("=" * 70)
print("V4 Retraining and Evaluation Pipeline Completed Successfully.")
