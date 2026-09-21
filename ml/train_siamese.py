import os
import json
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

# Adjust import paths for script execution
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from dataset_pairs import SignaturePairDataset
from siamese_model import SiameseNetwork, ContrastiveLoss

# Detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
FULL_ORG = "signatures/full_org"
FULL_FORG = "signatures/full_forg"
MODEL_SAVE_PATH = "ml_models/siamese_signature.pth"
MODEL_SAVE_V2 = "ml_models/siamese_signature_v2.pth"
SPLIT_MAPPING_PATH = "ml/writer_split.json"

# Hyper-parameters (tuned for quick demo runs)
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 0.001
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

def evaluate(model, dataloader):
    model.eval()
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for img1, img2, labels in dataloader:
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)
            out1, out2 = model(img1, img2)
            # Euclidean distance as similarity (lower distance = more similar)
            distances = torch.norm(out1 - out2, p=2, dim=1)
            # Convert to similarity score (higher = more similar)
            scores = -distances
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # Compute ROC-AUC (labels: 0=genuine, 1=forged)
    try:
        roc_auc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        roc_auc = float('nan')
    # Compute optimal threshold using Youden's J statistic (maximizing TPR-FPR)
    thresholds = np.sort(all_scores)
    best_j = -np.inf
    best_thr = None
    for thr in thresholds:
        preds = (np.array(all_scores) >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_labels, preds, labels=[0,1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_thr = thr
    # Precision, Recall, F1 for that threshold
    if best_thr is not None:
        preds = (np.array(all_scores) >= best_thr).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds, average='binary')
    else:
        precision = recall = f1 = float('nan')
    return {
        "roc_auc": roc_auc,
        "threshold": best_thr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "scores": all_scores,
        "labels": all_labels,
    }

def train():
    # ---------------------------------------------------------------
    # 1. Build deterministic writer split mapping (70/15/15)
    # ---------------------------------------------------------------
    if os.path.exists(SPLIT_MAPPING_PATH):
        with open(SPLIT_MAPPING_PATH, "r") as jf:
            split_map = json.load(jf)
    else:
        # Create mapping on-the-fly using a fixed seed
        dummy_dataset = SignaturePairDataset(FULL_ORG, FULL_FORG, split="train")
        all_writers = sorted(set(dummy_dataset.org_by_writer.keys()))
        random.shuffle(all_writers)
        n = len(all_writers)
        train_end = int(0.70 * n)
        val_end = train_end + int(0.15 * n)
        split_map = {
            "train": all_writers[:train_end],
            "val": all_writers[train_end:val_end],
            "test": all_writers[val_end:]
        }
        os.makedirs(os.path.dirname(SPLIT_MAPPING_PATH), exist_ok=True)
        with open(SPLIT_MAPPING_PATH, "w") as jf:
            json.dump(split_map, jf, indent=2)

    # ---------------------------------------------------------------
    # 2. Prepare datasets & loaders for train & validation
    # ---------------------------------------------------------------
    train_set = SignaturePairDataset(FULL_ORG, FULL_FORG, split="train", split_mapping_path=SPLIT_MAPPING_PATH)
    val_set = SignaturePairDataset(FULL_ORG, FULL_FORG, split="val", split_mapping_path=SPLIT_MAPPING_PATH)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = SiameseNetwork().to(DEVICE)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on {DEVICE}", flush=True)
    print(f"Train pairs: {len(train_set)}, Val pairs: {len(val_set)}", flush=True)

    best_val_auc = -np.inf
    best_metrics = None
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        total_steps = len(train_loader)
        for step, (img1, img2, labels) in enumerate(train_loader):
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (step + 1) % 500 == 0 or (step + 1) == total_steps:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{step+1}/{total_steps}] Running Loss: {running_loss / (step + 1):.4f}", flush=True)
        avg_loss = running_loss / total_steps
        # -------------------------------------------------------
        # 3. Validation after each epoch
        # -------------------------------------------------------
        print(f"Epoch [{epoch+1}/{EPOCHS}] Running validation...", flush=True)
        val_metrics = evaluate(model, val_loader)

        # Checkpoint saving BEFORE printing (so logging/encoding errors never prevent saving)
        # Safe checkpoint saving after every completed epoch: update V2 checkpoint
        os.makedirs(os.path.dirname(MODEL_SAVE_V2), exist_ok=True)
        if best_metrics is None or val_metrics['roc_auc'] >= best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            best_metrics = val_metrics
            torch.save(model.state_dict(), MODEL_SAVE_V2)

        # Also persist current threshold to config.json safely after every completed epoch
        config_path = "ml/config.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as cf:
            json.dump({"threshold": float(best_metrics['threshold'])}, cf, indent=2)

        # Epoch summary print with pure ASCII and flush=True
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f} | Val Threshold: {val_metrics['threshold']:.4f}", flush=True)

    # ---------------------------------------------------------------
    # 4. Final reporting and persist chosen threshold
    # ---------------------------------------------------------------
    if best_metrics is None:
        raise RuntimeError("Validation did not produce any metrics.")
    # Save threshold to a tiny JSON config for the verifier
    config_path = "ml/config.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as cf:
        json.dump({"threshold": float(best_metrics['threshold'])}, cf, indent=2)
    print("--- Training complete ---", flush=True)
    print(f"Best validation ROC-AUC: {best_val_auc:.4f}", flush=True)
    print(f"Selected similarity threshold (saved to {config_path}): {best_metrics['threshold']:.4f}", flush=True)
    print(f"Metrics - Precision: {best_metrics['precision']:.4f}, Recall: {best_metrics['recall']:.4f}, F1: {best_metrics['f1']:.4f}", flush=True)
    # Save final V2 model (do NOT overwrite original baseline)
    os.makedirs(os.path.dirname(MODEL_SAVE_V2), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_V2)
    print(f"V2 model saved to {MODEL_SAVE_V2} (baseline unchanged).", flush=True)
    # ---------------------------------------------------------------
    # 5. Test set evaluation (once only)
    # ---------------------------------------------------------------
    print("--- Evaluating held-out TEST set ---", flush=True)
    test_set = SignaturePairDataset(FULL_ORG, FULL_FORG, split="test", split_mapping_path=SPLIT_MAPPING_PATH)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    test_metrics = evaluate(model, test_loader)
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}", flush=True)
    print(f"Test Threshold (for reference): {test_metrics['threshold']:.4f}", flush=True)
    print(f"Test Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}", flush=True)


if __name__ == "__main__":
    train()