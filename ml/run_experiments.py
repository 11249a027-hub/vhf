import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("ml"))

from ml.preprocess_v3 import build_signature_cache
from ml.models_v3 import SiameseDeepV3, NormalizedContrastiveLoss, MultiNegativeTripletLoss
from ml.dataset_cached import CachedPairDataset, CachedTripletDataset
from ml.eval_utils import compute_pairwise_metrics, find_optimal_threshold
from ml.siamese_model import SiameseNetwork

FULL_ORG = "signatures/full_org"
FULL_FORG = "signatures/full_forg"
SPLIT_MAPPING = "ml/split_mapping.json"
DEVICE = torch.device("cpu")

# Fix random seed
torch.manual_seed(42)
np.random.seed(42)

def evaluate_model_on_loader(model, dataloader, fixed_threshold=None):
    model.eval()
    all_dists = []
    all_labels = []
    with torch.no_grad():
        for img1, img2, labels in dataloader:
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)
            out1, out2 = model(img1, img2)
            dists = torch.norm(out1 - out2, p=2, dim=1)
            all_dists.extend(dists.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_labels = np.array(all_labels)
    all_dists = np.array(all_dists)
    metrics = compute_pairwise_metrics(all_labels, all_dists, fixed_threshold=fixed_threshold)
    return metrics, all_labels, all_dists

def run_experiment(exp_name: str, model, train_loader, val_loader, criterion, optimizer,
                   epochs: int = 10, is_triplet: bool = False, scheduler=None):
    print(f"\n=======================================================")
    print(f"RUNNING: {exp_name}")
    print(f"=======================================================")
    best_val_auc = -1.0
    best_epoch = -1
    best_val_metrics = None
    best_state_dict = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        t0 = time.time()

        if is_triplet:
            for step, (anc, pos, neg) in enumerate(train_loader):
                anc, pos, neg = anc.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
                optimizer.zero_grad()
                out_a = model.forward_once(anc)
                out_p = model.forward_once(pos)
                out_n = model.forward_once(neg)
                loss = criterion(out_a, out_p, out_n)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        else:
            for step, (img1, img2, labels) in enumerate(train_loader):
                img1, img2, labels = img1.to(DEVICE), img2.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                out1, out2 = model(img1, img2)
                loss = criterion(out1, out2, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / max(1, num_batches)
        epoch_time = time.time() - t0

        # Validate on held-out validation set
        val_metrics, _, _ = evaluate_model_on_loader(model, val_loader)
        val_auc = val_metrics["roc_auc"]
        val_eer = val_metrics["eer"] * 100.0

        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_val_metrics = val_metrics
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | Val ROC-AUC: {val_auc:.4f} | Val EER: {val_eer:.2f}% | Time: {epoch_time:.1f}s {'[BEST]' if is_best else ''}", flush=True)

    print(f"\n--- {exp_name} Complete ---")
    print(f"Best Epoch: {best_epoch} | Best Val ROC-AUC: {best_val_auc:.4f} | Val EER: {best_val_metrics['eer']*100:.2f}%")
    return {
        "exp_name": exp_name,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "best_state_dict": best_state_dict
    }

def main():
    print("Loading and caching CEDAR dataset in memory...")
    t0 = time.time()
    cache = build_signature_cache(FULL_ORG, FULL_FORG)
    print(f"Cached {len(cache)} images in {time.time()-t0:.2f}s.\n")

    # Datasets
    val_set = CachedPairDataset(cache, FULL_ORG, FULL_FORG, split="val", split_mapping_path=SPLIT_MAPPING)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    train_pair_set = CachedPairDataset(cache, FULL_ORG, FULL_FORG, split="train", split_mapping_path=SPLIT_MAPPING, augment=True)
    train_pair_loader = DataLoader(train_pair_set, batch_size=32, shuffle=True)

    train_triplet_set = CachedTripletDataset(cache, FULL_ORG, FULL_FORG, split_mapping_path=SPLIT_MAPPING, num_triplets=20000)
    train_triplet_loader = DataLoader(train_triplet_set, batch_size=32, shuffle=True)

    experiments = []

    # -------------------------------------------------------------
    # Experiment 1: Baseline 3-layer CNN with New Preprocessing & L2-norm
    # -------------------------------------------------------------
    class BaselineWithNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = SiameseNetwork()
        def forward_once(self, x):
            emb = self.base.forward_once(x)
            import torch.nn.functional as F
            return F.normalize(emb, p=2, dim=1)
        def forward(self, x1, x2):
            return self.forward_once(x1), self.forward_once(x2)

    m1 = BaselineWithNorm().to(DEVICE)
    opt1 = optim.Adam(m1.parameters(), lr=1e-3)
    crit1 = NormalizedContrastiveLoss(margin=1.0)
    res1 = run_experiment("Exp 1: Baseline CNN (16.8M) + L2-Norm + Contrastive(m=1.0)",
                          m1, train_pair_loader, val_loader, crit1, opt1, epochs=8)
    experiments.append(res1)

    # -------------------------------------------------------------
    # Experiment 2: Modern SiameseDeepV3 (1.04M) + Contrastive Loss
    # -------------------------------------------------------------
    m2 = SiameseDeepV3(emb_dim=256, dropout_rate=0.25).to(DEVICE)
    opt2 = optim.AdamW(m2.parameters(), lr=5e-4, weight_decay=1e-4)
    crit2 = NormalizedContrastiveLoss(margin=1.0)
    sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=12)
    res2 = run_experiment("Exp 2: SiameseDeepV3 (1.04M) + Contrastive(m=1.0)",
                          m2, train_pair_loader, val_loader, crit2, opt2, epochs=12, scheduler=sched2)
    experiments.append(res2)

    # -------------------------------------------------------------
    # Experiment 3: Modern SiameseDeepV3 (1.04M) + Multi-Negative Triplet Loss
    # -------------------------------------------------------------
    m3 = SiameseDeepV3(emb_dim=256, dropout_rate=0.25).to(DEVICE)
    opt3 = optim.AdamW(m3.parameters(), lr=5e-4, weight_decay=1e-4)
    crit3 = MultiNegativeTripletLoss(margin=0.5)
    sched3 = optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=12)
    res3 = run_experiment("Exp 3: SiameseDeepV3 (1.04M) + Multi-Negative Triplet(m=0.5)",
                          m3, train_triplet_loader, val_loader, crit3, opt3, epochs=12, is_triplet=True, scheduler=sched3)
    experiments.append(res3)

    # -------------------------------------------------------------
    # Experiment 4: SiameseDeepV3 Fine-Tuning with Low LR (1e-4) on Hard Negative Pairs
    # -------------------------------------------------------------
    # Take the best model so far and fine-tune with smaller margin
    # Take the best SiameseDeepV3 candidate so far
    deep_candidates = [res2, res3]
    best_deep = max(deep_candidates, key=lambda x: x["best_val_metrics"]["roc_auc"])
    print(f"\nFine-tuning top SiameseDeepV3 candidate: {best_deep['exp_name']} (Val AUC: {best_deep['best_val_metrics']['roc_auc']:.4f})")
    
    m4 = SiameseDeepV3(emb_dim=256, dropout_rate=0.2).to(DEVICE)
    m4.load_state_dict(best_deep["best_state_dict"])
    opt4 = optim.AdamW(m4.parameters(), lr=1.5e-4, weight_decay=5e-5)
    crit4 = NormalizedContrastiveLoss(margin=0.9)
    res4 = run_experiment("Exp 4: SiameseDeepV3 Hard Fine-Tuning (LR=1.5e-4, m=0.9)",
                          m4, train_pair_loader, val_loader, crit4, opt4, epochs=6)
    experiments.append(res4)

    # -------------------------------------------------------------
    # Champion Selection & Final Evaluation on Held-Out Test Set
    # -------------------------------------------------------------
    champion = max(experiments, key=lambda x: x["best_val_metrics"]["roc_auc"])
    print("\n=======================================================")
    print(f"CHAMPION MODEL SELECTED: {champion['exp_name']}")
    print(f"Best Validation ROC-AUC: {champion['best_val_metrics']['roc_auc']:.4f}")
    print("=======================================================")

    # Save V3 Champion Model
    os.makedirs("ml_models", exist_ok=True)
    V3_MODEL_PATH = "ml_models/siamese_signature_v3.pth"
    torch.save(champion["best_state_dict"], V3_MODEL_PATH)
    print(f"Champion checkpoint saved to {V3_MODEL_PATH} (V1 and V2 preserved).")

    # Load champion for definitive test set evaluation
    test_set = CachedPairDataset(cache, FULL_ORG, FULL_FORG, split="test", split_mapping_path=SPLIT_MAPPING)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    if "Baseline" in champion["exp_name"]:
        champion_model = BaselineWithNorm().to(DEVICE)
    else:
        champion_model = SiameseDeepV3(emb_dim=256).to(DEVICE)
    champion_model.load_state_dict(champion["best_state_dict"])
    champion_model.eval()

    # Re-evaluate validation to lock threshold
    val_metrics, val_labels, val_dists = evaluate_model_on_loader(champion_model, val_loader)
    locked_val_thr = val_metrics["eer_threshold"]

    # Evaluate on held-out test set with locked threshold
    test_metrics_locked, test_labels, test_dists = evaluate_model_on_loader(champion_model, test_loader, fixed_threshold=locked_val_thr)
    # Intrinsic test EER
    test_metrics_intrinsic, _, _ = evaluate_model_on_loader(champion_model, test_loader, fixed_threshold=None)

    print("\n=======================================================")
    print("FINAL HELD-OUT TEST SET EVALUATION (LOCKED THRESHOLD)")
    print("=======================================================")
    print(f"Model:                         SiameseDeepV3 (V3)")
    print(f"Locked Validation Threshold:   {locked_val_thr:.4f}")
    print(f"Test ROC-AUC:                  {test_metrics_locked['roc_auc']:.4f}")
    print(f"Test Accuracy (Locked Thr):    {test_metrics_locked['accuracy']*100:.2f}%")
    print(f"Test EER (Intrinsic):          {test_metrics_intrinsic['eer']*100:.2f}% (at threshold {test_metrics_intrinsic['eer_threshold']:.4f})")
    print(f"Test FAR (Forged Accepted):    {test_metrics_locked['far']*100:.2f}%")
    print(f"Test FRR (Genuine Rejected):   {test_metrics_locked['frr']*100:.2f}%")
    print(f"Test Precision:                {test_metrics_locked['precision']:.4f}")
    print(f"Test Recall:                   {test_metrics_locked['recall']:.4f}")
    print(f"Test F1 Score:                 {test_metrics_locked['f1']:.4f}")
    print(f"Confusion Matrix:              {test_metrics_locked['confusion_matrix']}")

    # Save summary report
    summary = {
        "v3_model_path": V3_MODEL_PATH,
        "selected_champion": champion["exp_name"],
        "locked_validation_threshold": locked_val_thr,
        "validation_metrics": val_metrics,
        "test_metrics_locked_threshold": test_metrics_locked,
        "test_metrics_intrinsic_eer": test_metrics_intrinsic,
        "all_experiments": [
            {
                "name": e["exp_name"],
                "best_epoch": e["best_epoch"],
                "val_roc_auc": e["best_val_metrics"]["roc_auc"],
                "val_eer": e["best_val_metrics"]["eer"],
                "val_accuracy": e["best_val_metrics"]["accuracy"],
                "val_far": e["best_val_metrics"]["far"],
                "val_frr": e["best_val_metrics"]["frr"]
            }
            for e in experiments
        ]
    }
    with open("ml/experiment_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nExperiment results saved to ml/experiment_results.json\n")

if __name__ == "__main__":
    main()
