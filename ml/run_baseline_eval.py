import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("ml"))

from ml.siamese_model import SiameseNetwork
from ml.dataset_pairs import SignaturePairDataset
from ml.eval_utils import compute_pairwise_metrics, find_optimal_threshold

FULL_ORG = "signatures/full_org"
FULL_FORG = "signatures/full_forg"
SPLIT_MAPPING_PATH = "ml/split_mapping.json"
V2_MODEL_PATH = "ml_models/siamese_signature_v2.pth"
DEVICE = torch.device("cpu")

def extract_distances_and_labels(model, dataloader):
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
    return np.array(all_labels), np.array(all_dists)

def main():
    print("==================================================")
    print("PHASE 2 — TRUSTWORTHY BASELINE REPORT (V2 MODEL)")
    print("==================================================")

    model = SiameseNetwork().to(DEVICE)
    state_dict = torch.load(V2_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # 1. Validation Set
    val_set = SignaturePairDataset(FULL_ORG, FULL_FORG, split="val", split_mapping_path=SPLIT_MAPPING_PATH)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    val_labels, val_dists = extract_distances_and_labels(model, val_loader)

    # Find optimal threshold on validation set
    val_opt_thr = find_optimal_threshold(val_labels, val_dists, criterion='eer')
    val_metrics = compute_pairwise_metrics(val_labels, val_dists, fixed_threshold=val_opt_thr)

    print("\n--- VALIDATION SET (8 Unseen Writers, Calibrating Threshold) ---")
    print(f"Total Pairs: {val_metrics['total_pairs']} (Genuine: {val_metrics['genuine_pairs']}, Forged: {val_metrics['forged_pairs']})")
    print(f"Optimal Validation Threshold: {val_opt_thr:.4f}")
    print(f"Validation ROC-AUC:            {val_metrics['roc_auc']:.4f}")
    print(f"Validation EER:                {val_metrics['eer']*100:.2f}%")
    print(f"Validation Accuracy:           {val_metrics['accuracy']*100:.2f}%")
    print(f"Validation FAR (Forged Acc):   {val_metrics['far']*100:.2f}%")
    print(f"Validation FRR (Gen Rej):      {val_metrics['frr']*100:.2f}%")
    print(f"Validation Precision:          {val_metrics['precision']:.4f}")
    print(f"Validation Recall:             {val_metrics['recall']:.4f}")
    print(f"Validation F1:                 {val_metrics['f1']:.4f}")

    # 2. Test Set (Evaluated with locked validation threshold!)
    test_set = SignaturePairDataset(FULL_ORG, FULL_FORG, split="test", split_mapping_path=SPLIT_MAPPING_PATH)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    test_labels, test_dists = extract_distances_and_labels(model, test_loader)

    # Evaluate on test set with locked threshold
    test_metrics_locked = compute_pairwise_metrics(test_labels, test_dists, fixed_threshold=val_opt_thr)
    # Also evaluate test EER for reference
    test_metrics_eer = compute_pairwise_metrics(test_labels, test_dists, fixed_threshold=None)

    print("\n--- HELD-OUT TEST SET (9 Unseen Writers, LOCKED Validation Threshold) ---")
    print(f"Locked Threshold Applied:      {val_opt_thr:.4f}")
    print(f"Total Pairs: {test_metrics_locked['total_pairs']} (Genuine: {test_metrics_locked['genuine_pairs']}, Forged: {test_metrics_locked['forged_pairs']})")
    print(f"Test ROC-AUC:                  {test_metrics_locked['roc_auc']:.4f}")
    print(f"Test EER (Intrinsic):          {test_metrics_eer['eer']*100:.2f}% (at threshold {test_metrics_eer['eer_threshold']:.4f})")
    print(f"Test Accuracy (Locked Thr):    {test_metrics_locked['accuracy']*100:.2f}%")
    print(f"Test FAR (Forged Accepted):    {test_metrics_locked['far']*100:.2f}%")
    print(f"Test FRR (Genuine Rejected):   {test_metrics_locked['frr']*100:.2f}%")
    print(f"Test Precision:                {test_metrics_locked['precision']:.4f}")
    print(f"Test Recall:                   {test_metrics_locked['recall']:.4f}")
    print(f"Test F1:                       {test_metrics_locked['f1']:.4f}")
    print(f"Confusion Matrix:              {test_metrics_locked['confusion_matrix']}")

    # Save baseline report to JSON
    baseline_report = {
        "model_version": "V2 Baseline (siamese_signature_v2.pth)",
        "architecture": "3-layer Conv + 2-layer Dense (16.8M params, unnormalized)",
        "optimal_val_threshold": val_opt_thr,
        "val_metrics": val_metrics,
        "test_metrics_locked_threshold": test_metrics_locked,
        "test_metrics_at_test_eer": test_metrics_eer
    }
    with open("ml/baseline_v2_report.json", "w") as f:
        json.dump(baseline_report, f, indent=2)
    print("\nBaseline report saved to ml/baseline_v2_report.json\n")

if __name__ == "__main__":
    main()
