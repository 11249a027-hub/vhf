import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
    confusion_matrix
)

def compute_pairwise_metrics(labels: np.ndarray, distances: np.ndarray, fixed_threshold: float = None):
    """
    Computes comprehensive, mathematically sound signature verification metrics.
    
    Conventions:
      labels: 0 = Genuine, 1 = Forged / Negative
      distances: Euclidean / metric distance (lower = more similar, higher = more forged)
      Fraud score: distances (monotonic with Class 1 probability)
    """
    labels = np.asarray(labels, dtype=int)
    distances = np.asarray(distances, dtype=float)

    # 1. ROC-AUC (Positive class = 1 / Forged, higher distance = higher fraud score)
    try:
        roc_auc = float(roc_auc_score(labels, distances))
    except Exception:
        roc_auc = float('nan')

    # 2. Equal Error Rate (EER) and ROC Curve
    fpr, tpr, roc_thrs = roc_curve(labels, distances, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(roc_thrs[eer_idx])

    # 3. Decision Threshold: either fixed (passed from validation) or optimal EER
    eval_threshold = fixed_threshold if fixed_threshold is not None else eer_threshold

    # Predict: 1 (Forged) if distance >= eval_threshold, else 0 (Genuine)
    preds = (distances >= eval_threshold).astype(int)

    # Confusion matrix where label 0 = Genuine, label 1 = Forged
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    # tn: Genuine classified as Genuine (Correct Accept)
    # fp: Genuine classified as Forged  (False Rejection of Genuine -> FRR)
    # fn: Forged classified as Genuine  (False Acceptance of Forgery -> FAR)
    # tp: Forged classified as Forged   (Correct Rejection of Forgery)

    total_genuine = tn + fp
    total_forged = fn + tp

    far = float(fn / total_forged) if total_forged > 0 else 0.0
    frr = float(fp / total_genuine) if total_genuine > 0 else 0.0
    accuracy = float((tp + tn) / (total_genuine + total_forged))

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)

    return {
        "roc_auc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "evaluated_threshold": eval_threshold,
        "accuracy": accuracy,
        "far": far,
        "frr": frr,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": {
            "tn_genuine_correct": int(tn),
            "fp_genuine_rejected_frr": int(fp),
            "fn_forgery_accepted_far": int(fn),
            "tp_forgery_detected": int(tp)
        },
        "total_pairs": len(labels),
        "genuine_pairs": int(total_genuine),
        "forged_pairs": int(total_forged)
    }

def find_optimal_threshold(labels: np.ndarray, distances: np.ndarray, criterion: str = 'eer'):
    """
    Calibrate threshold strictly on validation data.
    criterion:
      - 'eer': minimizes |FAR - FRR|
      - 'youden': maximizes TPR - FPR (same as J statistic)
      - 'low_far': targets FAR <= 0.05 while minimizing FRR
    """
    labels = np.asarray(labels, dtype=int)
    distances = np.asarray(distances, dtype=float)

    fpr, tpr, thresholds = roc_curve(labels, distances, pos_label=1)
    fnr = 1.0 - tpr

    if criterion == 'eer':
        idx = np.nanargmin(np.abs(fnr - fpr))
        return float(thresholds[idx])
    elif criterion == 'youden':
        j_scores = tpr - fpr
        idx = np.nanargmax(j_scores)
        return float(thresholds[idx])
    elif criterion == 'low_far':
        # FAR is fn / (fn + tp) which in ROC corresponds to fnr (when class 1 is forged)
        # Here fpr is Genuine classified as Forged (FRR) and fnr is Forged classified as Genuine (FAR)
        valid_idxs = np.where(fnr <= 0.05)[0]
        if len(valid_idxs) > 0:
            best_sub_idx = np.argmin(fpr[valid_idxs])
            return float(thresholds[valid_idxs[best_sub_idx]])
        else:
            idx = np.argmin(fnr)
            return float(thresholds[idx])
    else:
        idx = np.nanargmin(np.abs(fnr - fpr))
        return float(thresholds[idx])
