from __future__ import annotations

"""
Train a simple linear probe on top of SAM 3 detection features.

This script expects a dataset created by `build_linear_probe_dataset.py`,
stored as:

    data/processed/linear_probe/sam3_linear_probe_<split>.npz

The .npz file must contain:
    - features: float32 array of shape (N, D)
    - targets: int64 array of shape (N,) with values {0, 1}
    - class_ids: int64 array of shape (N,) with values in [0, num_classes-1]

For each class c, we select all samples with class_ids == c and train a
separate logistic regression model:

    p(y=1 | x, c) = sigmoid( w_c^T x + b_c )

The resulting weight matrix and biases are saved to:

    data/processed/linear_probe/sam3_linear_probe_weights.npz

This file can later be used to re-score SAM 3 detections on any split.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompts import CLASS_PROMPTS  # noqa: E402


@dataclass
class LogisticRegressionWeights:
    """Weights and bias for a binary logistic regression model."""
    weights: np.ndarray  # shape (D,)
    bias: float          # scalar


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    # Clip to avoid overflow in exp
    x_clipped = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def train_ridge_regression(
    X: np.ndarray,
    Y: np.ndarray,
    l2: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a Ridge regression model using closed-form solution.
    
    Args:
        X: Input features of shape (N, M)
        Y: Target values of shape (N, K) where K is output dimension
        l2: L2 regularization coefficient
    
    Returns:
        Tuple of (weights, bias):
            - weights: shape (M, K)
            - bias: shape (K,)
    """
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    
    # Add bias column (column of ones)
    ones = np.ones((X.shape[0], 1), dtype=np.float32)
    Xb = np.concatenate([X, ones], axis=1)  # (N, M+1)
    
    # Ridge solution: (X^T X + λI)^-1 X^T Y
    M1 = Xb.shape[1]
    A = Xb.T @ Xb
    I = np.eye(M1, dtype=np.float32)
    I[-1, -1] = 0.0  # Don't regularize the bias term
    
    Wb = np.linalg.solve(A + l2 * I, Xb.T @ Y)  # (M+1, K)
    
    W = Wb[:-1, :]  # (M, K)
    b = Wb[-1, :]   # (K,)
    
    return W, b


def train_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    num_epochs: int = 500,
    learning_rate: float = 0.1,
    l2_weight: float = 1e-4,
) -> LogisticRegressionWeights:
    """
    Train a binary logistic regression model with L2 regularization.

    Args:
        x:
            Input features of shape (N, D).
        y:
            Binary targets of shape (N,), values in {0, 1}.
        num_epochs:
            Number of gradient descent iterations.
        learning_rate:
            Step size for gradient updates.
        l2_weight:
            L2 regularization coefficient (lambda).

    Returns:
        LogisticRegressionWeights with learned parameters.
    """
    # Ensure consistent dtypes for training
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    n_samples, n_features = x.shape
    # Initialize weights and bias to zero
    w = np.zeros(n_features, dtype=np.float32)
    b = 0.0

    for epoch in range(num_epochs):
        # Forward pass: compute logits and probabilities
        logits = x @ w + b
        probs = _sigmoid(logits)

        # Compute gradients of the loss w.r.t. weights and bias
        error = probs - y  # shape (N,)
        grad_w = (x.T @ error) / float(n_samples) + l2_weight * w
        grad_b = float(np.mean(error))

        # Gradient descent parameter update
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    return LogisticRegressionWeights(weights=w, bias=float(b))


def train_linear_probe(
    split: str = "train",
    num_epochs: int = 500,
    learning_rate: float = 0.1,
    l2_weight: float = 1e-4,
    bbox_l2: float = 1e-2,
) -> Path:
    """
    Train one logistic regression model per class on the specified split.

    Args:
        split:
            Dataset split used for training ('train' by default).
        num_epochs:
            Number of gradient descent iterations for each class model.
        learning_rate:
            Learning rate for gradient descent.
        l2_weight:
            L2 regularization strength for logistic regression.
        bbox_l2:
            L2 regularization strength for bbox Ridge regression.

    Returns:
        Path to the saved .npz file with all class-wise weights.
    """
    # Path to the linear-probe dataset (.npz) built from SAM 3 detections.
    in_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "linear_probe"
        / f"sam3_linear_probe_{split}.npz"
    )

    if not in_path.exists():
        # The dataset must be created beforehand by build_linear_probe_dataset.py
        raise FileNotFoundError(
            f"Linear probe dataset not found: {in_path}. "
            f"Please run 'build_linear_probe_dataset.py' first for split='{split}'."
        )

    # Load features, binary targets (TP/FP), class ids, and boxes.
    data = np.load(in_path)
    features = data["features"]          # (N, D)
    targets = data["targets"]            # (N,)
    class_ids = data["class_ids"]        # (N,)
    pred_boxes = data["pred_boxes"]      # (N, 4) in YOLO format (cx,cy,w,h)
    gt_boxes = data["gt_boxes"]          # (N, 4) in YOLO format, NaN for negatives

    num_samples, feature_dim = features.shape
    num_classes = len(CLASS_PROMPTS)

    print(f"Training linear probe on split='{split}'")
    print(f"  Input file:      {in_path}")
    print(f"  Num samples:     {num_samples}")
    print(f"  Feature dim:     {feature_dim}")
    print(f"  Num classes:     {num_classes}")
    print(f"  Num epochs:      {num_epochs}")
    print(f"  Learning rate:   {learning_rate}")
    print(f"  L2 weight:       {l2_weight}")
    
    # Sanity check: features should be 257-d (256-d query embeddings + 1-d score)
    if feature_dim != 257:
        print(
            f"\n  WARNING: Expected feature_dim=257 (256-d query + score), got {feature_dim}. "
            f"This may indicate old geometric features or a pipeline mismatch."
        )

    # Storage for class-wise weights and biases (classification)
    all_weights = np.zeros((num_classes, feature_dim), dtype=np.float32)
    all_biases = np.zeros((num_classes,), dtype=np.float32)
    
    # Storage for bbox refinement weights and biases
    # Input dimension: feature_dim + 4 (features + pred_box)
    # Output dimension: 4 (dx, dy, dw, dh)
    bbox_input_dim = feature_dim + 4
    all_bbox_weights = np.zeros((num_classes, bbox_input_dim, 4), dtype=np.float32)
    all_bbox_biases = np.zeros((num_classes, 4), dtype=np.float32)

    # For reporting purposes: class_id -> (num_pos, num_neg)
    class_stats: Dict[int, Tuple[int, int]] = {}

    # Train a separate logistic regression for each class.
    for class_id in range(num_classes):
        # Select only samples belonging to this class.
        mask = class_ids == class_id
        x_c = features[mask]
        y_c = targets[mask]

        num_c = x_c.shape[0]
        num_pos = int(np.sum(y_c))
        num_neg = int(num_c - num_pos)
        class_stats[class_id] = (num_pos, num_neg)

        print(
            f"\nClass {class_id} ({CLASS_PROMPTS[class_id]}): "
            f"{num_c} samples -> {num_pos} pos, {num_neg} neg"
        )

        # Determine if we can train logistic regression (need both pos and neg samples)
        train_cls = not (num_pos == 0 or num_neg == 0)
        
        if train_cls:
            # Train class-specific logistic regression on (x_c, y_c).
            model = train_logistic_regression(
                x_c,
                y_c,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                l2_weight=l2_weight,
            )

            all_weights[class_id] = model.weights
            all_biases[class_id] = model.bias

            # Simple training accuracy at threshold 0.5 (mainly for debugging).
            logits = x_c @ model.weights + model.bias
            preds = (logits >= 0.0).astype(np.int64)
            acc = float(np.mean(preds == y_c))
            print(f"  Training accuracy (thr=0.5): {acc:.4f}")
        else:
            print(
                "  WARNING: class has only one label type (all pos or all neg). "
                "Skipping logistic regression, but will attempt bbox regressor if positives available."
            )
        
        # Train bbox refinement regressor (only on positives)
        mask_pos = mask & (targets == 1)
        num_pos_bbox = int(np.sum(mask_pos))
        
        if num_pos_bbox > 0:
            # Get positive samples
            gt_pos = gt_boxes[mask_pos]      # (num_pos, 4)
            pred_pos = pred_boxes[mask_pos]  # (num_pos, 4)
            feats_pos = features[mask_pos]   # (num_pos, D)
            
            # Compute bbox deltas (dx, dy, dw, dh) using standard parametrization
            # Unpack YOLO format: (cx, cy, w, h)
            pred_cx, pred_cy, pred_w, pred_h = pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], pred_pos[:, 3]
            gt_cx, gt_cy, gt_w, gt_h = gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], gt_pos[:, 3]
            
            # Avoid division by zero or log of zero
            pred_w = np.maximum(pred_w, 1e-6)
            pred_h = np.maximum(pred_h, 1e-6)
            gt_w = np.maximum(gt_w, 1e-6)
            gt_h = np.maximum(gt_h, 1e-6)
            
            dx = (gt_cx - pred_cx) / pred_w
            dy = (gt_cy - pred_cy) / pred_h
            dw = np.log(gt_w / pred_w)
            dh = np.log(gt_h / pred_h)
            
            # Clamp deltas to reduce outliers and improve stability
            dx = np.clip(dx, -0.5, 0.5)
            dy = np.clip(dy, -0.5, 0.5)
            dw = np.clip(dw, -2.0, 2.0)
            dh = np.clip(dh, -2.0, 2.0)
            
            Y_deltas = np.stack([dx, dy, dw, dh], axis=1).astype(np.float32)  # (num_pos, 4)
            
            # Input: concatenate features and predicted box
            X_reg = np.concatenate([feats_pos, pred_pos], axis=1)  # (num_pos, D+4)
            
            # Train Ridge regression
            bbox_w, bbox_b = train_ridge_regression(X_reg, Y_deltas, l2=bbox_l2)
            
            all_bbox_weights[class_id] = bbox_w
            all_bbox_biases[class_id] = bbox_b
            
            # Compute training MSE for debugging
            Y_pred = X_reg @ bbox_w + bbox_b
            mse = float(np.mean((Y_pred - Y_deltas) ** 2))
            print(f"  BBox refinement: {num_pos_bbox} positives, training MSE = {mse:.6f}")
        else:
            print(f"  BBox refinement: no positives, skipping.")

    # Ensure output directory exists.
    out_dir = PROJECT_ROOT / "data" / "processed" / "linear_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save all class-wise weights and biases in a single .npz file.
    out_path = out_dir / "sam3_linear_probe_weights.npz"
    np.savez_compressed(
        out_path,
        weights=all_weights,
        biases=all_biases,
        bbox_weights=all_bbox_weights,
        bbox_biases=all_bbox_biases,
    )

    print(f"\nSaved linear-probe weights to: {out_path}")
    for class_id, (num_pos, num_neg) in class_stats.items():
        total = max(num_pos + num_neg, 1)
        ratio = num_pos / float(total)
        print(
            f"  Class {class_id}: {num_pos} pos, {num_neg} neg "
            f"(pos ratio = {ratio:.3f})"
        )

    return out_path


def main() -> None:
    """
    Entry point for command-line use.

    By default, this trains the linear probe on the 'train' split
    using a fixed set of hyperparameters.
    """
    split = "train"
    num_epochs = 500
    learning_rate = 0.1
    l2_weight = 1e-4
    bbox_l2 = 1e-2

    train_linear_probe(
        split=split,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        l2_weight=l2_weight,
        bbox_l2=bbox_l2,
    )


if __name__ == "__main__":
    main()
