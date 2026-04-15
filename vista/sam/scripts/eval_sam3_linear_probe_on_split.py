from __future__ import annotations

"""
[DEPRECATED] Evaluate SAM 3 detections AFTER applying the linear probe.

NOTE: This script is kept for backward compatibility, but the recommended way
to evaluate linear probe predictions is to use:

    python scripts/eval_sam3_on_split.py --mode probe [--split SPLIT] [--eval_threshold THRESHOLD]
    
This provides a unified interface for both baseline and probe evaluation.

---

This script is analogous to `eval_sam3_on_split.py`, but it reads
predictions from:

    data/processed/predictions/sam3_linear_probe_yolo/<split>/*.txt

instead of:

    data/processed/predictions/sam3_yolo/<split>/*.txt

Metrics are computed using `src.eval_yolo.evaluate_yolo_predictions`,
so they are directly comparable to:
- the original SAM 3 zero-shot results, and
- the YOLO-based models from the original project.

Usage examples:
  # Single threshold evaluation (default: test split, eval_threshold from config)
  python eval_sam3_linear_probe_on_split.py
  
  # Custom threshold
  python eval_sam3_linear_probe_on_split.py --split val --eval_threshold 0.30
  
  # Threshold sweep to find optimal operating point (RECOMMENDED on val)
  python eval_sam3_linear_probe_on_split.py --split val --sweep
  
RECOMMENDED ALTERNATIVE:
  python eval_sam3_on_split.py --mode probe --split val --sweep
"""

import sys
from pathlib import Path
import argparse

# Add project root to PYTHONPATH so that "src" can be imported when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    get_labels_dir,
    PREDICTIONS_DIR,
    EVAL_THRESHOLD_DEFAULT,
)
from src.prompts import CLASS_PROMPTS  # noqa: E402
from src.eval_yolo import (  # noqa: E402
    evaluate_yolo_predictions,
    print_evaluation_summary,
)


def eval_sam3_linear_probe_on_split(
    split: str = "test",
    eval_threshold: float = EVAL_THRESHOLD_DEFAULT,
):
    """
    Evaluate SAM 3 + linear probe detections on a given split.

    Args:
        split:
            Dataset split to evaluate ('train', 'val', or 'test').
        eval_threshold:
            Score threshold applied when loading the predictions (operating point).
            This is the EVALUATION threshold, separate from the export threshold
            used during prediction generation.
            
            IMPORTANT: Should be chosen via threshold sweep on validation set
            to find the optimal operating point for the linear probe predictions.
    
    Returns:
        EvaluationResult with metrics.
    """
    # Ground-truth labels (YOLO format) for this split.
    labels_dir = get_labels_dir(split)

    # Directory containing YOLO predictions AFTER applying the linear probe.
    preds_dir = PREDICTIONS_DIR / "sam3_linear_probe_yolo" / split

    # Basic run configuration logging.
    print(f"Evaluating split '{split}' (SAM 3 + linear probe)")
    print(f"Labels directory:     {labels_dir}")
    print(f"Predictions directory:{preds_dir}")
    print(f"Evaluation threshold: {eval_threshold} (operating point)")
    print(f"  Note: This threshold is used for EVALUATION only.")
    print(f"        Export threshold (used during prediction) is separate.\n")

    # Number of classes is inferred from the prompt mapping used in the project.
    num_classes = len(CLASS_PROMPTS)

    # Run generic YOLO-style evaluation (AP, mAP, precision/recall, ...).
    result = evaluate_yolo_predictions(
        labels_dir=labels_dir,
        preds_dir=preds_dir,
        num_classes=num_classes,
        confidence_threshold=eval_threshold,
    )

    # Map class ids to human-readable names (the same text prompts used for SAM 3).
    class_names = {cid: name for cid, name in CLASS_PROMPTS.items()}

    # Pretty-print a per-class and global summary of the evaluation metrics.
    print_evaluation_summary(result, class_names=class_names)
    
    return result


def main() -> None:
    """
    CLI entry point.

    Supports both single threshold evaluation and threshold sweep mode
    to find the optimal operating point for the linear probe predictions.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate SAM3 + linear probe predictions on a split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single evaluation on test with default threshold
  python eval_sam3_linear_probe_on_split.py
  
  # Evaluate validation set with custom threshold
  python eval_sam3_linear_probe_on_split.py --split val --eval_threshold 0.30
  
  # Find optimal threshold on validation set (RECOMMENDED before testing)
  python eval_sam3_linear_probe_on_split.py --split val --sweep
  
  # Sweep with custom range
  python eval_sam3_linear_probe_on_split.py --split val --sweep --sweep_start 0.10 --sweep_end 0.80 --sweep_step 0.05
        """
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (train/val/test). Use 'val' for threshold sweeps.",
    )
    parser.add_argument(
        "--eval_threshold",
        type=float,
        default=None,
        help=f"Evaluation threshold (operating point). Default: {EVAL_THRESHOLD_DEFAULT}. "
             "Should be chosen via --sweep mode on the validation set.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Enable threshold sweep mode: test multiple thresholds to find optimal operating point. "
             "STRONGLY RECOMMENDED to run on 'val' split before testing.",
    )
    parser.add_argument(
        "--sweep_start",
        type=float,
        default=0.05,
        help="Starting threshold for sweep (default: 0.05).",
    )
    parser.add_argument(
        "--sweep_end",
        type=float,
        default=0.95,
        help="Ending threshold for sweep (default: 0.95).",
    )
    parser.add_argument(
        "--sweep_step",
        type=float,
        default=0.05,
        help="Step size for threshold sweep (default: 0.05).",
    )
    args = parser.parse_args()

    split = args.split
    eval_threshold = args.eval_threshold if args.eval_threshold is not None else EVAL_THRESHOLD_DEFAULT

    if args.sweep:
        # ============================================================
        # SWEEP MODE: Test multiple thresholds to find optimal
        # ============================================================
        print("=" * 80)
        print(f"THRESHOLD SWEEP MODE (Linear Probe)")
        print(f"Split: {split}")
        print(f"Range: {args.sweep_start:.2f} to {args.sweep_end:.2f} step {args.sweep_step:.2f}")
        print("=" * 80)
        print()
        
        import numpy as np
        
        thresholds = np.arange(args.sweep_start, args.sweep_end + args.sweep_step / 2, args.sweep_step)
        
        results = []
        labels_dir = get_labels_dir(split)
        preds_dir = PREDICTIONS_DIR / "sam3_linear_probe_yolo" / split
        num_classes = len(CLASS_PROMPTS)
        
        for thresh in thresholds:
            result = evaluate_yolo_predictions(
                labels_dir=labels_dir,
                preds_dir=preds_dir,
                num_classes=num_classes,
                confidence_threshold=thresh,
            )
            results.append({
                "threshold": thresh,
                "micro_f1": result.micro_f1,
                "map_50": result.map_50,
                "micro_precision": result.micro_precision,
                "micro_recall": result.micro_recall,
                "total_gt": result.total_gt,
                "total_pred": result.total_pred,
            })
            print(f"Threshold {thresh:.3f}: micro-F1={result.micro_f1:.4f}, mAP@50={result.map_50:.4f}, "
                  f"P={result.micro_precision:.4f}, R={result.micro_recall:.4f}")
        
        print()
        print("=" * 80)
        print("SWEEP SUMMARY (Linear Probe)")
        print("=" * 80)
        
        # Find best thresholds
        best_f1_idx = max(range(len(results)), key=lambda i: results[i]["micro_f1"])
        best_map_idx = max(range(len(results)), key=lambda i: results[i]["map_50"])
        
        best_f1 = results[best_f1_idx]
        best_map = results[best_map_idx]
        
        print(f"\nBest threshold for micro-F1:")
        print(f"  Threshold: {best_f1['threshold']:.3f}")
        print(f"  micro-F1: {best_f1['micro_f1']:.4f}")
        print(f"  mAP@50: {best_f1['map_50']:.4f}")
        print(f"  Precision: {best_f1['micro_precision']:.4f}")
        print(f"  Recall: {best_f1['micro_recall']:.4f}")
        
        print(f"\nBest threshold for mAP@50:")
        print(f"  Threshold: {best_map['threshold']:.3f}")
        print(f"  mAP@50: {best_map['map_50']:.4f}")
        print(f"  micro-F1: {best_map['micro_f1']:.4f}")
        print(f"  Precision: {best_map['micro_precision']:.4f}")
        print(f"  Recall: {best_map['micro_recall']:.4f}")
        
        print("\n" + "=" * 80)
        print("RECOMMENDATION (Linear Probe)")
        print("=" * 80)
        print(f"For production use (best micro-F1): set EVAL_THRESHOLD_DEFAULT = {best_f1['threshold']:.2f} in config.py")
        print(f"For high precision (best mAP@50):    set EVAL_THRESHOLD_DEFAULT = {best_map['threshold']:.2f} in config.py")
        print("=" * 80)
        
        # Save sweep results to file
        sweep_output_dir = PROJECT_ROOT / "results" / "threshold_sweeps"
        sweep_output_dir.mkdir(parents=True, exist_ok=True)
        sweep_output_file = sweep_output_dir / f"linear_probe_sweep_{split}.txt"
        
        with open(sweep_output_file, "w", encoding="utf-8") as f:
            f.write("Threshold Sweep Results (Linear Probe)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Split: {split}\n")
            f.write(f"Range: {args.sweep_start:.2f} to {args.sweep_end:.2f} step {args.sweep_step:.2f}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Threshold\tmicro-F1\tmAP@50\tPrecision\tRecall\n")
            for r in results:
                f.write(f"{r['threshold']:.3f}\t{r['micro_f1']:.4f}\t{r['map_50']:.4f}\t"
                       f"{r['micro_precision']:.4f}\t{r['micro_recall']:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Best threshold for micro-F1: {best_f1['threshold']:.3f} (F1={best_f1['micro_f1']:.4f})\n")
            f.write(f"Best threshold for mAP@50: {best_map['threshold']:.3f} (mAP={best_map['map_50']:.4f})\n")
            f.write("=" * 80 + "\n")
        
        print(f"\nResults saved to: {sweep_output_file}")
        
    else:
        # ============================================================
        # SINGLE THRESHOLD MODE: Evaluate at fixed threshold
        # ============================================================
        eval_sam3_linear_probe_on_split(
            split=split,
            eval_threshold=eval_threshold,
        )


if __name__ == "__main__":
    main()
