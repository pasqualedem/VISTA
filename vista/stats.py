"""
Dataset statistics for Ultralytics YOLO-format datasets.

Main entry point: ``analyze_yolo_dataset(yaml_path, ...)``

Each "section" can be enabled/disabled independently, and virtually every
visual/output detail is customisable through keyword arguments.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Optional heavy imports – graceful degradation
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    from PIL import Image as PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _resolve_split_paths(
    yaml_path: Path,
    cfg: dict,
    splits: Sequence[str],
) -> Dict[str, List[Path]]:
    """
    Return ``{split: [image_path, ...]}`` for each requested split.

    Handles the three forms allowed by Ultralytics:
    * directory path          → glob for images
    * .txt file               → read lines
    * list of dirs/files
    """
    dataset_root = Path(cfg.get("path", yaml_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = yaml_path.parent / dataset_root

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def _collect(entry) -> List[Path]:
        if entry is None:
            return []
        if isinstance(entry, list):
            paths: List[Path] = []
            for e in entry:
                paths.extend(_collect(e))
            return paths
        p = Path(entry)
        if not p.is_absolute():
            p = dataset_root / p
        if p.is_dir():
            return sorted(f for f in p.rglob("*") if f.suffix.lower() in image_extensions)
        if p.suffix == ".txt" and p.is_file():
            lines = p.read_text().splitlines()
            return [Path(l.strip()) for l in lines if l.strip()]
        return []

    result: Dict[str, List[Path]] = {}
    for split in splits:
        entry = cfg.get(split)
        result[split] = _collect(entry)
    return result


def _label_path_for_image(image_path: Path) -> Path:
    """Convert an image path to its corresponding YOLO label .txt path."""
    parts = image_path.parts
    # Replace the last 'images' segment with 'labels'
    new_parts = []
    replaced = False
    for part in reversed(parts):
        if not replaced and part.lower() == "images":
            new_parts.append("labels")
            replaced = True
        else:
            new_parts.append(part)
    label_path = Path(*reversed(new_parts)).with_suffix(".txt")
    return label_path


def _parse_label_file(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Parse a YOLO label file and return a list of (class_id, cx, cy, w, h).
    Skips malformed lines silently.
    """
    annotations = []
    if not label_path.is_file():
        return annotations
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            annotations.append((cls, cx, cy, w, h))
        except ValueError:
            continue
    return annotations


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_svg(fig, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, format="svg", bbox_inches="tight", dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class AnnotationRecord:
    """One bounding-box annotation."""
    split: str
    image: str
    cls: int
    cx: float
    cy: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def aspect_ratio(self) -> float:
        return self.w / self.h if self.h > 0 else float("nan")

    @property
    def x1(self) -> float:
        return self.cx - self.w / 2

    @property
    def y1(self) -> float:
        return self.cy - self.h / 2


@dataclass
class ImageRecord:
    """One image entry."""
    split: str
    path: Path
    n_annotations: int
    width: Optional[int] = None
    height: Optional[int] = None

    @property
    def name(self) -> str:
        return self.path.name


# ---------------------------------------------------------------------------
# Section constants
# ---------------------------------------------------------------------------

SECTION_OVERVIEW        = "overview"
SECTION_CLASS_DIST      = "class_distribution"
SECTION_BBOX_STATS      = "bbox_stats"
SECTION_SPATIAL         = "spatial_heatmap"
SECTION_SIZE_SCATTER    = "size_scatter"
SECTION_ASPECT_RATIO    = "aspect_ratio"
SECTION_ANNOTATIONS_PER_IMAGE = "annotations_per_image"
SECTION_COOCCURRENCE    = "cooccurrence"
SECTION_SPLIT_COMPARE   = "split_comparison"
SECTION_IMAGE_SIZES     = "image_sizes"

ALL_SECTIONS = [
    SECTION_OVERVIEW,
    SECTION_CLASS_DIST,
    SECTION_BBOX_STATS,
    SECTION_SPATIAL,
    SECTION_SIZE_SCATTER,
    SECTION_ASPECT_RATIO,
    SECTION_ANNOTATIONS_PER_IMAGE,
    SECTION_COOCCURRENCE,
    SECTION_SPLIT_COMPARE,
    SECTION_IMAGE_SIZES,
]


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------

class YOLODatasetAnalyzer:
    """
    Compute and persist statistics for a YOLO-format dataset.

    Parameters
    ----------
    yaml_path : str | Path
        Path to the dataset YAML file.
    output_dir : str | Path
        Where to save outputs (plots, JSON, CSV, …).
    splits : list[str]
        Which splits to analyse.  Defaults to ``["train", "val", "test"]``.
    sections : list[str] | None
        Which analysis sections to run.  ``None`` → run all.
    read_image_sizes : bool
        Whether to open every image to read its pixel dimensions.
        Requires Pillow and is slow on large datasets.
    save_json : bool
        Save aggregate statistics as ``stats.json``.
    save_csv : bool
        Save per-annotation table as ``annotations.csv`` (requires pandas).
    verbose : bool
        Print progress messages.

    Plot customisation
    ------------------
    figsize : tuple[float, float]
        Default figure size for all plots.
    dpi : int
        DPI used when rasterising SVGs (affects detail in viewers that
        render SVGs at a fixed pixel size).
    style : str
        Matplotlib style, e.g. ``"seaborn-v0_8-whitegrid"``.
    palette : list[str] | None
        Colour palette used for per-class plots.  ``None`` → tab20.
    heatmap_bins : int
        Grid resolution for the spatial heatmap.
    heatmap_cmap : str
        Matplotlib colourmap name for the heatmap.
    scatter_alpha : float
        Opacity for scatter-plot markers.
    scatter_marker_size : float
        Marker size for the size scatter plot.
    hist_bins : int
        Number of bins for histogram plots.
    bar_orientation : str
        ``"vertical"`` or ``"horizontal"`` for class-distribution bar chart.
    show_values_on_bars : bool
        Annotate bar charts with numeric values.
    title_fontsize : int
        Font size for plot titles.
    label_fontsize : int
        Font size for axis labels.
    tick_fontsize : int
        Font size for tick labels.
    """

    def __init__(
        self,
        yaml_path: str | Path,
        output_dir: str | Path = "dataset_stats",
        *,
        splits: List[str] = None,
        sections: Optional[List[str]] = None,
        read_image_sizes: bool = False,
        save_json: bool = True,
        save_csv: bool = True,
        verbose: bool = True,
        # ── visual ──────────────────────────────────────────────────────────
        figsize: Tuple[float, float] = (10, 6),
        dpi: int = 150,
        style: str = "seaborn-v0_8-whitegrid",
        palette: Optional[List[str]] = None,
        heatmap_bins: int = 64,
        heatmap_cmap: str = "hot",
        scatter_alpha: float = 0.35,
        scatter_marker_size: float = 8.0,
        hist_bins: int = 40,
        bar_orientation: str = "horizontal",
        show_values_on_bars: bool = True,
        title_fontsize: int = 14,
        label_fontsize: int = 11,
        tick_fontsize: int = 9,
    ) -> None:
        self.yaml_path = Path(yaml_path).resolve()
        self.output_dir = Path(output_dir)
        self.splits = splits or ["train", "val", "test"]
        self.sections = set(sections) if sections is not None else set(ALL_SECTIONS)
        self.read_image_sizes = read_image_sizes
        self.save_json = save_json
        self.save_csv = save_csv
        self.verbose = verbose

        # visual
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        self.palette = palette
        self.heatmap_bins = heatmap_bins
        self.heatmap_cmap = heatmap_cmap
        self.scatter_alpha = scatter_alpha
        self.scatter_marker_size = scatter_marker_size
        self.hist_bins = hist_bins
        self.bar_orientation = bar_orientation
        self.show_values_on_bars = show_values_on_bars
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_fontsize = tick_fontsize

        # populated by _load
        self.cfg: dict = {}
        self.class_names: Dict[int, str] = {}
        self.images: List[ImageRecord] = []
        self.annotations: List[AnnotationRecord] = []
        self.stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Run the full analysis pipeline and return the statistics dict."""
        self._log("Loading dataset YAML …")
        self._load()
        self._log(f"Parsing {len(self.images)} images …")
        self._parse()

        _ensure_dir(self.output_dir)

        if SECTION_OVERVIEW in self.sections:
            self._section_overview()
        if SECTION_CLASS_DIST in self.sections and _HAS_MPL:
            self._section_class_distribution()
        if SECTION_BBOX_STATS in self.sections:
            self._section_bbox_stats()
        if SECTION_SPATIAL in self.sections and _HAS_MPL:
            self._section_spatial_heatmap()
        if SECTION_SIZE_SCATTER in self.sections and _HAS_MPL:
            self._section_size_scatter()
        if SECTION_ASPECT_RATIO in self.sections and _HAS_MPL:
            self._section_aspect_ratio()
        if SECTION_ANNOTATIONS_PER_IMAGE in self.sections and _HAS_MPL:
            self._section_annotations_per_image()
        if SECTION_COOCCURRENCE in self.sections and _HAS_MPL:
            self._section_cooccurrence()
        if SECTION_SPLIT_COMPARE in self.sections and _HAS_MPL:
            self._section_split_comparison()
        if SECTION_IMAGE_SIZES in self.sections and self.read_image_sizes and _HAS_MPL:
            self._section_image_sizes()

        if self.save_json:
            self._save_json()
        if self.save_csv and _HAS_PANDAS:
            self._save_csv()

        self._log(f"Done.  Outputs saved to: {self.output_dir}")
        return self.stats

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        self.cfg = _load_yaml(self.yaml_path)

        raw_names = self.cfg.get("names", {})
        if isinstance(raw_names, dict):
            self.class_names = {int(k): str(v) for k, v in raw_names.items()}
        elif isinstance(raw_names, list):
            self.class_names = {i: str(n) for i, n in enumerate(raw_names)}
        else:
            self.class_names = {}

        split_paths = _resolve_split_paths(self.yaml_path, self.cfg, self.splits)
        for split, paths in split_paths.items():
            for img_path in paths:
                self.images.append(ImageRecord(split=split, path=img_path, n_annotations=0))

    def _parse(self) -> None:
        for rec in self.images:
            label_path = _label_path_for_image(rec.path)
            annots = _parse_label_file(label_path)
            rec.n_annotations = len(annots)

            if self.read_image_sizes and _HAS_PIL and rec.path.is_file():
                try:
                    with PILImage.open(rec.path) as im:
                        rec.width, rec.height = im.size
                except Exception:
                    pass

            for cls, cx, cy, w, h in annots:
                self.annotations.append(
                    AnnotationRecord(
                        split=rec.split,
                        image=rec.name,
                        cls=cls,
                        cx=cx,
                        cy=cy,
                        w=w,
                        h=h,
                    )
                )

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _section_overview(self) -> None:
        n_classes = len(self.class_names)
        total_images = len(self.images)
        total_annots = len(self.annotations)

        per_split: Dict[str, dict] = {}
        for split in self.splits:
            imgs = [r for r in self.images if r.split == split]
            annots = [a for a in self.annotations if a.split == split]
            empty = sum(1 for r in imgs if r.n_annotations == 0)
            per_split[split] = {
                "images": len(imgs),
                "annotations": len(annots),
                "empty_images": empty,
                "annotations_per_image_mean": (
                    round(len(annots) / len(imgs), 3) if imgs else 0.0
                ),
            }

        overview = {
            "yaml": str(self.yaml_path),
            "num_classes": n_classes,
            "class_names": self.class_names,
            "total_images": total_images,
            "total_annotations": total_annots,
            "splits": per_split,
        }
        self.stats["overview"] = overview
        self._log_section("Overview", overview)

    def _section_class_distribution(self) -> None:
        class_counts: Dict[int, int] = defaultdict(int)
        for a in self.annotations:
            class_counts[a.cls] += 1

        sorted_ids = sorted(class_counts, key=lambda c: class_counts[c], reverse=True)
        names = [self._cls_name(c) for c in sorted_ids]
        counts = [class_counts[c] for c in sorted_ids]

        self.stats["class_distribution"] = {self._cls_name(c): class_counts[c] for c in sorted_ids}

        colors = self._palette(len(names))
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)
            if self.bar_orientation == "horizontal":
                bars = ax.barh(names, counts, color=colors)
                ax.set_xlabel("Annotation count", fontsize=self.label_fontsize)
                ax.set_ylabel("Class", fontsize=self.label_fontsize)
                if self.show_values_on_bars:
                    for bar, val in zip(bars, counts):
                        ax.text(
                            bar.get_width() + max(counts) * 0.005,
                            bar.get_y() + bar.get_height() / 2,
                            str(val),
                            va="center",
                            fontsize=self.tick_fontsize,
                        )
            else:
                bars = ax.bar(names, counts, color=colors)
                ax.set_ylabel("Annotation count", fontsize=self.label_fontsize)
                ax.set_xlabel("Class", fontsize=self.label_fontsize)
                plt.xticks(rotation=45, ha="right", fontsize=self.tick_fontsize)
                if self.show_values_on_bars:
                    for bar, val in zip(bars, counts):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(counts) * 0.005,
                            str(val),
                            ha="center",
                            fontsize=self.tick_fontsize,
                        )
            ax.tick_params(axis="both", labelsize=self.tick_fontsize)
            ax.set_title("Class Distribution", fontsize=self.title_fontsize)
            fig.tight_layout()
            _save_svg(fig, self.output_dir / "class_distribution.svg", dpi=self.dpi)
        self._log("  Saved class_distribution.svg")

    def _section_bbox_stats(self) -> None:
        if not self.annotations:
            return

        ws = np.array([a.w for a in self.annotations])
        hs = np.array([a.h for a in self.annotations])
        areas = ws * hs
        ars = np.where(hs > 0, ws / hs, np.nan)

        def _summary(arr: np.ndarray) -> dict:
            valid = arr[~np.isnan(arr)]
            return {
                "mean":   round(float(np.mean(valid)), 5),
                "std":    round(float(np.std(valid)),  5),
                "min":    round(float(np.min(valid)),  5),
                "max":    round(float(np.max(valid)),  5),
                "median": round(float(np.median(valid)), 5),
                "p25":    round(float(np.percentile(valid, 25)), 5),
                "p75":    round(float(np.percentile(valid, 75)), 5),
                "p95":    round(float(np.percentile(valid, 95)), 5),
            }

        # Per-class breakdown
        per_class: Dict[str, dict] = {}
        for cls_id in sorted(self.class_names):
            cls_annots = [a for a in self.annotations if a.cls == cls_id]
            if not cls_annots:
                continue
            cws = np.array([a.w for a in cls_annots])
            chs = np.array([a.h for a in cls_annots])
            per_class[self._cls_name(cls_id)] = {
                "count":  len(cls_annots),
                "width":  _summary(cws),
                "height": _summary(chs),
                "area":   _summary(cws * chs),
            }

        self.stats["bbox_stats"] = {
            "global": {
                "width":        _summary(ws),
                "height":       _summary(hs),
                "area":         _summary(areas),
                "aspect_ratio": _summary(ars),
            },
            "per_class": per_class,
        }
        self._log("  bbox_stats computed.")

    def _section_spatial_heatmap(self) -> None:
        if not self.annotations:
            return

        cxs = np.array([a.cx for a in self.annotations])
        cys = np.array([a.cy for a in self.annotations])
        bins = self.heatmap_bins

        heatmap, xedges, yedges = np.histogram2d(
            cxs, cys,
            bins=bins,
            range=[[0, 1], [0, 1]],
        )

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(7, 7))
            im = ax.imshow(
                heatmap.T,
                origin="lower",
                extent=[0, 1, 0, 1],
                cmap=self.heatmap_cmap,
                aspect="equal",
                interpolation="bilinear",
            )
            fig.colorbar(im, ax=ax, label="Annotation count")
            ax.set_xlabel("cx (normalised)", fontsize=self.label_fontsize)
            ax.set_ylabel("cy (normalised)", fontsize=self.label_fontsize)
            ax.set_title("Bounding-Box Centre Spatial Heatmap", fontsize=self.title_fontsize)
            ax.tick_params(labelsize=self.tick_fontsize)
            fig.tight_layout()
            _save_svg(fig, self.output_dir / "spatial_heatmap.svg", dpi=self.dpi)
        self._log("  Saved spatial_heatmap.svg")

        # Per-class heatmaps (one per class)
        cls_dir = _ensure_dir(self.output_dir / "spatial_heatmaps_per_class")
        for cls_id, cls_name in self.class_names.items():
            cls_ann = [a for a in self.annotations if a.cls == cls_id]
            if not cls_ann:
                continue
            cxs_c = np.array([a.cx for a in cls_ann])
            cys_c = np.array([a.cy for a in cls_ann])
            hm, _, _ = np.histogram2d(cxs_c, cys_c, bins=bins, range=[[0, 1], [0, 1]])
            safe_name = cls_name.replace(" ", "_").replace("/", "_")
            with plt.style.context(self.style):
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(hm.T, origin="lower", extent=[0, 1, 0, 1],
                               cmap=self.heatmap_cmap, aspect="equal",
                               interpolation="bilinear")
                fig.colorbar(im, ax=ax, label="Annotation count")
                ax.set_title(f"Spatial Heatmap – {cls_name}", fontsize=self.title_fontsize)
                ax.set_xlabel("cx", fontsize=self.label_fontsize)
                ax.set_ylabel("cy", fontsize=self.label_fontsize)
                ax.tick_params(labelsize=self.tick_fontsize)
                fig.tight_layout()
                _save_svg(fig, cls_dir / f"spatial_{safe_name}.svg", dpi=self.dpi)
            plt.close("all")
        self._log(f"  Saved per-class spatial heatmaps to {cls_dir.name}/")

    def _section_size_scatter(self) -> None:
        if not self.annotations:
            return

        ws = np.array([a.w for a in self.annotations])
        hs = np.array([a.h for a in self.annotations])
        cls_ids = np.array([a.cls for a in self.annotations])

        unique_cls = sorted(set(cls_ids.tolist()))
        colors = self._palette(len(unique_cls))
        cls_color = {c: colors[i] for i, c in enumerate(unique_cls)}

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)
            for cls_id in unique_cls:
                mask = cls_ids == cls_id
                ax.scatter(
                    ws[mask], hs[mask],
                    s=self.scatter_marker_size,
                    alpha=self.scatter_alpha,
                    color=cls_color[cls_id],
                    label=self._cls_name(cls_id),
                )
            ax.set_xlabel("Normalised width", fontsize=self.label_fontsize)
            ax.set_ylabel("Normalised height", fontsize=self.label_fontsize)
            ax.set_title("Bounding-Box Width vs Height", fontsize=self.title_fontsize)
            ax.tick_params(labelsize=self.tick_fontsize)
            if len(unique_cls) <= 20:
                ax.legend(fontsize=self.tick_fontsize, markerscale=3,
                          loc="upper right", framealpha=0.7)
            fig.tight_layout()
            _save_svg(fig, self.output_dir / "size_scatter.svg", dpi=self.dpi)
        self._log("  Saved size_scatter.svg")

    def _section_aspect_ratio(self) -> None:
        if not self.annotations:
            return

        ars = np.array([a.aspect_ratio for a in self.annotations])
        ars = ars[~np.isnan(ars) & ~np.isinf(ars)]

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.hist(ars, bins=self.hist_bins, color="#4c72b0", edgecolor="white", linewidth=0.4)
            ax.set_xlabel("Aspect ratio (w / h)", fontsize=self.label_fontsize)
            ax.set_ylabel("Count", fontsize=self.label_fontsize)
            ax.set_title("Bounding-Box Aspect-Ratio Distribution", fontsize=self.title_fontsize)
            ax.tick_params(labelsize=self.tick_fontsize)
            fig.tight_layout()
            _save_svg(fig, self.output_dir / "aspect_ratio_hist.svg", dpi=self.dpi)
        self._log("  Saved aspect_ratio_hist.svg")

    def _section_annotations_per_image(self) -> None:
        counts = [r.n_annotations for r in self.images]
        if not counts:
            return

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.hist(counts, bins=range(0, max(counts) + 2), align="left",
                    color="#55a868", edgecolor="white", linewidth=0.4)
            ax.set_xlabel("Annotations per image", fontsize=self.label_fontsize)
            ax.set_ylabel("Number of images", fontsize=self.label_fontsize)
            ax.set_title("Annotations-per-Image Distribution", fontsize=self.title_fontsize)
            ax.tick_params(labelsize=self.tick_fontsize)
            fig.tight_layout()
            _save_svg(fig, self.output_dir / "annotations_per_image.svg", dpi=self.dpi)

        self.stats["annotations_per_image"] = {
            "mean":   round(float(np.mean(counts)), 3),
            "std":    round(float(np.std(counts)),  3),
            "min":    int(np.min(counts)),
            "max":    int(np.max(counts)),
            "median": float(np.median(counts)),
            "empty_images": int(sum(1 for c in counts if c == 0)),
        }
        self._log("  Saved annotations_per_image.svg")

    def _section_cooccurrence(self) -> None:
        n_cls = len(self.class_names)
        if n_cls < 2:
            return

        cls_ids_sorted = sorted(self.class_names.keys())
        idx_map = {c: i for i, c in enumerate(cls_ids_sorted)}

        # Build image → set-of-classes map
        img_classes: Dict[str, set] = defaultdict(set)
        for a in self.annotations:
            img_classes[a.image].add(a.cls)

        matrix = np.zeros((n_cls, n_cls), dtype=int)
        for cls_set in img_classes.values():
            cls_list = sorted(cls_set)
            for i, ci in enumerate(cls_list):
                for cj in cls_list[i:]:
                    ii, jj = idx_map[ci], idx_map[cj]
                    matrix[ii, jj] += 1
                    if ii != jj:
                        matrix[jj, ii] += 1

        names = [self._cls_name(c) for c in cls_ids_sorted]

        # Normalise to fraction of images containing class i that also contain j
        diag = np.diag(matrix).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.where(diag[:, None] > 0, matrix / diag[:, None], 0.0)

        with plt.style.context(self.style):
            fig_w = max(7, n_cls * 0.55 + 1.5)
            fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.85))
            im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, label="Fraction of row-class images")
            ax.set_xticks(range(n_cls))
            ax.set_yticks(range(n_cls))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=self.tick_fontsize)
            ax.set_yticklabels(names, fontsize=self.tick_fontsize)
            ax.set_title("Class Co-occurrence Matrix\n(row: images containing class; value: fraction also containing column class)",
                         fontsize=self.title_fontsize)
            fig.tight_layout()
            _save_svg(fig, self.output_dir / "cooccurrence_matrix.svg", dpi=self.dpi)
        self._log("  Saved cooccurrence_matrix.svg")

    def _section_split_comparison(self) -> None:
        splits_present = [s for s in self.splits if any(r.split == s for r in self.images)]
        if len(splits_present) < 2:
            return

        # Per-split class counts
        cls_ids_sorted = sorted(self.class_names.keys())
        n_cls = len(cls_ids_sorted)
        idx_map_local = {c: i for i, c in enumerate(cls_ids_sorted)}
        split_counts: Dict[str, np.ndarray] = {}
        for split in splits_present:
            arr = np.zeros(n_cls, dtype=int)
            for a in self.annotations:
                if a.split == split and a.cls in idx_map_local:
                    arr[idx_map_local[a.cls]] += 1
            split_counts[split] = arr

        names = [self._cls_name(c) for c in cls_ids_sorted]
        x = np.arange(n_cls)
        bar_w = 0.8 / len(splits_present)
        colors = self._palette(len(splits_present))

        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=(max(10, n_cls * 0.5 + 3), 6))
            for i, (split, arr) in enumerate(split_counts.items()):
                offsets = x + i * bar_w - (len(splits_present) - 1) * bar_w / 2
                ax.bar(offsets, arr, width=bar_w * 0.9, label=split, color=colors[i], alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=self.tick_fontsize)
            ax.set_ylabel("Annotation count", fontsize=self.label_fontsize)
            ax.set_title("Per-Class Annotation Count by Split", fontsize=self.title_fontsize)
            ax.legend(fontsize=self.tick_fontsize)
            ax.tick_params(labelsize=self.tick_fontsize)
            fig.tight_layout()
            _save_svg(fig, self.output_dir / "split_comparison.svg", dpi=self.dpi)
        self._log("  Saved split_comparison.svg")

    def _section_image_sizes(self) -> None:
        if not _HAS_PIL:
            self._log("  Pillow not available – skipping image_sizes section.")
            return

        widths  = [r.width  for r in self.images if r.width  is not None]
        heights = [r.height for r in self.images if r.height is not None]
        if not widths:
            return

        self.stats["image_sizes"] = {
            "width":  {"mean": round(float(np.mean(widths)),  1),
                       "std":  round(float(np.std(widths)),   1),
                       "min":  int(np.min(widths)),
                       "max":  int(np.max(widths))},
            "height": {"mean": round(float(np.mean(heights)), 1),
                       "std":  round(float(np.std(heights)),  1),
                       "min":  int(np.min(heights)),
                       "max":  int(np.max(heights))},
        }

        with plt.style.context(self.style):
            fig, axes = plt.subplots(1, 2, figsize=self.figsize)
            for ax, data, label in zip(axes, [widths, heights], ["Width (px)", "Height (px)"]):
                ax.hist(data, bins=self.hist_bins, color="#c44e52", edgecolor="white", linewidth=0.4)
                ax.set_xlabel(label, fontsize=self.label_fontsize)
                ax.set_ylabel("Image count", fontsize=self.label_fontsize)
                ax.tick_params(labelsize=self.tick_fontsize)
            fig.suptitle("Image Dimensions Distribution", fontsize=self.title_fontsize)
            fig.tight_layout()
            _save_svg(fig, self.output_dir / "image_sizes.svg", dpi=self.dpi)
        self._log("  Saved image_sizes.svg")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_json(self) -> None:
        out_path = self.output_dir / "stats.json"
        with open(out_path, "w") as fh:
            json.dump(self.stats, fh, indent=2, default=str)
        self._log(f"  Saved {out_path.name}")

    def _save_csv(self) -> None:
        if not _HAS_PANDAS:
            self._log("  pandas not available – skipping CSV export.")
            return
        rows = [
            {
                "split": a.split,
                "image": a.image,
                "class_id":   a.cls,
                "class_name": self._cls_name(a.cls),
                "cx": a.cx,
                "cy": a.cy,
                "w":  a.w,
                "h":  a.h,
                "area":         round(a.area, 6),
                "aspect_ratio": round(a.aspect_ratio, 4) if not np.isnan(a.aspect_ratio) else None,
            }
            for a in self.annotations
        ]
        df = pd.DataFrame(rows)
        out_path = self.output_dir / "annotations.csv"
        df.to_csv(out_path, index=False)
        self._log(f"  Saved {out_path.name}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _cls_name(self, cls_id: int) -> str:
        return self.class_names.get(cls_id, str(cls_id))

    def _palette(self, n: int) -> List[str]:
        if self.palette and len(self.palette) >= n:
            return self.palette[:n]
        cmap = plt.get_cmap("tab20" if n <= 20 else "hsv")
        return [matplotlib.colors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _log_section(self, title: str, data: Any) -> None:
        if not self.verbose:
            return
        print(f"\n── {title} ──")
        print(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def analyze_yolo_dataset(
    yaml_path: str | Path,
    output_dir: str | Path = "dataset_stats",
    *,
    splits: Optional[List[str]] = None,
    sections: Optional[List[str]] = None,
    read_image_sizes: bool = False,
    save_json: bool = True,
    save_csv: bool = True,
    verbose: bool = True,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 150,
    style: str = "seaborn-v0_8-whitegrid",
    palette: Optional[List[str]] = None,
    heatmap_bins: int = 64,
    heatmap_cmap: str = "hot",
    scatter_alpha: float = 0.35,
    scatter_marker_size: float = 8.0,
    hist_bins: int = 40,
    bar_orientation: str = "horizontal",
    show_values_on_bars: bool = True,
    title_fontsize: int = 14,
    label_fontsize: int = 11,
    tick_fontsize: int = 9,
) -> Dict[str, Any]:
    """
    Analyse a YOLO-format dataset and save statistics + plots.

    Parameters
    ----------
    yaml_path : str | Path
        Path to the Ultralytics YOLO dataset YAML (must contain ``names``,
        ``train``/``val``/``test``, and optionally ``path``).
    output_dir : str | Path
        Directory where all outputs (SVG plots, JSON, CSV) are saved.
        Created if it does not exist.
    splits : list[str], optional
        Splits to analyse.  Defaults to ``["train", "val", "test"]``.
    sections : list[str], optional
        Subset of sections to run.  Pass ``None`` to run all.
        Available sections::

            "overview"               – image/annotation counts per split
            "class_distribution"     – bar chart of annotation counts per class
            "bbox_stats"             – numeric summary of w/h/area/AR globally and per class
            "spatial_heatmap"        – 2-D heatmap of bbox centres (global + per class)
            "size_scatter"           – w vs h scatter coloured by class
            "aspect_ratio"           – histogram of aspect ratios
            "annotations_per_image" – histogram of per-image annotation counts
            "cooccurrence"           – normalised class co-occurrence matrix
            "split_comparison"       – grouped bar chart comparing splits
            "image_sizes"            – histogram of image pixel dimensions
                                       (requires ``read_image_sizes=True`` and Pillow)

    read_image_sizes : bool
        Open every image to read pixel dimensions.  Slow on large datasets;
        requires Pillow.
    save_json : bool
        Persist aggregate stats as ``<output_dir>/stats.json``.
    save_csv : bool
        Persist per-annotation table as ``<output_dir>/annotations.csv``
        (requires pandas).
    verbose : bool
        Print progress messages.
    figsize : (float, float)
        Default figure size ``(width, height)`` in inches.
    dpi : int
        DPI for SVG rasterisation.
    style : str
        Matplotlib style name.
    palette : list[str] | None
        Explicit hex/named colour list for per-class plots.
    heatmap_bins : int
        Grid resolution for spatial heatmaps.
    heatmap_cmap : str
        Matplotlib colourmap for heatmaps.
    scatter_alpha : float
        Marker opacity for scatter plots.
    scatter_marker_size : float
        Marker size (``s``) for scatter plots.
    hist_bins : int
        Number of bins for histogram plots.
    bar_orientation : str
        ``"horizontal"`` (default) or ``"vertical"`` for bar charts.
    show_values_on_bars : bool
        Annotate bar charts with count labels.
    title_fontsize : int
        Font size for plot titles.
    label_fontsize : int
        Font size for axis labels.
    tick_fontsize : int
        Font size for tick labels.

    Returns
    -------
    dict
        Nested dictionary containing all computed statistics.

    Examples
    --------
    >>> stats = analyze_yolo_dataset("data/my_dataset.yaml", "out/stats")

    >>> # Only class distribution and bbox statistics, custom colours
    >>> stats = analyze_yolo_dataset(
    ...     "data/my_dataset.yaml",
    ...     sections=["class_distribution", "bbox_stats"],
    ...     bar_orientation="vertical",
    ...     palette=["#e63946", "#457b9d", "#2a9d8f"],
    ... )
    """
    analyser = YOLODatasetAnalyzer(
        yaml_path=yaml_path,
        output_dir=output_dir,
        splits=splits,
        sections=sections,
        read_image_sizes=read_image_sizes,
        save_json=save_json,
        save_csv=save_csv,
        verbose=verbose,
        figsize=figsize,
        dpi=dpi,
        style=style,
        palette=palette,
        heatmap_bins=heatmap_bins,
        heatmap_cmap=heatmap_cmap,
        scatter_alpha=scatter_alpha,
        scatter_marker_size=scatter_marker_size,
        hist_bins=hist_bins,
        bar_orientation=bar_orientation,
        show_values_on_bars=show_values_on_bars,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
    )
    return analyser.run()
