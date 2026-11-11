#!/usr/bin/env python3
"""Valutazione: mAP sulle classi + accuratezza del captioning sugli oggetti presi.

    python qwen_bench_eval.py --data ~/vista_bench --run ~/vista_bench/runs/Qwen3.8-27B

Il ground truth ha due livelli sullo stesso bbox (convenzione Roboflow): una
categoria 'class' e una 'caption'. La detection si valuta solo sulle 'class';
il captioning solo sulle predizioni che hanno già trovato un oggetto (IoU>=0.5),
così un caption sbagliato non viene punito due volte.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# sinonimi del modello -> classi del ground truth
CLASS_MAP = {
    "car": "car", "auto": "car", "sedan": "car", "suv": "car",
    "van": "van", "truck": "van", "minivan": "van",
    "motorcycle": "motorcycle", "motorbike": "motorcycle", "scooter": "scooter",
    "bicycle": "bicycle", "bike": "bicycle",
    "person": "person", "pedestrian": "person", "people": "person",
    "emergency vehicle": "emergency vehicle", "ambulance": "emergency vehicle",
    "police car": "emergency vehicle", "fire truck": "emergency vehicle",
    "vehicle": "vehicle",
}

# gruppi di didascalie: il confronto è sul concetto, non sulla stringa
CAPTION_GROUPS = {
    "involved": ["involved in accident", "involved", "crashed", "damaged",
                 "in accident", "accident"],
    "not_involved": ["not involved", "not involved in accident", "passing",
                     "in transit", "uninvolved"],
    "parked": ["parked", "stationary", "parked on street", "parked nearby"],
    "injured": ["injured", "hurt", "lying", "on ground", "unconscious",
                "lying down", "injured lying"],
    "helping": ["helping", "assisting", "helping injured", "giving aid",
                "rescuer", "paramedic", "helping others"],
    "calling": ["calling for help", "calling", "on phone", "phoning",
                "holding phone", "taking photo"],
    "standing": ["standing", "standing nearby", "bystander", "observing",
                 "watching", "speaking", "talking"],
    "walking": ["walking", "running", "moving"],
}
CAPTION_OF = {v: k for k, vs in CAPTION_GROUPS.items() for v in vs}


def group_of(text):
    t = (text or "").strip().lower().strip(".")
    if t in CAPTION_OF:
        return CAPTION_OF[t]
    for phrase, g in CAPTION_OF.items():          # match parziale
        if phrase in t:
            return g
    return None


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax0, ay0, ax1, ay1 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx0, by0, bx1, by1 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix = np.maximum(0, np.minimum(ax1[:, None], bx1) - np.maximum(ax0[:, None], bx0))
    iy = np.maximum(0, np.minimum(ay1[:, None], by1) - np.maximum(ay0[:, None], by0))
    inter = ix * iy
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3]) - inter
    return np.where(union > 0, inter / union, 0)


def ap_for(gt_by_img, pred_by_img, thr):
    """AP a soglia IoU fissa, predizioni senza score => ordine di emissione."""
    npos = sum(len(v) for v in gt_by_img.values())
    if npos == 0:
        return float("nan"), 0, 0
    tp, fp = [], []
    for img, preds in pred_by_img.items():
        g = gt_by_img.get(img, [])
        M = iou_matrix([p["bbox"] for p in preds], [x["bbox"] for x in g])
        taken = set()
        for i in range(len(preds)):
            j = -1
            if M.shape[1]:
                order = np.argsort(-M[i])
                for cand in order:
                    if M[i, cand] >= thr and cand not in taken:
                        j = int(cand)
                        break
            if j >= 0:
                taken.add(j)
                tp.append(1); fp.append(0)
            else:
                tp.append(0); fp.append(1)
    if not tp:
        return 0.0, 0, npos
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec, prec = tp / npos, tp / np.maximum(tp + fp, 1e-9)
    ap = 0.0
    for t in np.arange(0, 1.01, 0.01):           # 101-point, come COCO
        p = prec[rec >= t].max() if (rec >= t).any() else 0
        ap += p / 101
    return float(ap), int(tp[-1]), npos


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--data", default=str(Path.home() / "vista_bench"))
    ap_.add_argument("--run", required=True)
    ap_.add_argument("--n-images", type=int, default=0,
                     help="valuta solo le prime N immagini (0 = da meta.json)")
    a = ap_.parse_args()

    data, run = Path(a.data), Path(a.run)
    coco = json.loads((data / "annotations.json").read_text())
    preds = json.loads((run / "predictions.json").read_text())
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}

    # il run può coprire solo un prefisso del benchmark: restringere il ground
    # truth alle immagini effettivamente processate, altrimenti il recall (e
    # quindi l'AP) viene schiacciato dalle immagini mai viste dal modello.
    # preferire l'elenco esplicito delle immagini già processate: con l'ordine
    # interlacciato un semplice prefisso non corrisponde più al run
    done = run / "done_ids.json"
    keep = set(json.loads(done.read_text())) if done.exists() else None
    if keep is None:
        n_done = meta.get("n_images") or a.n_images
        keep = ({im["id"] for im in coco["images"][:n_done]}
                if n_done and n_done < len(coco["images"]) else None)
    if keep is not None and len(keep) < len(coco["images"]):
        n_done = len(keep)
        coco["images"] = [im for im in coco["images"] if im["id"] in keep]
        coco["annotations"] = [an for an in coco["annotations"]
                               if an["image_id"] in keep]
        preds = [p for p in preds if p["image_id"] in keep]
        print(f"(valutazione ristretta alle prime {n_done} immagini del benchmark)")
    name = {c["id"]: c["name"] for c in coco["categories"]}
    kind = {c["id"]: c["kind"] for c in coco["categories"]}

    # ground truth: bbox -> classe, e bbox -> didascalia (stesso riquadro)
    gt_cls, gt_cap = defaultdict(list), {}
    for an in coco["annotations"]:
        key = (an["image_id"], tuple(round(v, 1) for v in an["bbox"]))
        if kind[an["category_id"]] == "class":
            gt_cls[an["image_id"]].append(
                {"bbox": an["bbox"], "cls": name[an["category_id"]], "key": key})
        else:
            gt_cap[key] = group_of(name[an["category_id"]].lstrip("_ ").strip())

    pred_by_img = defaultdict(list)
    for p in preds:
        p = dict(p)
        p["cls"] = CLASS_MAP.get(p.get("class", ""), p.get("class", ""))
        pred_by_img[p["image_id"]].append(p)

    print(f"{len(coco['images'])} immagini · {sum(len(v) for v in gt_cls.values())} "
          f"oggetti GT · {len(preds)} predizioni\n")

    # ---- detection, class-agnostic e per classe ----
    rows = []
    for thr in (0.5, 0.75):
        v, tp, n = ap_for({i: g for i, g in gt_cls.items()}, pred_by_img, thr)
        rows.append((f"AP@{thr} class-agnostic", v, tp, n))
    classes = sorted({g["cls"] for v in gt_cls.values() for g in v})
    per_cls = {}
    for c in classes:
        g = {i: [x for x in v if x["cls"] == c] for i, v in gt_cls.items()}
        g = {i: v for i, v in g.items() if v}
        p = {i: [x for x in v if x["cls"] == c] for i, v in pred_by_img.items()}
        p = {i: v for i, v in p.items() if v}
        val, tp, n = ap_for(g, p, 0.5)
        per_cls[c] = (val, tp, n)

    print("DETECTION")
    for lab, v, tp, n in rows:
        print(f"  {lab:28s} {v*100:5.1f}   ({tp}/{n} oggetti presi)")
    valid = [v for v, _, n in per_cls.values() if n and not np.isnan(v)]
    print(f"  {'mAP@0.5 (media classi)':28s} {np.mean(valid)*100:5.1f}\n")
    print("  per classe (AP@0.5):")
    for c, (v, tp, n) in sorted(per_cls.items(), key=lambda x: -x[1][2]):
        if n:
            print(f"    {c:22s} {v*100:5.1f}   {tp:5d}/{n:5d}")

    # ---- captioning sugli oggetti correttamente rilevati ----
    hit = tot = 0
    conf = defaultdict(int)
    for img, plist in pred_by_img.items():
        g = gt_cls.get(img, [])
        M = iou_matrix([p["bbox"] for p in plist], [x["bbox"] for x in g])
        taken = set()
        for i, p in enumerate(plist):
            if not M.shape[1]:
                continue
            j = int(np.argmax(M[i]))
            if M[i, j] < 0.5 or j in taken:
                continue
            taken.add(j)
            want = gt_cap.get(g[j]["key"])
            if want is None:
                continue
            tot += 1
            got = group_of(p.get("caption", ""))
            hit += got == want
            conf[(want, got or "—")] += 1
    print(f"\nCAPTIONING (solo oggetti presi a IoU>=0.5)")
    if tot:
        print(f"  accuratezza {hit/tot*100:.1f}%  ({hit}/{tot})")
        print("  confusioni principali:")
        for (w, g), n in sorted(conf.items(), key=lambda x: -x[1])[:10]:
            mark = "ok " if w == g else "NO "
            print(f"    {mark}{w:14s} -> {g:14s} {n}")
    else:
        print("  nessun oggetto preso: niente da valutare")

    if meta:
        spi = meta.get("sec_per_image") or meta.get("sec_per_image_mean") or 0
        bad = meta.get("empty_responses", meta.get("parse_failures", 0))
        print(f"\nCOSTO  {spi:.1f}s per immagine · {meta.get('tokens_per_sec', 0):.0f} tok/s · "
              f"{bad} risposte vuote/non parsabili")


if __name__ == "__main__":
    main()
