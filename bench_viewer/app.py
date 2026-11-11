"""Visualizzatore del benchmark VISTA: ground truth vs predizioni, fianco a fianco.

    cd ~/projects/VISTA/DroneAmbulanceAI/bench_viewer
    ../.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8766

Il matching GT/predizioni e le metriche sono calcolati una volta all'avvio
(riusando la stessa logica di qwen_bench_eval.py) e poi serviti per immagine.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qwen_bench_eval import CLASS_MAP, group_of, iou_matrix, ap_for  # noqa: E402

DATA = Path(os.environ.get("BENCH_DATA", "/media/ssd/pasquale/vista_bench"))
RUNS = Path(os.environ.get("BENCH_RUNS",
                           str(Path.home() / "projects/VISTA/results")))
IOU = 0.5

app = FastAPI(title="VISTA benchmark viewer")

coco = json.loads((DATA / "annotations.json").read_text())
CAT = {c["id"]: c for c in coco["categories"]}
IMAGES = coco["images"]
IMG_BY_ID = {im["id"]: im for im in IMAGES}

# ---- ground truth: classe + didascalia sullo stesso riquadro ----
GT = defaultdict(list)
_cap_by_key = {}
for an in coco["annotations"]:
    key = (an["image_id"], tuple(round(v, 1) for v in an["bbox"]))
    c = CAT[an["category_id"]]
    if c["kind"] == "caption":
        _cap_by_key[key] = c["name"].lstrip("_ ").strip()
for an in coco["annotations"]:
    c = CAT[an["category_id"]]
    if c["kind"] != "class":
        continue
    key = (an["image_id"], tuple(round(v, 1) for v in an["bbox"]))
    GT[an["image_id"]].append({"bbox": an["bbox"], "cls": c["name"],
                               "caption": _cap_by_key.get(key)})

RUNS_CACHE = {}


def list_runs():
    return sorted(p.name for p in RUNS.iterdir()
                  if (p / "predictions.json").exists())


def load_run(name):
    """Carica un run e precalcola il matching per immagine."""
    if name in RUNS_CACHE:
        return RUNS_CACHE[name]
    d = RUNS / name
    if not (d / "predictions.json").exists():
        raise HTTPException(404, f"run '{name}' non trovato")
    preds = json.loads((d / "predictions.json").read_text())
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    done = d / "done_ids.json"
    done_ids = set(json.loads(done.read_text())) if done.exists() else \
        {p["image_id"] for p in preds}

    by_img = defaultdict(list)
    for p in preds:
        q = dict(p)
        q["cls"] = CLASS_MAP.get(q.get("class", ""), q.get("class", ""))
        by_img[q["image_id"]].append(q)

    # matching greedy per IoU, per immagine: etichetta ogni box TP/FP/FN
    per_img = {}
    conf = defaultdict(int)
    cap_hit = cap_tot = 0
    for iid in done_ids:
        g = GT.get(iid, [])
        p = by_img.get(iid, [])
        M = iou_matrix([x["bbox"] for x in p], [x["bbox"] for x in g])
        taken, pstat, gstat = {}, [], ["FN"] * len(g)
        for i in range(len(p)):
            j, best = -1, IOU
            for cand in range(len(g)):
                if cand in taken:
                    continue
                if M[i, cand] >= best:
                    best, j = M[i, cand], cand
            if j >= 0:
                taken[j] = i
                gstat[j] = "TP"
                cls_ok = p[i]["cls"] == g[j]["cls"]
                want = group_of(g[j]["caption"]) if g[j]["caption"] else None
                got = group_of(p[i].get("caption", ""))
                if want:
                    cap_tot += 1
                    cap_hit += want == got
                    conf[(want, got or "—")] += 1
                pstat.append({"status": "TP", "iou": round(float(M[i, j]), 3),
                              "match": j, "cls_ok": cls_ok,
                              "cap_ok": (want == got) if want else None,
                              "gt_caption": g[j]["caption"], "gt_cls": g[j]["cls"]})
            else:
                pstat.append({"status": "FP", "iou": 0.0, "match": None,
                              "cls_ok": None, "cap_ok": None,
                              "gt_caption": None, "gt_cls": None})
        per_img[iid] = {"pred_status": pstat, "gt_status": gstat,
                        "n_gt": len(g), "n_pred": len(p),
                        "tp": sum(1 for s in gstat if s == "TP"),
                        "fn": sum(1 for s in gstat if s == "FN"),
                        "fp": sum(1 for s in pstat if s["status"] == "FP")}

    gt_sub = {i: GT.get(i, []) for i in done_ids}
    pr_sub = {i: by_img.get(i, []) for i in done_ids if by_img.get(i)}
    metrics = {}
    for thr, lab in ((0.5, "AP50"), (0.75, "AP75")):
        v, tp, n = ap_for(gt_sub, pr_sub, thr)
        metrics[lab] = {"ap": round(v * 100, 2), "matched": tp, "total": n}
    per_cls = {}
    for c in sorted({x["cls"] for v in gt_sub.values() for x in v}):
        g = {i: [x for x in v if x["cls"] == c] for i, v in gt_sub.items()}
        g = {i: v for i, v in g.items() if v}
        pp = {i: [x for x in v if x["cls"] == c] for i, v in pr_sub.items()}
        pp = {i: v for i, v in pp.items() if v}
        v, tp, n = ap_for(g, pp, 0.5)
        if n:
            per_cls[c] = {"ap": round(v * 100, 2), "matched": tp, "total": n}
    vals = [x["ap"] for x in per_cls.values()]
    metrics["mAP50"] = round(float(np.mean(vals)), 2) if vals else 0.0
    metrics["per_class"] = per_cls
    metrics["caption_acc"] = round(cap_hit / cap_tot * 100, 1) if cap_tot else None
    metrics["caption_n"] = cap_tot
    metrics["confusions"] = sorted(
        ({"want": w, "got": g, "n": n, "ok": w == g} for (w, g), n in conf.items()),
        key=lambda x: -x["n"])[:15]
    metrics["n_images"] = len(done_ids)
    metrics["n_pred"] = len(preds)
    metrics["n_gt"] = sum(len(v) for v in gt_sub.values())
    metrics["cost"] = {k: meta.get(k) for k in
                       ("sec_per_image", "tokens_per_sec", "empty_responses",
                        "batch", "model", "max_new_tokens")}

    RUNS_CACHE[name] = {"preds": by_img, "per_img": per_img,
                        "metrics": metrics, "done": sorted(done_ids)}
    return RUNS_CACHE[name]


@app.get("/api/runs")
def api_runs():
    return list_runs()


@app.get("/api/run/{name}/metrics")
def api_metrics(name: str):
    return load_run(name)["metrics"]


@app.get("/api/run/{name}/images")
def api_images(name: str, video: str = "", order: str = "index"):
    r = load_run(name)
    out = []
    for iid in r["done"]:
        im = IMG_BY_ID[iid]
        if video and im["video"] != video:
            continue
        s = r["per_img"][iid]
        out.append({"id": iid, "file": im["file_name"], "video": im["video"],
                    "frame": im["frame"], "n_gt": s["n_gt"], "n_pred": s["n_pred"],
                    "tp": s["tp"], "fn": s["fn"], "fp": s["fp"]})
    if order == "worst":          # più oggetti mancati prima
        out.sort(key=lambda x: (-x["fn"], -x["fp"]))
    elif order == "best":
        out.sort(key=lambda x: (x["fn"], x["fp"]))
    return {"videos": sorted({im["video"] for im in IMAGES}), "images": out}


@app.get("/api/run/{name}/image/{iid}")
def api_image(name: str, iid: int):
    r = load_run(name)
    if iid not in r["per_img"]:
        raise HTTPException(404, "immagine non nel run")
    im = IMG_BY_ID[iid]
    s = r["per_img"][iid]
    gt = [dict(g, status=st) for g, st in zip(GT.get(iid, []), s["gt_status"])]
    pred = [dict(p, **st) for p, st in zip(r["preds"].get(iid, []), s["pred_status"])]
    return {"image": im, "gt": gt, "pred": pred, "stats":
            {k: s[k] for k in ("n_gt", "n_pred", "tp", "fn", "fp")}}


@app.get("/api/frame/{iid}")
def api_frame(iid: int):
    im = IMG_BY_ID.get(iid)
    if im is None:
        raise HTTPException(404, "immagine sconosciuta")
    return FileResponse(DATA / "images" / im["file_name"], media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True),
          name="static")
