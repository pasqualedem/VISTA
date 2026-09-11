"""Annotation store: loads COCO JSONs from JSON_FINALISSIMI, keeps a working copy
per video under labels/working/, supports edits + export."""
import json
import os
import re
import threading
import time

DATASET = "/media/ssd/pasquale/VISTADataset"
LABELS_SRC = os.path.join(DATASET, "labels", "JSON_FINALISSIMI")
LABELS_WORK = os.path.join(DATASET, "labels", "working")
LABELS_EXPORT = os.path.join(DATASET, "labels", "exports")


def find_videos():
    """All mp4s under train/ and test/ (skip video_recuperati duplicates)."""
    vids = []
    for split in ("train", "test"):
        base = os.path.join(DATASET, split)
        for day in sorted(os.listdir(base)):
            d = os.path.join(base, day)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(".mp4"):
                    stem = os.path.splitext(f)[0]
                    vids.append({
                        "id": stem,
                        "split": split,
                        "day": day,
                        "path": os.path.join(d, f),
                    })
    return vids


def label_file_for(stem):
    """Match a JSON_FINALISSIMI file to a video stem, e.g.
    Json_Finale_DJI_20251210134636_0001_S_IoU050.json <-> DJI_20251210134636_0001_S"""
    if not os.path.isdir(LABELS_SRC):
        return None
    for f in os.listdir(LABELS_SRC):
        if not f.endswith(".json") or f.endswith("_validation.json"):
            continue
        m = re.match(r"Json_Finale_(.+?)_IoU\d+\.json", f)
        if m and m.group(1) == stem:
            return os.path.join(LABELS_SRC, f)
    return None


class VideoAnnotations:
    """In-memory annotations for one video, COCO-compatible.
    frames: dict frame_idx -> list of ann dicts
    ann dict: {id, frame, category_id, bbox [x,y,w,h], source, run_id?}"""

    def __init__(self, stem, num_frames, width, height):
        self.stem = stem
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.categories = []        # [{id, name}]
        self.frames = {}            # frame_idx -> [ann]
        self.next_id = 1
        self.lock = threading.RLock()
        self.dirty = False
        self.info = {}

    # ---------- loading ----------
    def load(self):
        work = os.path.join(LABELS_WORK, f"{self.stem}.json")
        src = label_file_for(self.stem)
        path = work if os.path.exists(work) else src
        if path is None:
            self.categories = [{"id": 1, "name": "object", "raw": "object", "caption": False}]
            return "none"
        with open(path) as f:
            d = json.load(f)
        self.info = d.get("info", {})
        # "_ "-prefixed categories are captions (extra duplicated bbox per object);
        # bare names (bicycle/car/emergency vehicle/person) are the classes.
        self.categories = [{"id": c["id"],
                            "name": c["name"].lstrip("_ ").strip(),
                            "raw": c["name"],
                            "caption": c["name"].startswith("_")}
                           for c in d.get("categories", [])]
        # image_id -> frame index from file_name frame_%06d.jpg
        id2frame = {}
        for im in d.get("images", []):
            m = re.search(r"frame_(\d+)", im.get("file_name", ""))
            id2frame[im["id"]] = int(m.group(1)) if m else im["id"] - 1
        for a in d.get("annotations", []):
            fr = id2frame.get(a["image_id"], a["image_id"] - 1)
            ann = {
                "id": self.next_id,
                "frame": fr,
                "category_id": a["category_id"],
                "bbox": [round(v, 2) for v in a["bbox"]],
                "source": a.get("source", "original"),
            }
            if "run_id" in a:
                ann["run_id"] = a["run_id"]
            if "track_id" in a:
                ann["track_id"] = a["track_id"]
            self.next_id += 1
            self.frames.setdefault(fr, []).append(ann)
        return "working" if path == work else "original"

    # ---------- edits ----------
    def add(self, frame, category_id, bbox, source="manual", run_id=None, track_id=None):
        with self.lock:
            ann = {"id": self.next_id, "frame": frame, "category_id": category_id,
                   "bbox": [round(float(v), 2) for v in bbox], "source": source}
            if run_id is not None:
                ann["run_id"] = run_id
            if track_id is not None:
                ann["track_id"] = track_id
            self.next_id += 1
            self.frames.setdefault(frame, []).append(ann)
            self.dirty = True
            return ann

    def update(self, ann_id, bbox=None, category_id=None):
        with self.lock:
            for anns in self.frames.values():
                for a in anns:
                    if a["id"] == ann_id:
                        if bbox is not None:
                            a["bbox"] = [round(float(v), 2) for v in bbox]
                        if category_id is not None:
                            a["category_id"] = category_id
                        if a["source"] == "original":
                            a["source"] = "edited"
                        self.dirty = True
                        return a
        return None

    def delete(self, ann_id):
        with self.lock:
            for fr, anns in self.frames.items():
                for a in anns:
                    if a["id"] == ann_id:
                        anns.remove(a)
                        self.dirty = True
                        return True
        return False

    def delete_run(self, run_id):
        n = 0
        with self.lock:
            for fr in list(self.frames):
                keep = [a for a in self.frames[fr] if a.get("run_id") != run_id]
                n += len(self.frames[fr]) - len(keep)
                if keep:
                    self.frames[fr] = keep
                else:
                    del self.frames[fr]
            if n:
                self.dirty = True
        return n

    def get_range(self, start, end):
        with self.lock:
            out = {}
            for fr in range(start, end + 1):
                if fr in self.frames and self.frames[fr]:
                    out[fr] = self.frames[fr]
            return out

    def density(self):
        with self.lock:
            counts = [0] * self.num_frames
            manual = [0] * self.num_frames
            for fr, anns in self.frames.items():
                if 0 <= fr < self.num_frames:
                    counts[fr] = len(anns)
                    manual[fr] = sum(1 for a in anns
                                     if a["source"] in ("manual", "sam3", "edited"))
            return counts, manual

    def annotated_frames(self):
        with self.lock:
            return sorted(fr for fr, anns in self.frames.items() if anns)

    # ---------- persistence ----------
    def to_coco(self):
        with self.lock:
            images, annotations = [], []
            used = sorted(self.frames)
            fr2imgid = {}
            for i, fr in enumerate(used, 1):
                fr2imgid[fr] = fr + 1  # keep original convention: image_id = frame+1
            # emit every frame like the source files (image list covers all frames)
            images = [{"id": fr + 1, "file_name": f"frame_{fr:06d}.jpg",
                       "width": self.width, "height": self.height}
                      for fr in range(self.num_frames)]
            aid = 1
            for fr in sorted(self.frames):
                for a in self.frames[fr]:
                    ann = {"id": aid, "image_id": fr + 1,
                           "category_id": a["category_id"],
                           "bbox": a["bbox"],
                           "area": round(a["bbox"][2] * a["bbox"][3], 2),
                           "segmentation": [], "iscrowd": 0,
                           "source": a["source"]}
                    if "run_id" in a:
                        ann["run_id"] = a["run_id"]
                    if "track_id" in a:
                        ann["track_id"] = a["track_id"]
                    annotations.append(ann)
                    aid += 1
            info = dict(self.info)
            info["tool"] = "vista-annotator (SAM3 semi-supervised completion)"
            info["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            return {"info": info, "licenses": [],
                    "categories": [{"id": c["id"], "name": c.get("raw", c["name"]),
                                    "supercategory": "none"} for c in self.categories],
                    "images": images, "annotations": annotations}

    def save_working(self):
        os.makedirs(LABELS_WORK, exist_ok=True)
        path = os.path.join(LABELS_WORK, f"{self.stem}.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_coco(), f)
        os.replace(tmp, path)
        self.dirty = False
        return path

    def export(self):
        os.makedirs(LABELS_EXPORT, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LABELS_EXPORT, f"{self.stem}_{ts}.json")
        with open(path, "w") as f:
            json.dump(self.to_coco(), f)
        return path
