"""VISTA Annotator — semi-supervised annotation completion tool.

Run:  cd ~/projects/VISTA/DroneAmbulanceAI/vista-annotator && \
      ~/projects/VISTA/DroneAmbulanceAI/.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8765
"""
import os
import threading

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import store as store_mod
import sam_engine
from store import VideoAnnotations, find_videos

app = FastAPI(title="VISTA Annotator")

VIDEOS = {v["id"]: v for v in find_videos()}
STORES = {}        # video_id -> VideoAnnotations
JOBS = {}          # job_id -> Job
_readers = {}      # video_id -> (lock, decord.VideoReader, _)
_store_lock = threading.Lock()

os.makedirs("/media/ssd/pasquale/tmp", exist_ok=True)


def video_meta(v):
    cap = cv2.VideoCapture(v["path"])   # container metadata only
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n, w, h, fps


def get_store(vid) -> VideoAnnotations:
    with _store_lock:
        if vid not in STORES:
            v = VIDEOS.get(vid)
            if v is None:
                raise HTTPException(404, "unknown video")
            n, w, h, fps = video_meta(v)
            v["nframes"], v["fps"] = n, fps
            s = VideoAnnotations(vid, n, w, h)
            s.loaded_from = s.load()
            STORES[vid] = s
        return STORES[vid]


def read_frame(vid, idx):
    """Index-accurate decode via decord. The DJI clips are variable-frame-rate,
    so cv2's CAP_PROP_POS_FRAMES seek (which assumes the nominal rate) drifts
    progressively out of sync with the frame indices the labels refer to —
    tens of frames by the middle of a long clip."""
    v = VIDEOS.get(vid)
    if v is None:
        raise HTTPException(404, "unknown video")
    if vid not in _readers:
        from decord import VideoReader
        _readers[vid] = [threading.Lock(), VideoReader(v["path"], num_threads=2), -2]
    lk, vr, _ = _readers[vid]
    with lk:
        if not 0 <= idx < len(vr):
            raise HTTPException(404, f"frame {idx} out of range")
        rgb = vr[idx].asnumpy()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------- API ----------------

@app.get("/api/videos")
def list_videos():
    out = []
    for vid, v in VIDEOS.items():
        has_labels = store_mod.label_file_for(vid) is not None or \
            os.path.exists(os.path.join(store_mod.LABELS_WORK, f"{vid}.json"))
        out.append({"id": vid, "split": v["split"], "day": v["day"],
                    "has_labels": has_labels, "loaded": vid in STORES})
    return out


@app.get("/api/video/{vid}/meta")
def video_info(vid: str):
    s = get_store(vid)
    v = VIDEOS[vid]
    counts, manual = s.density()
    return {"id": vid, "nframes": s.num_frames, "width": s.width,
            "height": s.height, "fps": v.get("fps", 30.0),
            "categories": s.categories, "loaded_from": getattr(s, "loaded_from", "?"),
            "density": counts, "manual": manual}


@app.get("/api/video/{vid}/frame/{idx}")
def frame_jpg(vid: str, idx: int, q: int = 85):
    frame = read_frame(vid, idx)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=300"})


@app.get("/api/video/{vid}/annotations")
def get_annots(vid: str, start: int = 0, end: int = -1):
    s = get_store(vid)
    if end < 0:
        end = s.num_frames - 1
    return s.get_range(start, end)


class AnnCreate(BaseModel):
    frame: int
    category_id: int
    bbox: list


class AnnUpdate(BaseModel):
    bbox: list | None = None
    category_id: int | None = None


@app.post("/api/video/{vid}/annotations")
def create_annot(vid: str, body: AnnCreate):
    s = get_store(vid)
    ann = s.add(body.frame, body.category_id, body.bbox, source="manual")
    s.save_working()
    return ann


@app.patch("/api/video/{vid}/annotations/{ann_id}")
def update_annot(vid: str, ann_id: int, body: AnnUpdate):
    s = get_store(vid)
    ann = s.update(ann_id, bbox=body.bbox, category_id=body.category_id)
    if ann is None:
        raise HTTPException(404, "annotation not found")
    s.save_working()
    return ann


@app.delete("/api/video/{vid}/annotations/{ann_id}")
def delete_annot(vid: str, ann_id: int):
    s = get_store(vid)
    if not s.delete(ann_id):
        raise HTTPException(404, "annotation not found")
    s.save_working()
    return {"ok": True}


@app.delete("/api/video/{vid}/run/{run_id}")
def delete_run(vid: str, run_id: str):
    s = get_store(vid)
    n = s.delete_run(run_id)
    s.save_working()
    return {"deleted": n}


class PropagateReq(BaseModel):
    start: int
    end: int
    ann_ids: list | None = None   # None = all boxes on start frame
    exclude_ids: list | None = None   # caption twins hidden by caption mode
    extra_cats: dict | None = None    # ann_id(str) -> [caption category ids]


@app.post("/api/video/{vid}/propagate")
def propagate(vid: str, body: PropagateReq):
    s = get_store(vid)
    if body.end <= body.start:
        raise HTTPException(400, "end must be > start")
    if body.end >= s.num_frames:
        body.end = s.num_frames - 1
    if body.end - body.start > 3000:
        raise HTTPException(400, "chunk too large (max 3000 frames per run)")
    frame_anns = s.get_range(body.start, body.start).get(body.start, [])
    if body.ann_ids:
        frame_anns = [a for a in frame_anns if a["id"] in body.ann_ids]
    if body.exclude_ids:
        frame_anns = [a for a in frame_anns if a["id"] not in body.exclude_ids]
    if not frame_anns:
        raise HTTPException(400, f"no annotations on frame {body.start} to propagate")
    extra = body.extra_cats or {}
    boxes = [{"ann_id": a["id"], "category_id": a["category_id"], "bbox": a["bbox"],
              "extra_cats": extra.get(str(a["id"]), [])}
             for a in frame_anns]
    job = sam_engine.Job(vid, body.start, body.end, boxes)
    JOBS[job.id] = job
    t = threading.Thread(
        target=sam_engine.run_propagation,
        args=(job, VIDEOS[vid]["path"], s),
        kwargs={"save_cb": s.save_working}, daemon=True)
    t.start()
    return job.as_dict()


@app.get("/api/job/{job_id}")
def job_status(job_id: str, since_frame: int = -1):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    d = job.as_dict()
    if since_frame >= 0:
        d["new"] = [a for a in job.created if a["frame"] > since_frame]
    return d


@app.post("/api/job/{job_id}/cancel")
def job_cancel(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    job.cancel_flag = True
    return {"ok": True}


@app.post("/api/video/{vid}/save")
def save(vid: str):
    s = get_store(vid)
    return {"path": s.save_working()}


@app.post("/api/video/{vid}/export")
def export(vid: str):
    s = get_store(vid)
    path = s.export()
    return {"path": path, "download": f"/api/video/{vid}/export/download?f={os.path.basename(path)}"}


@app.get("/api/video/{vid}/export/download")
def export_download(vid: str, f: str):
    path = os.path.join(store_mod.LABELS_EXPORT, os.path.basename(f))
    if not os.path.exists(path):
        raise HTTPException(404, "export not found")
    return FileResponse(path, filename=f, media_type="application/json")


app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"),
                           html=True), name="static")
