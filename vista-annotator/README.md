# VISTA Annotator

Semi-supervised annotation completion for the VISTADataset: browse videos with
their COCO labels, find unannotated gaps, and fill them by propagating existing
boxes with **SAM3** (box-prompted video tracking).

## Run

```bash
cd ~/projects/VISTA/DroneAmbulanceAI/vista-annotator
# pin to the two RTX 4000 Ada cards (the 4090 is off limits); preferred card first
export CUDA_VISIBLE_DEVICES=GPU-284ce1c9-9672-01e8-f0c0-1cb2b34ba604,GPU-89ce7cc3-1e86-869f-bb98-2ac15b2f2c43
~/projects/VISTA/DroneAmbulanceAI/.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8765
```

Open http://localhost:8765 (or over Tailscale: http://100.103.109.76:8765).

## Data layout

- Videos: `/media/ssd/pasquale/VISTADataset/{train,test}/<day>/*.mp4`
- Source labels (read-only): `labels/JSON_FINALISSIMI/Json_Finale_<stem>_IoU050.json`
- Working copies (all edits go here, source never touched): `labels/working/<stem>.json`
- Exports: `labels/exports/<stem>_<timestamp>.json` (COCO, same schema as source
  plus `source` / `run_id` / `track_id` fields on annotations)

## Workflow

1. Pick a video (badge shows whether it has labels). Timeline: **red = no
   annotations** (gap), cyan = annotation density, orange strip = manual/SAM3 boxes.
2. `next gap ⇥` (or `G`) jumps to the next unannotated frame. Choose the start
   frame — typically the last annotated frame before the gap (`⌖ current`).
3. `→ gap end` suggests the end frame (gap end + 5 frames of overlap for sanity).
4. `▶ propagate` — SAM3 tracks every box on the start frame (or only the selected
   one) and writes a bbox per frame. Watch it live (`follow new frames`), cancel
   any time, `⎌ undo run` removes everything a run created.
   Boxes are **not** created on frames that already contain an equivalent box
   (IoU > 0.5), and a track stops when SAM3 loses the object.
5. Review: orange boxes = SAM3. Drag to move, drag handles to resize, `Del` to
   delete, `N` + drag to draw a new box (category from the dropdown).
6. `⬇ export COCO` writes a timestamped JSON and gives a download link.
   The working copy is auto-saved after every edit anyway.

## Notes

- SAM3 tracker (`facebook/sam3` checkpoint, ~3.4 GB) loads lazily on the first
  propagation, on the GPU with the most free memory. One job at a time.
- Max 3000 frames per propagation run; frames are extracted to
  `/media/ssd/pasquale/tmp` and deleted afterwards.
- The source JSONs contain many *duplicated* boxes (same bbox twice on a frame);
  the tracker treats each as an object, the duplicate usually dies immediately
  ("lost" at the first frame). Harmless, but worth deduplicating eventually.

## Caption mode (default: on)

The dataset encodes an object's caption as a **second, identical bbox** whose
category name is `_ `-prefixed (Roboflow workaround); bare categories
(`bicycle`, `car`, `emergency vehicle`, `person`) are the classes. With the
*caption mode* flag on, the app pairs the twins (IoU > 0.85, one class + one
caption): a single box is shown with the caption in quotes below it, the
annotation list shows one row per object, moving/resizing/deleting acts on both
boxes, new boxes can be given an optional caption, and **SAM3 propagation
tracks the object once but emits both boxes on every new frame** — so the
exported JSON keeps the class+caption duplication. Toggle it off to see the raw
boxes.
