#!/usr/bin/env python3
"""Estrae il sottoinsieme di benchmark: solo i fotogrammi annotati a mano.

Nei JSON_FINALISSIMI le annotazioni umane sono conservate verbatim sui
fotogrammi multipli di 30 (validation: `original_annotations_kept_on_multiple_
of_30_frames`); gli intermedi sono ricostruiti per matching con box Qwen e non
sono ground truth. Qui si tengono solo i primi.

I video sono a frame rate variabile: la decodifica va fatta per indice con
decord, non con il seek di cv2, altrimenti i fotogrammi scivolano rispetto agli
indici a cui si riferiscono le annotazioni.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

DS = Path("/media/ssd/pasquale/VISTADataset")
LAB = DS / "labels" / "JSON_FINALISSIMI"
STRIDE = 30


def video_for(stem):
    for split in ("test", "train"):
        for p in (DS / split).glob("*/*.mp4"):
            if p.stem == stem:
                return p
    raise FileNotFoundError(stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/media/ssd/pasquale/vista_bench")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--every", type=int, default=1,
                    help="tieni 1 keyframe ogni N (1 = tutti)")
    a = ap.parse_args()

    from decord import VideoReader
    from PIL import Image

    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    # tassonomia unificata per NOME: i quattro file usano id diversi per gli
    # stessi nomi, e nomi che iniziano con "_" sono didascalie, non classi.
    names = set()
    for lf in sorted(LAB.glob("*IoU050.json")):
        names |= {c["name"] for c in json.loads(lf.read_text())["categories"]}
    cls_names = sorted(n for n in names if not n.startswith("_"))
    cap_names = sorted(n for n in names if n.startswith("_"))
    categories = ([{"id": i, "name": n, "kind": "class"}
                   for i, n in enumerate(cls_names, 1)] +
                  [{"id": i, "name": n, "kind": "caption"}
                   for i, n in enumerate(cap_names, len(cls_names) + 1)])
    cat_id = {c["name"]: c["id"] for c in categories}
    print(f"tassonomia unificata: {len(cls_names)} classi, "
          f"{len(cap_names)} didascalie")

    images, annotations = [], []
    img_id = ann_id = 0
    stats = []

    for lf in sorted(LAB.glob("*IoU050.json")):
        stem = re.match(r"Json_Finale_(.+)_IoU\d+\.json", lf.name).group(1)
        d = json.loads(lf.read_text())
        local = {c["id"]: c["name"] for c in d["categories"]}

        # image_id -> indice di fotogramma, dal file_name
        id2frame = {}
        for im in d["images"]:
            m = re.search(r"frame_(\d+)", im["file_name"])
            id2frame[im["id"]] = int(m.group(1)) if m else im["id"] - 1

        by_frame = defaultdict(list)
        for an in d["annotations"]:
            by_frame[id2frame[an["image_id"]]].append(an)

        keys = [f for f in sorted(by_frame)
                if f % STRIDE == 0 and by_frame[f]]
        keys = keys[::a.every]

        vr = VideoReader(str(video_for(stem)), num_threads=8)
        kept_ann = 0
        for f in keys:
            if f >= len(vr):
                continue
            img_id += 1
            name = f"{stem}_{f:06d}.jpg"
            im = Image.fromarray(vr[f].asnumpy())
            im.save(out / "images" / name, quality=a.quality)
            images.append({"id": img_id, "file_name": name,
                           "width": im.width, "height": im.height,
                           "video": stem, "frame": f})
            for an in by_frame[f]:
                ann_id += 1
                annotations.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": cat_id[local[an["category_id"]]],
                    "bbox": an["bbox"], "area": an["area"],
                    "segmentation": [], "iscrowd": 0})
                kept_ann += 1
        stats.append((stem, len(vr), len(keys), kept_ann))
        print(f"{stem}: {len(keys)} keyframe, {kept_ann} annotazioni")

    n_cls = sum(1 for c in categories if c["kind"] == "class")
    coco = {"info": {"description": "VISTA benchmark - solo fotogrammi "
                                    f"annotati a mano (multipli di {STRIDE})",
                     "stride": STRIDE, "every": a.every,
                     "note": "categorie 'class' = classe dell'oggetto, "
                             "'caption' = descrizione contestuale sullo stesso "
                             "bbox duplicato (convenzione Roboflow)"},
            "licenses": [], "categories": categories,
            "images": images, "annotations": annotations}
    (out / "annotations.json").write_text(json.dumps(coco))

    mb = sum(p.stat().st_size for p in (out / "images").glob("*.jpg")) / 1e6
    print(f"\ntotale: {len(images)} fotogrammi, {len(annotations)} annotazioni, "
          f"{len(categories)} categorie, {mb:.0f} MB -> {out}")


if __name__ == "__main__":
    main()
