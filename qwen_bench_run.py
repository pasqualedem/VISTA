#!/usr/bin/env python3
"""Detection + brief captioning con Qwen3.8-27B, versione con batching.

Due scelte dettate dal costo:
  - formato di uscita compatto (una riga per oggetto invece di JSON): ~15 token
    per oggetto invece di ~40, quindi meno tempo e meno troncamenti;
  - generazione a batch: il decoding sequenziale lascia la GPU quasi ferma,
    un batch la riempie senza costare quasi nulla in più per immagine.

    python qwen_bench_run.py --gpu 0 --batch 8 --limit 24
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

SYSTEM_PROMPT = """You are an operator supervising a drone flying over a road accident scene. Detect every relevant object.

Objects to detect: cars, vans, trucks, motorcycles, bicycles, emergency vehicles, and people.

For each object output ONE line, nothing else:
x1,y1,x2,y2 <class> | <caption>

- x1,y1,x2,y2: integers, XYXY, normalised to the 0-1000 range of the image.
- <class>: one of car, van, truck, motorcycle, bicycle, emergency vehicle, person
- <caption>: 1-3 words on its state or role, e.g. involved in accident, not involved, parked, injured, standing, helping, calling for help, walking.

Example:
120,340,180,395 car | involved in accident
455,210,470,250 person | injured
No preamble, no numbering, no markdown. One object per line."""

LINE_RE = re.compile(
    r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s+([^|\n]+?)\s*\|\s*(.+?)\s*$")


def parse_lines(txt):
    out = []
    for line in txt.splitlines():
        m = LINE_RE.match(line)
        if m:
            x1, y1, x2, y2 = (int(m.group(i)) for i in range(1, 5))
            out.append((x1, y1, x2, y2, m.group(5).strip().lower(),
                        m.group(6).strip().lower()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "vista_bench"))
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stratify", action="store_true",
                    help="interlaccia i video: ogni prefisso è rappresentativo")
    ap.add_argument("--max-pixels", type=int, default=1280 * 720)
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    data = Path(a.data)
    coco = json.loads((data / "annotations.json").read_text())
    images = coco["images"]
    if a.stratify:
        # interlaccia i quattro video: così ogni prefisso dei risultati è già
        # un campione rappresentativo, e non serve un run separato ridotto
        from itertools import zip_longest
        byv = {}
        for im in images:
            byv.setdefault(im["video"], []).append(im)
        images = [x for grp in zip_longest(*byv.values()) for x in grp
                  if x is not None]
    if a.limit:
        images = images[:a.limit]
    out = Path(a.out) if a.out else data / "runs" / "batched"
    out.mkdir(parents=True, exist_ok=True)

    proc = AutoProcessor.from_pretrained(a.model, max_pixels=a.max_pixels,
                                         padding_side="left")
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa").eval()
    print(f"caricato in {time.time()-t0:.0f}s, "
          f"{torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)

    preds, times, toks, empty = [], [], [], 0
    for start in range(0, len(images), a.batch):
        chunk = images[start:start + a.batch]
        convs, pils = [], []
        for rec in chunk:
            img = Image.open(data / "images" / rec["file_name"]).convert("RGB")
            pils.append(img)
            convs.append([
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "List the objects."}]}])
        inputs = proc.apply_chat_template(
            convs, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", padding=True, enable_thinking=False
        ).to(model.device)

        t = time.time()
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=a.max_new_tokens,
                                 do_sample=False)
        dt = time.time() - t
        new = gen[:, inputs["input_ids"].shape[1]:]
        texts = proc.batch_decode(new, skip_special_tokens=True)
        toks.append(int((new != proc.tokenizer.pad_token_id).sum()))
        times.append(dt)

        for rec, txt in zip(chunk, texts):
            objs = parse_lines(txt)
            if not objs:
                empty += 1
            W, H = rec["width"], rec["height"]
            for x1, y1, x2, y2, cls, cap in objs:
                x0, x1_ = sorted((x1 / 1000 * W, x2 / 1000 * W))
                y0, y1_ = sorted((y1 / 1000 * H, y2 / 1000 * H))
                preds.append({"image_id": rec["id"],
                              "bbox": [x0, y0, x1_ - x0, y1_ - y0],
                              "score": 1.0, "class": cls, "caption": cap})

        done = start + len(chunk)
        per_img = sum(times) / done
        print(f"[{done}/{len(images)}] batch {dt:5.1f}s  "
              f"{per_img:5.2f}s/img  {sum(toks)/sum(times):5.1f} tok/s  "
              f"ETA {per_img*(len(images)-done)/60:5.1f} min  box {len(preds)}",
              flush=True)
        (out / "predictions.json").write_text(json.dumps(preds))
        (out / "done_ids.json").write_text(
            json.dumps([r["id"] for r in images[:done]]))

    (out / "predictions.json").write_text(json.dumps(preds))
    (out / "meta.json").write_text(json.dumps({
        "model": a.model, "batch": a.batch, "max_pixels": a.max_pixels,
        "max_new_tokens": a.max_new_tokens, "format": "compact-lines",
        "n_images": len(images), "n_preds": len(preds), "empty_responses": empty,
        "sec_per_image": sum(times) / len(images),
        "tokens_per_sec": sum(toks) / sum(times),
        "sec_total": sum(times), "stratified": a.stratify,
        "image_ids": [r["id"] for r in images],
        "system_prompt": SYSTEM_PROMPT}, indent=1))
    print(f"\n{len(images)} immagini, {len(preds)} box, "
          f"{sum(times)/len(images):.2f}s per immagine, "
          f"{empty} risposte vuote -> {out}")


if __name__ == "__main__":
    main()
