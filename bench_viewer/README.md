# VISTA benchmark viewer

Confronto fianco a fianco fra ground truth e predizioni sui keyframe annotati a
mano del VISTADataset, con le metriche del run.

## Avvio

```bash
cd ~/projects/VISTA/DroneAmbulanceAI/bench_viewer
../.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8766
```

http://localhost:8766

Percorsi sovrascrivibili con `BENCH_DATA` (default
`/media/ssd/pasquale/vista_bench`) e `BENCH_RUNS` (default
`~/projects/VISTA/results`). Ogni sottocartella di `BENCH_RUNS` che contiene
`predictions.json` compare nel menu a tendina dei run.

## Cosa mostra

- **Due riquadri sincronizzati**: a sinistra il ground truth, a destra le
  predizioni, stessa inquadratura, stesso zoom e stesso pan.
- **Colori**: verde = oggetto trovato (TP), arancio = falso positivo,
  rosso tratteggiato = oggetto mancato (FN). Sulle predizioni, `✗cls` segnala
  la classe sbagliata e `✗cap` la didascalia sbagliata.
- **Metriche del run** a destra: AP@0.5 e AP@0.75 class-agnostic, mAP@0.5 medio
  sulle classi, AP per classe, accuratezza del captioning con le confusioni più
  frequenti, e i numeri di costo (s/immagine, tok/s) letti da `meta.json`.
- **Statistiche del singolo fotogramma**: TP/FN/FP, recall e precisione locali.

## Navigazione

`←` `→` un fotogramma (con `Shift` dieci), rotella per lo zoom, trascinamento
per il pan, doppio clic per rientrare. L'elenco a sinistra si filtra per video e
si riordina per **peggiori prima** (più oggetti mancati), che è il modo rapido
per capire dove il modello sbaglia. La spunta *solo errori* nasconde i
fotogrammi perfetti.

Il matching GT/predizioni usa la stessa logica di `qwen_bench_eval.py`
(greedy per IoU, soglia 0.5), quindi i numeri mostrati coincidono con quelli
dello script da riga di comando.
