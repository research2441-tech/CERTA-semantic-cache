# CERTA: Verification-Weighted Conformal Risk Control and Trust-Gated Admission for Cooperative Semantic Caching in 6G Edge LLM Serving

Code for the manuscript of the same name. A seeded discrete-event simulator of a six-cell cooperative 6G MEC deployment that serves LLM queries from a semantic cache, offloads misses to an edge or cloud LLM tier, and is attacked by cache-poisoning campaigns and Byzantine calibration cells.

## Files

| File | Purpose |
|---|---|
| `certa_data.py` | Builds `certa_assets.npz`: QQP paraphrase intents, hard negatives, target-as-prefix adversarial keys, PAWS shift pairs, all-MiniLM-L6-v2 embeddings, and the Azure LLM inference trace 2023 (conversation) |
| `certa_sim.py` | Simulator, the six compared methods (Exact, Static-L, Static-C, CRC-Off, PE-Online, CERTA) and the ablation variants |
| `certa_experiments.py` | Experiments E1–E12 → `results/results.json` |
| `certa_diagrams.py` | Fig. 1 (architecture) and Fig. 2 (pipeline) |
| `certa_figures.py` | Result figures Fig. 3–12 (grayscale) |
| `CERTA_colab.ipynb` | One-click reproduction on the free Colab runtime |
| `requirements.txt` | Python dependencies |

## Run

```bash
pip install -r requirements.txt
python certa_data.py          # one-off, downloads QQP/PAWS/Azure trace and embeds the text
python certa_experiments.py   # all experiments, writes results/results.json
python certa_diagrams.py      # Fig. 1-2
python certa_figures.py       # Fig. 3-12
```

Set `CERTA_FAST=1` for a smoke run with 3 / 2 seeds instead of 10 / 5. The full run takes roughly 15–25 minutes on two CPU cores. All runs are deterministic for a given seed.

## Data sources

- Quora Question Pairs and PAWS (labeled_final), via Hugging Face `datasets` (GLUE/QQP, PAWS).
- Azure LLM inference trace 2023, conversation split (Azure/AzurePublicDataset on GitHub).
- Sentence embeddings: `sentence-transformers/all-MiniLM-L6-v2`.

## License

Released for research use. Dataset licenses follow their original providers.
