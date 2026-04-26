# Reproducibility Study: HyperKGR (EMNLP'25)

This repository contains my reproducibility study for the paper **HyperKGR: Knowledge Graph Reasoning in Hyperbolic Space with Graph Neural Network Encoding Symbolic Path** (EMNLP 2025).
Credit for the original ideas and framework fully belongs to the original authors: https://github.com/lihuiliullh/HyperKGR

The paper file used in this study:
- `https://aclanthology.org/2025.emnlp-main.1279.pdf`

## Repository Structure

- `sample/`: implementation and experiments with sampling setting.
- `not_sample/`: implementation and experiments without sampling setting.
- `ablation/`: ablation experiments for selected components.

Each main setting includes:
- `transductive/` for transductive reasoning tasks.
- `inductive/` for inductive reasoning tasks.

## Requirements

Recommended environment:
- Python 3.8+
- PyTorch `1.9.1+cu102`
- `torch_scatter==2.0.9`

Install the main dependencies first (according to your CUDA/CPU setup), then run the scripts below.

## Quick Reproduction

### 1) Transductive setting

Example:

```bash
cd sample/transductive
python -W ignore train.py --data_path data/WN18RR
```

You can also try other datasets in `data/` such as:
- `fb15k-237`
- `nell`
- `YAGO`
- `umls`
- `family`

### 2) Inductive setting

Example:

```bash
cd sample/inductive
python -W ignore train.py --data_path data/WN18RR_v1
```

You can run other splits similarly (`*_v1` to `*_v4`) for WN18RR, FB237, and NELL.

### 3) Ablation setting

Example:

```bash
cd ablation/sample/transductive
bash run_ablation.sh
```

## Notes on Data Split (Transductive)

Following common rule-mining settings, original training triples are split into `facts.txt` and `train.txt` (typically 3:1).  
This helps avoid leakage of query triples into the supporting fact graph.

## Expected Outputs

- Training logs printed in terminal.
- Performance summaries (e.g., `*_perf.txt`) inside `results/` folders.

## Citation

```bibtex
@inproceedings{liu2025hyperkgr,
  title={HyperKGR: Knowledge Graph Reasoning in Hyperbolic Space with Graph Neural Network Encoding Symbolic Path},
  author={Lihui Liu},
  booktitle={EMNLP},
  year={2025}
}
```
