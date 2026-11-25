# GraphSAGE GPU experiment results

- Source: `experiment/graphsage_finetuning`, commit `6f638d3`
- Run date: 2026-11-22 (Asia/Seoul)
- Device: NVIDIA GeForce RTX 4060 Laptop GPU
- Runtime: Python 3.12.16, PyTorch 2.12.1+cu130
- Seeds: 42, 43, 44, 45, 46 for every condition
- Selection rule: lowest mean validation RMSE; test RMSE is reported only after selection
- Stage 1: 65 runs complete; Stage 2: 70 runs complete

## Selected configurations

| Stage | Aggregator | Layers | Learning rate | Validation RMSE, mean ± SD | Test RMSE, mean |
|---|---|---:|---:|---:|---:|
| 1, recipe similarity | max pooling | 1 | 0.003 | 0.182604 ± 0.003613 | 0.188903 |
| 2, ingredient contribution | GRU, fixed order | 1 | 0.01 | 0.059705 ± 0.000394 | 0.057579 |

## Stage 1 comparisons

| Experiment | Best | Next | Validation RMSE, best vs next |
|---|---|---|---|
| Aggregator | max | weighted mean | 0.183582 vs 0.184894 |
| Layers | 1 | 2 | 0.182604 vs 0.183582 |
| Learning rate | 0.003 | 0.01 | 0.182604 vs 0.189345 |

The aggregator and layer differences are smaller than the best condition's across-seed standard deviation. The notebook's automatic selection still chooses max and one layer.

## Stage 2 comparisons

| Experiment | Best | Next | Validation RMSE, best vs next |
|---|---|---|---|
| Aggregator | GRU, fixed order | GRU, shuffled order | 0.059705 vs 0.060844 |
| Layers | 1 | 3 | 0.059705 vs 0.059738 |
| Learning rate | 0.01 | 0.001 | 0.059705 vs 0.059989 |

The one-layer and three-layer results are effectively tied at this precision. The learning-rate gap is also smaller than the best condition's across-seed standard deviation. The notebook's automatic selection chooses one layer and 0.01.

## Files

- Stage 1: `experiments_gpu/{01_aggregator,02_num_layers,03_learning_rate}/`
- Stage 2: `experiments_stage2_gpu/{01_aggregator,02_num_layers,03_learning_rate}/`
- Each experiment has `runs.csv`, `summary.csv`, and `summary.json`; each condition/seed has `metrics.json` and `history.json`.
- The stage 2 input embedding artifact was generated locally at `../artifacts/stage2_embeddings.pt`; it is not included in this result commit.
- CPU results from the earlier run remain local in `experiments/` and `experiments_stage2/`; they are not included in this result commit.

The experiments execute the checked-in `stage1_graph_experiments.ipynb` and `stage2_graph_experiments.ipynb` notebooks. The local CUDA setup and helper script are not included in this result commit.
