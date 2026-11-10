# Raw recipe GraphSAGE width sweep

Compare the three best raw-input encoders from the previous test-RMSE table:
MPNet, GTE, and multilingual E5. Their existing frozen sentence embeddings are
all 768-dimensional. Only the GraphSAGE hidden/output width changes:
16, 32, 64, 128, and 256. This gives 15 conditions. The width-32 checkpoints
and their metrics are imported from the earlier run; 12 conditions remain to
train. All conditions use the same recipe order, pair split, seed, fold schedule,
learning rate, maximum epochs, and early-stopping patience: seed 42, five folds,
learning rate 0.01, at most 100 epochs, and patience 10.

Run from the repository root with the project Python:

```powershell
$python = 'E:\__Programming\Language\Python\Python 3.10.0\python.exe'
$base = 'E:\대학\과제\3학년\소프트웨어응용\구현\graphsage_assembly_20261029'
$source = Join-Path $base 'stage1_matrix_texts'
$out = Join-Path $base 'stage1_hidden_sweep_raw_top3'
& $python -m training.stage1_hidden_sweep plan --source $source --out $out
```

`plan` validates the three source artifacts, width-32 checkpoint provenance,
and saved test metrics. It writes `plan.json` and a 15-row `results.csv`: three
`existing` rows with complete metrics and twelve `pending` rows. It never
encodes text or trains a model. This plan has already been created in the output
directory above; choose a fresh directory if you want to create another plan.

When training is requested later, run:

```powershell
& $python -m training.stage1_hidden_sweep run --out $out
& $python -m training.stage1_hidden_sweep report --out $out
```

`run` trains only the pending widths; `--hidden` and `--encoders` can select a
subset. Width 32 cannot be passed to `run`. Checkpoints, predictions, and logs
use both encoder and width in their filenames, so no previous result is
overwritten. `report` requires all 15 rows to be complete before writing the
metrics table and confusion matrices.
