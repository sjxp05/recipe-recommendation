# Stage 1 recipe similarity matrix

This experiment covers only recipe text, a 768-dimensional sentence encoder,
and the existing stage-1 GraphSAGE. It uses the same 0–4 labeled pairs, recipe
order, train/validation/test split, graph construction, and training settings
for every condition. There are four views and six encoders: 24 runs.

| View | Encoder input |
| --- | --- |
| `raw` | Original recipe name and original step strings from the pair CSV. |
| `masked` | The same strings with ingredient mentions replaced by current v5 category, role, and contribution tags. |
| `assembly` | Current compact v5 assembly, with reviewed unified ingredient IDs in `ING` records. |
| `assembly_masked` | The same assembly with `ING I0 [category, role, contribution]`; names, old IDs, unified IDs, and provenance fields removed. |

The masking tags come from the local `labeled_pairs_jihee_masked_v5.csv` output
of `mask_recipe_v5.py`. The same alias matching and redaction functions from
`mask_recipe.py` mask the *original* name and steps in `masked` and the assembly
fields in `assembly_masked`. The program checks ingredient order and reviewed
unified IDs against the v5 cache before creating any view. It fails if the cache
does not cover the same recipes. The raw and assembly masked texts were also
checked against every complete ingredient name of length at least four in the
current 2,916-recipe data; none remained.

Encoders: `BAAI/bge-base-en-v1.5`, `intfloat/e5-base-v2`,
`sentence-transformers/all-mpnet-base-v2`, `thenlper/gte-base`,
`intfloat/multilingual-e5-base`, and
`sentence-transformers/all-distilroberta-v1`. Both E5 models receive their
documented `query: ` prefix for symmetric similarity. All outputs must be
exactly 768 dimensions.
Long inputs are split into token-bounded windows; all windows are encoded,
combined by token-count-weighted mean, and normalized back to 768 dimensions.
The number of recipes requiring multiple windows and the total window count
are saved with each feature artifact and in `results.csv`.

Run from the repository root. Replace `$pairs` with the current labeled-pair
CSV path. `prepare` creates texts only; `encode` downloads/loads the selected
models; `fit` trains and evaluates all four views per selected encoder.

```powershell
$project = 'E:\대학\과제\3학년\소프트웨어응용'
$pairs = 'C:\Users\fjghq\Downloads\recipe_similarity_manual_20261106\labeled_pairs_jihee_plus_codex_3192.csv'
$out = Join-Path $project '구현\graphsage_assembly_20261029\stage1_matrix_texts'
python -m training.stage1_matrix prepare --project-root $project --pairs $pairs --assembly-db (Join-Path $project '데이터\recipe_assembly_v5.sqlite') --masked-cache (Join-Path $project '구현\labeled_pairs_jihee_masked_v5.csv') --id-map (Join-Path $project '데이터\ingredient_ids_reviewed\output\ingredient_name_old_new_simple.csv') --out $out
python -m training.stage1_matrix encode --out $out
python -m training.stage1_matrix fit --out $out
python -m training.stage1_report --out $out
```

For example, `--encoders gte` runs one encoder; multiple names select multiple
encoders. The encoded feature artifacts are cached. A checkpoint is reused only
when its feature artifact and training settings match its saved provenance;
`--overwrite` explicitly replaces existing artifacts or checkpoints.

`fit` writes each checkpoint, a test-pair prediction CSV, and a `results.csv`
containing train/validation/test RMSE, MAE, exact and within-one accuracy,
positive-pair precision/recall/F1, balanced accuracy, a 5×5 confusion matrix,
best epoch, and window counts. `stage1_report` writes the complete 24-condition
table and confusion matrices to `RESULTS.md`. The 3,192-pair source uses
pair-level splits, so recipes can occur in more than one split. The v5 tagger
has its own training history; these comparisons measure this fixed pipeline and do not constitute
independent generalization to entirely unseen recipes.
