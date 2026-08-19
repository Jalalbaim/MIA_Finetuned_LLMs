# E1 — Core validity & tightness

Scaled-up replacement for the workshop pipeline, covering **E1a–E1d** from
`extension.md`. Self-contained: its own config, its own sweep axes, its own
results table. It *imports* the workshop estimators rather than reimplementing
them, so the numbers stay comparable.

**RQ1.** Does the TV-based ceiling hold against the strongest known attacks,
and what fraction of it do they recover as a function of KL?

## Layout

| File | Role |
|---|---|
| `config_e1.py` | Every axis and fixed hyperparameter. Single source of truth. |
| `runspec.py` | `RunConfig` dataclass → `run_id`. Run identity, not filename regexes. |
| `models.py` | Pythia/GPT-Neo registry: batch sizes, dtype per platform, PEFT targets. |
| `corpora.py` | Enron / news / legal pools, seed-42 splits, 13-gram contamination check. |
| `train_e1.py` | E1a + E1b fine-tuning. fp16 + resume + session budget + inline caching. |
| `cache_logprobs.py` | Per-token log-prob cache (`.npz`). The efficiency lever. |
| `attacks.py` | Seven attacks as pure functions over the cache. |
| `stats.py` | AUROC / TPR@FPR / Adv / TV / KL / bounds + bootstrap CIs. |
| `eval_e1.py` | E1c: assembles `results/e1_metrics.csv` and runs the RQ1 check. |
| `neighbors.py` | Neighbour generation for the neighbourhood attack (subset only). |
| `kaggle/run_kaggle.py` | Session driver; copy-paste cells in its docstring. |
| `tests/test_e1.py` | The Week-1 "harness validated" gate. 28 tests, CPU, seconds. |

## Reused from the workshop pipeline

Imported unchanged, never copied:

- `metrics/compute_metrics.py` — `compute_kl_score_space`, `compute_bounds_from_kl`,
  `compute_auroc`, `compute_tpr_at_fpr`, `compute_advantage`, `compute_tv_empirical`
- `signals/s_{loss,ref,zlib,mink}.py` — the reference the vectorised attacks are tested against
- `data/membership_assignment.py` — `save_split` / `load_split`
- `data/prepare_enron.py` — `clean()`, applied to the news and legal corpora too

`token_level/` is **not** used. E1 is sequence-level only.

## Running it

```bash
# one-off
python data/prepare_enron.py                                    # if raw_data/pool.jsonl is absent
python E1_scaled_xps/corpora.py --prepare enron --splits --corpus enron
python -m pytest E1_scaled_xps/tests/test_e1.py -q              # the gate

# E1a — one model per Kaggle session
python E1_scaled_xps/train_e1.py --experiment e1a --model pythia-70m

# E1c — CPU
python E1_scaled_xps/eval_e1.py
```

On Kaggle, use `kaggle/run_kaggle.py`; its docstring has the notebook cells.

## Design decisions worth knowing

**Run identity is a dataclass, not a filename.** The workshop pipeline parses
`(seed, N, epoch, variant)` out of filenames with four parallel regexes in
`compute_metrics.py` plus matching if/elif chains in `compute_signals.py` and
`compute_kl.py` — three files to edit per new axis. E1 adds model and corpus,
E2 adds lr and dup_factor, E5 adds rank and epsilon. Here every run writes
`runs/<run_id>/config.json`, discovery reads those, and a new axis is a new
field. `expand_grid()` makes scaling to more seeds a for-loop.

**Each checkpoint is forward-passed exactly once.** `cache_logprobs.py` writes
per-token log-probs under both models, plus the per-position vocabulary mean
and standard deviation needed by Min-K%++ (unrecoverable afterwards). All seven
attacks, perplexity, next-token accuracy and every bootstrap CI then run on CPU
from that ~20 MB file. Adding an eighth attack costs nothing.

**Training caches inline.** The model is already resident when a grid
checkpoint is saved, so the cache is built right there. This is why
`--push-checkpoints` is off by default: the full grid is ~50 GB of checkpoints
versus ~2 GB of caches, and the caches are what E1c consumes. Keep checkpoints
for the configs that need the model itself — E1d (SPV-MIA), E5d
(inference-time defenses), and the neighbourhood attack — via `--keep-epochs`.

### Disk budget on Kaggle

Kaggle gives ~20 GB writable. One 410M run costs:

| artifact | size |
|---|---|
| 7 grid checkpoints @ fp16 | 5.7 GB |
| `training_state.pt` (fp32 weights + 2 AdamW moments = params × 12 B) | 4.9 GB |
| 7 log-prob caches | 0.15 GB |

That is 10.6 GB per run, so three runs cannot coexist — a first attempt died
with `ENOSPC` at run 2 epoch 19, which also destroyed the notebook's own output
file and with it the finished run 1. Two rules keep it bounded:

1. **A grid checkpoint is deleted once its cache is safely written.** The 20 MB
   cache is the artifact; the 810 MB checkpoint is only needed by
   E1d/E5d/neighbors. Override with `--keep-checkpoints` or `--keep-epochs`.
   Pruning never happens if the cache write failed.
2. **`training_state.pt` is deleted when the grid completes.** It exists only
   to resume.

Peak usage is then ~6 GB per run and ~0.2 GB persists after it finishes.
Because pruning removes the evidence a run happened, completion is recorded by
a `COMPLETED` marker in the run directory — rerunning a finished command skips
rather than retraining. `cached_epochs()` (not `completed_epochs()`) is what
tells you what is evaluable.

Training also stops early and checkpoints cleanly when free space drops below
`MIN_FREE_GB`, the same way it does at `--max-hours`.

**`P_pre` is computed once per run, not once per epoch.** The reference model
is epoch-invariant, but the first implementation re-ran it at every grid
checkpoint. At 410M, caching cost 507 s per checkpoint against 59 s of
training — so this was the single most expensive line in the whole experiment,
and half of it was redundant. The pass is now stored in
`cache/<run_id>/reference.npz` and reused, guarded by an exact
`(seq_id, split_code)` match so a mismatched reference cannot silently
misalign rows.

**fp16, verified by compute capability.** `torch.cuda.is_bf16_supported()`
defaults to `including_emulation=True` and returns `True` on a T4 (sm_75),
which has no bf16 hardware — the first Kaggle run duly selected an emulated
bfloat16 path. `resolve_dtype` now checks `get_device_capability()[0] >= 8`
directly, so P100 and T4 get fp16 + live GradScaler and only A100+ gets bf16.

**fp32 master weights, fp16 autocast.** The P100 (sm_60) has no bf16.
`finetune/train.py` loads weights in bfloat16, autocasts to bfloat16 on top,
and wraps the optimizer step in `GradScaler(enabled=False)`; that does not run
on E1's target hardware. Here weights stay fp32, autocast is fp16, and the
GradScaler is live.

**Bootstrap CIs everywhere, and they are CIs over *evaluation examples*.** With
one seed there is no training-variance estimate, and the paper must say so.
This matters most for TPR@0.1% FPR: 2000 non-members means 0.1% FPR is two
negatives, so the point estimate is quantised to multiples of 1/2000.

### `TV_N_BINS = 20`, not 500

The score-space KL is a plug-in histogram estimator, and at E1's pool sizes the
workshop's 500 bins makes it severely upward-biased. Measured against Gaussian
pairs with known true KL (mean of 40 replicates, n = 2000 vs 2000):

| true KL | 10 bins | 20 bins | 100 bins | 500 bins |
|---|---|---|---|---|
| 0.020 | 0.027 | 0.036 | 0.113 | **0.516** (26×) |
| 0.125 | 0.135 | 0.155 | 0.260 | **0.742** (6×) |
| 0.500 | 0.512 | 0.560 | 0.791 | **1.547** (3×) |
| 2.000 | 2.320 | 2.586 | 3.283 | **4.943** (2.5×) |

An inflated KL inflates the ceiling, so *"no attack exceeds the bound"* passes
trivially and the tightness gap becomes an artifact of the bin count. Both are
RQ1 headline numbers.

Fewer bins is not automatically safer. Coarsening can only *reduce* KL (data
processing), while the empirical advantage is computed on the raw continuous
scores — so too few bins pushes the ceiling below Adv and **manufactures** a
violation. Over 60 replicates × 9 separations × 4 score-distribution shapes
(Gaussian equal and unequal variance, bimodal members, unequal pool sizes),
false violations appear at 10 bins and vanish from 15 up.

20 bins is the smallest count with zero false violations and a positive safety
margin in every case tested. `test_e1.py` pins both halves: it fails if the
count is lowered enough to manufacture violations, and it fails if the count is
raised enough to inflate KL past 1.6× the truth.

Every row also carries `kl_bins_{15,20,30,50}` and `bound_bins_{...}` so the
paper can show the conclusion survives the choice. Below ~20 samples per bin
`stats.evaluate` emits a `RuntimeWarning` — a 24-vs-24 smoke run reports
KL ≈ 17 from pure discretisation noise.

## Known gaps

- **E1d (SPV-MIA)** is not implemented. It needs a self-prompt calibration
  model fine-tuned per attacked checkpoint; `extension.md` marks it REC and
  headline-config-only.
- **The neighbourhood attack does not scale to the full grid.** At z = 25 it
  multiplies a checkpoint's forward passes by 26. Run it on chosen
  checkpoints and say so; the other six attacks cover everything.
- **The news corpus source is unverified.** `CORPORA["news"].hf_candidates`
  needs a source that exposes publication dates, filtered to post-cutoff
  months — that date filter is the entire clean-room claim. Drop a scrape at
  `raw_data/e1/news/pool.jsonl` if the listed candidates do not resolve.
- **The 13-gram check runs against whatever reference you give it.** Making the
  claim *about the Pile* requires supplying Pile text. Running it against Enron
  is a cross-corpus sanity check and must not be described as the Pile result.
- **Canaries are deliberately absent**, so E4a / RQ4 has no ground truth.
