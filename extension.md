# Experimental Setup for ICLLR

## Research Questions

**RQ1 (Validity & tightness).** Does the TV-based ceiling hold against the _strongest known attacks_, and what fraction of the ceiling do they recover as a function of KL? — _Falsifiable claim: no attack exceeds min(Pinsker, BH); best attack recovers ≥X% of the ceiling in R2._

**RQ2 (Regime structure).** Is the R1/R2 transition governed by KL alone, regardless of _how_ KL is produced (epochs, corpus size, learning rate, dedup)? — _If curves from different knobs collapse onto one Adv-vs-KL curve, KL is a sufficient statistic for leakage risk. That's a real finding._

**RQ3 (Scaling).** How does the leakage ceiling scale with model size at fixed corpus and compute? — _Prediction: KL (hence ceiling) grows with parameters; quantify the exponent._

**RQ4 (Localization).** Does the token-level bound _quantitatively_ detect memorized spans and PII, not just anecdotally? — _Metric: AUPRC of KL_t as a span detector against ground truth._

**RQ5 (Interventions).** Which interventions move the ceiling (training-time) vs. only the observable signal at utility cost (inference-time, Thm 3.4)? — _Deliverable: one Pareto figure unifying Full FT / LoRA / MiCA / DP-SGD / inference-time defenses._

## Models

| Model                | Sizes                       | Role                                                          |
| -------------------- | --------------------------- | ------------------------------------------------------------- |
| **Pythia (deduped)** | 70M, 160M, 410M, 1.4B, 2.8B | Backbone. Scaling law (RQ3), all core experiments at 70M–410M |
| GPT-Neo              | 125M, 1.3B                  | Continuity with workshop version; one validity run each       |
| Pythia 6.9B          | one config only             | Stretch: single point extending the scaling figure            |

Use the **deduped** Pythia variants — it partially answers the Enron-in-Pile contamination objection and you should say so explicitly. Skip Llama/Qwen entirely; it was nice-to-have and it doesn't fit 4 weeks.

## Datasets

| Corpus                                             | What it tests                                             | Notes                                                                                                                                    |
| -------------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Enron** (email bodies)                           | Continuity + contamination discussion                     | Keep exact workshop preprocessing                                                                                                        |
| **Post-cutoff news** (2025–26 articles, ~10k docs) | Clean-room bound validity; zero pretraining contamination | Verify via 13-gram overlap check against Pile; report the overlap rate                                                                   |
| **Pile of Law – ECHR / court opinions subset**     | High-stakes domain; sells the audit/compliance framing    | Pick ONE legal subset; don't add clinical — access friction kills you in 4 weeks                                                         |
| **Canaries**                                       | Ground truth for RQ4                                      | 50 synthetic PII strings (fake names + emails + account numbers), injected at duplication counts {1, 4, 16} into each fine-tuning corpus |

Membership protocol unchanged: Bernoulli(½) randomization within a single corpus, 2000 fixed non-members, N ∈ {500, 2000, 6000}.

## Attacks & Metrics

- Signals: LOSS, Ref, zlib, Min-K% (existing) + **RMIA** + **SPV-MIA** [18] + neighborhood [17].
- Metrics: AUROC, **TPR@0.1% and 1% FPR** (log-log ROC), tightness gap = bound − best Adv, bootstrap 95% CIs on everything (1000 resamples), held-out **perplexity** for every fine-tuned checkpoint.
- Seeds: 1 seed everywhere is enough.

## The Experiment Grid

**E1 — Core validity & tightness (RQ1).**
Pythia {70M, 160M, 410M} × {Enron, news, legal} × N ∈ {500, 2000, 6000} × epochs {1,2,3,5,10,15,20} × all 7 attacks. One training run yields all checkpoints, so this is ~3×3×3 ≈ **~30 fine-tuning runs**, all small models — cheap. Attack evaluation is forward passes only.
_Expected:_ no violations; best attack (likely SPV-MIA or RMIA) recovers 50–80% of ceiling in R2, much less in R1; tightness gap shrinks monotonically with KL. If an attack _exceeds_ the bound anywhere, your KL estimator or member-distribution assumption is still broken — that's a fire alarm, not a result.

**E2 — Regime collapse (RQ2).**
At Pythia 410M / Enron, vary each KL-driver independently: LR ∈ {1e-5, 5e-5, 2e-4}, N ∈ {500,2000,6000}, epochs grid, dedup vs. 4×-duplicated corpus. Plot Adv vs. estimated KL, all runs on one axis.
_Expected:_ points collapse onto a single curve → KL is the sufficient statistic; the Pinsker/BH crossover at KL ≈ ln 2·2... (compute the exact crossover) marks the regime boundary regardless of knob. _If they don't collapse_ (e.g., duplication produces more Adv at equal KL), that's arguably an even better finding — sequence-level KL misses concentration of leakage — and your token-level bound explains it. Either outcome is publishable; say which you got.

**E3 — Scaling law (RQ3).**
Fixed Enron N=2000, 10 epochs, matched LR schedule: Pythia 70M → 2.8B (+6.9B if time).
_Expected:_ KL and ceiling grow with size; log-log plot of ceiling vs. params with fitted slope. This is your headline "larger models leak more — through the auditor's lens" figure.

**E4 — Token-level detection (RQ4).**
On canary-injected corpora (410M, heavy regime): compute KL*t over all member documents; ground truth = canary positions + verified verbatim 12-gram matches. Report **AUPRC of KL_t as span detector**, stratified by duplication count {1,4,16}. PII case study: NER over legal corpus, test whether KL_t mass concentrates on entity tokens (report lift over non-entity tokens).
\_Expected:* AUPRC well above prevalence baseline, increasing with duplication; entities carry 2–5× the average KL_t. Demote current Fig. 3 to an illustration next to these numbers.

**E5 — Interventions & the Pareto frontier (RQ5).**
Training-time: Full FT, LoRA r∈{4,16,64}, MiCA r∈{4,16,64}, DP-SGD ε∈{1,4,8} — all at Pythia 410M, Enron N=2000, 10 epochs. **Every run reports held-out perplexity + downstream proxy (next-token accuracy on domain held-out).**
Inference-time (Thm 3.4 validation): take the _worst_ leaking checkpoint (Full FT, 20 epochs) and apply temperature T∈{1.2, 1.5, 2.0}, top-k∈{10,50}, Gaussian logit noise σ∈{0.5,1,2}. Re-run all attacks on defended outputs; measure utility of defended model.
_Expected:_ training-time methods trace a genuine Pareto frontier (DP dominates at matched utility or vice versa — you don't know yet, and that's the point); inference-time defenses reduce observable Adv only along a utility-destruction path while the estimated ceiling is unchanged. One two-panel figure: (a) leakage vs. utility for training-time methods, (b) observable Adv vs. utility for inference-time, ceiling drawn as a horizontal line.
_Compute warning:_ DP-SGD at 410M with per-sample gradients is your single most expensive line item. If it doesn't fit, run DP at 160M and say so; don't silently drop ε=1.

---

# 4-Week Schedule

## Week 1 (Aug 17–23) — Infrastructure + launch everything small

The whole plan lives or dies on this week, because Week 2 is your defense.

- **Days 1–2:** Data pipeline. Assemble news corpus + contamination check; extract legal subset; write canary generator + injection at controlled duplication; freeze all splits and membership assignments (RNG-harnessed — you already have this pattern from the poisoning project, reuse it).
- **Days 2–3:** Attack harness. Implement RMIA, SPV-MIA, neighborhood on top of your existing four signals. Implement TPR@low-FPR + bootstrap CI machinery once, as a library, so every later experiment gets it for free. Unit-test attacks on a known workshop checkpoint — SPV-MIA should beat Ref; if it doesn't, your implementation is wrong.
- **Days 3–5:** Launch **E1 in full** (all small-model fine-tuning runs, checkpointing every grid epoch) and **E2** runs. These queue and run unattended.
- **Days 6–7:** Launch **E3** for 70M–410M; start 1.4B. Smoke-test DP-SGD at 160M so Week 3 isn't debugging Opacus under deadline.

**Gate at end of Week 1:** attack harness validated, all ≤410M training jobs running or done. If you're behind here, cut the legal corpus (not the news corpus, not canaries).

## Week 2 (Aug 24–30) — Defense week. Compute runs; you don't.

- Only unattended jobs: finish E3 (1.4B, 2.8B — these take days anyway), E1 attack evaluation sweeps over finished checkpoints, LoRA/MiCA grid of E5 (cheap, launch Monday).
- **Max 2–3 hours total of babysitting:** check job health, requeue failures. Nothing that requires thinking.
- Defend your thesis Aug 28. Do not open TensorBoard on Aug 27.

## Week 3 (Aug 31 – Sept 6) — Heavy analysis + interventions

- **Days 1–2:** E1/E2/E3 analysis. Produce the three core figures: validity+tightness, regime collapse, scaling law. This is where you find out whether E2 collapses or not — budget a day to chase whichever answer you get.
- **Days 2–4:** **E5 DP-SGD runs** (launch immediately Monday, they're slow) + inference-time defense evaluation (fast — forward passes on one checkpoint).
- **Days 4–6:** **E4 in full**: KL_t sweeps, canary AUPRC, NER lift analysis. This is pure evaluation on existing checkpoints, no training.
- **Day 7:** Statistical pass — 5-seed reruns of the LoRA-vs-MiCA headline comparison, paired tests, CI tables.

**Gate at end of Week 3:** all data in hand. Anything not finished gets cut, not extended — Week 4 admits no new experiments.

## Week 4 (Sept 7–13) — Figures, tables, writing, buffer

- **Days 1–3:** Final figures (the five listed under each experiment) and tables (attack roster × settings with CIs; rank ablation with CIs; Pareto table). Rewrite Section 4 following the outline from my previous message; rewrite Appendix F around the corrected estimators.
- **Days 3–5:** Rewrite the contributions and abstract _around what you actually found_ — especially E2's answer. Intro currently oversells theory; reposition as "auditing protocol + comprehensive empirical characterization."
- **Days 5–6:** One full buffer day for the rerun you will inevitably need, plus internal read-through by Mariam/Léa if they'll turn it around fast — send them the draft **by Day 4 at the latest**, not Day 6.
- **Day 7:** Abstract registration + polish.

---

Before the tables, two blunt corrections to your own constraints — then the plan works.

**1. Single seed changes what you're allowed to claim, and I won't pretend otherwise.** With one seed, the LoRA-vs-MiCA gap (0.600 vs 0.644 in your workshop version) is unclaimable as a finding — you don't know if it's method or noise. Mitigation that costs zero GPU-hours: report **bootstrap CIs over evaluation examples** (resample the 2000+2000 member/non-member pool, 1000×) for every AUROC, TPR, and KL estimate. That gives you honest uncertainty bands from a single run and is defensible in a rebuttal ("CIs reflect evaluation sampling; seed variance in the workshop version was ±0.01–0.02"). Write your claims as trends across the grid (which you have ~30 points of), not as pairwise method comparisons.

**2. Your compute caps model size, and that's fine.** P100 (16GB) full fine-tunes up to ~410M comfortably with AdamW. 1.4B full FT does **not** fit (optimizer states alone ≈ 22GB) — it needs 8-bit Adam + gradient checkpointing and even then it's miserable on a P100. So: everything core runs on Kaggle at 70M–410M; the 1.4B/2.8B scaling points and DP-SGD@410M go to RunPod for ~$25–60 total. Drop 6.9B entirely.

**Fixed hyperparameters everywhere** (so the tables don't repeat them): Pythia-deduped models; seq length 256; fp16; batch = largest fitting (grad-accum to effective 32); LR 5e-5 (except E2 sweep); checkpoints at epochs {1,2,3,5,10,15,20}; seed = 42 everywhere, frozen membership assignments.

---

## Master Experiment Table

| ID      | Experiment                                   | Configs                                                                                               | Model(s)                                                                         | Dataset(s)                                | Platform                                                   | VRAM                                   | Est. runtime                                                    | Priority                                                     |
| ------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------- | ---------------------------------------------------------- | -------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------ |
| **E1a** | Core validity & tightness, full grid         | N ∈ {500, 2000, 6000} × 7-epoch checkpoint grid                                                       | Pythia 70M, 160M, 410M                                                           | Enron                                     | Kaggle P100                                                | 3–12 GB                                | 9 runs ≈ **14 h** (70M ~25min, 160M ~1h, 410M ~2.5h; N=6000 ×2) | **MUST**                                                     |
| **E1b** | Cross-domain validity                        | N = 2000 only                                                                                         | Pythia 70M, 160M, 410M                                                           | Post-cutoff news; Pile-of-Law ECHR subset | Kaggle P100                                                | 3–12 GB                                | 6 runs ≈ **8 h**                                                | **MUST** (news) / REC (legal)                                |
| **E1c** | Attack evaluation over all checkpoints       | LOSS, Ref, zlib, Min-K%, RMIA, neighborhood                                                           | all E1 checkpoints (≈105)                                                        | as above                                  | Kaggle **2×T4** (target on GPU0, reference on GPU1)        | 2×6 GB                                 | ≈ **10 h** batched                                              | **MUST**                                                     |
| **E1d** | SPV-MIA                                      | self-prompt calibration model per target config — requires an extra fine-tune per attacked checkpoint | 410M targets only, 3 checkpoints (ep 3, 10, 20), Enron N=2000                    | Enron                                     | Kaggle P100                                                | 12 GB                                  | **6 h**                                                         | REC — run on headline config only; full grid is unaffordable |
| **E2**  | Regime collapse (KL as sufficient statistic) | LR ∈ {1e-5, 5e-5, 2e-4}; dedup vs. 4×-dup corpus                                                      | Pythia **160M** (not 410M — halves cost, same physics)                           | Enron N=2000                              | Kaggle P100                                                | 6 GB                                   | 5 extra runs ≈ **5 h**                                          | **MUST** — cheapest high-value experiment you have           |
| **E3**  | Scaling law of the ceiling                   | fixed N=2000, 10 ep; sizes 70M/160M/410M reused from E1a + **1.4B** + 2.8B                            | Pythia →2.8B                                                                     | Enron                                     | **RunPod A100 40GB** (1.4B: 8-bit Adam; 2.8B: + grad ckpt) | 24–38 GB                               | 1.4B ≈ 2h (~$3–5); 2.8B ≈ 5h (~$8–12)                           | 1.4B **MUST**, 2.8B OPTIONAL                                 |
| **E4a** | Token-level bound: canary AUPRC              | 50 canaries × dup {1,4,16}, injected into E1a corpora before training (plan in Week 1!)               | 410M, heavy-regime ckpt                                                          | Enron+canaries                            | Kaggle 2×T4                                                | 2×8 GB (full-vocab logits both models) | **4 h**                                                         | **MUST**                                                     |
| **E4b** | PII localization case study                  | NER (spaCy) × KL_t lift on entities                                                                   | 410M                                                                             | legal or Enron                            | Colab Pro / CPU-mostly                                     | 8 GB                                   | **3 h**                                                         | REC                                                          |
| **E5a** | PEFT grid                                    | LoRA r∈{4,16,64}, MiCA r∈{4,16,64}, 10 ep                                                             | Pythia 410M                                                                      | Enron N=2000                              | Kaggle P100                                                | 8 GB                                   | 6 runs ≈ **6 h**                                                | **MUST**                                                     |
| **E5b** | DP-SGD                                       | ε ∈ {1, 4, 8}                                                                                         | **160M on Kaggle** (410M per-sample grads won't fit P100) or 410M on RunPod A100 | Enron N=2000                              | Kaggle P100 / RunPod                                       | 14 GB / 30 GB                          | 3 runs ≈ **7 h** Kaggle (DP ≈ 2–3× slowdown) or ~$10 RunPod     | **MUST** (160M version)                                      |
| **E5c** | Utility for every run                        | held-out perplexity + next-token acc — piggybacks on every checkpoint                                 | all                                                                              | all                                       | free (folded into E1c eval pass)                           | —                                      | +**2 h** total                                                  | **MUST**                                                     |
| **E5d** | Inference-time defenses (Thm 3.4)            | temp {1.2,1.5,2.0}, top-k {10,50}, logit noise σ {0.5,1,2} on worst checkpoint                        | 410M, ep-20 Full FT                                                              | Enron                                     | Kaggle 2×T4                                                | 2×6 GB                                 | **4 h**                                                         | **MUST** — currently zero experiments back this theorem      |
| **E6**  | GPT-Neo continuity check                     | 125M, N=2000, one run + attacks                                                                       | GPT-Neo 125M                                                                     | Enron                                     | Kaggle P100                                                | 4 GB                                   | **1.5 h**                                                       | OPTIONAL                                                     |

**Totals: Kaggle ≈ 70 h** (fits in 120 h budget with ~40% slack for failures/reruns — you will need it), **RunPod ≈ $15–30 must-have**, +$12 optional 2.8B. Colab Pro: keep for interactive debugging and analysis notebooks only; don't burn it on training.

## Dataset Table

| Corpus                     | Type / size                                                                                      | Prep cost       | What it buys                                                                    |
| -------------------------- | ------------------------------------------------------------------------------------------------ | --------------- | ------------------------------------------------------------------------------- |
| Enron email bodies         | ~50k usable bodies; sample pools per N                                                           | Done (workshop) | Continuity; full N-grid                                                         |
| Post-cutoff news (2025–26) | ~8–10k articles, scraped/CC-News slice; 13-gram decontamination check vs. Pile reported in paper | ~1 day CPU      | Clean-room validity — kills the contamination objection                         |
| Pile of Law — ECHR subset  | ~5k documents, truncated to 256 tok                                                              | ~½ day          | Audit/compliance framing credibility                                            |
| Canaries                   | 50 unique fake PII strings, dup {1,4,16}                                                         | 2 h scripting   | Ground truth for RQ4 — **must be injected before E1a training runs, not after** |

## Weekly Schedule (mapped to Kaggle's 30 h/wk and your Aug 28 defense)

| Week                   | Kaggle hours | What runs                                                                                                           | What you do                                                                                                                                                                                                                       |
| ---------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **W1** (Aug 17–23)     | ~28 h        | E1a (all 9 runs, canaries pre-injected), E2, E1b-news                                                               | Days 1–2: data pipeline + canary injection + attack harness (RMIA, neighborhood) with bootstrap-CI library. Days 3–7: launch training in 12h-session-sized batches. **Gate:** harness validated on a workshop checkpoint by Day 3 |
| **W2** (Aug 24–30)     | ~20 h        | E1c attack sweeps (unattended, scripted), E5a PEFT grid, E1b-legal                                                  | ≤3 h babysitting. **Defense Aug 28.** Nothing requiring thought is scheduled                                                                                                                                                      |
| **W3** (Aug 31–Sept 6) | ~28 h        | E5b DP (launch Monday — slowest jobs), E4a, E5d, E1d SPV-MIA; RunPod session for E3 1.4B (+2.8B if W1–2 went clean) | Analysis of E1/E2/E3: produce validity, collapse, scaling figures. Chase whichever answer E2 gives                                                                                                                                |
| **W4** (Sept 7–13)     | ~10 h buffer | reruns only — **no new experiments admitted**                                                                       | Figures, tables, rewrite §4 + Appendix F, draft to supervisors by Day 4, abstract registration                                                                                                                                    |

## Practical infrastructure (do this in Week 1, it pays for itself tenfold)

- **Checkpointing around Kaggle's 12h kill:** save every epoch-grid checkpoint immediately to a private **HF Hub repo** (Kaggle Datasets versioning is clunkier); every training script must resume from `latest`. Assume every session dies at 11h59.
- **Cache logits, never recompute:** the attack harness's expensive object is per-sequence log-probs under both models. Compute once per (checkpoint, eval-pool), dump to `.npz` on HF Hub. All 6 signals, all metrics, all CIs, and E5c utility then run on CPU in minutes. This is the single biggest efficiency lever in the whole plan — it also means adding a new metric later costs nothing.
- **2×T4 trick:** for eval passes, pin P_ft to `cuda:0` and P_pre to `cuda:1`; you halve wall-clock on the attack sweeps versus P100.
- **Tracking:** W&B in offline mode on Kaggle (no reliable outbound), `wandb sync` from Colab after each session. One config dataclass per run, serialized to JSON, hash in the run name: `pythia410m_enron_N2000_lr5e-5_seed42_a3f2`.
- **Reproducibility:** freeze membership Bernoulli draws and canary positions as versioned artifacts _before any training_; fix `torch`, `transformers`, `peft`, `opacus` versions in a `requirements.txt` committed next to the code; log the git commit into every W&B run.
- **fp16 caution on P100:** no bf16 support — use fp16 with dynamic loss scaling; if DP-SGD becomes unstable in fp16 (it will be tempted to), run DP in fp32 at 160M, which still fits.

## Scaling to multiple seeds later (post-deadline / rebuttal / camera-ready)

Priority order, because you won't afford all of it:

1. **E5a LoRA-vs-MiCA and E5b DP** → 5 seeds. These are the pairwise claims that are naked without seed variance (~15 cheap runs, ~20 Kaggle-h).
2. **E1a at 410M, N=2000** (the headline validity config) → 3–5 seeds (~12 h).
3. **E3 scaling points** → 3 seeds at ≤410M; leave 1.4B+ at one seed and say so (~10 h + $10).
4. Everything else stays single-seed with bootstrap CIs, permanently — that's a defensible position if items 1–3 are covered.

Design your code for this _now_: seed as a config field, aggregation script that treats seeds as a groupby, so scaling up is a for-loop, not a refactor.

# Example of kaggle notebook so far:

!git clone https://github.com/Jalalbaim/MIA_Finetuned_LLMs.git

%cd /kaggle/working/MIA_Finetuned_LLMs

!pip install -r requirements.txt

!python -c "import torch; print(torch.**version**, torch.version.cuda); print('cap=', torch.cuda.get_device_capability());"

!python finetune/train.py --seed 0 --n 2000

!python signals/compute_signals.py --seed 0 --n 2000

!python metrics/compute_metrics.py

import shutil

shutil.make_archive('/kaggle/working/results', 'zip', '/kaggle/working/results')
