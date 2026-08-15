# Research Log

Format per entry: what changed, why, config used, results, lessons, next experiment.

---

## Phase 0 — Architecture Validation
- **What changed:** Confirmed task definition (128x128 -> 256x256 grayscale blind restoration + 2x SR),
  hardware constraints (RTX 3050 4GB dev / H100 target), model roster (Sanity CNN, NAFNet, Restormer, SwinIR),
  and framework design principles (Hydra configs, plugin-style model registry).
- **Why:** Establish a stable contract before any code is written, per project workflow rules.
- **Config used:** N/A (planning phase).
- **Results:** N/A.
- **Lessons:** Only 2 placeholder `.npy` files were available at project start, both 128x128
  (NoisyLR-shaped). No 256x256 GT files exist yet. User decided to proceed using these as
  placeholders while the framework is built, with the real dataset to follow.
- **Next experiment:** Phase 1 — scaffold environment and folder structure.

---

## Phase 1 — Environment + Folder Structure
- **What changed:** Created `AI_Restoration_Framework/` directory tree (framework, models, tasks,
  configs, datasets, losses, metrics, experiments, scripts, tests, docs, utils), `requirements.txt`,
  `environment.yml`, and a raw `dataset/` folder scaffold (`train/val/test` x `NoisyLR/GT`).
  Copied the 2 uploaded placeholder `.npy` files into `dataset/train/NoisyLR/` with an explicit
  `README_PLACEHOLDER.md` warning that this is not real data.
- **Why:** Freeze the module boundaries early so later phases (trainer, evaluator, model plugins)
  have a stable place to live, per the "framework must be frozen early" requirement.
- **Config used:** N/A (no configs authored yet — that starts in later phases).
- **Results:** Directory tree created and verified (see phase summary).
- **Lessons:** Distinguish `datasets/` (Python package for loading logic) from `dataset/`
  (raw data folder) — easy to confuse, kept as two separate top-level names intentionally
  matching the original spec.
- **Next experiment:** Phase 2 — dataset validator (`inspect_dataset.py`).

---

## Phase 2 — Dataset Validator
- **What changed:** Implemented `utils/dataset_validator.py` (core checks: sample count, shape
  verification, missing-pair detection, corrupted-file detection, dtype verification, min/max/mean/std,
  NaN/Inf detection, histograms, random visualizations, input-GT comparisons) and `inspect_dataset.py`
  (root-level CLI wrapper that writes a Markdown report to `docs/dataset_report/REPORT.md` plus images).
- **Why:** Per project rule, no training/evaluation code may be trusted until this exists and runs clean.
  Kept dependency-light (numpy + matplotlib only) so it can run before Hydra/model code is even installed.
- **Config used:** N/A (plain argparse CLI, not yet part of the Hydra config system by design — this
  tool runs *before* the config system is trusted).
- **Results:** Ran against the 2 placeholder `train/NoisyLR/*.npy` files (no GT, no val, no test).
  Report correctly flagged: 0 matched pairs, 2 NoisyLR files with no GT, global min/max
  -0.0031/1.4091, no NaN/Inf, no corruption, no shape/dtype mismatches on the files present.
  Report banner correctly escalated to 🟠 due to the missing-GT condition.
- **Lessons:** `pytest` could not be installed in this sandbox (no network egress). Wrote
  `tests/test_dataset_validator.py` for your real environment, and additionally ran the exact same
  assertions manually inline as a stand-in — all 7 checks passed (sample count, missing-pair
  detection, corrupted-file detection, shape mismatch, dtype mismatch, NaN/Inf detection, and
  graceful handling of a missing split directory).
- **Next experiment:** Phase 3 — dataset pipeline (PyTorch `Dataset`/`DataLoader` built on top of
  the same directory contract this validator checks).

---

## Phase 2.1 — Strict GT Resolution + Filename Pairing Rule (user addition)
- **What changed:** Added `validate_strict_pairing()` to `utils/dataset_validator.py`, formalizing
  the hard data contract: NoisyLR must be exactly `(128, 128)`, GT must be exactly `(256, 256)`,
  and NoisyLR/GT files must pair by **exact filename stem match** (no fuzzy/index-based matching).
  Wired this into `inspect_dataset.py` as an enforced rule: a new "Strict validation rule" section
  per split in the Markdown report, plus a `--strict`/`--no-strict` CLI flag (`--strict` is default)
  that makes the script exit with code 1 if any violation is found. `test/` split is correctly
  exempted from the missing-GT check (it has no GT by design) but still shape-checks NoisyLR.
- **Why:** User requested this be an explicit, named rule rather than only being implicit in the
  earlier missing-pair/shape-mismatch warnings — makes the contract testable and CI-enforceable.
- **Config used:** N/A (CLI flag, not yet in Hydra config system).
- **Results:** Re-ran `python inspect_dataset.py` against the current placeholder data
  (2 NoisyLR files, no GT) — correctly reported 2 `missing_gt` violations and exited with code 1.
  Manually verified (pytest unavailable offline, same caveat as Phase 2) 4 new test cases: missing-GT
  detection, bad-shape detection, a clean 0-violation case, and correct non-enforcement of missing-GT
  for the `test` split. All 4 passed.
- **Lessons:** Keeping this as a separate, explicitly named function (rather than folding it into
  the existing warning logic) makes it independently unit-testable and gives later phases (dataset
  pipeline, CI) a single clear function to call before trusting any data.
- **Next experiment:** Phase 3 — dataset pipeline.

---

## Phase 3 — Dataset Pipeline
- **What changed:** Split the pipeline into two layers:
  1. `utils/data_transforms.py` — pure numpy, torch-free: `normalize()` (3 modes: `none`,
     `clip_unit`, `dataset_stats`), `random_crop_pair()` (aligned NoisyLR/GT crop, GT always
     exactly 2x), `augment_pair()` (flip/90° rotation only — preserves noise statistics).
  2. `datasets/restoration_dataset.py` — thin torch wrapper: `RestorationDataset` (calls
     `validate_strict_pairing` from Phase 2.1 at init; `strict=True` raises `DatasetContractError`
     immediately, `strict=False` silently excludes bad/corrupted/unpaired samples but keeps clean
     ones) and `build_dataloader()` (defaults to batch_size=1, num_workers=0, matching the
     RTX3050 "prove correctness first" strategy).
- **Why:** Keeping array logic torch-free means it can be unit tested in *any* environment,
  including this build sandbox which has no network to install torch. Only the thin wrapper
  needs torch, and it's small enough to review carefully by hand plus test via a fake-torch
  harness.
- **Config used:** N/A (plain constructor args — Hydra wiring comes in Phase 4/configs/dataset/).
- **Results:**
  - `data_transforms.py`: all normalization modes verified against hand-computed expected values;
    crop pixel-alignment verified over 20 random samples using a synthetic exact-2x-upsample GT
    (crop of GT always exactly matched `2x` nearest-neighbor-upsample of the NoisyLR crop);
    augment shape preservation verified.
  - `restoration_dataset.py`: torch is not installed in this sandbox (no network egress), so a
    minimal fake `torch`/`torch.utils.data` module was injected via `sys.modules` to actually
    execute (not just review) the real class logic. This caught a real bug during development:
    corrupted files were not being excluded in lenient mode because `validate_strict_pairing`
    only shape-checks files that already loaded successfully — fixed by explicitly tracking
    corrupted stems separately in `RestorationDataset.__init__`. After the fix, 7/7 scenarios
    passed: strict-mode rejection, lenient-mode exclusion of bad samples, patch-crop shapes,
    val-split full-image behavior (patch_size ignored), test-split no-GT behavior, and
    DataLoader shuffle/drop_last defaults for train vs val.
- **Lessons:** Separating "logic that can be tested anywhere" from "thin framework glue" paid
  off immediately — it let real bugs be caught in an environment that can't even install the
  target framework. `tests/test_data_transforms.py` runs standalone; `tests/test_restoration_dataset.py`
  uses `pytest.importorskip("torch")` so it skips cleanly here but will run for real in your dev env.
- **Next experiment:** Phase 4 — framework core (registry, trainer/evaluator/inference/benchmark
  interfaces that models plug into).

---

## Phase 4 — Framework Core (task-agnostic interfaces + registries)
- **What changed:** Built the plugin architecture the whole project depends on:
  - `framework/registry/registry.py` — generic `Registry` class (register/get/build/list), plus
    7 global registries (MODEL, LOSS, METRIC, TASK, OPTIMIZER, SCHEDULER, CALLBACK) in
    `framework/registry/__init__.py`.
  - `tasks/base.py` — `BaseTask`: scale-factor-driven shape contract (`validate_shapes`),
    pre/postprocess hooks, and `check_model_compatibility` (fails fast on task/model scale mismatch).
    Torch-free by design, like Phase 3's transforms.
  - 3 concrete, registered tasks: `restoration_sr` (scale=2, the primary hackathon task),
    `super_resolution` (scale=2, separate config for future ablations), `denoising` (scale=1,
    same-resolution) — added `tasks/denoising/` and `configs/task/` as new folders at the *start*
    of this phase, before the freeze takes effect, per user request for restoration/SR/denoising
    interchangeability.
  - `models/base.py`, `losses/base.py` — abstract `BaseRestorationModel`/`BaseLoss` (torch `nn.Module`
    contracts; concrete subclasses arrive in Phases 5/11+).
  - `metrics/base.py` — `BaseMetric` (reset/update/compute), deliberately torch-*optional* (duck-types
    numpy vs torch tensors) so it's testable now.
  - `framework/callbacks/base.py` — `Callback`/`CallbackList` lifecycle hooks, pure Python.
  - `framework/trainer/base.py`, `framework/evaluator/base.py`, `framework/inference/base.py`,
    `framework/benchmark/base.py` — abstract interfaces only; concrete implementations are Phases
    7/8/9/10 respectively, built *against* these frozen contracts rather than from scratch.
  - `framework/trainer/step.py` — `compute_step()`: the shared forward(+loss) function that Phase 7's
    trainer and Phase 8's evaluator will both call. Zero torch dependency, so it's the one piece of
    "framework glue" logic that could be integration-tested immediately.
- **Why:** User asked explicitly that model/loss/metric/trainer/evaluator/inference be fully
  interchangeable via registries/configs, and that restoration/SR/denoising all be first-class
  tasks. The task abstraction (scale_factor-driven contract, not a hardcoded 128->256 assumption)
  is what makes same-resolution denoising a real citizen alongside 2x SR without special-casing
  it anywhere in trainer/evaluator code.
- **Config used:** `configs/task/{restoration_sr,super_resolution,denoising}.yaml` — plain YAML,
  validated with PyYAML (Hydra/OmegaConf not installed in this sandbox, but `Registry.build()` only
  needs dict-like `cfg["name"]`/`cfg.get(...)` access, which OmegaConf's DictConfig satisfies
  identically — no code change needed once Hydra is wired in).
- **Results:** 26/26 checks passed, run manually since pytest can't be installed offline (same
  caveat as Phases 2/3): 9 registry tests (register/get/build/duplicate-name/unknown-name/kwargs-merge),
  9 task tests (all 3 tasks build with correct scale_factor, shape validation accepts/rejects
  correctly per task — including denoising correctly *rejecting* an upscaled pair, which is the
  opposite of restoration_sr's rule), 4 callback tests, and — the most important one — 4 integration
  tests proving `compute_step()` handles two *completely different* (task, model, loss) triples
  (restoration_sr + doubling-model + L1-like-loss, vs. denoising + identity-model + same loss) with
  zero modification to `compute_step` itself. Also syntax-checked (`py_compile`) the torch-dependent
  files (`models/base.py`, `losses/base.py`, all 4 framework stubs) since torch isn't installed here;
  `metrics/base.py` was fully executed since it's torch-optional.
- **Lessons:** The task abstraction is the load-bearing piece here — once scale_factor lives on the
  task (not hardcoded as "2x" in the trainer), swapping in denoising is just a new registered class +
  config file, exactly as required. Keeping `compute_step` torch-free (it only ever calls whatever
  `model(x)`/`loss_fn(pred, gt)` it's given) meant the *actual* interchangeability claim could be
  integration-tested now instead of only in Phase 7 when torch is available.
- **Next experiment:** Phase 5 — loss framework (concrete L1/Charbonnier/SSIM/LPIPS/Frequency losses,
  registered under LOSS_REGISTRY, replacing today's dummy test losses).

---

## Phase 5 — Loss Framework
- **What changed:** Implemented 5 concrete losses, each a thin `BaseLoss` subclass registered under
  `LOSS_REGISTRY`: `l1_loss.py` (mean absolute error), `charbonnier_loss.py` (smooth L1 variant,
  `sqrt(diff^2 + eps^2)`), `ssim_loss.py` (1 - SSIM via a hand-implemented Gaussian-window conv2d,
  no external SSIM package dependency), `frequency_loss.py` (L1 on 2D FFT magnitude, for
  high-frequency detail recovery), and `lpips_loss.py` (wraps the `lpips` package, lazy-imported
  with a clear error if pretrained weights can't be fetched/found). Added `composite_loss.py`
  (`CompositeLoss`), which builds any weighted combination of the above **entirely through
  `LOSS_REGISTRY.build()`** — no special-casing — so ablations (`l1` alone vs. `l1 + 0.1*ssim` vs.
  `l1 + ssim + frequency`) are pure config changes (`configs/loss/*.yaml`, including two example
  composite configs).
- **Why:** Matches the project's loss framework spec exactly (L1, Charbonnier, SSIM, LPIPS,
  Frequency, config-driven ablations). SSIM was hand-implemented rather than pulled from a package
  to avoid a version-compatibility surface for ~30 lines of well-understood math; LPIPS is the one
  loss that genuinely needs an external asset (pretrained weights) and is therefore wrapped
  defensively rather than assumed to just work offline.
- **Config used:** `configs/loss/{l1,charbonnier,ssim,frequency,lpips}.yaml` (one per loss) plus
  `l1_ssim_composite.yaml` and `l1_ssim_frequency_composite.yaml` (example ablations) — all
  validated with PyYAML.
- **Results:** torch is still not installed in this sandbox, so nothing here could be executed as
  real `nn.Module` code. Instead:
  - L1, Charbonnier, and Frequency formulas were hand-verified against pure numpy mirrors of the
    exact same math (e.g. Charbonnier-at-zero-diff correctly equals `eps`, not 0, proving the
    smoothing behaves as designed; Frequency loss correctly returns 0 for identical images).
  - **SSIM was cross-checked against scikit-image's independent reference SSIM implementation**
    (`skimage.metrics.structural_similarity`) using a numpy mirror of my exact conv2d-based formula.
    Identical images gave SSIM=1.0 exactly. Two random-noise images and a blurred-vs-original pair
    matched skimage within <0.008 once boundary-padding convention was aligned (skimage defaults to
    reflect-padding; my torch code uses zero-padding via `F.conv2d`, which is the standard
    convention in deep-learning SSIM *losses* — e.g. pytorch-msssim, BasicSR — so this is an
    accepted convention difference, not a bug).
  - `CompositeLoss`'s weighted-sum + breakdown aggregation logic (the part specific to this class,
    as opposed to standard `nn.Module`/`nn.ModuleList` boilerplate) was verified by replicating its
    exact algorithm with plain-Python registry-built stand-ins: weighted sum and per-component
    breakdown both matched hand-computed expected values.
  - All 6 loss files syntax-checked clean with `py_compile`.
  - Wrote `tests/test_losses.py` (11 tests, `pytest.importorskip("torch")`) for your real environment.
- **Lessons:** Same pattern as Phases 3/4 — separate "the math, which can be verified independent
  of torch" from "the nn.Module wiring, which is standard boilerplate reviewed by eye." Getting an
  independent third-party reference (scikit-image) for SSIM specifically was worth the extra step
  since a self-consistent-but-wrong formula wouldn't have been caught by a numpy mirror of the same
  bug.
- **Next experiment:** Phase 6 — metrics (PSNR, SSIM, LPIPS as evaluation metrics — distinct from
  today's SSIM *loss* — with dataset-average and per-image reporting).

---

## Phase 6 — Metrics
- **What changed:** Implemented 3 concrete metrics registered under `METRIC_REGISTRY`:
  `psnr_metric.py` (`PSNRMetric`), `ssim_metric.py` (`SSIMMetric` — raw similarity value, distinct
  from Phase 5's `1 - SSIM` loss), `lpips_metric.py` (`LPIPSMetric`, wraps the same `lpips` package
  as the LPIPS loss but in eval/no-grad mode). All follow the Phase 4 `BaseMetric` reset/update/
  compute contract and store every per-sample value (`self.values`) alongside the aggregate, so
  Phase 8's per-image CSV export needs zero changes to these classes. Added `metrics/__init__.py`'s
  `build_metrics()` helper (name/config list -> `{name: BaseMetric}` dict) and 4 metric config files
  including a `default_benchmark_set.yaml` (PSNR+SSIM+LPIPS) for Phase 10.
- **Why:** PSNR and SSIM have no learned components, so — continuing the Phase 4 `BaseMetric`
  design choice to be torch-optional — they're implemented in pure numpy/scipy. This means they're
  not just formula-reviewed like Phase 5's losses, but **fully executable and independently
  verifiable right now**, in this sandbox, with no torch install. LPIPS is inherently torch- and
  network-dependent (learned network + downloaded weights) so it's wrapped the same defensive way
  as the LPIPS loss.
- **Config used:** `configs/metric/{psnr,ssim,lpips}.yaml` + `default_benchmark_set.yaml`, all
  validated with PyYAML.
- **Results:** 9/9 executable tests passed for real (not manual-review — actually run):
  - **PSNR matched `skimage.metrics.peak_signal_noise_ratio` to within 1e-6** across 5 random image
    pairs; identical-image case correctly returns `+inf` (standard convention), and an
    all-identical-samples dataset average also correctly resolves to `inf` rather than crashing.
  - **SSIM matched `skimage.metrics.structural_similarity` to within 0.007** (same reflect-boundary
    convention as skimage this time, unlike the SSIM loss which intentionally uses zero-padding);
    identical images gave exactly `1.0`; correctly handled `(1, 1, H, W)` batched input via squeeze,
    and correctly *rejected* an ambiguous `(2, 1, H, W)` batch-of-2 input with a clear error rather
    than silently averaging over images.
  - `build_metrics()` verified with mixed string/dict entries.
  - LPIPS syntax-checked (`py_compile`) only — needs torch + the `lpips` package + cached pretrained
    weights, none available in this sandbox; `tests/test_metrics.py`'s LPIPS test uses
    `pytest.importorskip` and skips gracefully if weights aren't cached even in a torch-enabled
    environment, so it fails soft rather than blocking the rest of the suite.
- **Lessons:** This phase had noticeably higher confidence than Phases 3-5 specifically because
  PSNR/SSIM's torch-optional design (a Phase 4 decision) meant they could be verified against a
  real independent library (`scikit-image`) end-to-end, not just formula-mirrored. Worth remembering
  for Phase 7+: wherever a component doesn't strictly need torch, keeping it torch-optional pays for
  itself in test confidence.
- **Next experiment:** Phase 7 — training engine (the full `BaseTrainer` implementation: optimizer
  stepping, AMP/BF16, gradient accumulation, EMA, checkpoint resume, DDP).

---

## Phase 7 — Training Engine
- **What changed:** Implemented the full `Trainer(BaseTrainer)` and its supporting pieces:
  - `framework/trainer/optim.py`, `scheduler.py` — AdamW/Adam/SGD and Cosine/CosineWarmRestarts/Step/
    None registered directly under OPTIMIZER_REGISTRY/SCHEDULER_REGISTRY (torch classes need no wrapping).
  - `framework/trainer/precision.py` — `PrecisionManager`: unifies FP32 (no autocast/scaler) / FP16
    (autocast + GradScaler, needed because FP16's limited dynamic range can underflow gradients) /
    BF16 (autocast, no scaler needed — same dynamic range as FP32) behind one interface so `Trainer`
    has no precision-specific branching.
  - `framework/trainer/ema.py` — `EMA` with the standard decay-warmup trick (ramps from ~0.1 toward
    the target decay early in training, avoiding a shadow pinned to near-random init weights).
  - `framework/trainer/checkpoint.py` — save/load full trainer state (model/optimizer/scheduler/
    scaler/EMA/epoch/step) plus reproducibility metadata (git commit hash, Python/torch/CUDA
    versions, GPU info) per the experiment-tracking spec.
  - `framework/trainer/trainer.py` — ties it together: `train_one_epoch` (autocast forward via
    `compute_step` from Phase 4, scaled backward, grad-accum boundary logic, gradient clipping,
    EMA update), `validate` (no-grad, per-sample metric accumulation via Phase 6's metrics),
    `fit`/`save_checkpoint`/`load_checkpoint`. Fails fast at construction if task/model
    `scale_factor` mismatch (reuses Phase 4's `check_model_compatibility`). DDP handled as
    "accept an already-wrapped model, guard callbacks/checkpointing behind `is_main_process`" —
    process-group setup is left to the launcher script, not buried in this class.
  - 3 concrete callbacks: `CheckpointCallback` (last.pt every N epochs, best.pt on metric
    improvement), `MemoryMonitorCallback` (GPU memory per epoch, no-ops cleanly without CUDA),
    `TensorBoardCallback` (offline scalar logging).
  - `configs/optimizer/adamw.yaml`, `configs/scheduler/{cosine,none}.yaml`, and 3 trainer presets:
    `rtx3050_dev.yaml` (fp32, no accum/EMA — prove correctness first), `rtx3050_dev_optimized.yaml`
    (fp16 + 8x grad accum + EMA, for fitting a larger effective batch into 4GB), `h100_full.yaml`
    (bf16, EMA on, DDP-compatible).
- **Why:** Matches the project's training-system spec (AdamW, cosine scheduler, EMA, grad clipping,
  checkpoint resume, best-model saving, memory monitoring, FP32/AMP/BF16, single-GPU/DDP) while
  keeping every precision/DDP/accumulation decision as config, not code branches scattered through
  the loop.
- **Config used:** All configs above, validated with PyYAML.
- **Results:** Per the project's own Phase Priority Rule, this phase's true end-to-end proof is
  Phase 11 (sanity model) — torch still isn't installed in this sandbox, so nothing requiring a real
  `nn.Module`/optimizer step could execute here. What COULD be verified for real, and was:
  - **`get_git_commit_hash()` returns this repo's actual current commit hash** (verified against
    real `git log` output) and degrades to `"unknown"` gracefully outside a repo.
  - `collect_env_info()` correctly reports `torch_version: "not installed"` / `cuda_available: False`
    in this genuinely torch-free sandbox — proving the lazy-import fallback works, not just that it
    looks right.
  - **EMA's decay-warmup formula** independently verified with a numpy simulation: step-0 decay is
    exactly 0.1, saturates at the target decay (0.999) for large step counts, and a simulated 20-step
    EMA correctly tracked noisy "model params" toward their true mean.
  - **Gradient-accumulation boundary counting** verified by simulation: `grad_accum_steps=3` over 7
    batches correctly produces optimizer steps at batches 3 and 6, with the trailing partial batch
    correctly left un-flushed (documented behavior, not a bug).
  - Found and fixed a real bug during review: `CheckpointCallback` was pre-computing
    `.state_dict()` before passing to `save_checkpoint`, which itself also calls `.state_dict()` —
    fixed to pass the `PrecisionManager` object directly.
  - All 13 new/touched files (`optim.py`, `scheduler.py`, `ema.py`, `checkpoint.py`, `precision.py`,
    `trainer.py`, 3 callbacks, 5 test files) are `py_compile`-clean.
  - Wrote `tests/test_trainer.py` (5 integration tests using a real tiny `nn.Module`), plus
    `test_ema.py`, `test_checkpoint.py`, `test_precision.py`, `test_optim_scheduler.py` — all guarded
    with `pytest.importorskip("torch")` where needed, ready for Phase 11 to actually execute them
    against the framework's real sanity model.
- **Lessons:** This is the largest phase so far and the first one where "torch genuinely isn't
  available" starts to bite — most of the interesting logic (the training loop itself) can only be
  code-reviewed here, not executed. Deliberately separated out every piece of logic that COULD be
  verified independent of a live model (git/env metadata, EMA math, grad-accum counting) and did so,
  rather than treating the whole phase as "untestable until Phase 11."
- **Next experiment:** Phase 8 — evaluation system (BaseEvaluator implementation: full-split metric
  aggregation, per-image CSV export, difficult-sample analysis, visual comparisons).

---

## Phase 8 — Evaluation System
- **What changed:** Implemented `Evaluator(BaseEvaluator)` and its reporting/visualization support:
  - `framework/evaluator/evaluator.py` — `evaluate()` (dataset-average metrics) and
    `evaluate_per_image()` (per-sample rows), both built on a single `_run()` pass reusing Phase 7's
    `compute_step` with `loss_fn=None`. `save_report()` writes CSV + Markdown + worst-K visual
    comparisons + difference maps. Deliberately two-pass for images: the main `_run()` never holds
    prediction images in memory (only scalar per-sample scores), and a second, cheap
    `_fetch_images_for_stems()` pass re-runs the model only for the identified worst-K samples —
    memory-conscious by design, matching the project's stated priority order (quality >
    reproducibility > modularity > benchmark clarity > runtime > **memory efficiency**).
  - `framework/evaluator/report.py` — `write_per_image_csv`, `find_worst_samples` (min/max mode),
    `write_summary_markdown`. Pure stdlib (`csv` module) + plain dicts — no torch dependency at all.
  - `framework/evaluator/visualize.py` — `plot_comparison_grid` (NoisyLR/Prediction/GT triplets) and
    `plot_difference_map` (|pred-gt| heatmap) — pure matplotlib, same headless-Agg pattern as Phase 2.
  - Raises a clear error (not a confusing crash) if `evaluate()` is called on a GT-less split,
    explicitly pointing to the Phase 9 Inferencer as the right tool for that case.
- **Why:** This phase is the direct payoff of Phase 4's shared-`compute_step` design — the evaluator
  needed zero new forward-pass logic, only metric accumulation and reporting around an already-proven
  function. Splitting report/visualize (torch-free) from evaluator.py (torch-dependent) continues the
  pattern that's proven valuable every phase since 3: it's what let 9 of this phase's components be
  fully executed rather than just reviewed.
- **Config used:** N/A this phase (Evaluator is constructed directly with a metrics dict from
  Phase 6's `build_metrics`) — CLI/config wiring for a standalone `evaluate.py` script is Phase 19.
- **Results:** **9/9 torch-free tests executed for real**: CSV round-trip (including the empty-split
  edge case), `find_worst_samples` in both min/max mode plus correct `None`-value skipping, Markdown
  summary content verification, and both plotting functions (including the empty-input no-op case for
  the comparison grid). The `Evaluator` class itself (2 tests: full end-to-end report generation, and
  the no-GT error path) needs torch and is written + guarded in `tests/test_evaluator.py`, ready for
  Phase 11.
- **Lessons:** No new bugs found this phase (unlike Phase 3's corrupted-file bug and Phase 7's
  double-`state_dict()` bug) — likely because this phase leaned harder on code already proven in
  earlier phases (`compute_step`, `build_metrics`) rather than introducing large amounts of new
  torch-dependent logic.
- **Next experiment:** Phase 9 — inference pipeline (BaseInferencer implementation: run on GT-less
  data, save predictions as .npy).

---

## Phase 9 — Inference Pipeline
- **What changed:** Implemented `Inferencer(BaseInferencer)`:
  - `run(data_loader)` — batched inference over a GT-less loader (matching the real `test/` split's
    contract: NoisyLR only, no "gt" key anywhere in the batch), built on `compute_step` with
    `loss_fn=None`. Saves each prediction as `{output_dir}/{stem}.npy` in float32 — the exact same
    on-disk convention as the dataset itself, so a prediction is directly diffable against a GT file
    if one's ever available for scoring.
  - `run_single(noisy)` — single in-memory array/tensor inference, accepting either a bare (H,W)
    array or an already-batched tensor, explicitly documented as NOT re-normalizing internally (a
    pure model-call wrapper, no hidden assumptions about input scale — the caller is responsible for
    matching whatever normalization mode training used).
- **Why:** This is the actual submission-generation pathway. Reusing `compute_step` again meant zero
  new forward-pass logic was needed (third phase in a row leaning on that Phase 4 design). Explicitly
  decided NOT to build `scripts/infer.py`/`scripts/train.py`/`scripts/evaluate.py` CLI wrappers yet —
  same reasoning as Phase 7: without a real trained model (Phase 11+) to exercise them against, a CLI
  script now would be untestable glue code. These will be built once there's something real to wire
  them to.
- **Config used:** N/A — inference is invoked directly with a task/model/checkpoint, not registry-
  config-driven in the same way losses/metrics/models are (there's nothing to swap via config here
  beyond the model itself, which is already config-driven via MODEL_REGISTRY).
- **Results:** All logic is torch-dependent (model forward pass), so nothing could execute in this
  sandbox — syntax-checked clean with `py_compile`. Wrote `tests/test_inference.py` (5 tests:
  correct output shapes/stems, `.npy` files actually written to disk with correct shape/dtype,
  `run_single` accepting both raw arrays and tensors, and an explicit test that a batch with no "gt"
  key anywhere doesn't crash — the exact shape of the real `test/` split). Ready for Phase 11.
- **Lessons:** Third phase running on the same `compute_step` foundation (Trainer, Evaluator, now
  Inferencer) — this is the concrete, repeated payoff of getting the Phase 4 task/model abstraction
  right early rather than writing three separate forward-pass implementations.
- **Next experiment:** Phase 10 — benchmark framework (params, FLOPs/MACs, model size, latency
  mean+std, GPU memory, throughput — the research-paper-style comparison table).

---

## Phase 10 — Benchmark Framework
- **What changed:** Implemented `Benchmark(BaseBenchmark)`:
  - `framework/benchmark/timing.py` — `compute_latency_stats()` (mean/std/min/max latency +
    throughput from a list of raw per-iteration timings) and `compute_model_size_mb()` (from
    `(numel, element_size)` pairs) — both pure arithmetic, no torch dependency, factored out
    specifically so the math is independently testable.
  - `framework/benchmark/benchmark.py` — `Benchmark.run()`: params (`model.num_parameters()` from
    Phase 4's `BaseRestorationModel`), model size, FLOPs/MACs (via fvcore's `FlopCountAnalysis`,
    wrapped in try/except — fvcore's operator coverage is architecture-dependent, so a failure
    degrades to `flops=None/macs=None` with a warning rather than crashing the whole benchmark),
    latency mean+std over `timed_iters` (after `warmup_iters` untimed passes, with
    `torch.cuda.synchronize()` around each timing when on GPU), GPU peak memory (skipped cleanly,
    not crashed, when CUDA isn't available), throughput, and an optional
    `training_time_per_epoch_seconds` the caller can supply from an actual training run's logs
    (this framework can't itself "benchmark" a quantity that only exists across a full epoch of
    real training).
- **Why:** Matches the project's research-paper-style benchmark table spec exactly. Splitting the
  arithmetic (timing.py) from the torch-dependent execution (benchmark.py) continues the pattern from
  every phase since 3 — and once again this was the difference between "fully executed" and
  "code-reviewed only" for this phase's components.
- **Config used:** N/A — same reasoning as Phase 9 (Inferencer): benchmark parameters are run-time
  arguments (input shape, iteration counts), not registry-swappable components.
- **Results:** **7/7 timing/size math tests executed for real**, including throughput correctly
  scaling linearly with batch size, single-sample std correctly resolving to 0.0 (not NaN, which
  `statistics.stdev` would raise on a length-1 input if not special-cased), and the empty-input
  error case. Confirmed `fvcore` is genuinely not installed in this sandbox (no network) — meaning
  the ImportError fallback path is exercising real, not hypothetical, behavior. `Benchmark.run()`
  itself needs torch (times a real forward pass) and is syntax-checked + written into
  `tests/test_benchmark.py` (2 tests: full key coverage on CPU, and training-time pass-through),
  ready for Phase 11.
- **Lessons:** Fourth phase in a row (Trainer's precision/EMA math, Evaluator's report/visualize,
  Inferencer's shape contracts, now Benchmark's arithmetic) where separating "torch-free math/logic"
  from "torch-dependent execution" into different files was the deciding factor in how much of the
  phase could be proven correct here versus deferred. This is now a consistent enough pattern across
  Phases 3-10 that it's worth calling out as a standing design principle for anything still to come.
- **Next experiment:** Phase 11 — sanity model integration. This is the big validation checkpoint:
  build the tiny CNN sanity model and prove the ENTIRE framework built in Phases 3-10 — dataset
  pipeline, task contract, losses, metrics, trainer (precision/EMA/checkpointing/callbacks),
  evaluator, inferencer, benchmark — actually works end-to-end with a real model and real gradients,
  per the project's own Phase Priority Rule ("the framework must be validated with the sanity model
  first"). Everything from here through Phase 10 has been built and math-verified but not yet proven
  as a working whole.

---

## Phase 11 — Sanity Model Integration
- **What changed:**
  - `models/sanity_cnn/model.py` — `SanityCNN`: a handful of 3x3 conv+ReLU layers followed by
    `nn.PixelShuffle` (sub-pixel convolution) for learned upsampling. `PixelShuffle(1)` is a
    well-defined identity-like rearrangement, so this same architecture also correctly supports
    `scale_factor=1` (denoising), not just the primary `scale_factor=2` task. Registered under
    `MODEL_REGISTRY` as `sanity_cnn`. Per the project's Sanity Model Rule, this must never appear in
    the final Phase 18 benchmark comparison.
  - `utils/experiment_dir.py` — `get_next_run_dir()`: returns a fresh `run_NNN` directory, correctly
    handling gaps in existing run numbers and ignoring non-run-named directories — the "never
    overwrite a past run" rule from Phase 1's experiment management spec, actually implemented.
  - `scripts/sanity_check.py` — the actual runnable validation script, checking all 8 Sanity Model
    Rule items in order (data loading, forward pass, backward pass, loss computation, metrics,
    checkpoint saving, resume, inference) plus a bonus Evaluator-report + Benchmark exercise. Falls
    back to a clearly-labeled SYNTHETIC dataset (matching the exact project contract) if the real
    dataset at `--data-root` isn't yet strictly valid — which is the case right now, since only 2
    placeholder NoisyLR files exist with no GT. Prints an explicit PASS/FAIL checklist and returns a
    non-zero exit code if any required check fails, with an explicit "do NOT proceed to Phase 12"
    message on failure.
- **Why:** This is the checkpoint the whole framework (Phases 3-10) has been building toward. Every
  earlier phase deferred its "real" proof to this one, per the project's own Phase Priority Rule.
- **Config used:** `configs/model/sanity_cnn.yaml`.
- **IMPORTANT — what could and could NOT be verified in this environment, stated plainly:**
  This sandbox has no network egress and torch could not be installed (re-confirmed by direct
  attempt at the start of this phase: `pip install torch` still fails with no matching distribution).
  Every previous phase found torch-free angles to verify real logic (math mirrors, fake-torch
  harnesses for simple classes). This phase is different in kind: its entire purpose is proving real
  gradient descent, real backward passes, and real checkpoint serialization work — faking that with
  a mock `torch` would defeat the point and give false confidence exactly where it matters most. So,
  honestly:
  - **What WAS verified for real:** `scripts/sanity_check.py`'s synthetic-data generator was fed
    through the actual `RestorationDataset` class (Phase 3) using the same fake-torch harness proven
    then — confirming the generated data genuinely satisfies the strict shape/pairing contract
    (Phase 2.1) with zero violations, correct split sizes (8 train / 4 val / 2 test), correct
    per-sample shapes, and correct absence of the "gt" key for the test split. This is real proof of
    item 1/8 (data loading) up to the point where a real model takes over.
  - `utils/experiment_dir.py` — all 6 scenarios (first run, sequential runs, gaps, non-run
    directories, `create=False` behavior) executed for real.
  - `models/sanity_cnn/model.py`'s `PixelShuffle`-based shape math (in → out channel/spatial
    relationship) was verified by hand against `PixelShuffle`'s documented behavior — not executed.
  - **What could NOT be verified here, and requires you to run it:** Items 2-8 of the Sanity Model
    Rule (forward pass, backward pass, loss computation, metrics on real tensors, checkpoint
    save/resume via `torch.save`/`torch.load`, and inference) all require a real torch install.
    `scripts/sanity_check.py` and `tests/test_sanity_cnn.py` are written, reviewed, and
    syntax-checked (`py_compile`) — but genuinely unexecuted.
  - **Action required before Phase 12:** per your own Phase Priority Rule, please run
    `python scripts/sanity_check.py` in your real WSL/RTX3050 environment and confirm all 8 checks
    pass before NAFNet integration begins. If anything fails, that needs fixing here first — an
    earlier phase's code, not just this script, may be at fault.
- **Lessons:** Found and fixed one real bug during review before it could mislead: the script
  originally called `train_one_epoch()` directly in a manual loop (to capture per-epoch loss) without
  updating `trainer.epoch`, which would have made the later checkpoint-resume assertion
  (`trainer2.epoch == trainer.epoch + 1`) trivially true regardless of whether resume logic was
  actually correct across multiple epochs. Fixed by explicitly advancing `trainer.epoch` in the loop,
  mirroring what `fit()` does internally.
- **Next experiment:** Phase 12 — NAFNet integration (official implementation, adapted via
  trainer/evaluator/inference wrappers only, no architecture modification) — **gated on you
  confirming the Phase 11 sanity check passes for real.**

---

## Phase 12 — NAFNet Integration
- **Note on process:** you replied "Continue" without confirming the Phase 11 sanity check had been
  run. I asked once more explicitly; you said "Continue" again. Proceeding on that basis — logged
  here plainly as an accepted risk, not a resolved one: Phases 12+ are now built on top of a
  training/evaluation/inference/checkpoint stack that has not been proven to actually work with real
  gradients in any environment yet.
- **IMPORTANT — deviation from your spec, flagged before proceeding:** your master prompt says
  "Use official implementations wherever possible. Do not rewrite published architectures
  unnecessarily." This sandbox has no network egress for `git clone`/`pip install` (confirmed: a
  direct `git clone https://github.com/megvii-research/NAFNet` attempt was blocked by the network
  allowlist). `web_search`/`web_fetch` could reach some of the real source, but Anthropic's copyright
  policy prohibits reproducing source code verbatim from search results into project files,
  regardless of the upstream license. So: **`models/nafnet/model.py` is a clean-room
  reimplementation** of the architecture from its published paper description (Chen et al., "Simple
  Baselines for Image Restoration", ECCV 2022) — not the vendored official repo. `ATTRIBUTION.md` in
  the same directory documents this explicitly, cites the paper, and gives you the exact commands to
  swap in the real official repo once you have network access, which you do and this sandbox doesn't.
- **What changed:** Implemented `LayerNorm2d`, `SimpleGate` (the paper's central finding —
  activation-free gating replaces GELU/ReLU/Sigmoid), `SimplifiedChannelAttention`, `NAFBlock` (two
  gated residual sub-blocks with learnable per-channel scales beta/gamma), and `NAFNet` (U-Net-style
  encoder-decoder of NAFBlocks, strided-conv downsampling, PixelShuffle upsampling, skip connections).
  Added an SR head (extra PixelShuffle upsampling + bilinear-upsample global residual) for
  `scale_factor > 1`, since the original architecture is same-resolution-only — documented as this
  project's adaptation, not part of the original paper. Registered as `nafnet` under `MODEL_REGISTRY`.
- **Why:** Matches your model-integration rule (preserve architecture correctness, adapt only
  trainer/config/logging/eval/inference/benchmark wrappers) as closely as this environment allows —
  the reimplementation preserves the paper's actual architectural ideas faithfully; only the SR head
  is a genuine addition, clearly marked as such.
- **Config used:** `configs/model/nafnet.yaml` (project defaults: width=32, 4 encoder/decoder stages,
  2 middle blocks — validated with PyYAML).
- **Results:** Torch still unavailable in this sandbox, so nothing here could execute. What WAS done:
  a full **symbolic shape trace** (pure Python, no torch) mechanically replaying the exact
  channel/spatial arithmetic through intro → 4 encoder+downsample stages → bottleneck → 4
  decoder+upsample stages → SR head, confirming every stage's shape matches its skip connection and
  the final output is exactly `(width, H, W)` before the SR head and `(out_channels, H*scale, W*scale)`
  after — this caught nothing wrong, but genuinely re-derived the arithmetic independently rather than
  trusting the by-hand trace alone. **Also verified the padding/cropping path** for a
  non-padder-size-aligned input (50x50 with 2 stages, padder_size=4) — correctly pads to 52x52,
  processes, and crops the final output to exactly 100x100. `tests/test_nafnet.py` (6 tests) written
  and syntax-checked, ready to run for real once torch is available.
- **Lessons:** This phase introduced a real, structural deviation from your spec (reimplementation
  instead of the official repo) driven by two independent hard constraints (no network, copyright
  policy) rather than a judgment call — flagged explicitly per your own "point out conflicts before
  changing anything" rule, with a concrete remediation path in `ATTRIBUTION.md`.
- **Next experiment:** Phase 13 — benchmark NAFNet (requires a real environment: this is where the
  sanity-check gate and this phase's unexecuted code both get their first real test simultaneously).

---

## Phase 13 — Benchmark NAFNet
- **What changed:** `framework/benchmark/report.py` (JSON persistence + single-model and multi-model
  Markdown tables — pure stdlib, model-agnostic by design so it's reused unchanged in Phases 15/17
  and the Phase 18 final comparison) and `scripts/benchmark_model.py` (CLI: `--model <registry name>`,
  builds task+model, runs Phase 10's `Benchmark`, writes results to a fresh
  `experiments/<model>/run_NNN/` directory via Phase 11's `get_next_run_dir`).
- **Why:** Same reasoning as every prior phase's report/logic split — `report.py` needed no torch and
  could be fully proven now; the actual benchmark numbers need a real forward pass on real hardware.
- **Results:** **`report.py` fully executed and verified**: JSON round-trip, Markdown table content
  (formatted with thousands separators, `N/A` for missing fields rather than crashing), and the
  multi-model comparison table correctly handling a model missing a field (`sanity_cnn` has no
  `flops` in the test — renders `N/A`, doesn't break the table). `scripts/benchmark_model.py` is
  syntax-checked only — it needs torch and real hardware for actual latency/FLOPs/memory numbers,
  and **no numbers are fabricated here**: this phase produces the tool, not the results. You'll need
  to run `python scripts/benchmark_model.py --model nafnet --device cuda` yourself once Phase 11's
  gate is genuinely satisfied.
- **Lessons:** None new — this phase mostly reused Phase 10/11's already-established patterns
  (registry-driven CLI, experiment run directories, report/logic separation) rather than introducing
  new design decisions.
- **Next experiment:** Phase 14 — Restormer integration (same official-vs-reimplementation
  constraint as NAFNet applies here too — will flag again if it does).

---

## Phase 14 — Restormer Integration
- **Same deviation as Phase 12, flagged again explicitly:** `git clone
  https://github.com/swz30/Restormer` is equally blocked here (no network egress), and the same
  copyright constraint applies to reproducing fetched source. `models/restormer/model.py` is a
  clean-room reimplementation from the published paper (Zamir et al., "Restormer: Efficient
  Transformer for High-Resolution Image Restoration", CVPR 2022), documented in
  `models/restormer/ATTRIBUTION.md` with the same swap-in-the-official-repo instructions.
- **What changed:** Implemented `MDTA` (Multi-Dconv Head Transposed Attention — the paper's key
  idea: self-attention computed across the channel dimension instead of the spatial dimension,
  making cost linear rather than quadratic in image resolution), `GDFN` (GELU-gated feed-forward
  with depth-wise convs), `TransformerBlock`, `Downsample`/`Upsample` (PixelUnshuffle/PixelShuffle,
  avoiding the checkerboard artifacts of strided/transposed convs), and the full 4-level
  encoder-decoder `Restormer` class with concat-then-1x1-reduce skip connections and a refinement
  stage — architecturally more involved than NAFNet's addition-based skips, so this phase's shape
  verification needed to cover the concat/reduce arithmetic specifically. Same SR-head adaptation
  pattern as NAFNet for `scale_factor > 1`. Registered as `restormer`. Default `dim`/`num_blocks`
  reduced from the paper's originals (dim=48 → 24, fewer blocks per stage) to fit the 4GB RTX3050
  constraint — a capacity choice, not an architectural deviation, and documented as such.
- **Why:** Same reasoning as Phase 12 — preserve the paper's actual architectural ideas faithfully
  within the hard constraints of this environment.
- **Config used:** `configs/model/restormer.yaml`, validated with PyYAML.
- **Results:** Torch still unavailable — nothing executed directly. **Symbolic shape trace** (pure
  Python) mechanically verified every stage: patch embed → 3 encoder+downsample levels → bottleneck
  → 3 decoder+upsample levels, **specifically checking the concat-then-reduce channel arithmetic**
  at each skip connection (e.g. `up4_3` output (dim×4) concatenated with `enc3` (dim×4) = dim×8,
  correctly reduced back to dim×4 by `reduce_chan_level3`) and confirming the finest level correctly
  keeps the concatenated dim×2 channels with no reduction, per the paper's actual design. Also
  verified the non-padder-aligned padding/cropping path (100×100 input, padder_size=8 → pads to
  104×104, crops SR output to exactly 200×200). `tests/test_restormer.py` (7 tests, including an
  `MDTA`-specific test for the `dim % num_heads == 0` guard) written and syntax-checked.
- **Lessons:** Restormer's concat-based skip connections (vs. NAFNet's addition-based ones) meant
  the shape-verification script for this phase needed to explicitly trace channel counts through
  concatenation and reduction, not just confirm matching shapes for an add — a slightly different
  and higher-value check than Phase 12's, since concat/reduce channel-count bugs are an easy mistake
  to make and wouldn't necessarily show up as an add-shape-mismatch crash the same way.
- **Next experiment:** Phase 15 — benchmark Restormer (same real-hardware requirement as Phase 13).

---

## Phase 15 — Benchmark Restormer
- **What changed:** One line: added `import models.restormer.model` to `scripts/benchmark_model.py`'s
  registration-trigger imports. `framework/benchmark/benchmark.py`, `framework/benchmark/report.py`,
  and the rest of the CLI script needed zero changes — `python scripts/benchmark_model.py --model
  restormer` now works the same way `--model nafnet` did in Phase 13.
- **Why worth noting despite being a 1-line change:** this is a direct, concrete demonstration of
  the "adding a model requires only models/new_model.py + registry update, no changes to
  trainer/evaluator/inference/benchmark" claim from Phase 4 — Restormer is architecturally very
  different from NAFNet (transformer blocks + concat skips vs. conv blocks + additive skips), and
  the benchmark tooling didn't need to know or care.
- **Results:** Same as Phase 13 — real numbers require your hardware; nothing fabricated here.
- **Next experiment:** Phase 16 — SwinIR integration (third and final official-vs-reimplementation
  case).

---

## Phase 16 — SwinIR Integration
- **Same deviation as Phases 12/14, flagged again:** `git clone https://github.com/JingyunLiang/SwinIR`
  equally blocked; same copyright constraint on fetched source. `models/swinir/model.py` is a
  clean-room reimplementation from the paper (Liang et al., "SwinIR: Image Restoration Using Swin
  Transformer", ICCVW 2021), documented in `models/swinir/ATTRIBUTION.md`.
- **What changed:** Implemented `window_partition`/`window_reverse`, `WindowAttention` (with learned
  relative position bias), `SwinTransformerLayer` (alternating regular/shifted-window attention via
  cyclic shift + attention masking), `RSTB` (stack of layers + conv, residual-wrapped), and the full
  `SwinIR` class. Unlike NAFNet/Restormer, SwinIR's deep feature extraction body operates at a
  **single constant spatial resolution** throughout (no U-Net downsampling) — and unlike NAFNet/
  Restormer, SwinIR's own published design **already includes** a native PixelShuffle upsampling
  module for SR, so no bolted-on SR head adaptation was needed here (a genuine architectural
  difference from the other two, not just an implementation detail). Registered as `swinir`.
  `embed_dim` reduced from the paper's lightweight-SR default (180→60) for the 4GB RTX3050 constraint.
- **Why:** Same reasoning as Phases 12/14.
- **Config used:** `configs/model/swinir.yaml`, validated with PyYAML.
- **Results:** This is the most thoroughly verified of the three models, because its trickiest parts
  (window partitioning, cyclic shifting, attention masking) are pure reshape/permute/arithmetic
  operations with **no gradient or learned-weight dependency** — meaning they could be mirrored in
  plain numpy and checked exactly, not just symbolically traced for shape:
  - **`window_partition`/`window_reverse` round-trip verified exactly** (not just shape-matched) on
    two cases (square 16×16 and non-square 32×24 with multiple windows per dimension) — reconstructed
    tensor bit-for-bit equal to the original via `np.allclose`.
  - **Shift-attention-mask logic verified property-by-property**: correct mask shape, exactly 9
    distinct region labels (the standard 3×3 slice grid), self-attention diagonal always unmasked
    (0) for every window, the mask actually blocks *something* (contains both 0 and −100, i.e. isn't
    an accidental no-op), and blocked-status is symmetric.
  - **Relative-position-index verified**: correct value range `[0, (2·ws−1)²−1]`, and every
    self-position (diagonal) entry mapping to the same fixed index, as it must (relative offset
    (0,0) is a single specific position in the bias table).
  - **Overall shape flow verified symbolically**: constant resolution through the RSTB body, correct
    single-PixelShuffle-stage output for `scale_factor=2`, and correct **two-stage** cascade for
    `scale_factor=4` (channel/spatial arithmetic re-derived, not assumed).
  - `tests/test_swinir.py` (11 tests, including direct calls to `window_partition`/`window_reverse`
    with real torch tensors) written and syntax-checked, ready for real execution.
- **Lessons:** This phase's shape-adjacent logic (window ops, masking, position indexing) was
  fundamentally more verifiable than NAFNet/Restormer's, precisely because none of it depends on
  actual learned weights or autograd — a useful distinction to keep in mind for future architecture
  work in any environment like this one: reshape/index/mask logic can always be exactly verified in
  plain numpy; only the conv/attention/gradient computations themselves need real torch.
- **Next experiment:** Phase 17 — benchmark SwinIR (same real-hardware requirement, 1-line
  registration addition like Phase 15).

---

## Phase 17 — Benchmark SwinIR
- **What changed:** One line, same as Phase 15: added `import models.swinir.model` to
  `scripts/benchmark_model.py`. No other file touched. `python scripts/benchmark_model.py --model
  swinir` now works identically to `--model nafnet`/`--model restormer`.
- **Why worth noting:** third consecutive architecturally-distinct model (conv U-Net, transformer
  with channel attention + concat skips, transformer with windowed spatial attention + constant
  resolution) added to the benchmark pipeline with zero changes beyond a registration import —
  the strongest cumulative evidence yet for the Phase 4 architecture's actual interchangeability
  claim, not just for one model but across three genuinely different architecture families.
- **Results:** Same as Phases 13/15 — real numbers require your hardware.
- **Next experiment:** Phase 18 — final comparison (sanity CNN excluded per the Sanity Model Rule;
  NAFNet vs. Restormer vs. SwinIR side-by-side, using `write_comparison_markdown` from Phase 13,
  already proven correct).

---

## Phase 18 — Final Comparison
- **What changed:** Small justified extension to Phase 8's `Evaluator.save_report()` — now also
  writes `aggregate_metrics.json` (reusing Phase 13's generic `write_benchmark_json`, since it's
  identical logic: dump a flat dict to JSON) alongside the existing CSV/Markdown/images, specifically
  so this phase could merge quality metrics with benchmark metrics automatically. `tests/
  test_evaluator.py`'s existing end-to-end test updated to assert this new file exists.
  `scripts/final_comparison.py` — reads each requested model's latest `experiments/<model>/run_NNN/`
  directory, merges `benchmark_results.json` (Phase 13/15/17) with `eval_report/aggregate_metrics.json`
  (this phase's addition) into one row per model, and writes a single comparison table via Phase 13's
  already-proven `write_comparison_markdown`. **Explicitly refuses to run if `sanity_cnn` is in the
  requested model list** — a hard-coded check, not just a convention, per the Sanity Model Rule.
- **Why:** Reuses two already-verified pieces (Phase 13's JSON I/O, Phase 8's evaluator) rather than
  building new aggregation logic from scratch — and this phase's whole job (merge + tabulate
  existing results) is naturally torch-free, unlike Phases 12-17's actual model code.
- **Results:** Because this script only reads/merges JSON files, it required **no torch and could be
  fully executed and tested for real** — a genuine exception to the "needs your hardware" pattern of
  the last several phases. **5/5 tests executed for real** via subprocess (not just reviewed):
  merging benchmark + quality metrics into one table, correctly showing `N/A` for a model missing a
  field relative to another that has it (caught and fixed a bug in my own *test* here — my first
  attempt asserted `N/A` with only one model and one field present, which can never produce an `N/A`
  since there's nothing to be missing relative to; fixed by adding a second model with the gap),
  hard rejection of `sanity_cnn` with a clear message, a clear "no results found, run the other
  scripts first" message when nothing exists yet, and correctly picking the latest `run_NNN` when
  multiple exist for the same model.
- **Lessons:** This phase is a useful reminder that not everything from Phase 12 onward is
  torch-blocked — anything that's pure aggregation/reporting over already-produced files stays
  fully testable, and it's worth checking that explicitly rather than assuming "we're in the
  model-integration phases now, so nothing can run here."
- **Next experiment:** Phase 19 — documentation (README, research journal is already continuously
  maintained, judge-facing outputs: demo script, pitch outline, slide outline).

---

## Phase 19 — Documentation
- **What changed:**
  - Built the three CLI entry points the project was still missing: `scripts/train.py` (found
    already present and well-formed at the start of this phase — reviewed, verified it syntax-checks
    clean, and added one missing piece: a friendly `DatasetContractError` message instead of a raw
    traceback, matching the pattern used everywhere else), `scripts/evaluate.py` (checkpoint -> full
    Phase 8 evaluation report), `scripts/infer.py` (checkpoint -> Phase 9 `.npy` predictions on the
    GT-less split). All three follow the same shape as `sanity_check.py`/`benchmark_model.py`:
    trigger registrations, build via registries, catch dataset errors with an actionable message.
  - `scripts/demo.py` — a short, live-demo-ready script: loads a checkpoint, restores a handful of
    samples, prints per-sample PSNR, saves a before/after comparison grid (reusing Phase 8's
    `plot_comparison_grid`).
  - `docs/PITCH_OUTLINE.md` (3-minute pitch, timed by section) and `docs/PRESENTATION_OUTLINE.md`
    (10 slides) — both explicitly mark every number that must be filled in from real measurements
    with `[INSERT: ...]` placeholders and a closing reminder not to present placeholder numbers as
    measured results.
  - Full `README.md`, replacing the Phase 1 stub: quick-start command sequence (all 9 scripts in
    order), project structure, task-agnostic design table, and an upfront **⚠️ Status and important
    caveats** section stating plainly that NAFNet/Restormer/SwinIR are reimplementations and that no
    training has actually been run yet — not buried in `research_log.md` where a judge is less likely
    to look first.
- **Why:** Per the project spec's judge-friendly-outputs and final-delivery requirements. The
  train/evaluate/infer scripts specifically were deferred from Phase 9 (no real model existed yet)
  and are exactly what "documentation of commands" needs to reference — writing the docs without
  them would mean documenting commands that don't exist.
- **Results:** `train.py`/`evaluate.py`/`infer.py`/`demo.py` are torch-dependent and syntax-checked
  only (`py_compile`, all clean). **`train.py`'s one torch-free piece — `_load_yaml_or_name`, which
  lets `--loss` accept either a bare registry name or a path to a composite-loss YAML — was actually
  executed and verified**: bare name wrapping, YAML-path loading of a real composite config, and
  default-params pass-through all confirmed correct. New test file `tests/
  test_train_script_helpers.py` (3 tests) added for this.
- **Lessons:** Documenting "the commands" honestly required admitting, prominently and near the top
  of the README (not just in the research log), that they're unexecuted — burying that caveat would
  have made the documentation actively misleading for anyone who reads the README first.
- **Next experiment:** Phase 20 — final audit (verify every file exists, imports are consistent,
  dependencies listed correctly, then package the final zip).

---

## Phase 20 — Final Audit
- **What was checked:**
  1. **Full syntax sweep**: all 101 `.py` files in the project compile clean (`py_compile`).
  2. **Dependency audit**: parsed every file's AST to extract every third-party import actually used
     anywhere in the codebase, cross-referenced against `requirements.txt`/`environment.yml`. Found
     and fixed **two real gaps**: `scipy` (used in `metrics/ssim_metric.py`'s `convolve2d`) and
     `pyyaml` (used in `scripts/train.py`'s config loading) were both genuinely imported in shipped
     code but missing from the dependency files — added both.
  3. **Config validation**: all 24 YAML files under `configs/` parse correctly with PyYAML.
  4. **Structure check**: confirmed all 4 model directories present (`sanity_cnn`, `nafnet`,
     `restormer`, `swinir`), with `ATTRIBUTION.md` correctly present for the 3 reimplemented papers
     and correctly absent for `sanity_cnn` (not a reimplementation).
  5. **Import sweep**: all 24 torch-free framework/task/metric/util modules import cleanly.
  6. **Git history**: all 20 phases committed in order, clean working tree after this audit's fixes.
  7. **Torch-free test re-verification**: re-ran the import step for every test file without a
     `pytest.importorskip("torch")` guard — 4/9 imported clean standalone; the other 5 fail only on
     `import pytest` itself (pytest genuinely can't be installed in this sandbox, the same
     already-documented constraint from every phase since Phase 2 — not a new or different issue,
     and the underlying logic in all 5 was independently hand-verified earlier in this project
     without pytest).
- **What this audit found:** exactly 2 real, fixable issues (the missing `scipy`/`pyyaml` deps) —
  everything else checked out. Worth noting this is the first phase where a systematic,
  whole-repository sweep (rather than per-phase spot checks) was run, and it still only surfaced two
  small dependency-list gaps rather than any deeper structural problem — reasonable evidence that
  the per-phase verification discipline held up across all 19 previous phases.
- **Final delivery status:** the repository is complete per the project's Final Delivery checklist —
  source code, configs, scripts, tests, documentation, and requirements are all present. Training,
  inference, evaluation, and benchmarking all have real, reviewed, syntax-checked code paths, but (as
  documented honestly throughout, especially in the README's status section) have not been executed
  with real torch/GPU — that remains the one outstanding item, gated on your real environment, not on
  anything further this sandbox can do.
- **Final commands** (also in README.md):
  ```bash
  # Installation
  pip install -r requirements.txt

  # Dataset validation (run first, always)
  python inspect_dataset.py --root dataset

  # Sanity check (run second, before trusting anything else)
  python scripts/sanity_check.py --data-root dataset --epochs 2 --device cuda

  # Training
  python scripts/train.py --model nafnet --epochs 100 --device cuda --precision fp16 --grad-accum-steps 8 --use-ema

  # Evaluation
  python scripts/evaluate.py --model nafnet --checkpoint experiments/nafnet/run_001/best.pt --split val --device cuda

  # Inference
  python scripts/infer.py --model nafnet --checkpoint experiments/nafnet/run_001/best.pt --device cuda

  # Benchmark
  python scripts/benchmark_model.py --model nafnet --device cuda

  # Model comparison
  python scripts/final_comparison.py
  ```

---

## Phase 10 — Benchmark Framework

---

## Phase 9 — Inference Pipeline
