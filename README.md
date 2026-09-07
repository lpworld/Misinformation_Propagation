# Reproducibility Package: The Engagement Fungibility Mechanism

This package accompanies the paper *Why Does Misinformation Propagate Faster? An Algorithmic Perspective on X*. It contains the full analysis code, the experiment configurations, and a 30,000-tweet data sample on which the entire workflow can be executed end to end.

## Package layout

```
ranker/         Reconstructed ranker: MaskNet architecture, features, training, score aggregation
simulation/     Cascade simulator: user pool, seed content, calibration, event model, ablation harness
analysis/       Pipeline entry points (run_*.py) and shared analysis modules
configs/        Experiment configurations (demo-scale and full-scale)
tests/          Unit tests for the core modules
data/
  labels/       Domain-credibility lists and the emotion lexicon
  processed/    part_sample.parquet, the bundled 30,000-tweet sample
reports/        Output reports from demonstration runs are written here
```

## Environment

Python 3.10 to 3.12. From the package root:

```
pip install -e .
```

This installs pandas, numpy, pyarrow, scipy, scikit-learn, torch, and the other dependencies. CPU-only torch is sufficient for the demonstration runs. To run the unit tests, additionally `pip install pytest`.

## Data

The bundled sample (`data/processed/part_sample.parquet`) contains 30,000 tweets drawn from part 1 of the USC X 2024 U.S. Election Dataset, stratified to include 2,000 low-credibility and 2,000 high-credibility labeled tweets so that the label-aware stage has material to work with. The full dataset is public at `https://github.com/sinking8/usc-x-24-us-election` (CC BY-NC-SA 4.0). To run at full scale, download the raw chunks, place them under `data/raw/usc-x-24/`, and run the phase 1 parser (`analysis/run_phase1.py`) to build the cached parquet the other stages consume.

Labeling resources in `data/labels/`:

- `iffy_plus.csv`, the Iffy+ list of low-credibility news domains (data under CC BY-SA).
- `mainstream_domains.txt`, a hand-curated list of high-credibility outlets based on Media Bias/Fact Check factual-reporting categories.
- `nrc_emotion_lexicon.txt`, the NRC emotion lexicon (free for research use with attribution).

## Demonstration workflow

The pipeline has two stages, kept in separate scripts by design. Stage 1 is label-free validation of the simulator, and Stage 2 is the label-aware hypothesis test. Run both from the package root.

**Step 1, train and validate (Stage 1).** Trains a small ranker on the sample, simulates 100 cascades for 1,000 users under the additive scoring regime, and writes the Stage 1 validation metrics:

```
python -m analysis.run_phase2 --config configs/experiment_demo_stage1.yaml
```

Outputs land in `data/processed/demo_stage1/` with a report at `reports/demo_stage1_report.md`. This step also saves the trained ranker that Step 2 loads.

**Step 2, hypothesis test (Stage 2).** Applies the domain-credibility labels, simulates all scoring regimes with label-stratified seeds at reduced scale, and reports the low-minus-high credibility gaps and the cross-regime contrasts that instantiate the paper's propositions:

```
python -m analysis.run_phase4 --config configs/experiment_demo_stage2.yaml
```

Outputs land in `data/processed/demo_stage2/` with a report at `reports/demo_stage2_report.md`.

**Step 3, unit tests.** The tests exercise the loader, labeling, scoring regimes, cascade simulator, and metric computations on synthetic inputs:

```
pytest tests/
```

## Full-scale pipeline

The `analysis/` directory contains every script used in the paper, in rough execution order: `run_phase1.py` (parse raw chunks), `run_phase2.py` (end-to-end loop), `run_phase3.py` and `run_phase3_sim.py` (calibration and Stage 1 validation), `run_phase4.py` (Stage 2 hypothesis test), `run_phase5.py` and `run_phase5_extra.py` (robustness checks), and the remaining `run_*.py` scripts for the mechanism decomposition, placebo and falsification checks, weight-space search, operating curve, and solution validation. The matching full-scale configurations are the `configs/experiment_phase*.yaml` files. These scripts expect the full corpus and, at full scale, benefit from a CUDA-capable GPU, though none strictly requires one.

The claim-level label validation (`analysis/run_claim_label_validation.py`) calls a commercial large-language-model API and reads its credentials from the `ANTHROPIC_API_KEY` environment variable, with the model identifier supplied through `LLM_MODEL_ID`.

## Notes

- All entry points set random seeds explicitly (default 1337), and the demonstration runs are deterministic given the bundled sample.
- Paths inside the scripts resolve relative to the package root, so run all commands from this directory.
