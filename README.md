# Drift-Aware Online Recommendation

The project simulates streaming user-item-category interactions, evaluates online
recommendation algorithms with prequential evaluation, compares user-level drift
detection strategies, and studies drift-aware reranking based on recent category
transition patterns.

## Core Components

- Synthetic streaming data generator with sudden and gradual user preference drift
  over a MIND-shaped catalog/profile stream.
- Online matrix factorization recommender trained with stochastic gradient updates.
- Time-decay, cumulative-popularity, and recent-popularity baselines.
- Time-decay decay-rate sweeps and reranking ablations.
- Pure MF and category-aware MF baselines.
- User-level KL-divergence and Jensen-Shannon divergence drift detectors.
- Recent pattern extraction with first-order category transitions.
- Drift-aware reranking module.
- Online user clustering that shares short-lived drift-adaptation rules among
  users with similar recent category behavior.
- Prequential evaluation with quality, phase-specific, and reactivity metrics.
- CSV experiment outputs and plots.

## Quick Start

From the repository root, use a Python virtual environment. The commands below
were verified with Python 3.13.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS and Linux, activate the environment with `source .venv/bin/activate`.

Download the **MIND-small Training Set** from the [official MIND site](https://msnews.github.io/#download)
after accepting Microsoft's terms. Extract `news.tsv` and `behaviors.tsv` into
`MINDsmall_train/` at the repository root. Both ready-to-run configurations
use these files. If you place the data elsewhere, set `data.mind_path` in the
synthetic config or `data.path` in the real-data config.

```text
MINDsmall_train/
├── behaviors.tsv
└── news.tsv
```

```powershell
python -m run_experiment.run_single_prequential_experiment --config config_synthetic_mind_shaped.yaml
python tool/generate_synthetic_plots.py --output-dir experiment-results/synthetic/latest
```

Raw metrics and figures are written to `experiment-results/`.

## MIND-Shaped Synthetic Runs

The default `config_synthetic_mind_shaped.yaml` keeps labelled synthetic drift while shaping the
synthetic stream from the local `MINDsmall_train` files. The generator reads
`news.tsv` for real MIND category labels and item/category proportions, then
uses `behaviors.tsv` histories and clicked impressions to sample user profile
priors and item popularity. This gives the controlled synthetic benchmark known
drift points while making its categories, item catalog, and candidate ranking
sets easier to compare with the real MIND run.

Useful knobs in the `data` block:

- `mind_item_limit` samples a compact MIND-like item catalog for faster runs;
  set it to `0` to use every MIND news item.
- `mind_profile_events` controls how many behavior rows estimate user/category
  priors; set it to `0` to use all rows.
- `candidate_set_size` creates fixed-size synthetic impression sets containing
  the clicked item, closer to MIND clicked-impression evaluation.

## MINDsmall Real-Data Runs

MINDsmall is handled as a real behavior-shift dataset, not as a labelled drift
benchmark. The loader uses `news.tsv` for item/category mappings and
`behaviors.tsv` for chronological clicked-impression events. For KL and JSD
detection, the category distribution in each row's `history` column is used as
the user's reference baseline, then compared with recent clicked labels observed
online. These history-divergence detectors are behavior-shift triggers rather
than supervised drift detectors.

```powershell
python -m run_experiment.run_single_prequential_experiment --config config_real_mind_behavior_shift.yaml
python tool/generate_real_mind_plots.py --output-dir experiment-results/mind/latest
```

Outputs are written to `experiment-results/mind/`. The MIND run also writes
`behavior_shifts.csv`, which contains interpretable KL/JSD activations: history
top label, recent top label, divergence score, and the temporary boost rule.

## Recency Ablation

The default experiment includes controls for whether the KL-reranker gain comes
from learned detection or from stronger recent-preference bias alone:

- `time_decay_mf` plus additional `time_decay_mf_decay_*` rates from
  `time_decay.decay_rates`;
- `global_popularity`, which keeps cumulative popularity without a recency
  window;
- `mf_rerank_always`, which applies the same reranker throughout the stream;
- `mf_rerank_always_preference_only` and
  `mf_rerank_always_transition_only`, which isolate the two reranking signals;
- `mf_rerank_oracle`, which activates reranking at the hidden synthetic drift;
- `drift_aware_kl`, which activates reranking only from KL detections.
- `drift_aware_kl_preference_only` and
  `drift_aware_kl_transition_only`, which isolate the two signals under learned
  KL activation.
- `drift_aware_jsd`, which uses the same categorical-window structure as KL but
  scores shifts with Jensen-Shannon divergence.
- `drift_aware_kl_clustered`, which stores each KL-triggered recent-pattern
  rule in the user's online behavioral cluster and temporarily applies it to
  other users assigned to that cluster.

## Online User Clustering

The optional `clustering` block in the YAML configs enables a scalable sharing
layer for drift-aware adaptation. It clusters users online from the same recent
category preference and transition features used by the reranker, then stores
cluster-level rules with a fixed TTL when a user-level drift detector fires.
Each cluster keeps a small bank of active rules: a new rule is merged only when
it is similar enough to an existing rule, otherwise it remains separate and users
match against the most compatible active rule. The implementation is
dataset-agnostic because it only requires compact label IDs from the stream, so
it works for synthetic and MIND categories.

## Tuning vs Final Test

Use `--phase both` to select a threshold on the tuning runs and pass it directly
to the final runs. When running the phases separately, check the tuning summary
and enter the selected threshold in the YAML before running `--phase test`.
Test-only runs read the threshold from the YAML.

For controlled synthetic drift claims:

```powershell
python -m run_experiment.run_synthetic_detector_tuning_and_test --config config_synthetic_mind_shaped.yaml --detector kl --phase both
```

For real MIND behavior-shift claims:

```powershell
python -m run_experiment.run_real_mind_behavior_shift_tuning_and_test --config config_real_mind_behavior_shift.yaml --detector jsd --phase both
```

The synthetic protocol defaults to tuning seeds `1,2,3` on sudden-strong and
gradual-strong streams, then final-test seeds `10,11,12,13,14` on sudden-strong,
sudden-weak, gradual-strong, noisy-sudden, and stable no-drift streams. MIND
runs use different model-initialization seeds on the same chronological log.
This measures sensitivity to initialization on a fixed replay. A held-out
real-data evaluation would require a separate data split. MIND metrics describe
the ranking of logged impressions and behavior-shift activations.

Use a fresh `--output-root` when changing the data or experiment code. The
runners reuse completed runs with matching saved YAML configurations.

Synthetic protocol outputs are written under `experiment-results/synthetic/`.
MIND protocol outputs are written under `experiment-results/mind/`.

## Main Outputs

- `experiment-results/synthetic/latest/summary.csv`: final recommendation-quality table.
- `experiment-results/synthetic/latest/phase_summary.csv`: recommendation quality split into
  pre-drift, transition, and post-drift phases.
- `experiment-results/synthetic/latest/detection_summary.csv`: KL and JSD detection-delay summary.
- `experiment-results/synthetic/latest/recovery.csv`: recovery-time table.
- `experiment-results/synthetic/latest/plots/`: experiment figures.
