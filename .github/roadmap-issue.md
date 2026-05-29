# Project Status / Roadmap

This issue tracks the current milestone split, near-term priorities, and longer-range backlog for DQN Framework.

## Current focus

- Finish repository health and release automation tasks before the next release cut.
- Tighten quality gates so CI and Sonar enforce the same expectations for coverage and workflow hygiene.
- Deepen CNN and wrapper test coverage before adding larger architecture changes.

## Milestones

- **3.0.x Repository Health & Release Automation**
  - Complete community health files and cross-links.
  - Generate release notes from the matching `CHANGELOG.md` section.
- **3.1.x Quality Gates & Workflow Hardening**
  - Unify repo-local script policy across CI, Sonar, and ignore rules.
  - Align `tuning_test.py` with the selected CI and coverage policy.
  - Enforce coverage thresholds and harden workflow dependencies with `actionlint` and refreshed action pins.
- **3.2.x Test Depth for CNN & Wrapper Paths**
  - Expand Atari and CNN preprocessing wrapper coverage.
  - Go beyond forward-pass smoke tests with deterministic initialization, gradient-flow, and backward-pass checks.
- **3.3.x Benchmarking & Experiment Traceability**
  - Add SumTree-backed prioritized replay sampling.
  - Benchmark CNN inference on CPU and GPU with reproducible inputs.
  - Persist experiment metadata manifests next to models and metrics.

## Priorities

1. Repository health files and release-note automation.
2. Coverage gates, workflow linting, and Sonar action maintenance.
3. CNN and wrapper test depth for the current `mlp`/`cnn` implementations.
4. Benchmarking and experiment traceability for reproducible comparisons.

## Future work

- **4.0.x Network Architecture Extensions**
  - Add a Transformer-based Q-network backend as a new `network_type` after the current testing and benchmarking baseline is in place.

## Known limitations

- `network_type` currently supports only `mlp` and `cnn`.
- The framework targets discrete-action environments only.
- Experiment metadata is still fragmented across models, metrics, and logs.
