# Release 2.0.0 DQN Framework — Research

## Task Details

| Field | Value |
|---|---|
| Jira ID | N/A (internal task) |
| Title | Release DQN Framework v2.0.0 |
| Description | Prepare and publish the first formal GitHub Release for DQN Framework — version 2.0.0. Includes version bump, CHANGELOG translation to English, GitHub Release + git tag, and release automation workflow. |
| Priority | High |
| Reporter | Owner |
| Created | 2026-04-07 |
| Due Date | — |
| Labels | release, versioning, ci-cd |
| Estimated Effort | M |
| Solution Research Complexity | N/A |

## Business Impact

DQN Framework has been developed iteratively (v1.0.0 released 2026-03-14, v1.0.1 unreleased) but never had a formal GitHub Release with tag. Version 2.0.0 marks the first officially published release, establishing:

- A clear baseline for external users and contributors to reference.
- Proper SemVer history with a tagged, downloadable artifact on GitHub.
- Automated release process for all future versions.
- Quality signal — SonarCloud integration, 101 unit tests, CI pipeline.

**Breaking changes justifying major bump (1.x → 2.0):**
- CNN DQN architecture (`models/cnn_dqn_network.py`) with configurable Conv2d layers and Dueling support.
- `ALE/Pong-v5` environment preset requiring new dependencies (`gymnasium[atari]`, `ale-py`, `opencv-python`).
- SonarCloud quality control integration (new CI workflows, quality gate enforcement).
- Restructured training internals (`utils/training.py` extracted from `train.py`/`tuning_test.py`).

## Collected Information

### Knowledge Base and Task Management Tools

No Jira/Confluence context — this is an internally defined task based on project maturity assessment.

### Codebase

#### Current Version State

| Item | Current Value | Target Value |
|------|--------------|-------------|
| `version.py` | `__version__ = "1.0.1"` | `__version__ = "2.0.0"` |
| README badge | `version-1.0.1` | `version-2.0.0` |
| CHANGELOG header | `[1.0.1] - Unreleased` | `[2.0.0] - 2026-04-XX` (release date) |
| Git tags | None | `v2.0.0` |
| GitHub Releases | None | v2.0.0 release with notes |

#### CHANGELOG Status

- Language: Polish (headers: "Dodane", "Naprawione", "Usunięte", "Zmienione").
- Target: English (headers: "Added", "Fixed", "Removed", "Changed") — consistent with English README.
- Sections `[1.0.0]` and `[1.0.1] - Unreleased` both need translation.
- `[1.0.1] - Unreleased` content merges into `[2.0.0]`.

#### Feature Scope (2.0.0 = full current feature set)

| Feature | Module | Status |
|---------|--------|--------|
| Double DQN training | `agents/dqn_agent.py` | ✅ Complete |
| Dueling DQN (MLP) | `models/dqn_network.py` | ✅ Complete |
| **CNN DQN** (new in 2.0) | `models/cnn_dqn_network.py` | ✅ Complete |
| Uniform Replay Buffer | `memory/replay_buffer.py` | ✅ Complete |
| Prioritized Experience Replay | `memory/replay_buffer.py` | ✅ Complete |
| N-step Returns Buffer | `memory/replay_buffer.py` | ✅ Complete |
| CartPole-v1 preset | `config/config.py` | ✅ Tuned (83% success rate) |
| MountainCar-v0 preset | `config/config.py` | ✅ Tuned (67% success rate) |
| Acrobot-v1 preset | `config/config.py` | ✅ Complete |
| **ALE/Pong-v5 preset** (new) | `config/config.py` | ✅ Complete |
| Eval-based early stopping | `train.py` | ✅ Complete |
| Reward shaping | `utils/training.py` | ✅ Complete |
| TensorBoard logging | `train.py` | ✅ Complete |
| CSV metrics | `train.py` | ✅ Complete |
| Unit tests (101) | `tests/` | ✅ Complete |
| CI (lint + smoke + tests) | `.github/workflows/ci.yml` | ✅ Complete |
| SonarCloud analysis | `.github/workflows/sonar.yml` | ✅ Complete |
| Pre-trained models | `*.pth` | ✅ 7 models in repo |

#### CI/CD Status

| Workflow | Purpose | Status |
|----------|---------|--------|
| `ci.yml` | Lint (ruff) + unit tests (pytest) + compile + smoke tests | ✅ Active |
| `sonar.yml` | SonarCloud quality analysis with coverage | ✅ Active |
| `release.yml` | **GitHub Release on tag push** | ❌ Missing — needs creation |

#### Test Coverage

- **51.66% line coverage** (326/631 lines) — from `coverage.xml`.
- 7 test modules + conftest + helpers.
- Excluded from coverage: CLI entry points (`train.py`, `evaluate.py`, `play.py`), `cnn_dqn_network.py` (requires GPU/Atari ROM), `utils/analyze.py` (standalone pandas tool).

#### Python Packaging Status

| File | Status |
|------|--------|
| `pyproject.toml` | Minimal — only `[tool.pytest.ini_options]`. No `[project]`, no `[build-system]`. |
| `setup.py` / `setup.cfg` | Not present |
| `MANIFEST.in` | Not present |
| `requirements.txt` | Present — pinned versions with CUDA torch |

**Decision**: PyPI publishing is out of scope for this release. The project is distributed via GitHub only.

#### Community Health Files

| File | Status |
|------|--------|
| `README.md` | ✅ Complete (English, badges, comprehensive) |
| `CHANGELOG.md` | ⚠️ Needs translation to English + restructuring |
| `LICENSE` | ✅ MIT |
| `CONTRIBUTING.md` | ❌ Not present |
| `CODE_OF_CONDUCT.md` | ❌ Not present |
| `SECURITY.md` | ❌ Not present |

**Decision**: CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md are optional for v2.0.0 — can be added in future versions.

### Related Links

- Repository: https://github.com/Finfinder/DQN_Framework
- SonarCloud: https://sonarcloud.io/summary/overall?id=Finfinder_DQN_Framework
- Gymnasium docs: https://gymnasium.farama.org/
- PyTorch docs: https://pytorch.org/docs/stable/
- Keep a Changelog: https://keepachangelog.com/en/1.1.0/
- Semantic Versioning: https://semver.org/

### Solution Research

Not performed — requirements are unambiguous, technology is already chosen. The task is a release engineering effort, not a technology selection.

### Related Charts and Diagrams

```
Current State                     Target State
─────────────                     ────────────
version.py: 1.0.1        →       version.py: 2.0.0
CHANGELOG: Polish, 1.0.1 →       CHANGELOG: English, 2.0.0
Git tags: none            →       Git tag: v2.0.0
GitHub Releases: none     →       GitHub Release: v2.0.0
Release workflow: none    →       .github/workflows/release.yml
README badge: 1.0.1       →       README badge: 2.0.0
```

## Current Implementation Status

### Existing Components

- `version.py` - `version.py` - Requires modification (1.0.1 → 2.0.0)
- `CHANGELOG.md` - `CHANGELOG.md` - Requires modification (translate to English, merge 1.0.1→2.0.0, set release date)
- `README.md` - `README.md` - Requires modification (badge version update)
- `ci.yml` - `.github/workflows/ci.yml` - Reuse as-is
- `sonar.yml` - `.github/workflows/sonar.yml` - Reuse as-is
- `pyproject.toml` - `pyproject.toml` - Reuse as-is (no packaging changes)

### Key Files and Directories

- `version.py` — single source of truth for version number
- `CHANGELOG.md` — change history, needs English translation and [1.0.1]→[2.0.0] merge
- `README.md` — version badge needs update
- `.github/workflows/` — CI workflows; release.yml needs to be created
- `config/config.py` — hyperparameter config with 4 env presets (no changes needed)
- `tests/` — 7 test modules, 101 tests (no changes needed)
- `*.pth` — 7 pre-trained model files (no changes needed)

## Gap Analysis

### Question 1
#### Should the CHANGELOG preserve the 1.0.0 section as a separate historical entry, or merge everything into a single 2.0.0 entry?
**Answer**: Preserve [1.0.0] as historical baseline, merge [1.0.1 - Unreleased] into [2.0.0]. The CHANGELOG should have two sections: [2.0.0] (with release date) containing all changes since 1.0.0, and [1.0.0] - 2026-03-14 as the original baseline. Both translated to English.

### Question 2
#### What is the release channel?
**Answer**: GitHub Release + git tag (v2.0.0). No PyPI publishing.

### Question 3
#### Should the release workflow trigger on tag push (e.g. v*) or manual dispatch?
**Answer**: Tag push trigger (on push tags: `v*`). This is the standard convention for GitHub Release workflows. The workflow should: create a GitHub Release from the tag, auto-generate release notes from CHANGELOG or commits, and attach relevant artifacts if needed.

### Question 4
#### Which pre-trained models should be attached to the GitHub Release as assets?
**Answer**: To be decided during implementation. Candidates: all 7 `.pth` files (total ~22MB estimated). Alternative: none — users can train their own models. Recommend including at least `dqn_cartpole_dueling.pth` as a quick-start artifact.

### Question 5
#### Should the release require passing CI + SonarCloud quality gate before publishing?
**Answer**: Yes — the release workflow should depend on CI passing. SonarCloud quality gate is informational but not blocking (it runs on a separate workflow).

## Deliverables Summary

| # | Deliverable | Description |
|---|-------------|-------------|
| 1 | Version bump | `version.py`: 1.0.1 → 2.0.0 |
| 2 | README badge update | Version badge: 1.0.1 → 2.0.0 |
| 3 | CHANGELOG translation + restructure | Polish → English, merge [1.0.1] into [2.0.0], set release date |
| 4 | Release workflow | `.github/workflows/release.yml` — trigger on `v*` tag push, create GitHub Release |
| 5 | Git tag + GitHub Release | Tag `v2.0.0`, GitHub Release with notes |
| 6 | Verification | CI green, all tests pass, CHANGELOG correctly formatted |
