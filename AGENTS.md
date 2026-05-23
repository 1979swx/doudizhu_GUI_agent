# Repository Guidelines

## Project Structure & Module Organization
Core library code lives in `verl/`, with training, workers, utilities, and model integrations grouped by domain. Agent-specific runtime pieces live in `agent_system/` (environments, memory, rollout, reward management). Use `examples/` and `recipe/` for runnable training scripts and experiment configs, `tests/` for automated coverage, `docs/` for Sphinx documentation, and `scripts/` plus `docker/` for maintenance and container workflows.

## Build, Test, and Development Commands
The main training environment, gym-cards, and sokoban environments are installed in the `verl-agent-bw-exp` conda environment. Use this environment to run Python scripts (activate it with `conda activate verl-agent-bw-exp`). Install the package for local development with `pip install -e .`; include test tooling with `pip install -e .[test]`. Run linting and formatting through `pre-commit install` once, then `pre-commit run --all-files` before opening a PR. Use targeted pytest runs that match CI, for example `pytest -s -x tests/sanity`, `pytest -s -x tests/utils/cpu_tests`, or `pytest -s tests/models/test_transformer.py`. Build docs from `docs/` with `make html`.

## Coding Style & Naming Conventions
Python is the primary language. Follow 4-space indentation, `snake_case` for modules/functions/scripts, `PascalCase` for classes, and keep public package names under `verl` or `agent_system` aligned with existing directory structure. Formatting and import ordering are enforced by `ruff` and `ruff-format` through pre-commit; the repository currently allows long lines, but keep new code readable and avoid unnecessary line growth. CI also checks naming consistency: use `verl`, not `veRL`, in code and docs.

## Testing Guidelines
Tests use `pytest` and are organized by subsystem and execution environment, such as `tests/sanity`, `tests/models`, `tests/utils/cpu_tests`, `tests/ray_gpu`, and `tests/e2e`. Name files `test_*.py` and place them near the feature area they validate. Prefer the smallest relevant test target locally, then expand to broader suites when touching shared trainer, model, or distributed code.

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects like `Update README`, `Add recipe/hgpo`, and scoped fixes such as `fix(ray): ensure Ray < 2.50.0`. Keep commits focused and descriptive; add a scope when it clarifies impact. PRs should explain the affected module, link related issues, list the commands you ran, and include CI workflow updates when adding new features or test surfaces.
