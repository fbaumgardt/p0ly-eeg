# Deferred CI and automation

Revisit when the repo no longer has legacy / WIP code paths (see checklist below).

## Prerequisites (done = safe to enable stricter automation)

- [ ] Metadata migration complete: `igt`, `intwm`, `simonfb` on `ExperimentSpec` + generic parser
- [ ] Remove or archive `src/p0ly_eeg/metadata/_legacy.py`
- [ ] `dump_*.py` modules implemented and covered (or removed)
- [ ] mypy runs on full `src/` and `tests/` with no path exclusions
- [ ] Ruff clean on full tree (already true today)

## Deferred automations

### Dependabot

- Add `.github/dependabot.yml` (uv + `github-actions`)
- Review MNE-related bumps manually before merge

### `ruff format --check` in CI

- CI step: `uv run ruff format --check src tests`
- One-time: `uv run ruff format src tests` if the first CI run fails
- Optional: add ruff format hook to pre-commit (overlaps with CI gate)

### Coverage `fail_under` gate

- Set `[tool.coverage.report] fail_under` in `pyproject.toml` to a realistic baseline, then raise
- Ensure tests cover `dump_*` and metadata paths you care about

### Codecov (optional)

- Upload coverage in CI; PR comments and history on codecov.io

### Branch protection (GitHub UI)

- Require CI workflow to pass before merge to `master`

### Release workflow (optional)

- Tag push → `uv build` → PyPI when publishing `p0ly-eeg` publicly

## mypy expansion (after legacy removal)

1. Replace scoped `[tool.mypy] files = [...]` with `files = ["src", "tests"]` or package mode
2. Remove `ignore_missing_imports` global flag; keep only `[[tool.mypy.overrides]]` for `mne` / `mne_icalabel` if needed
3. Enable `disallow_untyped_defs = true` for `p0ly_eeg.metadata` incrementally

## Reference

Initial CI plan: scoped mypy, pre-commit, no Dependabot/format gate/coverage gate. Branch: `master`.
