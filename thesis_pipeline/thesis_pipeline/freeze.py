"""Freeze a fitted feature-set + classifier to disk, alongside a spec file
that records everything needed to reproduce or audit the run: prompt hash,
gate thresholds, feature lists, CV protocol, and results.

assert_consistent() re-derives the same checks on reload. Call it right
after freezing, and again at the very start of the one-shot test-evaluation
notebook, before the sealed test set is touched — it is the last line of
defence against evaluating the wrong artifact, or an artifact frozen under
a prompt that has since changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib


def write_frozen_spec(spec_path: Path, spec: dict) -> None:
    spec_path.write_text(json.dumps(spec, indent=2, default=str))


def freeze_pipeline(pipeline, artifact_path: Path, *, features=None,
                    feature_source_run: str | None = None) -> None:
    """Dump a fitted pipeline to disk. When `features` is given, the artifact
    is a dict {"pipeline", "features", "feature_source_run"} so the exact
    input-column list (names and order) travels with the model and can be
    re-asserted on reload; otherwise the bare pipeline is dumped."""
    if features is None:
        joblib.dump(pipeline, artifact_path)
    else:
        joblib.dump(
            {"pipeline": pipeline, "features": list(features),
             "feature_source_run": feature_source_run},
            artifact_path,
        )


def load_frozen(spec_path: Path, artifact_path: Path):
    """Returns (spec, pipeline). Accepts either artifact shape written by
    freeze_pipeline() — a bare pipeline or the {"pipeline", ...} wrapper."""
    spec = json.loads(spec_path.read_text())
    art = joblib.load(artifact_path)
    pipeline = art["pipeline"] if isinstance(art, dict) and "pipeline" in art else art
    return spec, pipeline


def assert_consistent(spec_path: Path, artifact_path: Path, prompt_sha: str,
                       expected_prompt_sha: str) -> dict:
    """Reload spec + pipeline from disk and assert they agree with each
    other and with the prompt hash currently in force. Raises on any
    mismatch rather than silently proceeding with a stale or inconsistent
    artifact. Returns the loaded spec on success."""
    spec = json.loads(spec_path.read_text())
    art = joblib.load(artifact_path)
    pipeline = art["pipeline"] if isinstance(art, dict) and "pipeline" in art else art

    expected_features = spec["feature_sets"]["frozen_primary"]
    n_in = getattr(pipeline, "n_features_in_", None)
    if n_in is not None:
        assert n_in == len(expected_features), (
            f"artifact expects {n_in} features, spec's frozen_primary lists "
            f"{len(expected_features)} — pipeline and spec disagree"
        )
    if isinstance(art, dict) and art.get("features") is not None:
        assert list(art["features"]) == list(expected_features), (
            "artifact's stored feature list does not match spec's frozen_primary "
            "(names or order differ) — pipeline and spec disagree"
        )

    spec_sha = spec.get("prompt_sha256_16", expected_prompt_sha)
    assert prompt_sha == expected_prompt_sha == spec_sha, (
        f"prompt hash mismatch: running={prompt_sha} expected={expected_prompt_sha} "
        f"spec={spec_sha} — do not evaluate the test set with a pipeline frozen "
        "under a different prompt"
    )

    assert spec.get("test_set_used") is False, (
        "frozen_spec.json already claims the test set was used — stop and "
        "investigate before running any further evaluation"
    )
    return spec
