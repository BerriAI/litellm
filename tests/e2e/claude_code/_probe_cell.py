"""Shared body for the HTTP-probe compat cells (`count_tokens`, `tool_search`).

Every probe row does the same three things: fire one request per Claude
tier, shape-check each response, and report one `compat_result` row per
tier so the matrix builder's "all tiers must pass" aggregator still sees
three rows for the cell.

The tiers are independent HTTP round trips, so they run concurrently
here for the same reason the CLI rows fan out in
`run_claude_models_parallel`: a cell's wall time should be one probe,
not the sum of three. That matters more than the raw request latency
suggests, because every probe first blocks on the cross-process
per-provider token bucket (`rate_limiter.py`) -- a serial cell paid
three of those waits back to back while holding a pytest worker idle.

Report order follows the declared model order, not completion order, so
the results artifact is deterministic no matter which tier answers
first.

The leading underscore in the filename is what keeps pytest from
collecting this module as a test file.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence

import pytest
from pydantic import BaseModel

from e2e_http import NetworkError, Result


def probe_models_parallel[R: BaseModel](
    *,
    models: Sequence[str],
    probe: Callable[[str], Result[R]],
) -> dict[str, Result[R]]:
    """Run `probe` once per model concurrently and return outcomes keyed by model.

    Errors stay values: a probe that raises (the rate limiter does file
    I/O, and `infer_provider` can reject an edge-case model string)
    becomes a `NetworkError` entry rather than an exception that would
    discard the other tiers' outcomes.
    """
    if not models:
        raise ValueError("models must be a non-empty sequence")

    def _one(model: str) -> tuple[str, Result[R]]:
        try:
            return model, probe(model)
        except Exception as exc:
            return model, NetworkError(
                message=f"unexpected error probing {model!r}: {type(exc).__name__}: {exc}"
            )

    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        return dict(pool.map(_one, models))


def _failure_message(model: str, probe_name: str, error: str) -> str:
    return f"[{model}] {probe_name} probe failed: {error}"


def run_probe_cell[R: BaseModel](
    *,
    compat_result,
    models: Sequence[str],
    probe: Callable[[str], Result[R]],
    check_shape: Callable[[Result[R]], str | None],
    probe_name: str,
) -> None:
    """Run the shared HTTP-probe cell body across every tier in `models`.

    `probe` and `check_shape` are the per-feature halves: the caller
    binds the proxy client and api key into `probe`, and passes the
    matching `assert_*_shape` as `check_shape`. Both are plain callables
    so a test can inject fakes without a live proxy.
    """
    outcomes = probe_models_parallel(models=models, probe=probe)
    checked = tuple((model, check_shape(outcomes[model])) for model in models)

    for model, error in checked:
        compat_result.add(
            {"status": "pass"}
            if error is None
            else {
                "status": "fail",
                "error": _failure_message(model, probe_name, error),
            }
        )

    failures = tuple(
        _failure_message(model, probe_name, error)
        for model, error in checked
        if error is not None
    )
    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
