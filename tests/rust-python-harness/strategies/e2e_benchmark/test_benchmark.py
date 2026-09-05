from __future__ import annotations

import asyncio
import base64
from concurrent.futures import Future
from pathlib import Path
from time import sleep
from typing import Final

import httpx
import psutil
import pytest
from pydantic import ValidationError

from litellm.llms.base_llm.ocr.transformation import OCRResponse

from ...cli.catalog import load_catalog
from ...shared.reporting.models import RunStatus
from .execution import execute_phase, sample_rss
from .models import Invocation, Options
from .provider import PYTHON_SENTINEL, provider_process
from .reporting import percentile, render_measurements
from .runner import Report, parse_options, run_benchmark_cases
from .worker import measure_async, measure_sync
from .workloads import JSON_OBJECT, JSON_PAGES, ocr_workload, padded_pdf

REPO_ROOT: Final = Path(__file__).resolve().parents[4]


def invocation(*, phase: str = "timing") -> Invocation:
    return Invocation.model_validate(
        {
            "model": "mistral/mistral-ocr-latest",
            "document_url": "data:application/pdf;base64,AA==",
            "route": "ocr",
            "provider_url": "http://127.0.0.1:1",
            "iterations": 3,
            "warmup": 1,
            "phase": phase,
        }
    )


def test_request_and_response_sizes_vary_independently() -> None:
    small: Final = ocr_workload("small")
    request: Final = ocr_workload("request_large")
    response: Final = ocr_workload("response_large")
    small_body: Final = JSON_OBJECT.validate_json(small.response)
    request_body: Final = JSON_OBJECT.validate_json(request.response)
    response_body: Final = JSON_OBJECT.validate_json(response.response)

    assert small.document_bytes == 32 * 1024
    assert request.document_bytes == 2 * 1024 * 1024
    assert base64.b64decode(request.document_url.split(",", 1)[1]).startswith(b"%PDF-")
    assert small_body["pages"] == request_body["pages"]
    assert response.document_url == small.document_url
    assert len(response.response) > 100 * len(small.response)
    assert tuple(page["index"] for page in JSON_PAGES.validate_python(response_body["pages"])) == tuple(range(128))
    assert JSON_OBJECT.validate_python(response_body["usage_info"])["pages_processed"] == 128
    assert small.fixture_sha256 == request.fixture_sha256 == response.fixture_sha256


def test_pdf_padding_preserves_existing_offsets_and_exact_size() -> None:
    seed: Final = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\nstartxref\n9\n%%EOF\n"
    padded: Final = padded_pdf(seed, 1024)
    assert len(padded) == 1024
    assert padded.startswith(seed.split(b"startxref")[0])
    assert padded.endswith(b"\nstartxref\n9\n%%EOF\n")


@pytest.mark.parametrize("arguments", (("--iterations=0",), ("--warmup=0",), ("--route=chat",), ("--profile=unknown",)))
def test_invalid_benchmark_options_fail_before_running(arguments: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        parse_options(arguments)


def test_unknown_options_are_not_silently_ignored() -> None:
    with pytest.raises(ValueError, match="unknown benchmark arguments"):
        parse_options(("--concurrency=8",))


def test_percentiles_use_nearest_rank_without_dropping_the_tail() -> None:
    assert percentile(tuple(range(1, 101)), 0.95) == 95
    assert percentile((4, 1, 3, 2), 0.99) == 4
    with pytest.raises(ValueError, match="requires samples"):
        percentile((), 0.95)


def test_sync_timing_excludes_waiting_from_cpu_time() -> None:
    def call() -> OCRResponse:
        sleep(0.02)
        return OCRResponse(model="benchmark", pages=[])

    result: Final = measure_sync(call, invocation())
    assert len(result.latency_ms) == 3
    assert min(result.latency_ms) >= 20
    assert result.elapsed_ms >= sum(result.latency_ms)
    assert result.cpu_ms < result.elapsed_ms / 2


def test_async_timing_awaits_the_sdk_operation() -> None:
    async def call() -> OCRResponse:
        await asyncio.sleep(0.02)
        return OCRResponse(model="benchmark", pages=[])

    result: Final = asyncio.run(measure_async(call, invocation()))
    assert len(result.latency_ms) == 3
    assert min(result.latency_ms) >= 20
    assert result.cpu_ms < result.elapsed_ms / 2


def test_memory_pass_does_not_accumulate_latency_samples() -> None:
    result: Final = measure_sync(lambda: OCRResponse(model="benchmark", pages=[]), invocation(phase="memory"))
    assert result.latency_ms == ()


def test_memory_monitor_has_a_deadline() -> None:
    pending: Final[Future[str]] = Future()
    with pytest.raises(TimeoutError, match="memory measurement timed out"):
        tuple(sample_rss(psutil.Process(), pending, interval=0.001, timeout=0.01))


def test_replay_rejects_python_fallback_during_rust_measurement() -> None:
    workload: Final = ocr_workload("small")
    with provider_process(workload.response, "rust") as url:
        response: Final = httpx.post(url + "/v1/ocr", content=b"{}", headers={"user-agent": PYTHON_SENTINEL})
    assert response.status_code == 409
    assert "backend mismatch" in response.text


def test_worker_errors_are_reported_instead_of_counted_as_fast_calls() -> None:
    workload: Final = ocr_workload("small")
    with provider_process(workload.response, "rust") as url:
        request: Final = invocation().model_copy(update={"provider_url": url, "document_url": workload.document_url})
        with pytest.raises(RuntimeError, match="backend mismatch"):
            execute_phase(request, "python", Options(iterations=3, warmup=1), REPO_ROOT)


def test_strategy_runs_both_backends_and_exports_measurements(tmp_path: Path) -> None:
    strategy: Final = next(strategy for strategy in load_catalog() if strategy.id == "e2e_benchmark")
    case: Final = next(case for case in strategy.cases if case.sdk_function == "ocr")
    output: Final = tmp_path / "measurements.json"
    exit_code, run = run_benchmark_cases(
        (case,),
        REPO_ROOT,
        lambda _: None,
        ("--profile=small", "--route=aocr", "--iterations=3", "--warmup=1", "--repeats=1", f"--output={output}"),
    )
    assert exit_code == 0, run.failures
    assert run.results[case.key].status is RunStatus.PASSED
    report: Final = Report.model_validate_json(output.read_bytes())
    assert {value.backend for value in report.measurements} == {"python", "rust"}
    assert len({value.ready.response_digest for value in report.measurements}) == 1
    for value in report.measurements:
        assert len(value.timing.latency_ms) == 3
        assert value.timing.cpu_ms > 0
        assert min(value.timing.latency_ms) > 0
        assert value.memory.baseline_rss_bytes > 0
        assert value.memory.sampled_peak_rss_bytes >= value.memory.baseline_rss_bytes
        assert value.memory.sampled_peak_rss_bytes >= value.memory.retained_rss_bytes > 0
        assert (value.ready.native_sha256 is not None) == (value.backend == "rust")
    table: Final = render_measurements(report.measurements)
    assert "aocr/small | python" in table
    assert "aocr/small | rust" in table
    assert "CPU ms/call" in table
