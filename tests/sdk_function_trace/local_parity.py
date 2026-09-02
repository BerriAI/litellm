"""Local SDK parity report: uv run python -m tests.sdk_function_trace.local_parity."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import pytest
from _pytest._code.code import ExceptionRepr
from _pytest.terminal import TerminalReporter
from pydantic import BaseModel, JsonValue, TypeAdapter

from tests.sdk_function_trace.mock_provider import MockProviderResponse, mock_provider

Route = Literal["/ocr", "/chat/completions", "/v1/messages", "/audio/transcriptions", "/v1/responses (websocket)"]
Kind = Literal["SDK E2E", "Functions"]
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
ROUTES: Final[tuple[Route, ...]] = (
    "/ocr",
    "/chat/completions",
    "/v1/messages",
    "/audio/transcriptions",
    "/v1/responses (websocket)",
)
KINDS: Final[tuple[Kind, ...]] = ("SDK E2E", "Functions")
ANTHROPIC_RESPONSE: Final = MockProviderResponse(
    status_code=200,
    headers=(("content-type", "application/json"),),
    body=json.dumps(
        {
            "id": "msg_parity",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }
    ).encode(),
)


@dataclass(frozen=True, slots=True)
class Case:
    route: Route
    kind: Kind
    asynchronous: bool

    @property
    def label(self) -> str:
        return f"{self.route} | {self.kind} | {'async' if self.asynchronous else 'sync'}"


CASES: Final = tuple(
    Case(route, kind, asynchronous) for route in ROUTES for kind in KINDS for asynchronous in (False, True)
)


def invoke_sdk(case: Case, *, rust: bool, api_base: str) -> object:
    import litellm
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages,
        anthropic_messages_handler,
    )
    from tests.test_litellm.ocr import test_function_trace as ocr_trace

    match case.route:
        case "/ocr":
            if case.asynchronous:
                return asyncio.run(
                    litellm.aocr(
                        model=f"mistral/{ocr_trace.MODEL}",
                        document=ocr_trace.DOCUMENT,
                        api_key="test-key",
                        api_base=api_base,
                        pages=[0],
                        rust=rust,
                        timeout=5,
                    )
                )
            return litellm.ocr(
                model=f"mistral/{ocr_trace.MODEL}",
                document=ocr_trace.DOCUMENT,
                api_key="test-key",
                api_base=api_base,
                pages=[0],
                rust=rust,
                timeout=5,
            )
        case "/chat/completions":
            if case.asynchronous:
                return asyncio.run(
                    litellm.acompletion(
                        model="anthropic/claude-sonnet-4-6",
                        messages=[{"role": "user", "content": "hello"}],
                        max_tokens=16,
                        api_key="test-key",
                        api_base=api_base,
                        rust=rust,
                        timeout=5,
                        num_retries=0,
                    )
                )
            return litellm.completion(
                model="anthropic/claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=16,
                api_key="test-key",
                api_base=api_base,
                rust=rust,
                timeout=5,
                num_retries=0,
            )
        case "/v1/messages":
            if case.asynchronous:
                return asyncio.run(
                    anthropic_messages(
                        model="anthropic/claude-sonnet-4-6",
                        messages=[{"role": "user", "content": "hello"}],
                        max_tokens=16,
                        api_key="test-key",
                        api_base=api_base,
                        rust=rust,
                        timeout=5,
                    )
                )
            return anthropic_messages_handler(
                model="anthropic/claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=16,
                api_key="test-key",
                api_base=api_base,
                rust=rust,
                timeout=5,
            )
        case _:
            pytest.xfail("SDK parity scenario not implemented for this route")


def response_body(response: object, *, rust: bool, route: Route) -> dict[str, JsonValue]:
    raw: Final = response.model_dump(mode="json") if isinstance(response, BaseModel) else response
    body: Final = JSON_OBJECT.validate_python(raw)
    hidden: Final = JSON_OBJECT.validate_python(getattr(response, "_hidden_params", body.get("_hidden_params", {})))
    expected_engine: Final = "rust" if rust else "python"
    if rust or route != "/chat/completions":
        assert hidden.get("core_engine") == expected_engine, (
            f"expected {expected_engine} execution, got {hidden.get('core_engine')!r}"
        )
    excluded: Final = frozenset(
        {"_hidden_params", "id", "created"} if route == "/chat/completions" else {"_hidden_params"}
    )
    return {key: value for key, value in body.items() if key not in excluded}


def sdk_response(case: Case, *, rust: bool) -> dict[str, JsonValue]:
    from tests.test_litellm.ocr import test_function_trace as ocr_trace

    provider_response: Final = ocr_trace.PROVIDER_RESPONSE if case.route == "/ocr" else ANTHROPIC_RESPONSE
    with mock_provider(provider_response) as api_base:
        return response_body(invoke_sdk(case, rust=rust, api_base=api_base), rust=rust, route=case.route)


@pytest.mark.xfail(strict=False, reason="Local migration parity diagnostic")
@pytest.mark.parametrize("case", CASES, ids=tuple(case.label for case in CASES))
def test_parity(case: Case) -> None:
    if os.getenv("CI"):
        pytest.skip("Local-only parity report")
    from litellm.rust_bridge import get_native_bridge
    from tests.test_litellm.ocr import test_function_trace as ocr_trace

    if case.kind == "Functions":
        if case.route != "/ocr":
            pytest.xfail("Function trace parity not implemented for this route")
        ocr_trace.test_mistral_ocr_transformation_function_trace_parity(case.asynchronous)
        return
    if case.route == "/audio/transcriptions":
        pytest.xfail("Bedrock transcription is Rust-only; no Python reference implementation")
    if case.route == "/v1/responses (websocket)":
        pytest.xfail("Websocket SDK parity scenario not implemented")
    assert get_native_bridge() is not None, "Native Rust extension is required"
    python: Final = sdk_response(case, rust=False)
    native: Final = sdk_response(case, rust=True)
    differences: Final = tuple(
        key
        for key in sorted(python.keys() | native.keys())
        if key not in python or key not in native or python[key] != native[key]
    )
    assert python == native, f"Response mismatch in: {', '.join(differences)}\nPython: {python!r}\nRust: {native!r}"


class ParityReport:
    def __init__(self) -> None:
        self.reports: tuple[pytest.TestReport, ...] = ()

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
            self.reports = (*self.reports, report)

    def pytest_terminal_summary(self, terminalreporter: TerminalReporter) -> None:
        terminalreporter.section("Local SDK parity: XPASS = matched, XFAIL = mismatch or coverage gap")
        for route in ROUTES:
            terminalreporter.write_line(f"\n{route}", bold=True)
            for report in self.reports:
                if not report.nodeid.split("[", 1)[-1].startswith(f"{route} |"):
                    continue
                label: Final = report.nodeid.split(" | ", 1)[-1].removesuffix("]")
                status: Final = (
                    "XPASS" if report.passed else "XFAIL" if hasattr(report, "wasxfail") else report.outcome.upper()
                )
                terminalreporter.write_line(f"  {status:7} {label}", green=report.passed, yellow=not report.passed)
                if report.longrepr:
                    reason: Final = (
                        report.longrepr.reprcrash.message
                        if isinstance(report.longrepr, ExceptionRepr) and report.longrepr.reprcrash is not None
                        else str(report.longrepr)
                    )
                    terminalreporter.write_line(
                        f"          {reason.splitlines()[0].removeprefix('_pytest.outcomes.XFailed: ')}"
                    )


def main() -> int:
    if os.getenv("CI"):
        print("Local-only parity report; refusing to run in CI")
        return 0
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    return int(
        pytest.main(
            [
                str(Path(__file__).resolve()),
                "--disable-plugin-autoload",
                "-q",
                "--tb=no",
                "--no-header",
                "--disable-warnings",
                "-W",
                "ignore::pytest.PytestConfigWarning",
                "-r",
                "N",
            ],
            plugins=[ParityReport()],
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
