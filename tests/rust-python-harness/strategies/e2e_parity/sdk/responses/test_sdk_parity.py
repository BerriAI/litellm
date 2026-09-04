from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Coroutine
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict

from .....shared.parity.fixtures.store import recorded_fixtures
from .....shared.parity.models import SDKError, SDKReport, sdk_error_report, sdk_success
from .....shared.parity.normalization import NormalizationSpec
from .....shared.parity.recorded_http import (
    HttpHeader,
    RecordedExchange,
    RecordedHttpResponse,
    RecordedRequestMatcher,
    RecordedResponse,
    ReplayItem,
)
from .....shared.parity.runner import ExecutionVariant, parity_worker_main
from ...runner import E2ECheck
from ..contract import BaseSdkParityContract, contract_checks, execute_contract_command

API_KEY: Final = "test-key"
BASELINE_USER_AGENT: Final = "python-responses-parity-fallback"
BASELINE: Final = ExecutionVariant(name="Python", environment=(("LITELLM_RUST", "0"),))
CANDIDATE: Final = ExecutionVariant(name="Rust", environment=(("LITELLM_RUST", "1"),))


class ResponsesCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    model: str
    sdk_input: str | list[dict[str, object]]
    params: dict[str, object]
    provider_responses: tuple[RecordedResponse, ...]
    expected: Literal["success", "error"]

    def canonical_input(self) -> dict[str, object]:
        return cast(dict[str, object], self.model_dump(mode="json", exclude={"provider_responses"}))


def _response(status: int) -> RecordedHttpResponse:
    body: Final = (
        b'{"id":"msg_parity","type":"message","role":"assistant","model":"claude-sonnet-5",'
        b'"content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn",'
        b'"stop_sequence":null,"usage":{"input_tokens":2,"output_tokens":3}}'
        if status == 200
        else b'{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}'
    )
    return RecordedHttpResponse.from_bytes(
        status,
        (HttpHeader(name="content-type", value="application/json"), HttpHeader(name="retry-after", value="0")),
        body,
    )


CASES: Final = (
    ResponsesCase(
        name="anthropic:string",
        model="anthropic/claude-sonnet-5",
        sdk_input="hello",
        params={"max_output_tokens": 16},
        provider_responses=(_response(200),),
        expected="success",
    ),
    ResponsesCase(
        name="anthropic:instructions",
        model="anthropic/claude-sonnet-5",
        sdk_input=[{"role": "user", "content": "hello"}],
        params={"instructions": "be concise", "max_output_tokens": 32},
        provider_responses=(_response(200),),
        expected="success",
    ),
    ResponsesCase(
        name="anthropic:rate-limit",
        model="anthropic/claude-sonnet-5",
        sdk_input="hello",
        params={"max_output_tokens": 16},
        provider_responses=(_response(429),),
        expected="error",
    ),
)


class ResponsesContract(BaseSdkParityContract[ResponsesCase]):
    @property
    def name(self) -> str:
        return "Responses"

    @property
    def modes(self) -> tuple[str, ...]:
        return ("sync", "async")

    @property
    def baseline(self) -> ExecutionVariant:
        return BASELINE

    @property
    def candidate(self) -> ExecutionVariant:
        return CANDIDATE

    @property
    def baseline_user_agent(self) -> str:
        return BASELINE_USER_AGENT

    def cases(self) -> tuple[ResponsesCase, ...]:
        recorded: Final = recorded_fixtures(Path(__file__).with_name("fixtures") / "data", ResponsesCase)
        return tuple({case.name: case for case in (*CASES, *recorded)}.values())

    def dump_case(self, case: ResponsesCase) -> bytes:
        return case.model_dump_json().encode()

    def load_case(self, data: bytes) -> ResponsesCase:
        return ResponsesCase.model_validate_json(data)

    def case_name(self, case: ResponsesCase, mode: str) -> str:
        return f"{mode}:{case.name}"

    def responses(self, case: ResponsesCase) -> tuple[ReplayItem, ...]:
        return tuple(
            RecordedExchange(request=RecordedRequestMatcher(method="POST", path="/v1/messages"), response=response)
            for response in case.provider_responses
        )

    def normalization(self, case: ResponsesCase) -> NormalizationSpec:
        del case
        return NormalizationSpec(
            report_paths=(
                ("response", "created_at"),
                ("response", "id"),
            )
        )

    def execute(
        self,
        case: ResponsesCase,
        mode: str,
        mock_url: str,
        event_loop: asyncio.AbstractEventLoop,
    ) -> SDKReport:
        import litellm

        kwargs: Final = {
            "model": case.model,
            "input": case.sdk_input,
            **case.params,
            "use_chat_completions_api": True,
            "api_base": mock_url,
            "api_key": API_KEY,
            "extra_headers": {"x-litellm-parity-route": mode},
            "num_retries": 0,
        }
        try:
            if mode == "sync":
                sync_call: Final = cast(Callable[..., object], litellm.responses)
                return sdk_success(sync_call(**kwargs))
            if mode == "async":
                async_call: Final = cast(Callable[..., Coroutine[object, object, object]], litellm.aresponses)
                return sdk_success(event_loop.run_until_complete(async_call(**kwargs)))
            raise ValueError(f"unsupported Responses mode: {mode}")
        except Exception as error:
            return sdk_error_report(error)

    def assert_baseline(self, case: ResponsesCase, report: SDKReport) -> None:
        assert isinstance(report, SDKError) is (case.expected == "error")


CONTRACT: Final = ResponsesContract()


def parity_checks() -> AbstractContextManager[tuple[E2ECheck, ...]]:
    return contract_checks(CONTRACT, Path(__file__))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--parity-worker":
        raise SystemExit("usage: test_sdk_parity.py --parity-worker MOCK_URL")
    parity_worker_main(lambda line, url, loop: execute_contract_command(CONTRACT, line, url, loop), sys.argv[2])
