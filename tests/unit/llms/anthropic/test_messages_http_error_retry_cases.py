import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm.llms.anthropic.common_utils import (
    ToolsWithoutStrict,
    is_anthropic_invalid_thinking_block_error,
    is_anthropic_thinking_binding_mismatch_error,
    is_claude_strict_tools_rejection,
    tools_without_strict,
)
from litellm.llms.anthropic.messages.native.transformation import AnthropicMessagesConfig
from litellm.llms.deepseek.messages.transformation import DeepSeekAnthropicMessagesConfig
from litellm.llms.base_llm.anthropic_messages.transformation import BaseAnthropicMessagesConfig

FIXTURE_PATH: Final = (
    Path(__file__).parents[4] / "litellm-rust/crates/llms/tests/fixtures/anthropic_messages_http_error_retry.json"
)


class _TextCase(TypedDict):
    name: ReadOnly[str]
    error_text: ReadOnly[str]
    expected: ReadOnly[bool]


class _RetryCase(TypedDict):
    name: ReadOnly[str]
    status: ReadOnly[int]
    error_text: ReadOnly[str]
    thinking: ReadOnly[Mapping[str, object]]
    retried: ReadOnly[bool]
    stripped: ReadOnly[bool]


class _RetryTable(TypedDict):
    request: ReadOnly[Mapping[str, object]]
    stripped_request: ReadOnly[Mapping[str, object]]
    cases: ReadOnly[tuple[_RetryCase, ...]]


class _StrictRejectionCase(TypedDict):
    name: ReadOnly[str]
    status: ReadOnly[int]
    error_text: ReadOnly[str]
    expected: ReadOnly[bool]


class _ToolsWithoutStrictExpected(TypedDict):
    tools: ReadOnly[tuple[Mapping[str, object], ...]]
    dropped_count: ReadOnly[int]


class _ToolsWithoutStrictCase(TypedDict):
    name: ReadOnly[str]
    tools: ReadOnly[object]
    expected: ReadOnly[_ToolsWithoutStrictExpected | None]


class _StrictRetryCase(TypedDict):
    name: ReadOnly[str]
    status: ReadOnly[int]
    error_text: ReadOnly[str]
    thinking: ReadOnly[Mapping[str, object]]
    drops_unsupported_params: ReadOnly[bool]
    retried: ReadOnly[bool]
    stripped: ReadOnly[bool]
    strict_dropped: ReadOnly[bool]


class _StrictRetryTable(TypedDict):
    tools: ReadOnly[tuple[Mapping[str, object], ...]]
    tools_without_strict: ReadOnly[tuple[Mapping[str, object], ...]]
    dropped: ReadOnly[int]
    cases: ReadOnly[tuple[_StrictRetryCase, ...]]


class _Fixture(TypedDict):
    invalid_thinking_block_error: ReadOnly[tuple[_TextCase, ...]]
    thinking_binding_mismatch_error: ReadOnly[tuple[_TextCase, ...]]
    http_error_retry: ReadOnly[_RetryTable]
    claude_strict_tools_rejection: ReadOnly[tuple[_StrictRejectionCase, ...]]
    tools_without_strict: ReadOnly[tuple[_ToolsWithoutStrictCase, ...]]
    strict_tools_retry: ReadOnly[_StrictRetryTable]


FIXTURE: Final = TypeAdapter(_Fixture).validate_json(FIXTURE_PATH.read_text())
RETRY: Final = FIXTURE["http_error_retry"]
STRICT_RETRY: Final = FIXTURE["strict_tools_retry"]


@pytest.mark.parametrize(
    "case", [pytest.param(case, id=case["name"]) for case in FIXTURE["invalid_thinking_block_error"]]
)
def test_invalid_thinking_block_errors_match_the_shared_cases(case: _TextCase) -> None:
    assert is_anthropic_invalid_thinking_block_error(case["error_text"]) is case["expected"]


@pytest.mark.parametrize(
    "case", [pytest.param(case, id=case["name"]) for case in FIXTURE["thinking_binding_mismatch_error"]]
)
def test_binding_mismatch_errors_match_the_shared_cases(case: _TextCase) -> None:
    assert is_anthropic_thinking_binding_mismatch_error(case["error_text"]) is case["expected"]


@pytest.mark.parametrize(
    "case", [pytest.param(case, id=case["name"]) for case in FIXTURE["claude_strict_tools_rejection"]]
)
def test_strict_tools_rejections_match_the_shared_cases(case: _StrictRejectionCase) -> None:
    assert is_claude_strict_tools_rejection(case["status"], case["error_text"]) is case["expected"]


@pytest.mark.parametrize("case", [pytest.param(case, id=case["name"]) for case in FIXTURE["tools_without_strict"]])
def test_tools_without_strict_matches_the_shared_cases(case: _ToolsWithoutStrictCase) -> None:
    expected: Final = case["expected"]

    assert tools_without_strict(copy.deepcopy(case["tools"])) == (
        None
        if expected is None
        else ToolsWithoutStrict(tools=tuple(expected["tools"]), dropped_count=expected["dropped_count"])
    )


def _rejection(case: _RetryCase | _StrictRetryCase) -> httpx.HTTPStatusError:
    request: Final = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx.HTTPStatusError(
        "rejected",
        request=request,
        response=httpx.Response(case["status"], text=case["error_text"], request=request),
    )


@pytest.mark.parametrize(
    "config",
    [AnthropicMessagesConfig(), DeepSeekAnthropicMessagesConfig()],
    ids=["anthropic", "deepseek"],
)
@pytest.mark.parametrize("case", [pytest.param(case, id=case["name"]) for case in RETRY["cases"]])
def test_the_http_error_retry_matches_the_shared_cases(config: BaseAnthropicMessagesConfig, case: _RetryCase) -> None:
    body: Final = {**copy.deepcopy(RETRY["request"]), "thinking": case["thinking"]}

    retried: Final = config.should_retry_anthropic_messages_on_http_error(
        e=_rejection(case), litellm_params={}, request_body=body
    )
    transformed: Final = config.transform_anthropic_messages_request_on_http_error(
        e=_rejection(case), request_data=copy.deepcopy(body)
    )

    assert retried is case["retried"]
    assert transformed == (RETRY["stripped_request"] if case["stripped"] else body)


@pytest.mark.parametrize(
    "config",
    [AnthropicMessagesConfig(), DeepSeekAnthropicMessagesConfig()],
    ids=["anthropic", "deepseek"],
)
@pytest.mark.parametrize("case", [pytest.param(case, id=case["name"]) for case in STRICT_RETRY["cases"]])
def test_the_strict_tools_retry_matches_the_shared_cases(
    monkeypatch: pytest.MonkeyPatch, config: BaseAnthropicMessagesConfig, case: _StrictRetryCase
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    body: Final = {
        **copy.deepcopy(RETRY["request"]),
        "thinking": case["thinking"],
        "tools": copy.deepcopy(list(STRICT_RETRY["tools"])),
    }
    expected: Final = {
        **copy.deepcopy(RETRY["stripped_request"] if case["stripped"] else body),
        "tools": list(STRICT_RETRY["tools_without_strict"] if case["strict_dropped"] else STRICT_RETRY["tools"]),
    }

    retried: Final = config.should_retry_anthropic_messages_on_http_error(
        e=_rejection(case), litellm_params={"drop_params": case["drops_unsupported_params"]}, request_body=body
    )
    transformed: Final = config.transform_anthropic_messages_request_on_http_error(
        e=_rejection(case), request_data=copy.deepcopy(body)
    )

    assert retried is case["retried"]
    assert transformed == expected
