import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm.llms.anthropic.common_utils import (
    is_anthropic_invalid_thinking_block_error,
    is_anthropic_thinking_binding_mismatch_error,
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


class _Fixture(TypedDict):
    invalid_thinking_block_error: ReadOnly[tuple[_TextCase, ...]]
    thinking_binding_mismatch_error: ReadOnly[tuple[_TextCase, ...]]
    http_error_retry: ReadOnly[_RetryTable]


FIXTURE: Final = TypeAdapter(_Fixture).validate_json(FIXTURE_PATH.read_text())
RETRY: Final = FIXTURE["http_error_retry"]


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


def _rejection(case: _RetryCase) -> httpx.HTTPStatusError:
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
