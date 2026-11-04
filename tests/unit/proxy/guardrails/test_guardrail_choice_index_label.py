"""The per-choice label on guardrail spend logs is proxy-owned, never caller-supplied.

`guardrail_choice_index` names which choice a guardrail round inspected. The OpenAI
handler writes it around each round; `StandardLoggingGuardrailInformation` reads it and
spend logs persist it. A caller putting the key in its own request body would otherwise
have that value stored verbatim, with any type or size, on every guardrail row of the
request.
"""

from typing import Final
from unittest.mock import MagicMock

import pytest
from fastapi import Request

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request


def _request_mock() -> Request:
    request_mock = MagicMock(spec=Request)
    request_mock.url = MagicMock()
    request_mock.url.path = "/v1/chat/completions"
    request_mock.url.__str__.return_value = "http://localhost/v1/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"
    return request_mock


def _logged_choice_index(request_data: dict[str, object]) -> object:
    guardrail: Final = CustomGuardrail(guardrail_name="choice-label-test")
    guardrail.add_standard_logging_guardrail_information_to_request_data(
        guardrail_json_response={},
        request_data=request_data,
        guardrail_status="success",
    )
    metadata: Final = request_data["metadata"]
    assert isinstance(metadata, dict)
    rows: Final = metadata["standard_logging_guardrail_information"]
    assert isinstance(rows, list)
    return rows[0]["choice_index"]


@pytest.mark.asyncio
async def test_proxy_drops_a_caller_supplied_choice_index() -> None:
    data: Final[dict[str, object]] = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "guardrail_choice_index": "x" * 4096,
    }

    updated: Final = await add_litellm_data_to_request(
        data=data,
        request=_request_mock(),
        user_api_key_dict=UserAPIKeyAuth(api_key="hashed-key"),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert "guardrail_choice_index" not in updated


def test_a_non_integer_label_is_not_logged() -> None:
    assert _logged_choice_index({"guardrail_choice_index": "1; drop"}) is None


def test_an_integer_label_is_logged() -> None:
    assert _logged_choice_index({"guardrail_choice_index": 1}) == 1
