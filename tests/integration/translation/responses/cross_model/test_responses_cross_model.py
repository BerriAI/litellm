from dataclasses import replace
from pathlib import Path
from typing import Final
from unittest.mock import ANY
from uuid import uuid4

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.provider import SharedProvider
from integration._support.responses_vendor import same_response
from integration.translation.case import TranslationTestCase
from integration.translation.response_contracts import responses_structure
from integration.translation.response_scenarios import two_messages_before_tools
from integration.translation.responses.bases.anthropic import CLAUDE_SONNET_5_5_TEST_CASE
from integration.translation.responses.bases.azure_ai import (
    CLAUDE_SONNET_5_5_TEST_CASE as AZURE_CLAUDE_SONNET_5_5_TEST_CASE,
)
from integration.translation.responses.bases.openai_responses import GPT_5_6_SOL_TEST_CASE, GPT_6_1_SOL_TEST_CASE
from integration.translation.runner import assert_translation

pytestmark: Final = pytest.mark.shares_worker_state

GPT_6_1_SOL_TWO_MESSAGES_BEFORE_TOOLS_TEST_CASE: Final = two_messages_before_tools(GPT_6_1_SOL_TEST_CASE)
AZURE_CLAUDE_SONNET_5_5_CROSS_MODEL_TEST_CASE: Final = replace(
    AZURE_CLAUDE_SONNET_5_5_TEST_CASE,
    scenario="cross_model_rich_request",
    litellm_request={**CLAUDE_SONNET_5_5_TEST_CASE.litellm_request, "model": "azure_ai/claude-sonnet-5-5"},
    expected_provider_request=CLAUDE_SONNET_5_5_TEST_CASE.expected_provider_request,
    mock_provider_response=CLAUDE_SONNET_5_5_TEST_CASE.mock_provider_response,
    expected_litellm_response={
        **CLAUDE_SONNET_5_5_TEST_CASE.expected_litellm_response,
        "model": "azure_ai/claude-sonnet-5-5",
    },
)
GPT_6_1_SOL_PREVIOUS_RESPONSE_ID_TEST_CASE: Final = replace(
    GPT_5_6_SOL_TEST_CASE,
    scenario="previous_response_id",
    litellm_request={**GPT_5_6_SOL_TEST_CASE.litellm_request, "model": "openai/responses/gpt-6.1-sol"},
    expected_provider_request={**GPT_5_6_SOL_TEST_CASE.expected_provider_request, "model": "gpt-6.1-sol"},
    mock_provider_response={**GPT_5_6_SOL_TEST_CASE.mock_provider_response, "model": "gpt-6.1-sol"},
    expected_litellm_response={
        **GPT_5_6_SOL_TEST_CASE.expected_litellm_response,
        "model": GPT_6_1_SOL_TEST_CASE.expected_litellm_response["model"],
    },
)


@pytest.mark.parametrize(
    "case",
    (CLAUDE_SONNET_5_5_TEST_CASE, AZURE_CLAUDE_SONNET_5_5_CROSS_MODEL_TEST_CASE),
    ids=(CLAUDE_SONNET_5_5_TEST_CASE.id, AZURE_CLAUDE_SONNET_5_5_CROSS_MODEL_TEST_CASE.id),
)
def test_responses_keep_the_same_public_contract_on_native_and_bridged_routes(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert {key: value for key, value in case.litellm_request.items() if key != "model"} == {
        key: value for key, value in GPT_6_1_SOL_TEST_CASE.litellm_request.items() if key != "model"
    }
    native: Final = assert_translation(GPT_6_1_SOL_TEST_CASE, gateway, provider)
    compared: Final = assert_translation(case, gateway, provider)
    assert responses_structure(compared) == responses_structure(native)


def test_responses_preserves_multiple_message_items_before_tools(gateway: Gateway, provider: SharedProvider) -> None:
    body: Final = assert_translation(GPT_6_1_SOL_TWO_MESSAGES_BEFORE_TOOLS_TEST_CASE, gateway, provider)
    responses_structure(body)


@pytest.mark.parametrize(
    "case",
    (GPT_6_1_SOL_PREVIOUS_RESPONSE_ID_TEST_CASE, AZURE_CLAUDE_SONNET_5_5_TEST_CASE),
    ids=(GPT_6_1_SOL_PREVIOUS_RESPONSE_ID_TEST_CASE.id, AZURE_CLAUDE_SONNET_5_5_TEST_CASE.id),
)
def test_previous_response_id_replays_the_turn_without_leaking_internal_routing_fields(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider, tmp_path: Path
) -> None:
    stock: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = tmp_path / "stored-responses.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                **stock,
                "general_settings": {**object_value(stock["general_settings"]), "store_prompts_in_spend_logs": True},
            }
        )
    )
    with owned_proxy_process(gateway, tmp_path, {}, config=config) as owned:
        assert_previous_response_replay(case, owned.gateway, provider)


def assert_previous_response_replay(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    identity: Final = f"resp_contract_{uuid4().hex}"
    first_case: Final = replace(
        case,
        mock_provider_response={**case.mock_provider_response, "id": identity},
        expected_litellm_response={**case.expected_litellm_response, "id": ANY},
    )
    first: Final = assert_translation(first_case, gateway, provider)
    responses_structure(first)
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE model_group=%s',
            (string_value(case.litellm_request["model"]),),
        ),
        lambda rows: any(same_response(str(row["request_id"]), str(first["id"])) for row in rows),
        seconds=70,
    )
    previous_identifier: Final = next(
        str(row["request_id"]) for row in rows if same_response(str(row["request_id"]), str(first["id"]))
    )
    expected_request: Final = (
        {**case.expected_provider_request, "input": "Say hello again.", "previous_response_id": identity}
        if case is GPT_6_1_SOL_PREVIOUS_RESPONSE_ID_TEST_CASE
        else {
            **case.expected_provider_request,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Say hello."}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hello."}]},
                {"role": "user", "content": [{"type": "text", "text": "Say hello again."}]},
            ],
        }
    )
    followup: Final = replace(
        first_case,
        litellm_request={**case.litellm_request, "input": "Say hello again.", "previous_response_id": first["id"]},
        expected_provider_request=expected_request,
        mock_provider_response={**case.mock_provider_response, "id": f"{identity}_second"},
        expected_litellm_response={
            **case.expected_litellm_response,
            "id": ANY,
            "previous_response_id": None if case is GPT_6_1_SOL_PREVIOUS_RESPONSE_ID_TEST_CASE else previous_identifier,
        },
    )
    responses_structure(assert_translation(followup, gateway, provider))
