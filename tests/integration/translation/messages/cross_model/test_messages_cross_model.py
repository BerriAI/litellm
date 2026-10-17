import json
from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway
from integration._support.provider import SharedProvider
from integration._support.wire import Reply
from integration.translation.case import TranslationTestCase
from integration.translation.messages.bases.anthropic import CLAUDE_SONNET_5_5_TEST_CASE
from integration.translation.messages.bases.azure_ai import (
    CLAUDE_SONNET_5_5_TEST_CASE as AZURE_CLAUDE_SONNET_5_5_TEST_CASE,
)
from integration.translation.messages.bases.openai_responses import GPT_6_1_SOL_TEST_CASE
from integration.translation.response_contracts import messages_structure
from integration.translation.response_scenarios import text_completion, two_messages_before_tools
from integration.translation.runner import assert_provider_request, assert_translation

GPT_6_1_SOL_TWO_MESSAGES_BEFORE_TOOLS_TEST_CASE: Final = two_messages_before_tools(GPT_6_1_SOL_TEST_CASE)
AZURE_CLAUDE_SONNET_5_5_CROSS_MODEL_TEST_CASE: Final = replace(
    AZURE_CLAUDE_SONNET_5_5_TEST_CASE,
    scenario="cross_model_rich_request",
    litellm_request={**CLAUDE_SONNET_5_5_TEST_CASE.litellm_request, "model": "azure_ai/claude-sonnet-5-5"},
    expected_provider_request={
        **CLAUDE_SONNET_5_5_TEST_CASE.expected_provider_request,
        "system": [{"type": "text", "text": CLAUDE_SONNET_5_5_TEST_CASE.expected_provider_request["system"]}],
    },
    mock_provider_response=CLAUDE_SONNET_5_5_TEST_CASE.mock_provider_response,
    expected_litellm_response={
        **CLAUDE_SONNET_5_5_TEST_CASE.expected_litellm_response,
        "model": "azure_ai/claude-sonnet-5-5",
    },
)


@pytest.mark.parametrize(
    "case",
    (AZURE_CLAUDE_SONNET_5_5_CROSS_MODEL_TEST_CASE, GPT_6_1_SOL_TEST_CASE),
    ids=(AZURE_CLAUDE_SONNET_5_5_CROSS_MODEL_TEST_CASE.id, GPT_6_1_SOL_TEST_CASE.id),
)
def test_messages_keep_the_same_public_contract_on_native_and_bridged_routes(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert {key: value for key, value in case.litellm_request.items() if key != "model"} == {
        key: value for key, value in CLAUDE_SONNET_5_5_TEST_CASE.litellm_request.items() if key != "model"
    }
    native: Final = assert_translation(CLAUDE_SONNET_5_5_TEST_CASE, gateway, provider)
    compared: Final = assert_translation(case, gateway, provider)
    assert messages_structure(compared) == messages_structure(native)


def test_messages_preserves_multiple_responses_messages_before_tools(
    gateway: Gateway, provider: SharedProvider
) -> None:
    body: Final = assert_translation(GPT_6_1_SOL_TWO_MESSAGES_BEFORE_TOOLS_TEST_CASE, gateway, provider)
    messages_structure(body)


@pytest.mark.parametrize(
    "case",
    (CLAUDE_SONNET_5_5_TEST_CASE, GPT_6_1_SOL_TEST_CASE),
    ids=(CLAUDE_SONNET_5_5_TEST_CASE.id, GPT_6_1_SOL_TEST_CASE.id),
)
def test_messages_forwards_the_configured_provider_affinity_header_on_native_and_bridged_routes(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    if case is CLAUDE_SONNET_5_5_TEST_CASE:
        pytest.skip("BUG: LIT-8405 native Messages drops the configured provider affinity header")
    alias: Final = (
        "contract-affinity-anthropic" if case is CLAUDE_SONNET_5_5_TEST_CASE else "contract-affinity-responses"
    )
    configured: Final = replace(
        case,
        scenario="provider_affinity_header",
        litellm_request={**case.litellm_request, "model": alias},
        expected_provider_headers={**case.expected_provider_headers, "x-session-affinity": "contract-session-1"},
        expected_litellm_response={**case.expected_litellm_response, "model": alias},
    )
    body: Final = assert_translation(
        configured, gateway, provider, headers={"x-litellm-session-id": "contract-session-1"}
    )
    messages_structure(body)


def test_messages_metadata_does_not_leak_unsupported_prompt_cache_key_to_vertex_gemma(
    gateway: Gateway, provider: SharedProvider
) -> None:
    with gateway.scenario() as scenario:
        alias: Final = scenario.model(
            model="vertex_ai/gemma/gemma-4-26b-a4b-it",
            api_base=f"{provider.wire.url}/v1/projects/synthetic-vertex-project/locations/global/endpoints/contract:predict",
            vertex_project="synthetic-vertex-project",
            vertex_location="global",
            vertex_credentials=json.dumps(
                {
                    "type": "external_account",
                    "audience": "synthetic-vertex-audience",
                    "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
                    "token_url": f"{gateway.upstream_url}/_oauth/token",
                    "credential_source": {"url": f"{gateway.upstream_url}/health"},
                }
            ),
        )
        provider.expect(Reply(body=json.dumps({"predictions": text_completion("gemma-4-26b-a4b-it")}).encode()))
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": alias,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Say hello."}],
                "metadata": {"user_id": "contract-session-1"},
            },
        )
        assert response.status_code == 200, response.text
        assert_provider_request(
            provider.received(),
            "/v1/projects/synthetic-vertex-project/locations/global/endpoints/contract:predict",
            {"content-type": "application/json", "authorization": "Bearer scripted-token"},
            {
                "instances": [
                    {
                        "@requestFormat": "chatCompletions",
                        "messages": [{"role": "user", "content": "Say hello."}],
                        "max_tokens": 64,
                    }
                ]
            },
        )
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        messages_structure(body)
        assert body == {
            "id": "chatcmpl-contract-text",
            "type": "message",
            "role": "assistant",
            "model": alias,
            "content": [{"type": "text", "text": "Hello."}],
            "stop_reason": "end_turn",
            "stop_details": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
