from typing import Final

import pytest

from litellm.llms.azure_ai.agents.transformation import AzureAIAgentsConfig

MESSAGES: Final = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]},
]
FLATTENED_MESSAGES: Final = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "hello world"},
]


@pytest.mark.parametrize(
    ("optional_params", "expected"),
    [
        pytest.param(
            {},
            {
                "agent_id": "asst_from_model",
                "messages": FLATTENED_MESSAGES,
                "api_version": AzureAIAgentsConfig.DEFAULT_API_VERSION,
            },
            id="agent-id-from-model-and-default-api-version",
        ),
        pytest.param(
            {
                "agent_id": "asst_override",
                "api_version": "2025-05-15-preview",
                "thread_id": "thread_9",
                "instructions": "use the search tool",
            },
            {
                "agent_id": "asst_override",
                "messages": FLATTENED_MESSAGES,
                "api_version": "2025-05-15-preview",
                "thread_id": "thread_9",
                "instructions": "use the search tool",
            },
            id="caller-agent-id-api-version-thread-and-instructions",
        ),
        pytest.param(
            {"assistant_id": "asst_legacy_name"},
            {
                "agent_id": "asst_legacy_name",
                "messages": FLATTENED_MESSAGES,
                "api_version": AzureAIAgentsConfig.DEFAULT_API_VERSION,
            },
            id="assistant-id-names-the-agent",
        ),
    ],
)
def test_transform_request_builds_the_agent_run_payload(
    optional_params: dict[str, object], expected: dict[str, object]
) -> None:
    payload = AzureAIAgentsConfig().transform_request(
        model="azure_ai/agents/asst_from_model",
        messages=MESSAGES,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert payload == expected
