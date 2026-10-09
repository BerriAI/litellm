import json
from typing import Final
from unittest.mock import MagicMock

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


def test_azure_ai_agents_build_model_response_with_annotations():
    """
    Test that _build_model_response includes annotations in the Message object.
    """
    from litellm.llms.azure_ai.agents.handler import AzureAIAgentsHandler
    from litellm.types.utils import ModelResponse

    handler = AzureAIAgentsHandler()
    model_response = ModelResponse()

    annotations = [
        {
            "type": "url_citation",
            "url_citation": {
                "url": "https://example.com",
                "title": "Example",
                "start_index": 0,
                "end_index": 5,
            },
        }
    ]

    result = handler._build_model_response(
        model="azure_ai/agents/asst_123",
        content="Hello [1]",
        model_response=model_response,
        thread_id="thread_abc",
        messages=[{"role": "user", "content": "test"}],
        annotations=annotations,
    )

    assert result.choices[0].message.content == "Hello [1]"
    assert result.choices[0].message.annotations is not None
    assert len(result.choices[0].message.annotations) == 1
    assert result.choices[0].message.annotations[0]["type"] == "url_citation"


def test_azure_ai_agents_build_model_response_without_annotations():
    """
    Test that _build_model_response works correctly without annotations.
    """
    from litellm.llms.azure_ai.agents.handler import AzureAIAgentsHandler
    from litellm.types.utils import ModelResponse

    handler = AzureAIAgentsHandler()
    model_response = ModelResponse()

    result = handler._build_model_response(
        model="azure_ai/agents/asst_123",
        content="Hello",
        model_response=model_response,
        thread_id="thread_abc",
        messages=[{"role": "user", "content": "test"}],
    )

    assert result.choices[0].message.content == "Hello"
    assert getattr(result.choices[0].message, "annotations", None) is None


def test_azure_ai_agents_config_get_agent_id():
    """
    Test agent ID extraction via config method.
    """
    from litellm.llms.azure_ai.agents.transformation import AzureAIAgentsConfig

    config = AzureAIAgentsConfig()

    agent_id = config.get_agent_id("azure_ai/agents/asst_abc123", {})
    assert agent_id == "asst_abc123"

    agent_id = config.get_agent_id("azure_ai/agents/asst_abc123", {"agent_id": "asst_override"})
    assert agent_id == "asst_override"

    agent_id = config.get_agent_id("azure_ai/agents/asst_abc123", {"assistant_id": "asst_assistant"})
    assert agent_id == "asst_assistant"


def test_azure_ai_agents_config_get_complete_url():
    """
    Test that AzureAIAgentsConfig correctly generates base URLs.
    """
    from litellm.llms.azure_ai.agents.transformation import AzureAIAgentsConfig

    config = AzureAIAgentsConfig()

    url = config.get_complete_url(
        api_base="https://test-project.services.ai.azure.com",
        api_key=None,
        model="agents/asst_123",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == "https://test-project.services.ai.azure.com"

    url_with_slash = config.get_complete_url(
        api_base="https://test-project.services.ai.azure.com/",
        api_key=None,
        model="agents/asst_123",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url_with_slash == "https://test-project.services.ai.azure.com"


def test_azure_ai_agents_config_transform_request():
    """
    Test that AzureAIAgentsConfig correctly transforms requests.
    """
    from litellm.llms.azure_ai.agents.transformation import AzureAIAgentsConfig

    config = AzureAIAgentsConfig()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]

    request = config.transform_request(
        model="azure_ai/agents/asst_123",
        messages=messages,
        optional_params={},
        litellm_params={"stream": False},
        headers={},
    )

    assert request["agent_id"] == "asst_123"
    assert "messages" in request
    assert len(request["messages"]) == 2
    assert request["messages"][0]["role"] == "system"
    assert request["messages"][1]["role"] == "user"
    assert "api_version" in request
    assert request["api_version"] == "2025-05-01"


def test_azure_ai_agents_extract_content_from_messages():
    """
    Test content extraction from Azure Agents message response.
    """
    from litellm.llms.azure_ai.agents.handler import AzureAIAgentsHandler

    handler = AzureAIAgentsHandler()

    messages_data = {
        "data": [
            {
                "id": "msg_123",
                "role": "assistant",
                "content": [{"type": "text", "text": {"value": "The answer is 100."}}],
            },
            {
                "id": "msg_122",
                "role": "user",
                "content": [{"type": "text", "text": {"value": "What is 25 * 4?"}}],
            },
        ]
    }

    content, annotations = handler._extract_content_from_messages(messages_data)
    assert content == "The answer is 100."
    assert annotations is None

    empty_data = {"data": []}
    content, annotations = handler._extract_content_from_messages(empty_data)
    assert content == ""
    assert annotations is None


def test_azure_ai_agents_extract_content_with_annotations():
    """
    Test that annotations (e.g., Bing Search citations) are extracted from
    Azure Agents message responses and transformed to OpenAI-compatible format.

    Ref: https://github.com/BerriAI/litellm/issues/19126
    """
    from litellm.llms.azure_ai.agents.handler import AzureAIAgentsHandler

    handler = AzureAIAgentsHandler()

    messages_data = {
        "data": [
            {
                "id": "msg_abc",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": {
                            "value": "According to sources [1], the answer is yes.",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "text": "[1]",
                                    "start_index": 22,
                                    "end_index": 25,
                                    "url_citation": {
                                        "url": "https://example.com/source",
                                        "title": "Example Source",
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        ]
    }

    content, annotations = handler._extract_content_from_messages(messages_data)
    assert content == "According to sources [1], the answer is yes."
    assert annotations is not None
    assert len(annotations) == 1
    assert annotations[0]["type"] == "url_citation"
    assert annotations[0]["url_citation"]["url"] == "https://example.com/source"
    assert annotations[0]["url_citation"]["title"] == "Example Source"

    assert annotations[0]["url_citation"]["start_index"] == 22
    assert annotations[0]["url_citation"]["end_index"] == 25


def test_azure_ai_agents_get_agent_id_from_model():
    """
    Test agent ID extraction from model name.
    """
    from litellm.llms.azure_ai.agents.transformation import AzureAIAgentsConfig

    agent_id = AzureAIAgentsConfig.get_agent_id_from_model("azure_ai/agents/asst_abc123")
    assert agent_id == "asst_abc123"

    agent_id = AzureAIAgentsConfig.get_agent_id_from_model("agents/asst_xyz789")
    assert agent_id == "asst_xyz789"

    agent_id = AzureAIAgentsConfig.get_agent_id_from_model("asst_plain")
    assert agent_id == "asst_plain"


def test_azure_ai_agents_handler_url_builders():
    """
    Test the URL building methods in the handler.

    Azure Foundry Agents API uses direct paths without /openai/ prefix.
    See: https://learn.microsoft.com/en-us/azure/ai-foundry/agents/quickstart
    """
    from litellm.llms.azure_ai.agents.handler import AzureAIAgentsHandler

    handler = AzureAIAgentsHandler()
    api_base = "https://test.services.ai.azure.com/api/projects/test-project"
    api_version = "2025-05-01"
    thread_id = "thread_abc123"
    run_id = "run_xyz789"

    thread_url = handler._build_thread_url(api_base, api_version)
    assert thread_url == f"{api_base}/threads?api-version={api_version}"

    messages_url = handler._build_messages_url(api_base, thread_id, api_version)
    assert messages_url == f"{api_base}/threads/{thread_id}/messages?api-version={api_version}"

    runs_url = handler._build_runs_url(api_base, thread_id, api_version)
    assert runs_url == f"{api_base}/threads/{thread_id}/runs?api-version={api_version}"

    status_url = handler._build_run_status_url(api_base, thread_id, run_id, api_version)
    assert status_url == f"{api_base}/threads/{thread_id}/runs/{run_id}?api-version={api_version}"


def test_azure_ai_agents_is_agents_route():
    """
    Test the is_azure_ai_agents_route detection method.
    """
    from litellm.llms.azure_ai.agents.transformation import AzureAIAgentsConfig

    assert AzureAIAgentsConfig.is_azure_ai_agents_route("azure_ai/agents/asst_123") is True
    assert AzureAIAgentsConfig.is_azure_ai_agents_route("agents/asst_123") is True

    assert AzureAIAgentsConfig.is_azure_ai_agents_route("azure_ai/gpt-4") is False
    assert AzureAIAgentsConfig.is_azure_ai_agents_route("gpt-4") is False


def test_azure_ai_agents_provider_detection():
    """
    Test that the azure_ai provider is correctly detected from model name.
    """
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider(
        model="azure_ai/agents/asst_abc123",
        api_base="https://test.services.ai.azure.com",
    )

    assert provider == "azure_ai"
    assert model == "agents/asst_abc123"


@pytest.mark.asyncio
async def test_azure_ai_agents_streaming_accumulates_annotations_from_multiple_text_items():
    """
    Test that annotations from multiple text content items in thread.message.completed
    are accumulated (not overwritten).

    Ref: Greptile review on PR #23849
    """
    from litellm.llms.azure_ai.agents.handler import AzureAIAgentsHandler

    handler = AzureAIAgentsHandler()

    completed_data = {
        "content": [
            {
                "type": "text",
                "text": {
                    "value": "First source [1].",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "text": "[1]",
                            "start_index": 12,
                            "end_index": 15,
                            "url_citation": {
                                "url": "https://example.com/first",
                                "title": "First",
                            },
                        }
                    ],
                },
            },
            {
                "type": "text",
                "text": {
                    "value": "Second source [2].",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "text": "[2]",
                            "start_index": 13,
                            "end_index": 16,
                            "url_citation": {
                                "url": "https://example.com/second",
                                "title": "Second",
                            },
                        }
                    ],
                },
            },
        ]
    }

    sse_lines = [
        "event: thread.created",
        "",
        'data: {"id": "thread_multi"}',
        "",
        "event: thread.message.completed",
        "",
        f"data: {json.dumps(completed_data)}",
        "",
        "data: [DONE]",
    ]

    async def mock_aiter_lines():
        for line in sse_lines:
            yield line

    mock_response = MagicMock()
    mock_response.aiter_lines = MagicMock(return_value=mock_aiter_lines())

    chunks = []
    async for chunk in handler._process_sse_stream(mock_response, "azure_ai/agents/asst_123"):
        chunks.append(chunk)

    final_chunk = chunks[-1]
    assert final_chunk.choices[0].delta.annotations is not None
    assert len(final_chunk.choices[0].delta.annotations) == 2
    urls = [a["url_citation"]["url"] for a in final_chunk.choices[0].delta.annotations]
    assert "https://example.com/first" in urls
    assert "https://example.com/second" in urls


@pytest.mark.asyncio
async def test_azure_ai_agents_streaming_annotations_from_completed_message():
    """
    Test that annotations from thread.message.completed SSE events are collected
    and attached to the final chunk's delta.

    Ref: https://github.com/BerriAI/litellm/issues/19126
    """
    from litellm.llms.azure_ai.agents.handler import AzureAIAgentsHandler

    handler = AzureAIAgentsHandler()

    completed_data = {
        "content": [
            {
                "type": "text",
                "text": {
                    "value": "According to [1], the answer is 42.",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "text": "[1]",
                            "start_index": 12,
                            "end_index": 15,
                            "url_citation": {
                                "url": "https://example.com/citation",
                                "title": "Citation Source",
                            },
                        }
                    ],
                },
            }
        ]
    }

    sse_lines = [
        "event: thread.created",
        "",
        'data: {"id": "thread_stream_123"}',
        "",
        "event: thread.message.delta",
        "",
        'data: {"delta": {"content": [{"type": "text", "text": {"value": "According to [1], the answer is 42."}}]}}',
        "",
        "event: thread.message.completed",
        "",
        f"data: {json.dumps(completed_data)}",
        "",
        "data: [DONE]",
    ]

    async def mock_aiter_lines():
        for line in sse_lines:
            yield line

    mock_response = MagicMock()
    mock_response.aiter_lines = MagicMock(return_value=mock_aiter_lines())

    chunks = []
    async for chunk in handler._process_sse_stream(mock_response, "azure_ai/agents/asst_123"):
        chunks.append(chunk)

    assert len(chunks) >= 1
    final_chunk = chunks[-1]
    assert final_chunk.choices[0].finish_reason == "stop"
    assert final_chunk.choices[0].delta.annotations is not None
    assert len(final_chunk.choices[0].delta.annotations) == 1
    ann = final_chunk.choices[0].delta.annotations[0]
    assert ann["type"] == "url_citation"
    assert ann["url_citation"]["url"] == "https://example.com/citation"
    assert ann["url_citation"]["title"] == "Citation Source"


def test_azure_ai_agents_validate_environment():
    """
    Test that headers are correctly set up with Bearer token authentication.

    Azure Foundry Agents uses Bearer token authentication (Azure AD tokens).
    """
    from litellm.llms.azure_ai.agents.transformation import AzureAIAgentsConfig

    config = AzureAIAgentsConfig()

    headers = config.validate_environment(
        headers={},
        model="agents/asst_123",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="test-azure-ad-token",
        api_base="https://test.services.ai.azure.com/api/projects/test-project",
    )

    assert headers["Content-Type"] == "application/json"
    assert headers["Authorization"] == "Bearer test-azure-ad-token"


def test_azure_ai_get_azure_ai_route():
    """
    Test the get_azure_ai_route dispatch method.
    """
    from litellm.llms.azure_ai.common_utils import AzureFoundryModelInfo

    assert AzureFoundryModelInfo.get_azure_ai_route("agents/asst_123") == "agents"
    assert AzureFoundryModelInfo.get_azure_ai_route("azure_ai/agents/asst_abc") == "agents"

    assert AzureFoundryModelInfo.get_azure_ai_route("gpt-4") == "default"
    assert AzureFoundryModelInfo.get_azure_ai_route("claude-3-sonnet") == "default"
    assert AzureFoundryModelInfo.get_azure_ai_route("azure_ai/gpt-4o") == "default"
