import pytest

from litellm.llms.azure_ai.agents.handler import AzureAIAgentsHandler


def test_should_encode_thread_id_in_azure_ai_agent_urls():
    handler = AzureAIAgentsHandler()

    assert (
        handler._build_messages_url(
            "https://example.services.ai.azure.com/api/projects/proj",
            "../../threads/other?x=1#frag",
            "2024-05-01-preview",
        )
        == "https://example.services.ai.azure.com/api/projects/proj/threads/..%2F..%2Fthreads%2Fother%3Fx%3D1%23frag/messages?api-version=2024-05-01-preview"
    )
    assert (
        handler._build_runs_url(
            "https://example.services.ai.azure.com/api/projects/proj",
            "thread/abc",
            "2024-05-01-preview",
        )
        == "https://example.services.ai.azure.com/api/projects/proj/threads/thread%2Fabc/runs?api-version=2024-05-01-preview"
    )


def test_should_encode_thread_and_run_ids_in_azure_ai_agent_status_url():
    handler = AzureAIAgentsHandler()

    assert (
        handler._build_run_status_url(
            "https://example.services.ai.azure.com/api/projects/proj",
            "thread/abc",
            "../runs/other#frag",
            "2024-05-01-preview",
        )
        == "https://example.services.ai.azure.com/api/projects/proj/threads/thread%2Fabc/runs/..%2Fruns%2Fother%23frag?api-version=2024-05-01-preview"
    )


def test_should_reject_dot_segments_in_azure_ai_agent_urls():
    handler = AzureAIAgentsHandler()

    with pytest.raises(ValueError, match="thread_id cannot be a dot path segment"):
        handler._build_messages_url(
            "https://example.services.ai.azure.com/api/projects/proj",
            "..",
            "2024-05-01-preview",
        )

    with pytest.raises(ValueError, match="run_id cannot be a dot path segment"):
        handler._build_run_status_url(
            "https://example.services.ai.azure.com/api/projects/proj",
            "thread_123",
            "..",
            "2024-05-01-preview",
        )
