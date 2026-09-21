from litellm.anthropic_beta_headers_manager import update_headers_with_filtered_beta
from litellm.llms.bedrock.claude_platform.messages_transformation import (
    BedrockClaudePlatformMessagesConfig,
)


def test_bedrock_claude_platform_filters_anthropic_beta_headers():
    assert BedrockClaudePlatformMessagesConfig().should_filter_anthropic_beta_headers() is True


def test_bedrock_claude_platform_beta_filter_drops_and_renames():
    headers = {
        "anthropic-beta": ",".join(
            (
                "code-execution-2025-08-25",
                "advanced-tool-use-2025-11-20",
                "context-management-2025-06-27",
            )
        )
    }

    filtered = update_headers_with_filtered_beta(headers=dict(headers), provider="bedrock")
    values = filtered["anthropic-beta"].split(",")

    assert "code-execution-2025-08-25" not in values
    assert "advanced-tool-use-2025-11-20" not in values
    assert "tool-search-tool-2025-10-19" in values
    assert "context-management-2025-06-27" in values
