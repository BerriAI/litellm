from unittest.mock import patch

import pytest

from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.experimental_pass_through.transformation import (
    VertexAIPartnerModelsAnthropicMessagesConfig,
)


def test_validate_environment_uses_vertex_ai_location():
    config = VertexAIPartnerModelsAnthropicMessagesConfig()
    headers = {}
    litellm_params = {
        "vertex_ai_project": "test-project",
        "vertex_ai_location": "europe-west1",
        "vertex_credentials": "{}",
    }
    optional_params = {}

    with patch.object(
        config, "_ensure_access_token", return_value=("token", "test-project")
    ), patch.object(
        config, "get_complete_vertex_url", return_value="https://mock-url"
    ) as mock_get_url:
        config.validate_anthropic_messages_environment(
            headers=headers,
            model="claude-3-sonnet",
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_base=None,
        )
        assert mock_get_url.call_args.kwargs["vertex_location"] == "europe-west1"
