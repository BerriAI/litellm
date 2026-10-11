import json
import os
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

GREENPT_BASE_URL = "https://api.greenpt.ai/v1"


def _get_config():
    provider = JSONProviderRegistry.get("greenpt")
    assert provider is not None
    config_class = create_config_class(provider)
    return config_class()


def test_greenpt_provider_registered():
    provider = JSONProviderRegistry.get("greenpt")
    assert provider is not None
    assert provider.base_url == GREENPT_BASE_URL
    assert provider.api_key_env == "GREENPT_API_KEY"
    assert provider.api_base_env == "GREENPT_API_BASE"


def test_greenpt_resolves_env_api_key(monkeypatch):
    config = _get_config()
    monkeypatch.setenv("GREENPT_API_KEY", "test-key")
    api_base, api_key = config._get_openai_compatible_provider_info(None, None)
    assert api_base == GREENPT_BASE_URL
    assert api_key == "test-key"


def test_greenpt_maps_max_completion_tokens():
    config = _get_config()
    params = config.map_openai_params(
        non_default_params={"max_completion_tokens": 256},
        optional_params={},
        model="greenpt/glm-5.2",
        drop_params=False,
    )
    assert params == {"max_tokens": 256}


def test_greenpt_complete_url_appends_endpoint():
    config = _get_config()
    url = config.get_complete_url(
        api_base=GREENPT_BASE_URL,
        api_key="test-key",
        model="greenpt/glm-5.2",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == f"{GREENPT_BASE_URL}/chat/completions"


def test_greenpt_provider_resolution():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, _, api_base = get_llm_provider(
        model="greenpt/glm-5.2",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "glm-5.2"
    assert provider == "greenpt"
    assert api_base == GREENPT_BASE_URL


def test_greenpt_embedding_dispatches():
    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
        "model": "green-embedding",
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }
    client = HTTPHandler()

    with patch.object(HTTPHandler, "post", return_value=response) as mock_post:
        result = litellm.embedding(
            model="greenpt/green-embedding",
            input=["renewable inference"],
            api_key="test-key",
            client=client,
        )

    assert result.data[0]["embedding"] == [0.1, 0.2]
    assert mock_post.call_args.args[0] == f"{GREENPT_BASE_URL}/embeddings"
    assert json.loads(mock_post.call_args.kwargs["data"]) == {
        "model": "green-embedding",
        "input": ["renewable inference"],
    }


def test_greenpt_rerank_config():
    config = ProviderConfigManager.get_provider_rerank_config(
        model="green-rerank",
        provider=LlmProviders.GREENPT,
        api_base=GREENPT_BASE_URL,
        present_version_params=[],
    )
    assert config.get_complete_url(GREENPT_BASE_URL, "green-rerank") == (f"{GREENPT_BASE_URL}/rerank")
