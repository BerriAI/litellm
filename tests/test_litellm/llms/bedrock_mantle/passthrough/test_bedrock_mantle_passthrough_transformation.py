import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from botocore.credentials import Credentials

from litellm.llms.bedrock.passthrough.transformation import BedrockPassthroughConfig
from litellm.llms.bedrock_mantle.passthrough.transformation import BedrockMantlePassthroughConfig
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.passthrough.main import llm_passthrough_route
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

MANTLE_API_BASE = "https://bedrock-mantle.us-east-2.api.aws"
INVOKE_ENDPOINT = "model/us.openai.gpt-5.6-sol/invoke"
CONVERSE_ENDPOINT = "model/us.openai.gpt-5.6-sol/converse"
REQUEST_BODY = {"messages": [{"role": "user", "content": "say pong"}], "max_completion_tokens": 64}


@pytest.fixture
def no_ambient_aws(monkeypatch):
    for name in (
        "AWS_BEARER_TOKEN_BEDROCK",
        "BEDROCK_MANTLE_API_KEY",
        "BEDROCK_MANTLE_API_BASE",
        "BEDROCK_MANTLE_REGION",
        "AWS_BEDROCK_RUNTIME_ENDPOINT",
        "AWS_REGION_NAME",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    ):
        monkeypatch.delenv(name, raising=False)


def test_bedrock_mantle_registers_its_own_bedrock_passthrough_config():
    config = ProviderConfigManager.get_provider_passthrough_config(
        model="us.openai.gpt-5.6-sol", provider=LlmProviders.BEDROCK_MANTLE
    )
    assert isinstance(config, BedrockMantlePassthroughConfig)
    assert isinstance(config, BedrockPassthroughConfig)


def test_mantle_api_base_only_lends_its_region_to_the_runtime_url(no_ambient_aws):
    url, base_url = BedrockMantlePassthroughConfig().get_complete_url(
        api_base=MANTLE_API_BASE,
        api_key=None,
        model="us.openai.gpt-5.6-sol",
        endpoint=INVOKE_ENDPOINT,
        request_query_params=None,
        litellm_params={"api_base": MANTLE_API_BASE},
    )
    assert str(url) == f"https://bedrock-runtime.us-east-2.amazonaws.com/{INVOKE_ENDPOINT}"
    assert base_url == "https://bedrock-runtime.us-east-2.amazonaws.com"


def test_explicit_region_and_non_mantle_api_base_are_kept(no_ambient_aws):
    vpc_endpoint = "https://vpce-0123.bedrock-runtime.us-east-1.vpce.amazonaws.com"
    url, base_url = BedrockMantlePassthroughConfig().get_complete_url(
        api_base=vpc_endpoint,
        api_key=None,
        model="us.openai.gpt-5.6-sol",
        endpoint=INVOKE_ENDPOINT,
        request_query_params=None,
        litellm_params={"api_base": vpc_endpoint, "aws_region_name": "us-east-1"},
    )
    assert str(url) == f"{vpc_endpoint}/{INVOKE_ENDPOINT}"
    assert base_url == vpc_endpoint


def test_region_falls_back_to_the_mantle_default_without_any_hint(no_ambient_aws):
    url, _ = BedrockMantlePassthroughConfig().get_complete_url(
        api_base=None,
        api_key=None,
        model="us.openai.gpt-5.6-sol",
        endpoint=INVOKE_ENDPOINT,
        request_query_params=None,
        litellm_params={},
    )
    assert str(url) == f"https://bedrock-runtime.us-east-1.amazonaws.com/{INVOKE_ENDPOINT}"


@pytest.mark.parametrize(
    ("litellm_params", "env", "expected_bearer"),
    [
        ({"api_key": "deployment-bedrock-api-key"}, {}, "deployment-bedrock-api-key"),
        ({}, {"BEDROCK_MANTLE_API_KEY": "mantle-env-key"}, "mantle-env-key"),
        ({}, {"AWS_BEARER_TOKEN_BEDROCK": "aws-env-key"}, "aws-env-key"),
    ],
)
def test_sign_request_uses_the_deployment_bearer_token(no_ambient_aws, monkeypatch, litellm_params, env, expected_bearer):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    headers, body = BedrockMantlePassthroughConfig().sign_request(
        headers={},
        litellm_params=litellm_params,
        request_data=REQUEST_BODY,
        api_base=f"https://bedrock-runtime.us-east-1.amazonaws.com/{INVOKE_ENDPOINT}",
        model="us.openai.gpt-5.6-sol",
    )
    assert headers["Authorization"] == f"Bearer {expected_bearer}"
    assert body is not None
    assert json.loads(body) == REQUEST_BODY


def test_sign_request_falls_back_to_sigv4_scoped_to_the_mantle_region(no_ambient_aws):
    config = BedrockMantlePassthroughConfig()
    with patch.object(config, "get_credentials", return_value=Credentials("AKIA", "secret")):
        headers, body = config.sign_request(
            headers={},
            litellm_params={"api_base": MANTLE_API_BASE},
            request_data=REQUEST_BODY,
            api_base=f"https://bedrock-runtime.us-east-2.amazonaws.com/{INVOKE_ENDPOINT}",
            model="us.openai.gpt-5.6-sol",
        )
    assert headers["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIA/")
    assert "/us-east-2/bedrock/aws4_request" in headers["Authorization"]
    assert body is not None
    assert json.loads(body) == REQUEST_BODY


@pytest.mark.parametrize(
    ("route_kwargs", "env", "expected_bearer"),
    [
        ({"api_key": "deployment-bedrock-api-key"}, {}, "deployment-bedrock-api-key"),
        ({}, {"BEDROCK_MANTLE_API_KEY": "mantle-env-key"}, "mantle-env-key"),
    ],
)
def test_invoke_passthrough_route_reaches_bedrock_runtime_for_a_mantle_deployment(
    no_ambient_aws, monkeypatch, route_kwargs, env, expected_bearer
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    client = HTTPHandler()
    with (
        patch.object(client.client, "send", return_value=MagicMock(status_code=200)),
        patch.object(client.client, "build_request", wraps=client.client.build_request) as build_request,
    ):
        response = llm_passthrough_route(
            model="bedrock_mantle/us.openai.gpt-5.6-sol",
            endpoint=INVOKE_ENDPOINT,
            method="POST",
            api_base=MANTLE_API_BASE,
            json=dict(REQUEST_BODY),
            client=client,
            litellm_logging_obj=MagicMock(),
            **route_kwargs,
        )
    assert response.status_code == 200
    sent = build_request.call_args.kwargs
    assert str(sent["url"]) == f"https://bedrock-runtime.us-east-2.amazonaws.com/{INVOKE_ENDPOINT}"
    assert sent["headers"]["Authorization"] == f"Bearer {expected_bearer}"
    assert json.loads(sent["content"]) == REQUEST_BODY


def _logged_model_response(endpoint, body):
    request = httpx.Request("POST", f"https://bedrock-runtime.us-east-1.amazonaws.com/{endpoint}")
    return BedrockMantlePassthroughConfig().logging_non_streaming_response(
        model="us.openai.gpt-5.6-sol",
        custom_llm_provider="bedrock_mantle",
        httpx_response=httpx.Response(200, json=body, request=request),
        request_data={"messages": [{"role": "user", "content": [{"text": "say pong"}]}]},
        logging_obj=MagicMock(),
        endpoint=endpoint,
    )


def test_converse_logging_parses_the_converse_response_shape():
    result = _logged_model_response(
        CONVERSE_ENDPOINT,
        {
            "metrics": {"latencyMs": 800.0},
            "output": {"message": {"content": [{"text": "pong"}], "role": "assistant"}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 8, "outputTokens": 5, "totalTokens": 13},
        },
    )
    assert result.choices[0].message.content == "pong"
    assert result.usage.prompt_tokens == 8
    assert result.usage.completion_tokens == 5


def test_invoke_logging_parses_the_openai_chat_response_shape():
    result = _logged_model_response(
        INVOKE_ENDPOINT,
        {
            "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "pong", "role": "assistant"}}],
            "created": 1787677792,
            "id": "chatcmpl-regression",
            "model": "us.openai.gpt-5.6-sol",
            "object": "chat.completion",
            "usage": {"completion_tokens": 5, "prompt_tokens": 8, "total_tokens": 13},
        },
    )
    assert result.choices[0].message.content == "pong"
    assert result.usage.prompt_tokens == 8
    assert result.usage.completion_tokens == 5
