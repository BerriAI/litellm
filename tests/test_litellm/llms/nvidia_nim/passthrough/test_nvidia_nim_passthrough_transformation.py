import json
from types import MappingProxyType

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.nvidia_nim.passthrough.transformation import (
    NvidiaNimPassthroughConfig,
    nvidia_nim_router_model_in_endpoint,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

NIM_BASE = "http://nim.internal:8000"
INFER_BODY = {
    "input": [
        {"type": "image_url", "url": "data:image/png;base64,AAAA"},
        {"type": "image_url", "url": "data:image/png;base64,BBBB"},
    ]
}


@pytest.fixture(autouse=True)
def clear_nvidia_nim_env(monkeypatch):
    for env_var in ("NVIDIA_NIM_API_BASE", "NVIDIA_NIM_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.setattr(litellm, "api_key", None)


def test_provider_config_manager_resolves_nvidia_nim_passthrough_config():
    config = ProviderConfigManager.get_provider_passthrough_config(
        model="nvidia/nemoretriever-page-elements-v2", provider=LlmProviders.NVIDIA_NIM
    )

    assert isinstance(config, NvidiaNimPassthroughConfig)


@pytest.mark.parametrize(
    "api_base, endpoint, litellm_params, expected",
    [
        (NIM_BASE, "nim-page/v1/infer", {"litellm_metadata": {"model_group": "nim-page"}}, f"{NIM_BASE}/v1/infer"),
        (
            f"{NIM_BASE}/v1",
            "nim-page/v1/infer",
            {"litellm_metadata": {"model_group": "nim-page"}},
            f"{NIM_BASE}/v1/infer",
        ),
        (f"{NIM_BASE}/v1/", "/v1/infer", {}, f"{NIM_BASE}/v1/infer"),
        (NIM_BASE, "v1/infer", {}, f"{NIM_BASE}/v1/infer"),
        (f"{NIM_BASE}/v2", "v1/infer", {}, f"{NIM_BASE}/v2/v1/infer"),
        (f"{NIM_BASE}/infer", "infer", {}, f"{NIM_BASE}/infer/infer"),
        (NIM_BASE, "nvidia/nemoretriever-page-elements-v2/v1/infer", {}, f"{NIM_BASE}/v1/infer"),
    ],
)
def test_relay_url_strips_the_model_group_and_never_doubles_the_api_version(
    api_base, endpoint, litellm_params, expected
):
    url, base = NvidiaNimPassthroughConfig().get_complete_url(
        api_base=api_base,
        api_key=None,
        model="nvidia/nemoretriever-page-elements-v2",
        endpoint=endpoint,
        request_query_params=None,
        litellm_params=litellm_params,
    )

    assert str(url) == expected
    assert base == expected.removesuffix("/v1/infer").removesuffix("/infer")


def test_query_params_are_forwarded_on_the_relay_url():
    url, _ = NvidiaNimPassthroughConfig().get_complete_url(
        api_base=NIM_BASE,
        api_key=None,
        model="nvidia/nemoretriever-page-elements-v2",
        endpoint="v1/infer",
        request_query_params={"timeout": "30"},
        litellm_params={},
    )

    assert str(url) == f"{NIM_BASE}/v1/infer?timeout=30"


def test_env_api_base_is_used_when_the_deployment_has_none(monkeypatch):
    monkeypatch.setenv("NVIDIA_NIM_API_BASE", f"{NIM_BASE}/v1")

    url, _ = NvidiaNimPassthroughConfig().get_complete_url(
        api_base=None,
        api_key=None,
        model="nvidia/nemoretriever-page-elements-v2",
        endpoint="v1/infer",
        request_query_params=None,
        litellm_params={},
    )

    assert str(url) == f"{NIM_BASE}/v1/infer"


def test_missing_api_base_raises_instead_of_building_a_relative_url():
    with pytest.raises(ValueError, match="NVIDIA_NIM_API_BASE"):
        NvidiaNimPassthroughConfig().get_complete_url(
            api_base=None,
            api_key=None,
            model="nvidia/nemoretriever-page-elements-v2",
            endpoint="v1/infer",
            request_query_params=None,
            litellm_params={},
        )


def test_deployment_key_becomes_a_bearer_token_and_caller_headers_are_kept():
    caller_headers = MappingProxyType({"x-request-id": "abc"})

    headers = NvidiaNimPassthroughConfig().validate_environment(
        headers=caller_headers,
        model="nvidia/nemoretriever-page-elements-v2",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="nvapi-secret",
    )

    assert headers == {"x-request-id": "abc", "Authorization": "Bearer nvapi-secret"}


def test_self_hosted_nim_without_a_key_sends_no_authorization_header():
    headers = NvidiaNimPassthroughConfig().validate_environment(
        headers={}, model="nvidia/x", messages=[], optional_params={}, litellm_params={}, api_key=None
    )

    assert "Authorization" not in headers


def test_env_api_key_fills_in_when_the_deployment_has_none(monkeypatch):
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "nvapi-from-env")

    assert NvidiaNimPassthroughConfig.get_api_key(None) == "nvapi-from-env"
    assert NvidiaNimPassthroughConfig.get_api_key("nvapi-deployment") == "nvapi-deployment"


@pytest.mark.parametrize(
    "endpoint, router_models, expected",
    [
        ("nim-page/v1/infer", ("nim-page", "nim-table"), "nim-page"),
        ("/nim-page/v1/infer", ("nim-page",), "nim-page"),
        (
            "nvidia/nemoretriever-page-elements-v2/v1/infer",
            ("nvidia/nemoretriever-page-elements-v2",),
            "nvidia/nemoretriever-page-elements-v2",
        ),
        ("nim/v1/infer", ("nim", "nim/v1"), "nim/v1"),
        ("v1/infer", ("nim-page",), None),
        ("nim-page-elements/v1/infer", ("nim-page",), None),
        ("", ("nim-page",), None),
    ],
)
def test_router_model_in_endpoint_takes_the_longest_leading_model_group(endpoint, router_models, expected):
    assert nvidia_nim_router_model_in_endpoint(endpoint, frozenset(router_models)) == expected


@pytest.mark.parametrize("request_data, expected", [({"stream": True}, True), ({"stream": False}, False), ({}, False)])
def test_is_streaming_request_reads_the_stream_flag(request_data, expected):
    assert NvidiaNimPassthroughConfig().is_streaming_request("v1/infer", request_data) is expected


def test_non_streaming_relay_logs_the_upstream_json_body():
    response = httpx.Response(
        200,
        json={"data": [{"index": 0, "bounding_boxes": {}}]},
        request=httpx.Request("POST", f"{NIM_BASE}/v1/infer"),
    )

    result = NvidiaNimPassthroughConfig().logging_non_streaming_response(
        model="nvidia/nemoretriever-page-elements-v2",
        custom_llm_provider="nvidia_nim",
        httpx_response=response,
        request_data=INFER_BODY,
        logging_obj=None,  # pyright: ignore[reportArgumentType]  # not read for a plain passthrough body
        endpoint="v1/infer",
    )

    assert result == {"response": {"data": [{"index": 0, "bounding_boxes": {}}]}}


@pytest.mark.asyncio
async def test_object_detection_relay_sends_the_native_body_unchanged_to_v1_infer():
    upstream_requests: list[httpx.Request] = []

    def nim(request: httpx.Request) -> httpx.Response:
        upstream_requests.append(request)
        return httpx.Response(200, json={"data": [{"index": 0}, {"index": 1}]}, headers={"x-nim": "1"})

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(nim))

    response = await litellm.allm_passthrough_route(
        model="nvidia_nim/nvidia/nemoretriever-page-elements-v2",
        endpoint="nim-page/v1/infer",
        method="POST",
        api_base=f"{NIM_BASE}/v1",
        api_key="nvapi-secret",
        json=dict(INFER_BODY),
        litellm_metadata={"model_group": "nim-page"},
        client=client,
    )

    (sent,) = upstream_requests
    assert str(sent.url) == f"{NIM_BASE}/v1/infer"
    assert json.loads(sent.content) == INFER_BODY
    assert sent.headers["authorization"] == "Bearer nvapi-secret"
    assert response.status_code == 200
    assert response.headers["x-nim"] == "1"
    assert response.json() == {"data": [{"index": 0}, {"index": 1}]}
