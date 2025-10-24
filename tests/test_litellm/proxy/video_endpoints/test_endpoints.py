import asyncio
import copy
from types import SimpleNamespace
from typing import Any, Dict

import orjson
import pytest
from starlette.requests import Request
from starlette.responses import Response

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.video_endpoints import endpoints


@pytest.mark.asyncio
async def test_video_generation_prompt_rerouting(monkeypatch):
    """Ensure video prompts are exposed to guardrails and restored afterwards."""

    async def fake_add_litellm_data_to_request(**kwargs):
        return kwargs["data"]

    async def fake_update_request_status(**_: Any) -> None:
        await asyncio.sleep(0)

    proxy_logger_calls: Dict[str, Any] = {}

    async def fake_pre_call_hook(*, user_api_key_dict, data, call_type):  # type: ignore[override]
        proxy_logger_calls["pre_call_input"] = copy.deepcopy(data)
        modified = {
            **data,
            "messages": [
                {
                    "role": "user",
                    "content": "sanitized video prompt",
                }
            ],
        }
        return modified

    async def fake_post_call_failure_hook(**_: Any) -> None:
        return None

    fake_proxy_logger = SimpleNamespace(
        pre_call_hook=fake_pre_call_hook,
        update_request_status=fake_update_request_status,
        post_call_failure_hook=fake_post_call_failure_hook,
    )

    async def fake_base_process_llm_request(self, **kwargs):  # type: ignore[override]
        class FakeResponse(dict):
            _hidden_params = {}

        return FakeResponse(result="ok")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/videos",
        "headers": [],
    }
    body = orjson.dumps({"prompt": "original video prompt"})

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(scope, receive)
    response = Response()
    user_api_key = UserAPIKeyAuth()

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.add_litellm_data_to_request",
        fake_add_litellm_data_to_request,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", fake_proxy_logger)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.version", "test-version")
    monkeypatch.setattr(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.get_custom_headers",
        classmethod(lambda *args, **kwargs: {}),
    )
    monkeypatch.setattr(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.base_process_llm_request", fake_base_process_llm_request
    )

    result = await endpoints.video_generation(
        request=request,
        fastapi_response=response,
        input_reference=None,
        user_api_key_dict=user_api_key,
    )
    await asyncio.sleep(0)

    assert result == {"result": "ok"}
    pre_call_input = proxy_logger_calls["pre_call_input"]
    assert pre_call_input["messages"][0]["content"] == "original video prompt"
    # The key test: guardrails should have processed the prompt
    assert "messages" in pre_call_input
    assert pre_call_input["messages"][0]["content"] == "original video prompt"


@pytest.mark.asyncio
async def test_video_remix_prompt_rerouting(monkeypatch):
    """Ensure video remix prompts are exposed to guardrails and restored afterwards."""

    async def fake_add_litellm_data_to_request(**kwargs):
        return kwargs["data"]

    async def fake_update_request_status(**_: Any) -> None:
        await asyncio.sleep(0)

    proxy_logger_calls: Dict[str, Any] = {}

    async def fake_pre_call_hook(*, user_api_key_dict, data, call_type):  # type: ignore[override]
        proxy_logger_calls["pre_call_input"] = copy.deepcopy(data)
        modified = {
            **data,
            "messages": [
                {
                    "role": "user",
                    "content": "sanitized remix prompt",
                }
            ],
        }
        return modified

    async def fake_post_call_failure_hook(**_: Any) -> None:
        return None

    fake_proxy_logger = SimpleNamespace(
        pre_call_hook=fake_pre_call_hook,
        update_request_status=fake_update_request_status,
        post_call_failure_hook=fake_post_call_failure_hook,
    )

    async def fake_base_process_llm_request(self, **kwargs):  # type: ignore[override]
        class FakeResponse(dict):
            _hidden_params = {}

        return FakeResponse(result="ok")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/videos/video_123/remix",
        "headers": [],
    }
    body = orjson.dumps({"prompt": "original remix prompt"})

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(scope, receive)
    response = Response()
    user_api_key = UserAPIKeyAuth()

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.add_litellm_data_to_request",
        fake_add_litellm_data_to_request,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", fake_proxy_logger)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.version", "test-version")
    monkeypatch.setattr(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.get_custom_headers",
        classmethod(lambda *args, **kwargs: {}),
    )
    monkeypatch.setattr(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.base_process_llm_request", fake_base_process_llm_request
    )

    result = await endpoints.video_remix(
        video_id="video_123",
        request=request,
        fastapi_response=response,
        user_api_key_dict=user_api_key,
    )
    await asyncio.sleep(0)

    assert result == {"result": "ok"}
    pre_call_input = proxy_logger_calls["pre_call_input"]
    assert pre_call_input["messages"][0]["content"] == "original remix prompt"
    # The key test: guardrails should have processed the prompt
    assert "messages" in pre_call_input
    assert pre_call_input["messages"][0]["content"] == "original remix prompt"


@pytest.mark.asyncio
async def test_video_generation_without_prompt(monkeypatch):
    """Test video generation when no prompt is provided."""

    async def fake_add_litellm_data_to_request(**kwargs):
        return kwargs["data"]

    async def fake_update_request_status(**_: Any) -> None:
        await asyncio.sleep(0)

    proxy_logger_calls: Dict[str, Any] = {}

    async def fake_pre_call_hook(*, user_api_key_dict, data, call_type):  # type: ignore[override]
        proxy_logger_calls["pre_call_input"] = copy.deepcopy(data)
        return data

    async def fake_post_call_failure_hook(**_: Any) -> None:
        return None

    fake_proxy_logger = SimpleNamespace(
        pre_call_hook=fake_pre_call_hook,
        update_request_status=fake_update_request_status,
        post_call_failure_hook=fake_post_call_failure_hook,
    )

    async def fake_base_process_llm_request(self, **kwargs):  # type: ignore[override]
        class FakeResponse(dict):
            _hidden_params = {}

        return FakeResponse(result="ok")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/videos",
        "headers": [],
    }
    body = orjson.dumps({"model": "sora-2"})  # No prompt

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(scope, receive)
    response = Response()
    user_api_key = UserAPIKeyAuth()

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.add_litellm_data_to_request",
        fake_add_litellm_data_to_request,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", fake_proxy_logger)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.version", "test-version")
    monkeypatch.setattr(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.get_custom_headers",
        classmethod(lambda *args, **kwargs: {}),
    )
    monkeypatch.setattr(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.base_process_llm_request", fake_base_process_llm_request
    )

    result = await endpoints.video_generation(
        request=request,
        fastapi_response=response,
        input_reference=None,
        user_api_key_dict=user_api_key,
    )
    await asyncio.sleep(0)

    assert result == {"result": "ok"}
    pre_call_input = proxy_logger_calls["pre_call_input"]
    assert "messages" not in pre_call_input


@pytest.mark.asyncio
async def test_video_remix_without_prompt(monkeypatch):
    """Test video remix when no prompt is provided."""

    async def fake_add_litellm_data_to_request(**kwargs):
        return kwargs["data"]

    async def fake_update_request_status(**_: Any) -> None:
        await asyncio.sleep(0)

    proxy_logger_calls: Dict[str, Any] = {}

    async def fake_pre_call_hook(*, user_api_key_dict, data, call_type):  # type: ignore[override]
        proxy_logger_calls["pre_call_input"] = copy.deepcopy(data)
        return data

    async def fake_post_call_failure_hook(**_: Any) -> None:
        return None

    fake_proxy_logger = SimpleNamespace(
        pre_call_hook=fake_pre_call_hook,
        update_request_status=fake_update_request_status,
        post_call_failure_hook=fake_post_call_failure_hook,
    )

    async def fake_base_process_llm_request(self, **kwargs):  # type: ignore[override]
        class FakeResponse(dict):
            _hidden_params = {}

        return FakeResponse(result="ok")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/videos/video_123/remix",
        "headers": [],
    }
    body = orjson.dumps({"model": "sora-2"})  # No prompt

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(scope, receive)
    response = Response()
    user_api_key = UserAPIKeyAuth()

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.add_litellm_data_to_request",
        fake_add_litellm_data_to_request,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", fake_proxy_logger)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.version", "test-version")
    monkeypatch.setattr(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.get_custom_headers",
        classmethod(lambda *args, **kwargs: {}),
    )
    monkeypatch.setattr(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.base_process_llm_request", fake_base_process_llm_request
    )

    result = await endpoints.video_remix(
        video_id="video_123",
        request=request,
        fastapi_response=response,
        user_api_key_dict=user_api_key,
    )
    await asyncio.sleep(0)

    assert result == {"result": "ok"}
    pre_call_input = proxy_logger_calls["pre_call_input"]
    assert "messages" not in pre_call_input