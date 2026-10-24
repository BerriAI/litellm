import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import litellm
from litellm import CustomLLM, Router
from litellm.proxy import proxy_server
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.image_endpoints import endpoints
from litellm.types.utils import ImageObject


class Upstream(CustomLLM):
    def __init__(self, fail: bool = False):
        super().__init__()
        self.calls = 0
        self.fail = fail

    def _maybe_fail(self):
        self.calls += 1
        if self.fail:
            raise litellm.ServiceUnavailableError("503", model="x", llm_provider="x")

    async def acompletion(self, *args, **kwargs):
        self._maybe_fail()
        return litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "ok"}}])

    async def aimage_generation(self, model, prompt, model_response, **kwargs):
        self._maybe_fail()
        model_response.data = [ImageObject(url="https://example.com/out.png")]
        return model_response


@pytest.mark.asyncio
async def test_image_generation_applies_key_router_settings_fallbacks(monkeypatch):
    """Ensure key router_settings fallbacks are applied to /v1/images/generations."""
    primary, fallback = Upstream(fail=True), Upstream(fail=False)
    monkeypatch.setattr(
        litellm,
        "custom_provider_map",
        [
            {"provider": "failing", "custom_handler": primary},
            {"provider": "ok", "custom_handler": fallback},
        ],
    )
    litellm.utils.custom_llm_setup()

    edge = [{"A": ["B"]}]
    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        Router(
            model_list=[
                {"model_name": "A", "litellm_params": {"model": "failing/x", "num_retries": 0}},
                {"model_name": "B", "litellm_params": {"model": "ok/x", "num_retries": 0}},
            ],
            num_retries=0,
            fallbacks=[],
        ),
    )
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        proxy_server.ProxyLogging(user_api_key_cache=proxy_server.user_api_key_cache),
    )

    key = UserAPIKeyAuth(api_key="sk-test", router_settings={"fallbacks": edge})
    app = FastAPI()
    app.include_router(endpoints.router)
    app.add_exception_handler(ProxyException, proxy_server.openai_exception_handler)
    app.dependency_overrides[endpoints.user_api_key_auth] = lambda: key
    client = TestClient(app)

    r = client.post("/v1/images/generations", json={"model": "A", "prompt": "hi"})
    assert r.status_code == 200, r.text
    assert primary.calls == 1
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_image_generation_applies_key_router_settings_model_alias(monkeypatch):
    """Ensure key router_settings model_group_alias is applied to /v1/images/generations."""
    upstream = Upstream()
    monkeypatch.setattr(
        litellm,
        "custom_provider_map",
        [
            {"provider": "ok", "custom_handler": upstream},
        ],
    )
    litellm.utils.custom_llm_setup()

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        Router(
            model_list=[
                {"model_name": "real-model", "litellm_params": {"model": "ok/x", "num_retries": 0}},
            ],
            num_retries=0,
        ),
    )
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        proxy_server.ProxyLogging(user_api_key_cache=proxy_server.user_api_key_cache),
    )

    key = UserAPIKeyAuth(
        api_key="sk-test",
        models=["real-model"],
        router_settings={"model_group_alias": {"alias-model": "real-model"}},
    )
    app = FastAPI()
    app.include_router(endpoints.router)
    app.add_exception_handler(ProxyException, proxy_server.openai_exception_handler)
    app.dependency_overrides[endpoints.user_api_key_auth] = lambda: key
    client = TestClient(app)

    r = client.post("/v1/images/generations", json={"model": "alias-model", "prompt": "hi"})
    assert r.status_code == 200, r.text
    assert upstream.calls == 1
