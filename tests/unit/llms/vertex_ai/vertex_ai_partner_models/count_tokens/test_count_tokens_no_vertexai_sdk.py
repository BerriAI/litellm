"""
Regression tests for #28084:

`VertexAIPartnerModels.count_tokens` (for Claude / Mistral / Llama on Vertex)
used to gate on `import vertexai` even though the actual count-tokens path goes
through `VertexAIPartnerModelsTokenCounter.handle_count_tokens_request`, which
talks to the publisher's `:rawPredict` endpoint over plain httpx and never
touches the Gemini SDK. The unused gate broke `/v1/messages/count_tokens` for
any LiteLLM install that did not pull in `google-cloud-aiplatform` (which is
not in the default `proxy` / `proxy-dev` extras).

These tests pin the absence of that gate by:

1. simulating `vertexai` being unimportable and verifying the partner-model
   path does not raise the historical "vertexai import failed" error before
   reaching the network/auth layer, and
2. asserting that import of the partner-model count-tokens handler module by
   itself does not pull `vertexai` into `sys.modules`.
"""

import sys

import pytest

from litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler import (
    VertexAIPartnerModelsTokenCounter,
)
from litellm.llms.vertex_ai.vertex_ai_partner_models.main import VertexAIPartnerModels


@pytest.mark.asyncio
async def test_count_tokens_does_not_require_vertexai_sdk(monkeypatch):
    """Even when `import vertexai` would fail, count_tokens must not raise the
    historical "vertexai import failed" gate. The downstream handler talks to
    `:rawPredict` over httpx with an access token — no Gemini SDK needed."""

    # Simulate `vertexai` being unimportable, regardless of what is actually on
    # the test environment's sys.path.
    monkeypatch.setitem(sys.modules, "vertexai", None)
    monkeypatch.setitem(sys.modules, "vertexai.preview", None)

    captured = {}

    async def fake_ensure_access_token(
        self, credentials, project_id, custom_llm_provider
    ):
        return "fake-token", "fake-project"

    def fake_build_endpoint(self, model, project_id, vertex_location, api_base=None):
        captured["model_to_endpoint"] = model
        return "https://fake-endpoint"

    monkeypatch.setattr(
        VertexAIPartnerModelsTokenCounter,
        "_ensure_access_token_async",
        fake_ensure_access_token,
    )
    monkeypatch.setattr(
        VertexAIPartnerModelsTokenCounter,
        "_build_count_tokens_endpoint",
        fake_build_endpoint,
    )

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"input_tokens": 9}

    class FakeClient:
        async def post(self, url, headers=None, json=None, **kwargs):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResponse()

    import litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler as handler_mod

    monkeypatch.setattr(
        handler_mod, "get_async_httpx_client", lambda **kwargs: FakeClient()
    )

    result = await VertexAIPartnerModels().count_tokens(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hello"}],
        litellm_params={"vertex_location": "us-east5"},
        vertex_project="test-project",
        vertex_location="us-east5",
        vertex_credentials=None,
    )

    # We should reach the publisher endpoint and parse its response, not raise
    # the vertexai-import gate.
    assert result == {
        "input_tokens": 9,
        "tokenizer_used": "vertex_ai_partner_models",
    }
    assert captured["headers"] == {"Authorization": "Bearer fake-token"}
    assert captured["model_to_endpoint"] == "claude-sonnet-4-6"


def test_handler_module_does_not_import_vertexai_sdk():
    """Importing the partner-model count-tokens handler must not load the
    Gemini SDK into sys.modules. Operators who only need Claude-on-Vertex
    token counting should not pay for `google-cloud-aiplatform`."""

    # Force-evict any prior load so this assertion measures what THIS module
    # pulls in, not what an unrelated earlier test did.
    for mod in list(sys.modules):
        if mod == "vertexai" or mod.startswith("vertexai."):
            sys.modules.pop(mod, None)

    # Re-import the handler module to verify it stays SDK-free.
    import importlib

    import litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler as handler_mod

    importlib.reload(handler_mod)

    leaked = [m for m in sys.modules if m == "vertexai" or m.startswith("vertexai.")]
    assert leaked == [], f"unexpected vertexai SDK imports: {leaked}"
