import pytest
import litellm
from litellm.router import Router
from litellm.types.utils import ModelResponse


@pytest.mark.asyncio
async def test_router_builder_param_propagation(monkeypatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        nonlocal captured
        captured = kwargs.copy()
        # Return a minimal non-streaming response
        return ModelResponse(model="dummy", choices=[{"message": {"content": "ok"}}])

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    router = Router(
        model_list=[
            {
                "model_name": "alias-a",
                "litellm_params": {"model": "openai/gpt-3.5-turbo", "api_key": "sk-test"},
            }
        ]
    )

    resp = await router.acompletion(
        model="alias-a",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.33,
        tool_choice="auto",
        timeout=1.23,
    )
    assert isinstance(resp, ModelResponse)
    assert captured.get("temperature") == 0.33
    assert captured.get("tool_choice") == "auto"
    # timeout may be overridden by deployment/router settings; ensure it's present at least
    assert "timeout" in captured or "request_timeout" in captured
