import pytest
import litellm
from litellm.router import Router
from litellm.types.utils import ModelResponse


@pytest.mark.asyncio
async def test_router_teardown_smoke(monkeypatch):
    async def fake_acompletion(**kwargs):
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
        model="alias-a", messages=[{"role": "user", "content": "hi"}]
    )
    assert isinstance(resp, ModelResponse)

    # Ensure discard does not raise
    router.discard()
