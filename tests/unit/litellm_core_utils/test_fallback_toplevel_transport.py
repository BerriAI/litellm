import litellm
import pytest
from litellm.litellm_core_utils.fallback_utils import async_completion_with_fallbacks
from litellm.llms.openai.openai import OpenAIChatCompletion

BASE = "http://127.0.0.1:9/v1"
HDR = "x-litellm-attempted-fallbacks"


@pytest.fixture
def upstream(monkeypatch):
    attempts: list[str] = []
    real = OpenAIChatCompletion.acompletion

    async def spy(self, *args, **kwargs):
        attempts.append(kwargs.get("model"))
        return await real(self, *args, **kwargs)

    monkeypatch.setattr(OpenAIChatCompletion, "acompletion", spy)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    return attempts


def _common():
    return dict(
        messages=[{"role": "user", "content": "hi"}],
        api_base=BASE,
        api_key="none",
        num_retries=0,
        max_retries=0,
        timeout=5,
    )


def _fb(name):
    return {"model": f"openai/{name}", "api_base": BASE, "api_key": "none"}


@pytest.mark.asyncio
async def test_toplevel_plus_nested_no_repeated_fallback(upstream):
    with pytest.raises(Exception, match="All fallback attempts failed"):
        await async_completion_with_fallbacks(
            model="openai/primary",
            fallbacks=[_fb("fb1")],
            kwargs={"fallbacks": [_fb("fb2")]},
            **_common(),
        )
    assert upstream == ["primary", "fb1", "fb2"]


@pytest.mark.asyncio
async def test_toplevel_and_nested_duplicate_fallback_deduplication(upstream):
    with pytest.raises(Exception, match="All fallback attempts failed"):
        await async_completion_with_fallbacks(
            model="openai/primary",
            fallbacks=[_fb("fb1")],
            kwargs={"fallbacks": [_fb("fb1"), _fb("fb2")]},
            **_common(),
        )
    assert upstream == ["primary", "fb1", "fb2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("form", ["top_level", "nested"])
async def test_attempted_fallbacks_header_is_1(upstream, form):
    fb = {"model": "openai/fb1", "mock_response": "fallback ok"}
    extra = {"fallbacks": [fb]} if form == "top_level" else {"kwargs": {"fallbacks": [fb]}}
    resp = await async_completion_with_fallbacks(model="openai/primary", **extra, **_common())
    assert resp.choices[0].message.content == "fallback ok"
    assert upstream == ["primary"]  # fb1 served by mock_response, never reaches transport
    headers = resp._hidden_params["additional_headers"]
    assert int(headers[HDR]) == 1
