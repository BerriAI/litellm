import asyncio
from typing import Any, Dict

import litellm


class DummyRouter:
    """
    Minimal stand-in for Router to test the aspeech method in isolation.
    It has only the attributes and methods that aspeech uses.
    """

    def __init__(self) -> None:
        self.default_litellm_params = {}

    async def async_get_available_deployment(
        self,
        model: str,
        messages,
        specific_deployment=None,
        request_kwargs=None,
    ):
        # Return a minimal deployment dict with litellm_params
        return {
            "litellm_params": {
                "model": model,
            }
        }

    def _update_kwargs_before_fallbacks(self, model: str, kwargs: Dict[str, Any]) -> None:
        # No-op for this test
        return

    def _get_client(self, deployment, kwargs: Dict[str, Any], client_type: str):
        # No client needed for this test
        return None

    async def aspeech(self, model: str, input: str, voice=None, **kwargs):
        # This body should match the aspeech implementation you modified in router.py
        try:
            kwargs["input"] = input

            if voice is not None:
                kwargs["voice"] = voice

            extra_body = kwargs.pop("extra_body", None)
            if isinstance(extra_body, dict):
                for k, v in extra_body.items():
                    if k not in kwargs:
                        kwargs[k] = v

            deployment = await self.async_get_available_deployment(
                model=model,
                messages=[{"role": "user", "content": "prompt"}],
                specific_deployment=kwargs.pop("specific_deployment", None),
                request_kwargs=kwargs,
            )
            self._update_kwargs_before_fallbacks(model=model, kwargs=kwargs)
            data = deployment["litellm_params"].copy()
            data["model"]  # keep same side effect as real code

            for k, v in self.default_litellm_params.items():
                if k not in kwargs:
                    kwargs[k] = v
                elif k == "metadata":
                    kwargs[k].update(v)

            # litellm.aspeech will be mocked in the test
            response = await litellm.aspeech(
                **{
                    **data,
                    "client": None,
                    **kwargs,
                }
            )
            return response
        except Exception as e:
            raise e


async def _call_aspeech_with_extra_body(monkeypatch):
    captured_kwargs: Dict[str, Any] = {}

    async def mock_aspeech(**kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        class DummyResponse:
            pass
        return DummyResponse()

    # Patch litellm.aspeech so we don't make real API calls
    monkeypatch.setattr(litellm, "aspeech", mock_aspeech)

    router = DummyRouter()

    extra_body = {
        "ref_audio": "dummy-audio",
        "ref_text": "dummy-text",
        "task_type": "Base",
    }

    # Call aspeech without voice, but with extra_body
    await router.aspeech(
        model="qwen3-tts",
        input="hello world",
        extra_body=extra_body,
    )

    return captured_kwargs


def test_aspeech_forwards_extra_body(monkeypatch):
    """
    Ensure aspeech forwards extra_body fields (e.g. ref_audio, ref_text, task_type)
    to litellm.aspeech, even when voice is omitted.
    """
    captured_kwargs = asyncio.run(_call_aspeech_with_extra_body(monkeypatch))

    # Standard fields should be present
    assert captured_kwargs["model"] == "qwen3-tts"
    assert captured_kwargs["input"] == "hello world"

    # Custom fields from extra_body must also be present
    assert captured_kwargs["ref_audio"] == "dummy-audio"
    assert captured_kwargs["ref_text"] == "dummy-text"
    assert captured_kwargs["task_type"] == "Base"
