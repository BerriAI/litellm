"""
Tests for issue #20389: RuntimeError: dictionary changed size during iteration
during model_copy(deep=True) on streaming response.

Root cause: model_copy(deep=True) delegates to copy.deepcopy which iterates
pydantic_private.  If another coroutine mutates the model concurrently,
Python 3.13+ raises RuntimeError.

Fix: _safe_model_deep_copy() catches RuntimeError and falls back to
model_validate(model_dump()) which serialises to a plain dict first.
"""

import threading
from unittest.mock import patch

from pydantic import BaseModel

from litellm.litellm_core_utils.streaming_handler import _safe_model_deep_copy
from litellm.types.utils import (
    Choices,
    Delta,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)


class TestSafeModelDeepCopy:
    """_safe_model_deep_copy produces a correct copy."""

    def test_basic_model_response(self):
        resp = ModelResponse(
            id="chatcmpl-test",
            choices=[
                Choices(
                    index=0,
                    message=Message(role="assistant", content="Hello"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        copy = _safe_model_deep_copy(resp)
        assert copy.id == "chatcmpl-test"
        assert copy.choices[0].message.content == "Hello"
        assert copy is not resp

    def test_mutation_does_not_affect_copy(self):
        resp = ModelResponse(
            id="chatcmpl-original",
            choices=[
                Choices(
                    index=0,
                    message=Message(role="assistant", content="original"),
                    finish_reason="stop",
                )
            ],
        )
        copy = _safe_model_deep_copy(resp)
        resp.choices[0].message.content = "mutated"
        assert copy.choices[0].message.content == "original"

    def test_streaming_response(self):
        resp = ModelResponseStream(
            id="chatcmpl-stream",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(role="assistant", content="chunk"),
                    finish_reason=None,
                )
            ],
        )
        copy = _safe_model_deep_copy(resp)
        assert copy.id == "chatcmpl-stream"
        assert copy.choices[0].delta.content == "chunk"

    def test_fallback_on_runtime_error(self):
        """Simulate the race condition: model_copy raises RuntimeError."""
        resp = ModelResponse(
            id="chatcmpl-race",
            choices=[
                Choices(
                    index=0,
                    message=Message(role="assistant", content="race"),
                    finish_reason="stop",
                )
            ],
        )
        resp._hidden_params = {"custom_llm_provider": "openai", "is_finished": True}
        resp._response_headers = {"x-request-id": "abc123"}
        with patch.object(
            ModelResponse,
            "model_copy",
            side_effect=RuntimeError("dictionary changed size during iteration"),
        ):
            copy = _safe_model_deep_copy(resp)
        assert copy.id == "chatcmpl-race"
        assert copy.choices[0].message.content == "race"
        assert copy is not resp
        assert copy._hidden_params["custom_llm_provider"] == "openai"
        assert copy._response_headers["x-request-id"] == "abc123"

    def test_non_dict_runtime_error_reraises(self):
        """RuntimeError unrelated to dict iteration must not be swallowed."""
        resp = ModelResponse(
            id="chatcmpl-other",
            choices=[
                Choices(
                    index=0,
                    message=Message(role="assistant", content="other"),
                    finish_reason="stop",
                )
            ],
        )
        import pytest

        with patch.object(
            ModelResponse,
            "model_copy",
            side_effect=RuntimeError("some other error"),
        ):
            with pytest.raises(RuntimeError, match="some other error"):
                _safe_model_deep_copy(resp)

    def test_usage_preserved_in_copy(self):
        resp = ModelResponse(
            id="chatcmpl-usage",
            choices=[
                Choices(
                    index=0,
                    message=Message(role="assistant", content="test"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )
        copy = _safe_model_deep_copy(resp)
        assert copy.usage.prompt_tokens == 100
        assert copy.usage.completion_tokens == 50
        assert copy.usage.total_tokens == 150

    def test_none_fields_preserved(self):
        resp = ModelResponse(
            id="chatcmpl-none",
            choices=[
                Choices(
                    index=0,
                    message=Message(role="assistant", content=None),
                    finish_reason="stop",
                )
            ],
        )
        copy = _safe_model_deep_copy(resp)
        assert copy.choices[0].message.content is None

    def test_custom_pydantic_model(self):
        """Works with any BaseModel, not just ModelResponse."""

        class SimpleModel(BaseModel):
            name: str
            value: int

        m = SimpleModel(name="test", value=42)
        copy = _safe_model_deep_copy(m)
        assert copy.name == "test"
        assert copy.value == 42
        assert copy is not m

    def test_concurrent_mutation_does_not_crash(self):
        """Simulate concurrent mutation while copying."""
        resp = ModelResponse(
            id="chatcmpl-concurrent",
            choices=[
                Choices(
                    index=0,
                    message=Message(role="assistant", content="concurrent"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        errors = []
        copies = []

        def mutator():
            for i in range(100):
                try:
                    # Mutate __pydantic_private__ directly — this is the dict
                    # that deepcopy iterates over and where the race occurs.
                    resp.__pydantic_private__["_hidden_params"] = {
                        "api_key": f"key-{i}"
                    }
                except Exception as e:
                    errors.append(e)

        def copier():
            for _ in range(50):
                try:
                    copy = _safe_model_deep_copy(resp)
                    copies.append(copy)
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=mutator)
        t2 = threading.Thread(target=copier)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # No errors should have occurred
        assert len(errors) == 0, f"Errors during concurrent copy: {errors}"
        assert len(copies) == 50
