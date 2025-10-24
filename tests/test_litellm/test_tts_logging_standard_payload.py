import pytest

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.types.utils import CallTypes

class _DeferredTTSAdapter:
    _hidden_params = {}
    async def aiter_bytes(self, chunk_size: int = 1024):
        async def _gen():
            yield b"bytes"
        return _gen()

@pytest.mark.asyncio
async def test_aspeech_logging_builds_standard_payload_for_tts():
    logging_obj = LiteLLMLogging(
        model="gpt-4o-mini-tts",
        messages=[],
        stream=False,
        litellm_call_id="test-call",
        function_id="test-func",
        call_type=CallTypes.aspeech.value,
        start_time=None,
        kwargs={"input": "hello world"},
    )

    result = _DeferredTTSAdapter()
    await logging_obj.async_success_handler(result=result)

    assert "standard_logging_object" in logging_obj.model_call_details, (
        "standard_logging_object should be built for TTS/aspeech responses"
    )
    sl = logging_obj.model_call_details["standard_logging_object"]
    assert sl is None or isinstance(sl, dict)
