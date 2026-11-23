from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Response, UploadFile

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.proxy_server import _transcription_error_headers, audio_transcriptions


def test_transcription_error_headers_extracts_model_id_from_logging_obj() -> None:
    logging_obj = LiteLLMLoggingObj(
        model="openai/whisper-1",
        messages=[],
        stream=False,
        call_type="atranscription",
        start_time=None,  # type: ignore[arg-type]
        litellm_call_id="call-123",
        function_id="fn-1",
    )
    logging_obj.litellm_params = {"model_info": {"id": "dep-whisper-1"}}
    headers = _transcription_error_headers(
        "call-123",
        {"litellm_logging_obj": logging_obj},
        RuntimeError("upstream 500"),
    )
    assert headers == {
        "x-litellm-call-id": "call-123",
        "x-litellm-model-id": "dep-whisper-1",
    }


@pytest.mark.asyncio
async def test_audio_transcriptions_5xx_preserves_x_litellm_model_id_header(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    async def fake_add_litellm_data_to_request(**kwargs: object) -> dict:
        data = dict(kwargs["data"])  # type: ignore[arg-type]
        logging_obj = LiteLLMLoggingObj(
            model="whisper",
            messages=[],
            stream=False,
            call_type="atranscription",
            start_time=None,  # type: ignore[arg-type]
            litellm_call_id=data["litellm_call_id"],
            function_id="fn-1",
        )
        logging_obj.litellm_params = {"model_info": {"id": "deployment-whisper-1"}}
        data["litellm_logging_obj"] = logging_obj
        return data

    async def fake_route_request(**kwargs: object) -> object:
        raise litellm.InternalServerError(
            message="boom",
            llm_provider="openai",
            model="whisper-1",
        )

    monkeypatch.setattr(proxy_server, "add_litellm_data_to_request", fake_add_litellm_data_to_request)
    monkeypatch.setattr(proxy_server, "route_request", fake_route_request)
    monkeypatch.setattr(proxy_server.proxy_logging_obj, "pre_call_hook", AsyncMock(side_effect=lambda **kw: kw["data"]))
    monkeypatch.setattr(proxy_server.proxy_logging_obj, "post_call_failure_hook", AsyncMock())

    req = MagicMock()
    req.headers = {"x-litellm-call-id": "call-audio-500"}
    req.form = AsyncMock(return_value={"model": "whisper"})
    upload = UploadFile(filename="audio.wav", file=io.BytesIO(b"RIFF0000WAVE"))
    user = UserAPIKeyAuth(api_key="test-key")

    with pytest.raises(ProxyException) as exc_info:
        await audio_transcriptions(
            request=req,
            fastapi_response=Response(),
            file=upload,
            user_api_key_dict=user,
        )

    assert exc_info.value.headers["x-litellm-call-id"] == "call-audio-500"
    assert exc_info.value.headers["x-litellm-model-id"] == "deployment-whisper-1"
