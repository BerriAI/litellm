"""Public voice-cloning API.

The operation is intentionally separate from :func:`litellm.speech`: cloning
creates a provider voice and returns its generated ``voice_id``.  The returned
ID can then be passed to MiniMax text-to-speech calls as the voice value.
"""

import httpx

from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.llms.minimax.voice_clone.transformation import (
    MinimaxVoiceCloneConfig,
    VoiceCloneResponse,
)
from litellm.types.llms.openai import FileTypes
from litellm.types.utils import LlmProviders


def _validate_provider(custom_llm_provider: str | None) -> None:
    if (custom_llm_provider or "minimax").lower() != "minimax":
        raise ValueError("voice cloning is currently supported only for MiniMax")


def voice_clone(
    file: FileTypes,
    voice_id: str,
    model: str = "speech-2.8-hd",
    *,
    purpose: str = "voice_clone",
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = "minimax",
    extra_headers: dict[str, str] | None = None,
    client: httpx.Client | None = None,
) -> VoiceCloneResponse:
    """Upload an audio sample and create a MiniMax custom voice.

    Args:
        file: Audio bytes, a path, an open binary file, or a
            ``(filename, content, content_type)`` tuple.
        voice_id: Caller-chosen custom voice identifier.
        model: One of MiniMax's supported speech models.
        purpose: ``voice_clone`` for cloning audio or ``prompt_audio`` for
            prompt audio uploads.
        api_base: Optional MiniMax host (international or China), with or
            without a trailing ``/v1``.
    """
    _validate_provider(custom_llm_provider)
    config = MinimaxVoiceCloneConfig
    headers = config.validate_environment(headers=extra_headers, api_key=api_key)
    upload_files, upload_data = config.transform_upload_request(file=file, purpose=purpose)
    clone_body = config.transform_clone_request(file_id="pending", voice_id=voice_id, model=model)
    # ``pending`` is replaced after the upload; validating the other required
    # clone fields before network I/O keeps malformed requests deterministic.
    clone_body.pop("file_id")

    timeout_value = timeout if timeout is not None else 600.0
    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout_value)
    try:
        upload_response = http_client.post(
            config.get_complete_url(api_base=api_base, operation="upload"),
            headers=headers,
            files=upload_files,
            data=upload_data,
            timeout=timeout_value,
        )
        file_id = config.transform_upload_response(upload_response)
        clone_body = config.transform_clone_request(file_id=file_id, voice_id=voice_id, model=model)
        clone_response = http_client.post(
            config.get_complete_url(api_base=api_base, operation="clone"),
            headers={**headers, "Content-Type": "application/json"},
            json=clone_body,
            timeout=timeout_value,
        )
        return config.transform_clone_response(clone_response, file_id=file_id, model=model)
    finally:
        if owns_client:
            http_client.close()


async def avoice_clone(
    file: FileTypes,
    voice_id: str,
    model: str = "speech-2.8-hd",
    *,
    purpose: str = "voice_clone",
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = "minimax",
    extra_headers: dict[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> VoiceCloneResponse:
    """Asynchronously upload and clone a MiniMax voice."""
    _validate_provider(custom_llm_provider)
    config = MinimaxVoiceCloneConfig
    headers = config.validate_environment(headers=extra_headers, api_key=api_key)
    upload_files, upload_data = config.transform_upload_request(file=file, purpose=purpose)
    config.transform_clone_request(file_id="pending", voice_id=voice_id, model=model)
    timeout_value = timeout if timeout is not None else 600.0
    http_client = client or get_async_httpx_client(
        llm_provider=LlmProviders.MINIMAX,
        params={"timeout": timeout_value},
    )
    upload_response = await http_client.post(
        config.get_complete_url(api_base=api_base, operation="upload"),
        headers=headers,
        files=upload_files,
        data=upload_data,
        timeout=timeout_value,
    )
    file_id = config.transform_upload_response(upload_response)
    clone_body = config.transform_clone_request(file_id=file_id, voice_id=voice_id, model=model)
    clone_response = await http_client.post(
        config.get_complete_url(api_base=api_base, operation="clone"),
        headers={**headers, "Content-Type": "application/json"},
        json=clone_body,
        timeout=timeout_value,
    )
    return config.transform_clone_response(clone_response, file_id=file_id, model=model)
