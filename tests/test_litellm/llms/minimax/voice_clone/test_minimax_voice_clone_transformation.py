import httpx
import pytest

from litellm.llms.minimax.voice_clone.transformation import (
    MinimaxVoiceCloneConfig,
    MinimaxVoiceCloneError,
)
from litellm.voice_clone.main import voice_clone


def response(payload: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=payload)


def test_regional_urls_support_host_and_v1_api_base() -> None:
    assert (
        MinimaxVoiceCloneConfig.get_complete_url("https://api.minimax.io", operation="upload")
        == "https://api.minimax.io/v1/files/upload"
    )
    assert (
        MinimaxVoiceCloneConfig.get_complete_url("https://api.minimaxi.com/v1", operation="clone")
        == "https://api.minimaxi.com/v1/voice_clone"
    )


def test_upload_request_contains_audio_and_voice_clone_purpose() -> None:
    files, data = MinimaxVoiceCloneConfig.transform_upload_request(
        ("sample.wav", b"audio", "audio/wav"),
    )

    assert files == {"file": ("sample.wav", b"audio", "audio/wav")}
    assert data == {"purpose": "voice_clone"}


def test_prompt_audio_purpose_is_supported() -> None:
    _, data = MinimaxVoiceCloneConfig.transform_upload_request(("sample.mp3", b"audio"), "prompt_audio")
    assert data == {"purpose": "prompt_audio"}


def test_upload_response_extracts_nested_file_id() -> None:
    file_id = MinimaxVoiceCloneConfig.transform_upload_response(
        response({"file": {"file_id": "file-123"}, "base_resp": {"status_code": 0}})
    )
    assert file_id == "file-123"


def test_clone_request_contains_only_documented_required_fields() -> None:
    assert MinimaxVoiceCloneConfig.transform_clone_request("file-123", "my_voice", "speech-2.8-hd") == {
        "file_id": "file-123",
        "voice_id": "my_voice",
        "model": "speech-2.8-hd",
    }


def test_clone_response_extracts_voice_id_and_preserves_file_and_model() -> None:
    result = MinimaxVoiceCloneConfig.transform_clone_response(
        response({"data": {"voice_id": "voice-123"}, "base_resp": {"status_code": 0}}),
        file_id="file-123",
        model="speech-2.6-hd",
    )
    assert result == {"file_id": "file-123", "voice_id": "voice-123", "model": "speech-2.6-hd"}


def test_nonzero_base_response_status_is_rejected() -> None:
    with pytest.raises(MinimaxVoiceCloneError):
        MinimaxVoiceCloneConfig.transform_clone_response(
            response(
                {
                    "base_resp": {"status_code": 1004, "status_msg": "authentication failed"},
                }
            ),
            file_id="file-123",
            model="speech-2.8-hd",
        )


def test_voice_clone_performs_upload_then_clone() -> None:
    seen: list[tuple[str, str, str | None]] = []

    def transport(request: httpx.Request) -> httpx.Response:
        body = request.content
        if request.url.path == "/v1/files/upload":
            assert b'name="purpose"' in body
            assert b"voice_clone" in body
            seen.append((request.method, request.url.path, None))
            return httpx.Response(200, json={"file_id": "file-123", "base_resp": {"status_code": 0}})
        seen.append((request.method, request.url.path, request.content.decode()))
        return httpx.Response(200, json={"voice_id": "voice-123", "base_resp": {"status_code": 0}})

    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        result = voice_clone(
            ("sample.wav", b"audio", "audio/wav"),
            "my_voice",
            model="speech-2.8-hd",
            api_key="test-key",
            client=client,
        )

    assert result == {"file_id": "file-123", "voice_id": "voice-123", "model": "speech-2.8-hd"}
    assert [(method, path) for method, path, _ in seen] == [
        ("POST", "/v1/files/upload"),
        ("POST", "/v1/voice_clone"),
    ]
    assert '"file_id":"file-123"' in (seen[1][2] or "")


@pytest.mark.parametrize("model", ["speech-2.8-turbo", "unknown"])
def test_unsupported_model_is_rejected(model: str) -> None:
    with pytest.raises(ValueError, match="Unsupported MiniMax voice-clone model"):
        MinimaxVoiceCloneConfig.transform_clone_request("file-123", "my_voice", model)
