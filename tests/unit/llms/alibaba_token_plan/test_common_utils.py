import base64
from io import BytesIO
from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm.llms.alibaba_token_plan.common_utils import (
    DEFAULT_API_BASE,
    IMAGE_PATH,
    MESSAGES_PATH,
    get_api_base,
    get_api_url,
    image_reference,
    validate_headers,
)

PNG: Final = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


@pytest.fixture(autouse=True)
def clear_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALIBABA_TOKEN_PLAN_API_BASE", raising=False)
    monkeypatch.delenv("ALIBABA_TOKEN_PLAN_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)


@pytest.mark.parametrize(
    ("api_base", "expected_image_url", "expected_messages_url"),
    [
        (
            None,
            DEFAULT_API_BASE.replace("compatible-mode/v1", IMAGE_PATH),
            DEFAULT_API_BASE.replace("compatible-mode/v1", MESSAGES_PATH),
        ),
        (
            "https://gateway.example/plan/compatible-mode/v1/",
            f"https://gateway.example/plan/{IMAGE_PATH}",
            f"https://gateway.example/plan/{MESSAGES_PATH}",
        ),
    ],
)
def test_native_urls_share_the_openai_compatible_base(
    api_base: str | None, expected_image_url: str, expected_messages_url: str
) -> None:
    assert get_api_url(api_base, IMAGE_PATH) == expected_image_url
    assert get_api_url(api_base, MESSAGES_PATH) == expected_messages_url


def test_environment_base_applies_when_no_api_base_is_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_BASE", "https://env.example/compatible-mode/v1")
    assert get_api_base(None) == "https://env.example/compatible-mode/v1"
    assert get_api_base("https://explicit.example/compatible-mode/v1") == "https://explicit.example/compatible-mode/v1"


def test_headers_use_the_token_plan_key_and_fail_without_one(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(litellm.AuthenticationError, match="ALIBABA_TOKEN_PLAN_API_KEY"):
        validate_headers({}, None)
    monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_KEY", "env-key")
    assert validate_headers({"x-trace": "1"}, None) == {
        "x-trace": "1",
        "Authorization": "Bearer env-key",
        "Content-Type": "application/json",
    }
    assert validate_headers({}, "explicit-key")["Authorization"] == "Bearer explicit-key"


def test_image_reference_turns_uploads_into_data_uris(tmp_path: Path) -> None:
    data_uri: Final = f"data:image/png;base64,{base64.b64encode(PNG).decode()}"
    path: Final = tmp_path / "image.png"
    path.write_bytes(PNG)
    assert image_reference("https://images.example/a.png") == "https://images.example/a.png"
    assert image_reference(PNG) == data_uri
    assert image_reference(BytesIO(PNG)) == data_uri
    assert image_reference(("image.png", PNG, "image/png")) == data_uri
    assert image_reference(path) == data_uri
    with pytest.raises(TypeError):
        image_reference(42)
