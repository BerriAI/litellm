from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pytest

try:
    from google import genai
    from google.genai import types

    GOOGLE_GENAI_SDK_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_SDK_AVAILABLE = False

MASTER_KEY = "sk-1234"
PROMPT = "Reply with only the single word: pong"


def has_vertex_credentials() -> bool:
    credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if credentials_file and os.path.isfile(credentials_file):
        return True
    return bool(
        os.environ.get("VERTEX_AI_PRIVATE_KEY", "")
        and os.environ.get("VERTEX_AI_PRIVATE_KEY_ID", "")
    )


def _make_client(proxy_url: str) -> "genai.Client":
    return genai.Client(
        api_key=MASTER_KEY,
        http_options={"base_url": proxy_url},
    )


def _generation_config() -> "types.GenerateContentConfig":
    return types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        top_k=20,
    )


def _collect_stream_text(chunks: List["types.GenerateContentResponse"]) -> str:
    return "".join(chunk.text for chunk in chunks if chunk.text)


class BaseGoogleGenAIProxySDKTest(ABC):
    @property
    @abstractmethod
    def proxy_model_name(self) -> str: ...

    @property
    @abstractmethod
    def model_config(self) -> Dict[str, Any]: ...

    def _skip_reason_if_credentials_missing(self) -> Optional[str]:
        model = self.model_config.get("model", "")
        if model.startswith("gemini/"):
            if not os.getenv("GEMINI_API_KEY"):
                return "GEMINI_API_KEY not set — skipping Gemini proxy SDK tests"
            return None

        if "vertex_ai" in model:
            if has_vertex_credentials():
                return None
            return "Vertex AI credentials not set — skipping Vertex AI proxy SDK tests"

        return f"Unsupported model for proxy SDK tests: {model}"

    def _require_proxy_sdk(self) -> None:
        if not GOOGLE_GENAI_SDK_AVAILABLE:
            pytest.skip("google-genai SDK not installed")
        reason = self._skip_reason_if_credentials_missing()
        if reason:
            pytest.skip(reason)

    def test_proxy_genai_sdk_non_streaming(self, google_genai_proxy_url: str) -> None:
        self._require_proxy_sdk()

        client = _make_client(google_genai_proxy_url)
        response = client.models.generate_content(
            model=self.proxy_model_name,
            contents=types.Part.from_text(text=PROMPT),
            config=_generation_config(),
        )

        assert response is not None
        assert response.text is not None
        assert len(response.text.strip()) > 0

    def test_proxy_genai_sdk_streaming_completes_without_errors(
        self, google_genai_proxy_url: str
    ) -> None:
        self._require_proxy_sdk()

        client = _make_client(google_genai_proxy_url)
        stream = client.models.generate_content_stream(
            model=self.proxy_model_name,
            contents=types.Part.from_text(text=PROMPT),
            config=_generation_config(),
        )

        chunks: List[types.GenerateContentResponse] = []
        stream_error: Optional[Exception] = None

        try:
            for chunk in stream:
                chunks.append(chunk)
        except Exception as exc:
            stream_error = exc

        assert (
            stream_error is None
        ), f"Streaming raised {type(stream_error).__name__}: {stream_error}"
        assert len(chunks) > 0, "Expected at least one streaming chunk"
        assert _collect_stream_text(chunks).strip(), "Expected non-empty streamed text"

    def test_proxy_genai_sdk_streaming_dict_style(
        self, google_genai_proxy_url: str
    ) -> None:
        self._require_proxy_sdk()

        client = _make_client(google_genai_proxy_url)
        stream = client.models.generate_content_stream(
            model=self.proxy_model_name,
            contents={"text": PROMPT},
            config={
                "temperature": 0,
                "top_p": 0.95,
                "top_k": 20,
            },
        )

        chunks = list(stream)
        assert len(chunks) > 0
        assert _collect_stream_text(chunks).strip()
