"""Types for the compiled ``litellm.rust_bridge._native`` PyO3 extension.

Keep in sync with the surface-pinning test in
litellm-rust/crates/python-bridge/src/lib.rs and the ``__text_signature__``
contract tests in litellm-rust/crates/python-bridge/src/routes/definition.rs.

``_panic_for_test`` is only built with the ``panic-test`` cargo feature and is
absent from default wheels, so it is intentionally not stubbed.
"""

from collections.abc import Coroutine, Mapping, Sequence

__version__: str

class RustBridgeDeclined(Exception):
    """The route declined before calling the provider; the host may retry on its own path."""

    def __init__(self, message: str) -> None: ...

class RustUpstreamError(Exception):
    """The provider call was issued and failed.

    Args are ``(status, message)``; ``status`` is 0 when there was no HTTP response.
    """

    def __init__(self, status: int, message: str) -> None: ...

class ResponsesWebSocketConnection:
    @classmethod
    def connect(
        cls,
        url: str,
        headers: Mapping[str, str] | None = ...,
        timeout_seconds: float | None = ...,
    ) -> Coroutine[object, object, ResponsesWebSocketConnection]: ...
    def send_text(self, text: str) -> Coroutine[object, object, None]: ...
    def recv_text(self) -> Coroutine[object, object, str | None]: ...
    def close(self) -> Coroutine[object, object, None]: ...

def ocr(
    model: str,
    document: dict[str, object],
    api_key: str | None = ...,
    api_base: str | None = ...,
    custom_llm_provider: str | None = ...,
    extra_headers: Mapping[str, object] | None = ...,
    optional_params: Mapping[str, object] | None = ...,
    timeout_seconds: float | None = ...,
    trace: bool = ...,
) -> dict[str, object]: ...
def aocr(
    model: str,
    document: dict[str, object],
    api_key: str | None = ...,
    api_base: str | None = ...,
    custom_llm_provider: str | None = ...,
    extra_headers: Mapping[str, object] | None = ...,
    optional_params: Mapping[str, object] | None = ...,
    timeout_seconds: float | None = ...,
    trace: bool = ...,
) -> Coroutine[object, object, dict[str, object]]: ...
def transcription(
    model: str,
    audio: dict[str, object],
    api_key: str | None = ...,
    api_base: str | None = ...,
    custom_llm_provider: str | None = ...,
    extra_headers: Mapping[str, object] | None = ...,
    optional_params: Mapping[str, object] | None = ...,
    timeout_seconds: float | None = ...,
    trace: bool = ...,
) -> dict[str, object]: ...
def atranscription(
    model: str,
    audio: dict[str, object],
    api_key: str | None = ...,
    api_base: str | None = ...,
    custom_llm_provider: str | None = ...,
    extra_headers: Mapping[str, object] | None = ...,
    optional_params: Mapping[str, object] | None = ...,
    timeout_seconds: float | None = ...,
    trace: bool = ...,
) -> Coroutine[object, object, dict[str, object]]: ...
def messages(
    model: str,
    body: dict[str, object],
    api_key: str | None = ...,
    api_base: str | None = ...,
    custom_llm_provider: str | None = ...,
    extra_headers: Mapping[str, object] | None = ...,
    timeout_seconds: float | None = ...,
    trace: bool = ...,
) -> dict[str, object]: ...
def amessages(
    model: str,
    body: dict[str, object],
    api_key: str | None = ...,
    api_base: str | None = ...,
    custom_llm_provider: str | None = ...,
    extra_headers: Mapping[str, object] | None = ...,
    timeout_seconds: float | None = ...,
    trace: bool = ...,
) -> Coroutine[object, object, dict[str, object]]: ...
def chat_completions_decline(
    model: str,
    messages: Sequence[object],
    optional_params: Mapping[str, object] | None = ...,
    custom_llm_provider: str | None = ...,
) -> str | None: ...
def chat_completions(
    model: str,
    messages: Sequence[object],
    optional_params: Mapping[str, object] | None = ...,
    api_key: str | None = ...,
    api_base: str | None = ...,
    custom_llm_provider: str | None = ...,
    extra_headers: Mapping[str, object] | None = ...,
    timeout_seconds: float | None = ...,
    trace: bool = ...,
) -> dict[str, object]: ...
def achat_completions(
    model: str,
    messages: Sequence[object],
    optional_params: Mapping[str, object] | None = ...,
    api_key: str | None = ...,
    api_base: str | None = ...,
    custom_llm_provider: str | None = ...,
    extra_headers: Mapping[str, object] | None = ...,
    timeout_seconds: float | None = ...,
    trace: bool = ...,
) -> Coroutine[object, object, dict[str, object]]: ...
def gil_stats() -> dict[str, object]: ...
def build_info() -> dict[str, object]: ...
