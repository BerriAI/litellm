from types import MappingProxyType
from typing import Final

import httpx
import pytest

import litellm
from litellm.rust_bridge import failures


class UpstreamRateLimited(Exception):
    status_code = 429
    message = "rate limited"


class NativeHttpFailure(Exception):
    def __init__(self, status: int, body: str, headers: httpx.Headers | list[tuple[bytes, bytes]]) -> None:
        super().__init__(status, body)
        self.headers: Final = headers


@pytest.mark.parametrize("as_bytes", (False, True))
def test_native_http_failure_retains_raw_response_headers_and_body(as_bytes: bool) -> None:
    headers: Final = httpx.Headers([(b"x-repeat", b"first"), (b"x-repeat", b"second"), (b"x-opaque", b"\x80\xff")])
    native: Final = NativeHttpFailure(429, '{"message":"slow down"}', headers.raw if as_bytes else headers)

    mapped: Final = failures.map_native_failure(native, "test-model", "openai", MappingProxyType({}))

    assert isinstance(mapped, litellm.RateLimitError)
    assert mapped.response.status_code == native.args[0]
    assert mapped.response.text == native.args[1]
    names: Final = frozenset(name for name, _ in headers.raw)
    assert tuple((name, value) for name, value in mapped.response.headers.raw if name in names) == tuple(headers.raw)
    assert mapped.__context__ is native


def test_upstream_status_maps_onto_the_public_exception_contract() -> None:
    upstream: Final = UpstreamRateLimited("rate limited")

    mapped: Final = failures.map_failure(upstream, "anthropic/claude-sonnet-4-5", "anthropic", MappingProxyType({}))

    assert isinstance(mapped, litellm.RateLimitError)
    assert mapped.llm_provider == "anthropic"
    assert mapped.model == "claude-sonnet-4-5"


def test_mapper_failure_keeps_the_native_error_as_context(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(**_kwargs: object) -> Exception:
        raise ValueError("mapper broke")

    monkeypatch.setattr(litellm, "exception_type", explode)
    native_error: Final = RuntimeError("native")

    mapped: Final = failures.map_failure(native_error, "mistral/mistral-ocr-latest", "mistral", MappingProxyType({}))

    assert isinstance(mapped, ValueError)
    assert mapped.__context__ is native_error


def test_kwargs_are_handed_to_the_mapper_as_owned_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: Final[list[dict[str, object]]] = []

    def record(**kwargs: object) -> Exception:
        seen.append(dict(kwargs))
        return RuntimeError("mapped")

    monkeypatch.setattr(litellm, "exception_type", record)
    request_kwargs: Final = MappingProxyType({"metadata": {"user_id": "u"}})

    failures.map_failure(RuntimeError("native"), "gpt-4o", "openai", request_kwargs)

    assert seen[0]["completion_kwargs"] == {"metadata": {"user_id": "u"}}
    assert seen[0]["extra_kwargs"] == {"metadata": {"user_id": "u"}}
    assert seen[0]["completion_kwargs"] is not request_kwargs
    assert seen[0]["model"] == "gpt-4o"
    assert seen[0]["custom_llm_provider"] == "openai"
