from types import MappingProxyType
from typing import Final

import pytest

import litellm
from litellm.rust_bridge import failures


class UpstreamRateLimited(Exception):
    status_code = 429
    message = "rate limited"


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


@pytest.mark.parametrize("provider", ("github_copilot", "bedrock", "edenai"))
def test_native_http_failure_preserves_public_identity_and_response_headers(provider: str) -> None:
    class NativeFailure(Exception):
        headers: Final = (("Retry-After", "17"), ("x-provider-trace", "first"), ("x-provider-trace", "second"))

    upstream: Final = NativeFailure(429, "rate limited")

    mapped: Final = failures.map_native_failure(upstream, f"{provider}/claude-test", provider, MappingProxyType({}))

    assert isinstance(mapped, litellm.RateLimitError)
    assert mapped.status_code == upstream.args[0]
    assert mapped.llm_provider == provider
    assert mapped.model == "claude-test"
    assert mapped.response.text == upstream.args[1]
    assert mapped.response.headers["retry-after"] == upstream.headers[0][1]
    assert mapped.response.headers.get_list("x-provider-trace") == [value for _, value in upstream.headers[1:]]
    assert mapped.__context__ is upstream
