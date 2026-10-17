from types import MappingProxyType
from typing import Final

import pytest

import litellm
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.rust_bridge import failures


class UpstreamRateLimited(Exception):
    status_code = 429
    message = "rate limited"


class NativeUpstreamFailure(Exception):
    headers: Final = (("retry-after", "7"),)


@pytest.mark.parametrize(
    ("status", "body", "error_type"),
    (
        (400, '{"error":{"message":"bad request"}}', litellm.BadRequestError),
        (429, '{"error":{"message":"rate limited"}}', litellm.RateLimitError),
    ),
)
def test_native_upstream_message_matches_the_provider_error_without_tuple_formatting(
    status: int, body: str, error_type: type[Exception]
) -> None:
    native: Final = NativeUpstreamFailure(status, body)
    model: Final = "anthropic/claude-sonnet-5"
    request: Final = MappingProxyType[str, object]({})
    expected: Final = failures.map_failure(AnthropicError(status, body), model, "anthropic", request)

    mapped: Final = failures.map_native_failure(native, model, "anthropic", request)

    assert type(mapped) is error_type
    assert str(mapped) == str(expected)
    assert isinstance(mapped, (litellm.BadRequestError, litellm.RateLimitError))
    assert mapped.status_code == status
    assert mapped.response.text == body
    assert mapped.response.headers["retry-after"] == "7"
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
