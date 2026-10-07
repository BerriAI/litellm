from __future__ import annotations

from collections.abc import Generator, Mapping
from types import MappingProxyType, SimpleNamespace
from typing import Final

import pytest

import litellm
from litellm.rust_bridge import bindings, failures


class RustFailure(Exception):
    pass


@pytest.fixture(autouse=True)
def native_failure(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: SimpleNamespace(RustFailure=RustFailure))
    yield


def report(stage: str, kind: Mapping[str, object], message: str = "native failed") -> RustFailure:
    return RustFailure({"stage": stage, "kind": dict(kind), "message": message})


def upstream(status: int, body: str = "slow down", url: str | None = "https://upstream.invalid/v1") -> RustFailure:
    return report(
        "upstream",
        {"kind": "upstream", "status": status, "headers": [["retry-after", "7"]], "body": body, "url": url},
        f"upstream request failed with status {status}: {body}",
    )


REQUEST: Final = MappingProxyType({"model": "anthropic/claude-sonnet-4-5", "api_base": "https://caller.invalid/v1"})


def test_report_reads_the_bare_native_exception_only() -> None:
    native: Final = report("prepare", {"kind": "request"}, "top_k is not supported")
    public: Final = ValueError("public")
    public.__cause__ = native

    decoded: Final = failures.report(native)

    assert decoded is not None
    assert (decoded.stage, decoded.kind.kind, decoded.message) == ("prepare", "request", "top_k is not supported")
    assert failures.report(public) is None, "a settled public exception is final, its cause is not re-read"


@pytest.mark.parametrize(
    "error",
    (
        ValueError("plain"),
        ValueError({"stage": "prepare", "kind": {"kind": "request"}, "message": "shaped like a report"}),
        RustFailure(),
        RustFailure({"stage": "nowhere"}),
    ),
)
def test_report_returns_nothing_for_other_exceptions_and_malformed_reports(error: BaseException) -> None:
    assert failures.report(error) is None


def test_upstream_failures_map_through_the_public_status_contract_with_the_provider_answer() -> None:
    native: Final = upstream(429)

    public: Final = failures.public_exception(native, REQUEST, "anthropic")

    assert isinstance(public, litellm.RateLimitError)
    assert public.status_code == 429
    assert public.response.headers["retry-after"] == "7"
    assert public.response.text == "slow down"
    assert str(public.response.request.url) == "https://upstream.invalid/v1"
    assert public.llm_provider == "anthropic"
    assert public.model == "claude-sonnet-4-5"
    assert public.__cause__ is native


@pytest.mark.parametrize(
    ("fields", "expected"),
    (
        ({"model": "m", "api_base": "https://base.invalid/v1"}, "https://base.invalid/v1"),
        ({"model": "m", "api_base": None, "base_url": "https://alias.invalid/v1"}, "https://alias.invalid/v1"),
        ({"model": "m", "api_base": "", "base_url": "https://alias.invalid/v1"}, "https://alias.invalid/v1"),
        ({"model": "m"}, "https://docs.litellm.ai/docs"),
    ),
)
def test_upstream_failures_without_a_url_fall_back_to_the_callers_endpoint(
    fields: Mapping[str, object], expected: str
) -> None:
    public: Final = failures.public_exception(upstream(401, "nope", url=None), fields, "openai")

    assert isinstance(public, litellm.AuthenticationError)
    assert str(public.response.request.url) == expected


@pytest.mark.parametrize(
    ("kind", "expected"),
    (
        ("request", litellm.BadRequestError),
        ("unsupported", litellm.UnsupportedParamsError),
        ("auth", litellm.AuthenticationError),
        ("timeout", litellm.Timeout),
        ("connection", litellm.APIConnectionError),
        ("response", litellm.APIError),
        ("internal", litellm.APIError),
    ),
)
def test_each_kind_has_one_public_exception_naming_the_bare_model_and_provider(
    kind: str, expected: type[Exception]
) -> None:
    native: Final = report("prepare", {"kind": kind}, "what went wrong")

    public: Final = failures.public_exception(native, REQUEST, None)

    assert type(public) is expected
    assert "what went wrong" in str(public)
    assert getattr(public, "model") == "claude-sonnet-4-5"
    assert getattr(public, "llm_provider") == "anthropic"
    assert public.__cause__ is native


def test_public_exception_leaves_non_native_errors_unchanged() -> None:
    error: Final = RuntimeError("bridge exploded")

    assert failures.public_exception(error, REQUEST, "mistral") is error


def test_mapper_failure_keeps_the_native_error_as_context(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(**_kwargs: object) -> Exception:
        raise ValueError("mapper broke")

    monkeypatch.setattr(litellm, "exception_type", explode)
    native_error: Final = RuntimeError("native")

    mapped: Final = failures.map_failure(native_error, "mistral/mistral-ocr-latest", "mistral", MappingProxyType({}))

    assert isinstance(mapped, ValueError)
    assert mapped.__context__ is native_error
