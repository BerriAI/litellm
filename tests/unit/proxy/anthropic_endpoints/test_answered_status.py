from typing import Final

import openai
import pytest

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.proxy._types import ProxyException
from litellm.proxy.anthropic_endpoints.answered_status import with_answered_status


def _anthropic_internal_server_error(provider_status_code: int | None) -> litellm.InternalServerError:
    return litellm.InternalServerError(
        message="AnthropicError - Overloaded",
        llm_provider="anthropic",
        model="claude-opus-5-5",
        provider_status_code=provider_status_code,
    )


@pytest.mark.parametrize("code", [400, 500, 503])
def test_a_proxy_exception_passes_through_unchanged(code: int) -> None:
    exc: Final = ProxyException(message="budget service unavailable", type="None", param=None, code=code)

    answered_exc: Final = with_answered_status(exc)

    assert answered_exc is exc
    assert exc.code == str(code)


@pytest.mark.parametrize(
    ("exc", "answered", "copy"),
    [
        pytest.param(
            litellm.ServiceUnavailableError(message="BedrockException", llm_provider="bedrock", model="m"),
            503,
            False,
            id="service-unavailable-503",
        ),
        pytest.param(_anthropic_internal_server_error(529), 529, True, id="anthropic-overload"),
        pytest.param(_anthropic_internal_server_error(500), 500, True, id="anthropic-500"),
        pytest.param(_anthropic_internal_server_error(None), 500, False, id="internal-server-error"),
        pytest.param(
            MidStreamFallbackError(
                message="fallbacks failed",
                model="m",
                llm_provider="anthropic",
                original_exception=_anthropic_internal_server_error(529),
            ),
            529,
            True,
            id="fallback-over-an-anthropic-overload",
        ),
        pytest.param(
            MidStreamFallbackError(
                message="fallbacks failed",
                model="m",
                llm_provider="anthropic",
                original_exception=MidStreamFallbackError(
                    message="retry failed before content",
                    model="m",
                    llm_provider="anthropic",
                    original_exception=_anthropic_internal_server_error(529),
                ),
            ),
            529,
            True,
            id="nested-fallback-over-an-anthropic-overload",
        ),
        pytest.param(
            litellm.RateLimitError(message="slow down", llm_provider="anthropic", model="m"),
            429,
            False,
            id="rate-limit",
        ),
    ],
)
def test_a_litellm_exception_answers_the_status_anthropic_gives_its_failure(
    exc: openai.APIStatusError, answered: int, copy: bool
) -> None:
    raised_status: Final = exc.status_code

    answered_exc: Final = with_answered_status(exc)

    assert (answered_exc is not exc) is copy
    assert (type(answered_exc), answered_exc.status_code, str(answered_exc)) == (type(exc), answered, str(exc))
    assert exc.status_code == raised_status


def _raise_a_bedrock_overload_caused_by_a_reset() -> None:
    try:
        raise ConnectionError("upstream reset")
    except ConnectionError as upstream:
        raise litellm.InternalServerError(
            message="AnthropicError - Overloaded",
            llm_provider="anthropic",
            model="m",
            provider_status_code=529,
        ) from upstream


def test_the_answered_copy_keeps_the_raised_exceptions_cause_and_traceback() -> None:
    with pytest.raises(litellm.InternalServerError) as raised:
        _raise_a_bedrock_overload_caused_by_a_reset()

    answered_exc: Final = with_answered_status(raised.value)

    assert isinstance(raised.value.__cause__, ConnectionError)
    assert (answered_exc.__cause__, answered_exc.__traceback__) == (raised.value.__cause__, raised.value.__traceback__)


def test_an_exception_without_a_status_passes_through_unchanged() -> None:
    exc: Final = ValueError("no status")

    assert with_answered_status(exc) is exc
    assert not hasattr(exc, "status_code")
