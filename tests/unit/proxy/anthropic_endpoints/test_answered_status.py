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


@pytest.mark.parametrize(("code", "answered"), [(503, "529"), (400, "400"), (500, "500")])
def test_a_proxy_exception_answers_its_503_as_529_and_keeps_every_other_code(code: int, answered: str) -> None:
    exc = ProxyException(message="budget service unavailable", type="None", param=None, code=code)

    answered_exc = with_answered_status(exc)

    assert (type(answered_exc), answered_exc.code, answered_exc.message) == (ProxyException, answered, exc.message)
    assert exc.code == str(code)


@pytest.mark.parametrize(
    ("exc", "answered"),
    [
        pytest.param(
            litellm.ServiceUnavailableError(message="BedrockException", llm_provider="bedrock", model="m"),
            529,
            id="service-unavailable-503",
        ),
        pytest.param(_anthropic_internal_server_error(529), 529, id="anthropic-overload"),
        pytest.param(_anthropic_internal_server_error(500), 500, id="anthropic-500"),
        pytest.param(_anthropic_internal_server_error(None), 500, id="internal-server-error"),
        pytest.param(
            MidStreamFallbackError(
                message="fallbacks failed",
                model="m",
                llm_provider="anthropic",
                original_exception=_anthropic_internal_server_error(529),
            ),
            529,
            id="fallback-over-an-anthropic-overload",
        ),
        pytest.param(
            litellm.RateLimitError(message="slow down", llm_provider="anthropic", model="m"), 429, id="rate-limit"
        ),
    ],
)
def test_a_litellm_exception_answers_the_status_anthropic_gives_its_failure(
    exc: openai.APIStatusError, answered: int
) -> None:
    raised_status = exc.status_code

    answered_exc = with_answered_status(exc)

    assert (type(answered_exc), answered_exc.status_code, str(answered_exc)) == (type(exc), answered, str(exc))
    assert exc.status_code == raised_status


def _raise_a_bedrock_overload_caused_by_a_reset() -> None:
    try:
        raise ConnectionError("upstream reset")
    except ConnectionError as upstream:
        raise litellm.ServiceUnavailableError(
            message="BedrockException", llm_provider="bedrock", model="m"
        ) from upstream


def test_the_answered_copy_keeps_the_raised_exceptions_cause_and_traceback() -> None:
    with pytest.raises(litellm.ServiceUnavailableError) as raised:
        _raise_a_bedrock_overload_caused_by_a_reset()

    answered_exc = with_answered_status(raised.value)

    assert isinstance(raised.value.__cause__, ConnectionError)
    assert (answered_exc.__cause__, answered_exc.__traceback__) == (raised.value.__cause__, raised.value.__traceback__)


def test_an_exception_without_a_status_passes_through_unchanged() -> None:
    exc = ValueError("no status")

    assert with_answered_status(exc) is exc
    assert not hasattr(exc, "status_code")
