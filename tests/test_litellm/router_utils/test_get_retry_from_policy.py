from types import MappingProxyType
from typing import Final

import pytest

import litellm
from litellm.router_utils.get_retry_from_policy import get_num_retries_from_retry_policy
from litellm.types.router import RetryPolicy

_EXCEPTION_FOR_FIELD: Final = MappingProxyType(
    {
        "BadRequestErrorRetries": litellm.BadRequestError,
        "AuthenticationErrorRetries": litellm.AuthenticationError,
        "TimeoutErrorRetries": litellm.Timeout,
        "RateLimitErrorRetries": litellm.RateLimitError,
        "ContentPolicyViolationErrorRetries": litellm.ContentPolicyViolationError,
        "InternalServerErrorRetries": litellm.InternalServerError,
        "ServiceUnavailableErrorRetries": litellm.ServiceUnavailableError,
    }
)

_SPECIFIC_FIELDS: Final = tuple(name for name in RetryPolicy.model_fields if name != "DefaultRetries")


def _error(exception_type: type[Exception]) -> Exception:
    return exception_type(message="boom", llm_provider="openai", model="gpt-5.6")


@pytest.mark.parametrize("field", _SPECIFIC_FIELDS)
def test_every_specific_field_controls_retries_for_its_exception(field: str):
    exception: Final = _error(_EXCEPTION_FOR_FIELD[field])

    assert get_num_retries_from_retry_policy(exception=exception, retry_policy=RetryPolicy(**{field: 0})) == 0
    assert get_num_retries_from_retry_policy(exception=exception, retry_policy=RetryPolicy(**{field: 4})) == 4


@pytest.mark.parametrize("field", _SPECIFIC_FIELDS)
def test_specific_field_does_not_apply_to_unrelated_exceptions(field: str):
    policy: Final = RetryPolicy(**{field: 0})
    unrelated: Final = tuple(
        exception_type
        for name, exception_type in _EXCEPTION_FOR_FIELD.items()
        if name != field and not issubclass(exception_type, _EXCEPTION_FOR_FIELD[field])
    )

    for exception_type in unrelated:
        assert get_num_retries_from_retry_policy(exception=_error(exception_type), retry_policy=policy) is None


def test_subclass_prefers_its_own_field_over_the_parent_field():
    policy: Final = RetryPolicy(BadRequestErrorRetries=5, ContentPolicyViolationErrorRetries=1)

    assert (
        get_num_retries_from_retry_policy(exception=_error(litellm.ContentPolicyViolationError), retry_policy=policy)
        == 1
    )
    assert get_num_retries_from_retry_policy(exception=_error(litellm.BadRequestError), retry_policy=policy) == 5


def test_subclass_falls_back_to_the_parent_field():
    policy: Final = RetryPolicy(BadRequestErrorRetries=5)

    assert (
        get_num_retries_from_retry_policy(exception=_error(litellm.ContentPolicyViolationError), retry_policy=policy)
        == 5
    )


@pytest.mark.parametrize("exception_type", (litellm.BadGatewayError, litellm.NotFoundError))
def test_default_retries_covers_exceptions_without_a_specific_field(exception_type: type[Exception]):
    exception: Final = _error(exception_type)

    assert get_num_retries_from_retry_policy(exception=exception, retry_policy=RetryPolicy(DefaultRetries=0)) == 0
    assert (
        get_num_retries_from_retry_policy(
            exception=exception, retry_policy=RetryPolicy(ServiceUnavailableErrorRetries=0)
        )
        is None
    )


def test_specific_field_wins_over_default_retries():
    policy: Final = RetryPolicy(DefaultRetries=0, RateLimitErrorRetries=3)

    assert get_num_retries_from_retry_policy(exception=_error(litellm.RateLimitError), retry_policy=policy) == 3
    assert get_num_retries_from_retry_policy(exception=_error(litellm.BadGatewayError), retry_policy=policy) == 0


def test_default_retries_applies_when_the_specific_field_is_unset():
    policy: Final = RetryPolicy(DefaultRetries=2)

    assert (
        get_num_retries_from_retry_policy(exception=_error(litellm.ServiceUnavailableError), retry_policy=policy) == 2
    )


def test_empty_policy_matches_nothing():
    assert (
        get_num_retries_from_retry_policy(exception=_error(litellm.ServiceUnavailableError), retry_policy=RetryPolicy())
        is None
    )
    assert (
        get_num_retries_from_retry_policy(exception=_error(litellm.ServiceUnavailableError), retry_policy=None) is None
    )


def test_dict_policy_is_accepted():
    assert (
        get_num_retries_from_retry_policy(
            exception=_error(litellm.ServiceUnavailableError),
            retry_policy={"ServiceUnavailableErrorRetries": 0},
        )
        == 0
    )


def test_model_group_policy_replaces_the_global_policy():
    exception: Final = _error(litellm.ServiceUnavailableError)
    global_policy: Final = RetryPolicy(ServiceUnavailableErrorRetries=5)

    assert (
        get_num_retries_from_retry_policy(
            exception=exception,
            retry_policy=global_policy,
            model_group="gpt-5.6",
            model_group_retry_policy={"gpt-5.6": {"ServiceUnavailableErrorRetries": 1}},
        )
        == 1
    )
    assert (
        get_num_retries_from_retry_policy(
            exception=exception,
            retry_policy=global_policy,
            model_group="gpt-5.6",
            model_group_retry_policy={"gpt-5.6": RetryPolicy(RateLimitErrorRetries=1)},
        )
        is None
    )
    assert (
        get_num_retries_from_retry_policy(
            exception=exception,
            retry_policy=global_policy,
            model_group="other-group",
            model_group_retry_policy={"gpt-5.6": RetryPolicy(ServiceUnavailableErrorRetries=1)},
        )
        == 5
    )
