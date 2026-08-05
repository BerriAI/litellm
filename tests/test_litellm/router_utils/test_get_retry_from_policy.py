import litellm
from litellm.router_utils.get_retry_from_policy import (
    get_num_retries_from_retry_policy,
)
from litellm.types.router import RetryPolicy


def _service_unavailable_error() -> litellm.ServiceUnavailableError:
    return litellm.ServiceUnavailableError(
        message="model is down",
        llm_provider="openai",
        model="gpt-5.6",
    )


def _internal_server_error() -> litellm.InternalServerError:
    return litellm.InternalServerError(
        message="upstream 500",
        llm_provider="openai",
        model="gpt-5.6",
    )


def test_service_unavailable_error_retries_honored():
    policy = RetryPolicy(ServiceUnavailableErrorRetries=0)

    assert (
        get_num_retries_from_retry_policy(
            exception=_service_unavailable_error(),
            retry_policy=policy,
        )
        == 0
    )


def test_service_unavailable_error_retries_nonzero():
    policy = RetryPolicy(ServiceUnavailableErrorRetries=4)

    assert (
        get_num_retries_from_retry_policy(
            exception=_service_unavailable_error(),
            retry_policy=policy,
        )
        == 4
    )


def test_internal_server_error_retries_honored():
    policy = RetryPolicy(InternalServerErrorRetries=0)

    assert (
        get_num_retries_from_retry_policy(
            exception=_internal_server_error(),
            retry_policy=policy,
        )
        == 0
    )


def test_service_unavailable_not_covered_by_internal_server_error_retries():
    policy = RetryPolicy(InternalServerErrorRetries=0)

    assert (
        get_num_retries_from_retry_policy(
            exception=_service_unavailable_error(),
            retry_policy=policy,
        )
        is None
    )


def test_internal_server_error_not_covered_by_service_unavailable_retries():
    policy = RetryPolicy(ServiceUnavailableErrorRetries=0)

    assert (
        get_num_retries_from_retry_policy(
            exception=_internal_server_error(),
            retry_policy=policy,
        )
        is None
    )


def test_service_unavailable_error_retries_from_dict_policy():
    assert (
        get_num_retries_from_retry_policy(
            exception=_service_unavailable_error(),
            retry_policy={"ServiceUnavailableErrorRetries": 0},
        )
        == 0
    )


def test_service_unavailable_error_retries_from_model_group_policy():
    assert (
        get_num_retries_from_retry_policy(
            exception=_service_unavailable_error(),
            model_group="gpt-5.6",
            model_group_retry_policy={"gpt-5.6": RetryPolicy(ServiceUnavailableErrorRetries=1)},
        )
        == 1
    )
