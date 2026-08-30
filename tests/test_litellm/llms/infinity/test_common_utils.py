from litellm.llms.infinity.common_utils import InfinityError


def test_infinity_error_default_headers_are_not_shared() -> None:
    first_error = InfinityError(status_code=500, message="first")
    second_error = InfinityError(status_code=500, message="second")

    assert first_error.headers is None
    assert second_error.headers is None
