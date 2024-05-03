"""Test batch_completion's return_exceptions."""
import pytest
import litellm

msg1 = [{"role": "user", "content": "hi 1"}]
msg2 = [{"role": "user", "content": "hi 2"}]


def test_batch_completion_return_exceptions_default():
    """Test batch_completion's return_exceptions."""
    with pytest.raises(Exception):
        _ = litellm.batch_completion(
            model="gpt-3.5-turbo",
            messages=[msg1, msg2],
            api_key="sk_xxx",  # deliberately set invalid key
            # return_exceptions=False,
        )


def test_batch_completion_return_exceptions_true():
    """Test batch_completion's return_exceptions."""
    res = litellm.batch_completion(
        model="gpt-3.5-turbo",
        messages=[msg1, msg2],
        api_key="sk_xxx",  # deliberately set invalid key
        return_exceptions=True,
    )

    assert isinstance(res[0], litellm.exceptions.AuthenticationError)
