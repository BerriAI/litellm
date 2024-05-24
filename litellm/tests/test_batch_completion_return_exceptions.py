"""https://github.com/BerriAI/litellm/pull/3397/commits/a7ec1772b1457594d3af48cdcb0a382279b841c7#diff-44852387ceb00aade916d6b314dfd5d180499e54f35209ae9c07179febe08b4b."""
"""Test batch_completion's return_exceptions."""
import litellm

msg1 = [{"role": "user", "content": "hi 1"}]
msg2 = [{"role": "user", "content": "hi 2"}]


def test_batch_completion_return_exceptions_true():
    """Test batch_completion's return_exceptions."""
    res = litellm.batch_completion(
        model="gpt-3.5-turbo",
        messages=[msg1, msg2],
        api_key="sk_xxx",  # deliberately set invalid key
    )

    assert isinstance(res[0], litellm.exceptions.AuthenticationError)