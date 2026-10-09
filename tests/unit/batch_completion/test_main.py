import httpx
import respx

import litellm

msg1 = [{"role": "user", "content": "hi 1"}]
msg2 = [{"role": "user", "content": "hi 2"}]


def test_batch_completion_return_exceptions_true(respx_mock: respx.MockRouter):
    """Test batch_completion's return_exceptions.

    With an invalid API key, we expect an error to be returned rather than raised.
    The error type may be AuthenticationError (from API) or InternalServerError
    (from connection issues), depending on network conditions.
    """
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            401,
            json={
                "error": {
                    "message": "Incorrect API key provided: sk_xxx.",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )
    )
    res = litellm.batch_completion(
        model="gpt-3.5-turbo",
        messages=[msg1, msg2],
        api_key="sk_xxx",
    )

    assert isinstance(
        res[0],
        (
            litellm.exceptions.AuthenticationError,
            litellm.exceptions.InternalServerError,
        ),
    ), f"Expected AuthenticationError or InternalServerError, got {type(res[0])}"
