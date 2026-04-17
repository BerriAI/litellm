import pytest

from litellm.proxy.utils import _get_openapi_url


@pytest.mark.parametrize(
    "env_vars, expected_url",
    [
        ({}, "/openapi.json"),  # default case
        ({"NO_OPENAPI": "True"}, None),  # OpenAPI disabled
    ],
)
def test_get_openapi_url(monkeypatch, env_vars, expected_url):
    # Clear relevant environment variables
    monkeypatch.delenv("NO_OPENAPI", raising=False)

    # Set test environment variables
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    result = _get_openapi_url()
    assert result == expected_url
