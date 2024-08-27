import pytest
import litellm
from litellm.litellm_core_utils.ssl_utils import get_ssl_verify, get_ssl_certificate
from typing import Any


@pytest.fixture
def reset_litellm_sessions():
    # Setup: Reset the global settings before each test
    litellm.ssl_verify = True
    litellm.ssl_certificate = None

    yield

    # Teardown: Reset the global settings after each test
    litellm.ssl_verify = True
    litellm.ssl_certificate = None


@pytest.mark.parametrize(
    "env_value, expected",
    [
        # SSL_VERIFY is not set
        (None, True),
        # SSL_VERIFY is set to "true"
        ("true", True),
        ("True", True),
        # SSL_VERIFY is set to "false"
        ("false", False),
        ("False", False),
        ("path/to/custom/certificate.pem", "path/to/custom/certificate.pem"),  # SSL_VERIFY is set to a custom string
    ]
)
def test_get_ssl_verify_with_env_var_set(env_value: str, expected: Any, monkeypatch):
    if env_value is not None:
        monkeypatch.setenv("SSL_VERIFY", env_value)
    else:
        monkeypatch.delenv("SSL_VERIFY", raising=False)

    assert get_ssl_verify() == expected


@pytest.mark.usefixtures("reset_litellm_sessions")
def test_get_ssl_verify_set_through_global_var():
    litellm.ssl_verify = "path/to/certificate.pem"
    assert get_ssl_verify() == "path/to/certificate.pem"


@pytest.mark.parametrize(
    "env_value, expected",
    [
        # SSL_CERTIFICATE is not set
        (None, None),
        # SSL_CERTIFICATE is set to a custom string
        ("path/to/custom/certificate.pem", "path/to/custom/certificate.pem"),
    ]
)
def test_get_ssl_certificate_with_env_var_set(monkeypatch, env_value: str, expected: Any):
    if env_value is not None:
        monkeypatch.setenv("SSL_CERTIFICATE", env_value)
    else:
        monkeypatch.delenv("SSL_CERTIFICATE", raising=False)

    assert get_ssl_certificate() == expected


@pytest.mark.usefixtures("reset_litellm_sessions")
def test_get_ssl_certificate_set_through_global_var():
    litellm.ssl_certificate = "path/to/certificate.pem"
    assert get_ssl_certificate() == "path/to/certificate.pem"
