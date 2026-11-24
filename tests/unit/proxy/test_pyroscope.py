"""Unit tests for ProxyStartupEvent._init_pyroscope (Grafana Pyroscope profiling)."""

import os
import sys
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy.proxy_server import ProxyStartupEvent
from litellm.secret_managers.main import get_secret_str as real_get_secret_str


def _mock_pyroscope_module():
    """Return a mock module so 'import pyroscope' succeeds in _init_pyroscope."""
    m = MagicMock()
    m.configure = MagicMock()
    return m


def _patch_pyroscope_grafana_secrets(user: Optional[str], token: Optional[str]):
    """Patch proxy_server.get_secret_str for Grafana keys; defer other secrets to the real helper."""

    def side_effect(secret_name: str, default_value=None):
        if secret_name == "PYROSCOPE_GRAFANA_USER":
            return user
        if secret_name == "PYROSCOPE_GRAFANA_API_TOKEN":
            return token
        return real_get_secret_str(secret_name, default_value)

    return patch(
        "litellm.proxy.proxy_server.get_secret_str",
        side_effect=side_effect,
    )


def test_init_pyroscope_returns_cleanly_when_disabled():
    """When LITELLM_ENABLE_PYROSCOPE is false, _init_pyroscope returns without error."""
    with (
        patch(
            "litellm.proxy.proxy_server.get_secret_bool",
            return_value=False,
        ),
        patch.dict(
            os.environ,
            {"LITELLM_ENABLE_PYROSCOPE": "false"},
            clear=False,
        ),
    ):
        ProxyStartupEvent._init_pyroscope()


def test_init_pyroscope_raises_when_enabled_but_missing_app_name():
    """When LITELLM_ENABLE_PYROSCOPE is true but PYROSCOPE_APP_NAME is not set, raises ValueError."""
    mock_pyroscope = _mock_pyroscope_module()
    with (
        patch(
            "litellm.proxy.proxy_server.get_secret_bool",
            return_value=True,
        ),
        patch.dict(
            sys.modules,
            {"pyroscope": mock_pyroscope},
        ),
        patch.dict(
            os.environ,
            {
                "LITELLM_ENABLE_PYROSCOPE": "true",
                "PYROSCOPE_APP_NAME": "",
                "PYROSCOPE_SERVER_ADDRESS": "http://localhost:4040",
            },
            clear=False,
        ),
    ):
        with pytest.raises(ValueError, match="PYROSCOPE_APP_NAME"):
            ProxyStartupEvent._init_pyroscope()


def test_init_pyroscope_raises_when_enabled_but_missing_server_address():
    """When LITELLM_ENABLE_PYROSCOPE is true but PYROSCOPE_SERVER_ADDRESS is not set, raises ValueError."""
    mock_pyroscope = _mock_pyroscope_module()
    with (
        patch(
            "litellm.proxy.proxy_server.get_secret_bool",
            return_value=True,
        ),
        patch.dict(
            sys.modules,
            {"pyroscope": mock_pyroscope},
        ),
        patch.dict(
            os.environ,
            {
                "LITELLM_ENABLE_PYROSCOPE": "true",
                "PYROSCOPE_APP_NAME": "myapp",
                "PYROSCOPE_SERVER_ADDRESS": "",
            },
            clear=False,
        ),
    ):
        with pytest.raises(ValueError, match="PYROSCOPE_SERVER_ADDRESS"):
            ProxyStartupEvent._init_pyroscope()


def test_init_pyroscope_raises_when_sample_rate_invalid():
    """When PYROSCOPE_SAMPLE_RATE is not a number, raises ValueError."""
    mock_pyroscope = _mock_pyroscope_module()
    with (
        patch(
            "litellm.proxy.proxy_server.get_secret_bool",
            return_value=True,
        ),
        patch.dict(
            sys.modules,
            {"pyroscope": mock_pyroscope},
        ),
        patch.dict(
            os.environ,
            {
                "LITELLM_ENABLE_PYROSCOPE": "true",
                "PYROSCOPE_APP_NAME": "myapp",
                "PYROSCOPE_SERVER_ADDRESS": "http://localhost:4040",
                "PYROSCOPE_SAMPLE_RATE": "not-a-number",
                "PYROSCOPE_GRAFANA_API_TOKEN": "",
                "PYROSCOPE_GRAFANA_USER": "",
            },
            clear=False,
        ),
    ):
        with pytest.raises(ValueError, match="PYROSCOPE_SAMPLE_RATE"):
            ProxyStartupEvent._init_pyroscope()


def test_init_pyroscope_accepts_integer_sample_rate():
    """When enabled with valid config and integer sample rate, configures pyroscope."""
    mock_pyroscope = _mock_pyroscope_module()
    with (
        patch(
            "litellm.proxy.proxy_server.get_secret_bool",
            return_value=True,
        ),
        patch.dict(
            sys.modules,
            {"pyroscope": mock_pyroscope},
        ),
        patch.dict(
            os.environ,
            {
                "LITELLM_ENABLE_PYROSCOPE": "true",
                "PYROSCOPE_APP_NAME": "myapp",
                "PYROSCOPE_SERVER_ADDRESS": "http://localhost:4040",
                "PYROSCOPE_SAMPLE_RATE": "100",
                "PYROSCOPE_GRAFANA_API_TOKEN": "",
                "PYROSCOPE_GRAFANA_USER": "",
            },
            clear=False,
        ),
    ):
        ProxyStartupEvent._init_pyroscope()
    mock_pyroscope.configure.assert_called_once()
    call_kw = mock_pyroscope.configure.call_args[1]
    assert call_kw["application_name"] == "myapp"
    assert call_kw["server_address"] == "http://localhost:4040"
    assert call_kw["sample_rate"] == 100


def test_init_pyroscope_accepts_float_sample_rate_parsed_as_int():
    """PYROSCOPE_SAMPLE_RATE can be a float string; it is parsed as integer."""
    mock_pyroscope = _mock_pyroscope_module()
    with (
        patch(
            "litellm.proxy.proxy_server.get_secret_bool",
            return_value=True,
        ),
        patch.dict(
            sys.modules,
            {"pyroscope": mock_pyroscope},
        ),
        patch.dict(
            os.environ,
            {
                "LITELLM_ENABLE_PYROSCOPE": "true",
                "PYROSCOPE_APP_NAME": "myapp",
                "PYROSCOPE_SERVER_ADDRESS": "http://localhost:4040",
                "PYROSCOPE_SAMPLE_RATE": "100.7",
                "PYROSCOPE_GRAFANA_API_TOKEN": "",
                "PYROSCOPE_GRAFANA_USER": "",
            },
            clear=False,
        ),
    ):
        ProxyStartupEvent._init_pyroscope()
    call_kw = mock_pyroscope.configure.call_args[1]
    assert call_kw["sample_rate"] == 100


def test_init_pyroscope_configures_grafana_cloud_basic_auth():
    """When Grafana Cloud credentials are set, passes them as Pyroscope basic auth."""
    mock_pyroscope = _mock_pyroscope_module()
    with (
        patch(
            "litellm.proxy.proxy_server.get_secret_bool",
            return_value=True,
        ),
        _patch_pyroscope_grafana_secrets("123456", "glc_test_token"),
        patch.dict(
            sys.modules,
            {"pyroscope": mock_pyroscope},
        ),
        patch.dict(
            os.environ,
            {
                "LITELLM_ENABLE_PYROSCOPE": "true",
                "PYROSCOPE_APP_NAME": "myapp",
                "PYROSCOPE_SERVER_ADDRESS": "https://profiles-prod-001.grafana.net",
            },
            clear=False,
        ),
    ):
        ProxyStartupEvent._init_pyroscope()
    call_kw = mock_pyroscope.configure.call_args[1]
    assert call_kw["basic_auth_username"] == "123456"
    assert call_kw["basic_auth_password"] == "glc_test_token"


def test_init_pyroscope_raises_when_grafana_token_missing_user():
    """When Grafana token is set without a Pyroscope user, raises ValueError."""
    mock_pyroscope = _mock_pyroscope_module()
    with (
        patch(
            "litellm.proxy.proxy_server.get_secret_bool",
            return_value=True,
        ),
        _patch_pyroscope_grafana_secrets("", "glc_test_token"),
        patch.dict(
            sys.modules,
            {"pyroscope": mock_pyroscope},
        ),
        patch.dict(
            os.environ,
            {
                "LITELLM_ENABLE_PYROSCOPE": "true",
                "PYROSCOPE_APP_NAME": "myapp",
                "PYROSCOPE_SERVER_ADDRESS": "https://profiles-prod-001.grafana.net",
            },
            clear=False,
        ),
    ):
        with pytest.raises(ValueError, match="PYROSCOPE_GRAFANA_USER"):
            ProxyStartupEvent._init_pyroscope()


def test_init_pyroscope_raises_when_grafana_user_missing_token():
    """When Grafana Pyroscope user is set without a token, raises ValueError."""
    mock_pyroscope = _mock_pyroscope_module()
    with (
        patch(
            "litellm.proxy.proxy_server.get_secret_bool",
            return_value=True,
        ),
        _patch_pyroscope_grafana_secrets("123456", ""),
        patch.dict(
            sys.modules,
            {"pyroscope": mock_pyroscope},
        ),
        patch.dict(
            os.environ,
            {
                "LITELLM_ENABLE_PYROSCOPE": "true",
                "PYROSCOPE_APP_NAME": "myapp",
                "PYROSCOPE_SERVER_ADDRESS": "https://profiles-prod-001.grafana.net",
            },
            clear=False,
        ),
    ):
        with pytest.raises(ValueError, match="PYROSCOPE_GRAFANA_API_TOKEN"):
            ProxyStartupEvent._init_pyroscope()


def test_init_pyroscope_raises_when_grafana_user_whitespace_only_with_token():
    """Whitespace-only user id does not satisfy Grafana mutual exclusion."""
    mock_pyroscope = _mock_pyroscope_module()
    with (
        patch(
            "litellm.proxy.proxy_server.get_secret_bool",
            return_value=True,
        ),
        _patch_pyroscope_grafana_secrets("   \t", "glc_test_token"),
        patch.dict(
            sys.modules,
            {"pyroscope": mock_pyroscope},
        ),
        patch.dict(
            os.environ,
            {
                "LITELLM_ENABLE_PYROSCOPE": "true",
                "PYROSCOPE_APP_NAME": "myapp",
                "PYROSCOPE_SERVER_ADDRESS": "https://profiles-prod-001.grafana.net",
            },
            clear=False,
        ),
    ):
        with pytest.raises(ValueError, match="PYROSCOPE_GRAFANA_USER"):
            ProxyStartupEvent._init_pyroscope()


def test_init_pyroscope_strips_grafana_credentials_for_basic_auth():
    """Leading/trailing whitespace on Grafana secrets is trimmed before configure."""
    mock_pyroscope = _mock_pyroscope_module()
    with (
        patch(
            "litellm.proxy.proxy_server.get_secret_bool",
            return_value=True,
        ),
        _patch_pyroscope_grafana_secrets("  123456  ", "  glc_test_token\n"),
        patch.dict(
            sys.modules,
            {"pyroscope": mock_pyroscope},
        ),
        patch.dict(
            os.environ,
            {
                "LITELLM_ENABLE_PYROSCOPE": "true",
                "PYROSCOPE_APP_NAME": "myapp",
                "PYROSCOPE_SERVER_ADDRESS": "https://profiles-prod-001.grafana.net",
            },
            clear=False,
        ),
    ):
        ProxyStartupEvent._init_pyroscope()
    call_kw = mock_pyroscope.configure.call_args[1]
    assert call_kw["basic_auth_username"] == "123456"
    assert call_kw["basic_auth_password"] == "glc_test_token"
