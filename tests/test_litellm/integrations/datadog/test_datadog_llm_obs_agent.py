import os
from unittest.mock import patch

import httpx

from litellm.integrations.datadog.datadog_llm_obs import DataDogLLMObsLogger


def test_datadog_llm_obs_agent_configuration():
    """
    Test that DataDog LLM Obs logger correctly configures agent endpoint.
    """
    test_env = {
        "LITELLM_DD_AGENT_HOST": "localhost",
        "LITELLM_DD_LLM_OBS_PORT": "10518",
        "DD_API_KEY": "test-api-key",  # Optional, but checking if it's preserved
    }

    # Ensure DD_SITE is NOT set to verify we don't need it in agent mode

    with patch.dict(os.environ, test_env, clear=True):
        with patch("asyncio.create_task"):  # Prevent periodic flush task from running
            dd_logger = DataDogLLMObsLogger()

        expected_url = "http://localhost:10518/api/intake/llm-obs/v1/trace/spans"
        assert dd_logger.intake_url == expected_url
        assert dd_logger.DD_API_KEY == "test-api-key"


def test_datadog_llm_obs_agent_no_api_key_ok():
    """
    Test that agent mode works WITHOUT DD_API_KEY (agent handles auth).
    """
    test_env = {
        "LITELLM_DD_AGENT_HOST": "localhost",
        # No DD_API_KEY
    }

    with patch.dict(os.environ, test_env, clear=True):
        with patch("asyncio.create_task"):
            # Should NOT raise exception anymore
            dd_logger = DataDogLLMObsLogger()

            assert dd_logger.DD_API_KEY is None
            # Default port is 8126 if not set
            expected_url = "http://localhost:8126/api/intake/llm-obs/v1/trace/spans"
            assert dd_logger.intake_url == expected_url


def test_datadog_llm_obs_direct_api_configuration():
    """
    Test that direct API configuration still works as expected.
    """
    test_env = {
        "DD_API_KEY": "direct-api-key",
        "DD_SITE": "us5.datadoghq.com",
    }

    with patch.dict(os.environ, test_env, clear=True):
        with patch("asyncio.create_task"):
            dd_logger = DataDogLLMObsLogger()

        expected_url = "https://api.us5.datadoghq.com/api/intake/llm-obs/v1/trace/spans"
        assert dd_logger.intake_url == expected_url
        assert dd_logger.DD_API_KEY == "direct-api-key"


def test_datadog_llm_obs_agent_uds_configuration():
    """
    Test that DD_TRACE_AGENT_URL with a unix:// scheme configures the logger
    to ship LLM Obs spans over a Unix domain socket. Matches the env var
    that ddtrace and the Datadog Operator's APM auto-inject already use.
    """
    test_env = {
        "DD_TRACE_AGENT_URL": "unix:///var/run/datadog/apm.socket",
    }

    # Spy on AsyncHTTPTransport so we can assert the socket path without
    # poking at httpx private attributes (which move between versions).
    original_transport_init = httpx.AsyncHTTPTransport.__init__
    captured_uds = []

    def spy_init(self, *args, **kwargs):
        captured_uds.append(kwargs.get("uds"))
        return original_transport_init(self, *args, **kwargs)

    with patch.dict(os.environ, test_env, clear=True):
        with patch("asyncio.create_task"):
            with patch.object(httpx.AsyncHTTPTransport, "__init__", spy_init):
                dd_logger = DataDogLLMObsLogger()

        # URL uses localhost since the request is delivered straight over the socket.
        expected_url = "http://localhost/api/intake/llm-obs/v1/trace/spans"
        assert dd_logger.intake_url == expected_url

        # Agent mode does not require an API key — agent handles auth.
        assert dd_logger.DD_API_KEY is None

        # Logger holds the UDS transport built from DD_TRACE_AGENT_URL.
        assert isinstance(dd_logger.uds_transport, httpx.AsyncHTTPTransport)
        assert "/var/run/datadog/apm.socket" in captured_uds


def test_datadog_llm_obs_agent_host_takes_precedence_over_uds():
    """
    If both LITELLM_DD_AGENT_HOST (TCP) and DD_TRACE_AGENT_URL (UDS) are set,
    the explicit LITELLM_DD_AGENT_HOST wins for backward compatibility.
    """
    test_env = {
        "LITELLM_DD_AGENT_HOST": "127.0.0.1",
        "DD_TRACE_AGENT_URL": "unix:///var/run/datadog/apm.socket",
    }

    with patch.dict(os.environ, test_env, clear=True):
        with patch("asyncio.create_task"):
            dd_logger = DataDogLLMObsLogger()

        expected_url = "http://127.0.0.1:8126/api/intake/llm-obs/v1/trace/spans"
        assert dd_logger.intake_url == expected_url


def test_datadog_llm_obs_agent_uds_rejects_non_absolute_path():
    """
    DD_TRACE_AGENT_URL must use the canonical `unix:///absolute/path` form
    (three slashes). The non-standard `unix://hostname/path` form is accepted
    by httpx at construction time but fails with an obscure OSError on the
    first flush — we refuse it early with an actionable error instead.
    """
    test_env = {
        "DD_TRACE_AGENT_URL": "unix://run/datadog/apm.socket",  # only 2 slashes
    }

    with patch.dict(os.environ, test_env, clear=True):
        with patch("asyncio.create_task"):
            try:
                DataDogLLMObsLogger()
            except ValueError as e:
                assert "unsupported unix:// form" in str(e)
            else:
                raise AssertionError("Expected ValueError for non-absolute uds path")


def test_datadog_llm_obs_tcp_dd_trace_agent_url_falls_back_to_direct():
    """
    A non-unix DD_TRACE_AGENT_URL (e.g. http://...) is ignored by this
    integration; the logger falls back to direct intake. Avoids accidentally
    sending LLM Obs traffic to the ddtrace HTTP endpoint when the operator
    has only injected the TCP form.
    """
    test_env = {
        "DD_TRACE_AGENT_URL": "http://datadog-agent.datadog.svc:8126",
        "DD_API_KEY": "direct-api-key",
        "DD_SITE": "us5.datadoghq.com",
    }

    with patch.dict(os.environ, test_env, clear=True):
        with patch("asyncio.create_task"):
            dd_logger = DataDogLLMObsLogger()

        expected_url = "https://api.us5.datadoghq.com/api/intake/llm-obs/v1/trace/spans"
        assert dd_logger.intake_url == expected_url
