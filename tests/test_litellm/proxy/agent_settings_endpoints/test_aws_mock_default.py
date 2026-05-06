"""
The `_aws_mock_enabled` flag controls whether `POST /v2/agent-vm-config/
test-connection` returns a synthetic success response or actually calls
`sts:GetCallerIdentity`. The default MUST be off, so a freshly deployed
production proxy does not silently green-light invalid AWS credentials.

This is the fix for the P1 Greptile flagged on PR #27332 — previously the
default was `"1"` (mock-on), so an operator who saved bad credentials
would see a fake success and only discover the failure when VMs failed
to launch.
"""

import pytest

from litellm.proxy.agent_settings_endpoints.vm_config_endpoints import (
    _aws_mock_enabled,
)


class TestAwsMockDefault:
    def test_mock_disabled_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("LITELLM_CLOUD_AGENT_MOCK_AWS", raising=False)
        assert _aws_mock_enabled() is False

    def test_mock_disabled_when_env_zero(self, monkeypatch):
        monkeypatch.setenv("LITELLM_CLOUD_AGENT_MOCK_AWS", "0")
        assert _aws_mock_enabled() is False

    def test_mock_disabled_for_arbitrary_truthy_strings(self, monkeypatch):
        # We require an EXPLICIT "1" — no fuzzy-truthy parsing. A typo
        # like "true" or "yes" must not silently flip on the mock.
        for bad_value in ("true", "yes", "on", "TRUE", " 1 "):
            monkeypatch.setenv("LITELLM_CLOUD_AGENT_MOCK_AWS", bad_value)
            assert (
                _aws_mock_enabled() is False
            ), f"value {bad_value!r} unexpectedly enabled mock mode"

    def test_mock_enabled_only_with_explicit_one(self, monkeypatch):
        monkeypatch.setenv("LITELLM_CLOUD_AGENT_MOCK_AWS", "1")
        assert _aws_mock_enabled() is True


@pytest.fixture(autouse=True)
def _no_op_fixture():
    yield
