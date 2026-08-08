"""
Mocked tests for the `AgentVMProvider` factory.

Covers Validation #1 from LIT-2878: factory returns the right impl per
config value and rejects unknown values. Validation #6 (provider swap is
config-only) is also exercised here — switching `vm_provider` from `noop` to
`ec2` requires no code path change beyond the factory.
"""

from __future__ import annotations

import pytest

from litellm.proxy.agent_session_endpoints.vm_providers import (
    EC2Provider,
    NoopProvider,
    SUPPORTED_PROVIDERS,
    get_vm_provider,
)


def test_factory_default_is_noop():
    """No `agent_settings` block at all → noop provider."""
    provider = get_vm_provider(None)
    assert isinstance(provider, NoopProvider)
    assert provider.name == "noop"


def test_factory_explicit_noop():
    provider = get_vm_provider({"vm_provider": "noop"})
    assert isinstance(provider, NoopProvider)


def test_factory_ec2_with_settings():
    settings = {
        "vm_provider": "ec2",
        "ec2": {
            "default_region": "us-west-2",
            "default_ami_id": "ami-deadbeef",
            "default_instance_type": "t3.large",
            "use_spot": True,
        },
    }
    provider = get_vm_provider(settings)
    assert isinstance(provider, EC2Provider)
    assert provider.default_region == "us-west-2"
    assert provider.default_ami_id == "ami-deadbeef"
    assert provider.default_instance_type == "t3.large"
    assert provider.default_use_spot is True


def test_factory_ec2_uses_defaults_when_block_missing():
    """`vm_provider: ec2` with no `ec2:` block still works using built-in defaults."""
    provider = get_vm_provider({"vm_provider": "ec2"})
    assert isinstance(provider, EC2Provider)
    assert provider.default_region == "us-west-2"  # baked-in default
    assert provider.default_instance_type == "t3.large"


def test_factory_unknown_raises():
    with pytest.raises(ValueError) as exc_info:
        get_vm_provider({"vm_provider": "firecracker-someday"})
    msg = str(exc_info.value)
    assert "Unknown vm_provider" in msg
    # Make sure the error message lists the supported providers so callers
    # can fix their config without grep.
    for name in SUPPORTED_PROVIDERS:
        assert name in msg


def test_factory_supported_providers_set():
    assert set(SUPPORTED_PROVIDERS) >= {"noop", "ec2"}


def test_provider_swap_is_config_only():
    """Switching `vm_provider` is the only change required (Validation #6)."""
    cfg_noop = {"vm_provider": "noop"}
    cfg_ec2 = {**cfg_noop, "vm_provider": "ec2"}
    p1 = get_vm_provider(cfg_noop)
    p2 = get_vm_provider(cfg_ec2)
    # Both implement AgentVMProvider — same call site can use either.
    assert hasattr(p1, "provision") and hasattr(p2, "provision")
    assert hasattr(p1, "terminate") and hasattr(p2, "terminate")
    assert hasattr(p1, "status") and hasattr(p2, "status")
    assert p1.name != p2.name
