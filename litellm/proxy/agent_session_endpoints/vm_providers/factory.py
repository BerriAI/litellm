"""
Factory for `AgentVMProvider` implementations.

Reads `agent_settings.vm_provider` from the loaded proxy config. Default is
`noop` so the proxy starts cleanly without AWS configured.

Validation criterion #1 (`test_factory`) covers:
- factory returns the right impl for each value
- unknown values raise `ValueError`
- defaults to `noop` when no config is set
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from litellm.proxy.agent_session_endpoints.vm_providers.base import AgentVMProvider
from litellm.proxy.agent_session_endpoints.vm_providers.ec2 import EC2Provider
from litellm.proxy.agent_session_endpoints.vm_providers.noop import NoopProvider

# Registry of {name: factory_callable}. The factory takes the raw settings
# dict (e.g. `agent_settings.ec2`) so per-provider config stays inside the
# provider class, not the factory.
_PROVIDER_REGISTRY = {
    "noop": lambda settings: NoopProvider(),
    "ec2": lambda settings: EC2Provider(settings=settings),
}


SUPPORTED_PROVIDERS = tuple(_PROVIDER_REGISTRY.keys())


def get_vm_provider(
    agent_settings: Optional[Dict[str, Any]] = None,
) -> AgentVMProvider:
    """
    Build the configured provider.

    `agent_settings` is the `agent_settings` block from `config.yaml`:

    ```yaml
    agent_settings:
      vm_provider: ec2
      ec2:
        default_region: us-west-2
        default_ami_id: ami-...
        ...
    ```

    Defaults to `noop` if `agent_settings` is None or `vm_provider` is missing.
    Raises `ValueError` for unknown provider names.
    """
    settings = agent_settings or {}
    provider_name = settings.get("vm_provider", "noop")

    if provider_name not in _PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown vm_provider {provider_name!r}. "
            f"Supported: {', '.join(sorted(_PROVIDER_REGISTRY.keys()))}."
        )

    provider_settings = settings.get(provider_name, {}) or {}
    return _PROVIDER_REGISTRY[provider_name](provider_settings)
