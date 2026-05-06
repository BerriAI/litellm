"""SSM RunCommand hydrate transport (LIT-2890).

Pushes a ``HydratePayload`` to a warm VM by issuing
``ssm.send_command(DocumentName='AWS-RunShellScript', InstanceIds=[vm_id])``
with a small shell script that:

  1. Writes the JSON payload to ``/var/run/litellm-agent/hydrate.json``
     (mode 0600, root-only)
  2. Sends ``SIGUSR1`` to the daemon, which it interprets as
     "switch from warm mode to session mode using the new file"

The payload is base64-encoded so SSM doesn't have to escape JSON quotes.

Why SSM (not long-poll): B0 measured 1700ms median, 2029ms p95 for the
full RunCommand round-trip on a pre-warmed VM (LIT-2888). That fits the
3000ms session-create gate with ~970ms of headroom for the proxy DB
write + JWT mint + daemon hydrate handler.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Protocol

from litellm._logging import verbose_proxy_logger
from litellm.proxy.agent_session_endpoints.vm_providers.base import AwsCreds
from litellm.proxy.agent_session_endpoints.warm_pool.types import HydratePayload

# Silence boto3 debug logging (same defensive measure as ec2.py).
for _name in ("boto3", "botocore", "urllib3.connectionpool"):
    logging.getLogger(_name).setLevel(logging.WARNING)


class HydrateTransport(Protocol):
    """Pluggable hydrate transport. SSM is the only impl today."""

    async def push(
        self,
        *,
        vm_id: str,
        region: str,
        aws_creds: AwsCreds,
        payload: HydratePayload,
    ) -> None: ...


class HydrateTransportError(RuntimeError):
    """Transport failed to deliver the hydrate payload."""


def _hydrate_script(payload_b64: str) -> str:
    """Shell script that lands on the VM via SSM.

    NOTE: ``payload_b64`` is interpolated into a single-quoted shell string,
    base64 chars are safe inside single quotes (no escape needed). The
    daemon listens for SIGUSR1 in warm mode and re-reads
    ``/var/run/litellm-agent/hydrate.json`` on receipt.
    """
    return f"""#!/bin/bash
set -e
mkdir -p /var/run/litellm-agent
umask 077
echo '{payload_b64}' | base64 -d > /var/run/litellm-agent/hydrate.json
chmod 600 /var/run/litellm-agent/hydrate.json
chown root:root /var/run/litellm-agent/hydrate.json
# Signal the daemon. The systemd MainPID file is written by the unit.
if [ -f /run/litellm-agent.pid ]; then
  kill -USR1 "$(cat /run/litellm-agent.pid)" 2>/dev/null || true
else
  pkill -USR1 -f litellm-agent-runtime || true
fi
"""


class SSMHydrateTransport:
    """Production hydrate transport using AWS SSM RunCommand."""

    name = "ssm"

    def __init__(self, max_concurrent_aws_calls: int = 16) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent_aws_calls,
            thread_name_prefix="ssm-hydrate",
        )

    async def push(
        self,
        *,
        vm_id: str,
        region: str,
        aws_creds: AwsCreds,
        payload: HydratePayload,
    ) -> None:
        """Send the hydrate script. Raises ``HydrateTransportError`` on failure."""
        if not vm_id:
            raise HydrateTransportError("ssm.push: vm_id is required")
        if not region:
            raise HydrateTransportError("ssm.push: region is required")

        payload_b64 = base64.b64encode(
            payload.model_dump_json().encode("utf-8")
        ).decode("ascii")
        script = _hydrate_script(payload_b64)

        try:
            await asyncio.get_running_loop().run_in_executor(
                self._executor,
                self._send_command_sync,
                aws_creds,
                region,
                vm_id,
                script,
            )
        except HydrateTransportError:
            raise
        except Exception as exc:
            verbose_proxy_logger.exception(
                "ssm.hydrate failed vm=%s: %s", vm_id, _safe_aws_error(exc)
            )
            raise HydrateTransportError(
                f"SSM RunCommand failed: {_safe_aws_error(exc)}"
            ) from exc

    # ---------- sync (boto3) ----------

    def _send_command_sync(
        self,
        creds: AwsCreds,
        region: str,
        vm_id: str,
        script: str,
    ) -> Dict[str, Any]:
        client = self._build_ssm_client(creds, region)
        response = client.send_command(
            InstanceIds=[vm_id],
            DocumentName="AWS-RunShellScript",
            Comment="litellm warm-pool hydrate",
            Parameters={"commands": [script]},
            TimeoutSeconds=60,
        )
        cmd_id = (response.get("Command") or {}).get("CommandId")
        if not cmd_id:
            raise HydrateTransportError(
                f"send_command returned no CommandId: {response!r}"
            )
        return response

    def _build_ssm_client(self, creds: AwsCreds, region: str) -> Any:
        try:
            import boto3  # noqa: PLC0415
            from botocore.config import Config  # noqa: PLC0415
        except ImportError as e:
            raise HydrateTransportError(
                "boto3 is required for the SSM hydrate transport. "
                "Install with: pip install boto3."
            ) from e

        return boto3.client(
            "ssm",
            region_name=region,
            aws_access_key_id=creds.access_key_id,
            aws_secret_access_key=creds.secret_access_key,
            aws_session_token=creds.session_token,
            config=Config(
                retries={"max_attempts": 3, "mode": "standard"},
                region_name=region,
            ),
        )


def _safe_aws_error(exc: Exception) -> str:
    """Format a boto3 error without leaking the request payload."""
    response = getattr(exc, "response", None) or {}
    err = response.get("Error", {}) if isinstance(response, dict) else {}
    code = err.get("Code") or type(exc).__name__
    message = err.get("Message") or "(no message)"
    return f"{code}: {message}"


# Module-level singleton — reused across attach calls so we don't churn the
# thread pool. Tests reset via ``reset_default_transport``.
_DEFAULT_TRANSPORT: Optional[SSMHydrateTransport] = None


def get_default_transport() -> SSMHydrateTransport:
    global _DEFAULT_TRANSPORT
    if _DEFAULT_TRANSPORT is None:
        _DEFAULT_TRANSPORT = SSMHydrateTransport()
    return _DEFAULT_TRANSPORT


def set_default_transport(transport: Any) -> None:
    """Test helper / DI hook: swap the default transport (e.g. an in-memory fake)."""
    global _DEFAULT_TRANSPORT
    _DEFAULT_TRANSPORT = transport


def reset_default_transport() -> None:
    global _DEFAULT_TRANSPORT
    _DEFAULT_TRANSPORT = None
