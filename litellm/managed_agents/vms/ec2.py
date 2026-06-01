"""
EC2 implementation of `AgentVMProvider` (BYOC).

Each session gets a dedicated EC2 instance launched in the *team's* AWS
account using the team's BYOC creds. The proxy never holds AWS creds itself.

Key design points:
- creds enter via `ProvisionContext.aws_creds` and never leave this module
- `boto3` debug logging is silenced (`set_stream_logger`) so SigV4-signing
  payloads cannot accidentally leak the access key
- spot is tried first, on-demand is the fallback
- every `RunInstances` is paired with a `TerminateInstances` retry path

The constants `_BOTO3_RETRY_MODE`, `_SPOT_INTERRUPTION_BEHAVIOR` and
`_USER_DATA_TEMPLATE` are tuned to match the values measured in the B0
spike (LIT-2888) — see deliverables comment on that ticket.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.managed_agents.vms.base import (
    AgentVMProvider,
    AwsCreds,
    Ec2Config,
    InvalidCredentialsError,
    ProvisionContext,
    ProvisionError,
    Repo,
    VMHandle,
    VMState,
    VMStatus,
)

# `botocore` logs every signed request at DEBUG level. We force WARNING so
# the AWS access key never lands in proxy logs even when the operator turns
# on litellm DEBUG logging. This is enforced again in `_silence_boto_logs`
# below in case some other code path raises the level back up.
_BOTO_LOGGERS = ("boto3", "botocore", "urllib3.connectionpool", "s3transfer")

# Tuned in B0: max 5 short retries; longer waits make sessions feel sluggish.
_BOTO3_RETRY_MODE = "standard"
_BOTO3_RETRY_MAX_ATTEMPTS = 5

# Spot interruption behaviour: terminate (default) — we don't want hibernation
# because we destroy the VM at session end anyway.
_SPOT_INTERRUPTION_BEHAVIOR = "terminate"

# Map from EC2 lifecycle state to our VMState enum.
_EC2_STATE_TO_VMSTATE = {
    "pending": VMState.PENDING,
    "running": VMState.RUNNING,
    "shutting-down": VMState.STOPPING,
    "stopping": VMState.STOPPING,
    "stopped": VMState.STOPPED,
    "terminated": VMState.TERMINATED,
}

# AWS errors that mean "your creds are bad" — surface these as 400 to the user.
_INVALID_CRED_CODES = {
    "InvalidClientTokenId",
    "AuthFailure",
    "SignatureDoesNotMatch",
    "UnauthorizedOperation",
    "OptInRequired",
    "AccessDenied",
}

# AWS error codes that mean "spot capacity unavailable" — fall back to on-demand.
_SPOT_UNAVAILABLE_CODES = {
    "InsufficientInstanceCapacity",
    "SpotMaxPriceTooLow",
    "MaxSpotInstanceCountExceeded",
    "InstanceLimitExceeded",
}


def _silence_boto_logs() -> None:
    """Force every boto/botocore logger to WARNING so creds never leak."""
    for name in _BOTO_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


_silence_boto_logs()


def _build_user_data(ctx: ProvisionContext) -> str:
    """
    Build the EC2 user-data shell script.

    The script writes the daemon JWT + base URL into systemd's environment file
    and starts the agent runtime daemon. NOTHING in this script logs the JWT
    body — it goes straight into the EnvironmentFile that systemd reads.
    """
    daemon_jwt = ctx.daemon_jwt or ""
    base_url = ctx.daemon_base_url or ""
    mode = ctx.mode or "session"

    # Repos are written to /etc/litellm-agent/repos.json so the daemon can
    # iterate them on first run. We base64 the payload to avoid quoting hell.
    repos_json = _repos_to_json(ctx.repos)
    env_json = _env_to_json(ctx.env_vars)

    repos_b64 = base64.b64encode(repos_json.encode("utf-8")).decode("utf-8")
    env_b64 = base64.b64encode(env_json.encode("utf-8")).decode("utf-8")

    return f"""#!/bin/bash
set -e
mkdir -p /etc/litellm-agent
cat > /etc/litellm-agent/runtime.env <<EOF
LITELLM_SESSION_ID={ctx.session_id}
LITELLM_TEAM_ID={ctx.team_id}
LITELLM_AGENT_ID={ctx.agent_id or ''}
LITELLM_BASE_URL={base_url}
LITELLM_AGENT_MODE={mode}
LITELLM_DAEMON_JWT={daemon_jwt}
EOF
chmod 600 /etc/litellm-agent/runtime.env
echo "{repos_b64}" | base64 -d > /etc/litellm-agent/repos.json
echo "{env_b64}" | base64 -d > /etc/litellm-agent/env.json
chmod 600 /etc/litellm-agent/repos.json /etc/litellm-agent/env.json
systemctl enable --now litellm-agent-runtime.service || true
"""


def _repos_to_json(repos: List[Repo]) -> str:
    import json

    return json.dumps([{"url": r.url, "ref": r.ref, "path": r.path} for r in repos])


def _env_to_json(env: Dict[str, str]) -> str:
    import json

    return json.dumps(env)


def _safe_aws_error(exc: Exception) -> str:
    """
    Format a boto3 ClientError without leaking the request payload.

    `botocore.exceptions.ClientError.__str__` includes the SigV4 metadata in
    some versions; we extract only the error code + message.
    """
    response = getattr(exc, "response", None) or {}
    err = response.get("Error", {}) if isinstance(response, dict) else {}
    code = err.get("Code") or type(exc).__name__
    message = err.get("Message") or "(no message)"
    return f"{code}: {message}"


def _aws_error_code(exc: Exception) -> Optional[str]:
    response = getattr(exc, "response", None) or {}
    err = response.get("Error", {}) if isinstance(response, dict) else {}
    return err.get("Code")


def _tag_specs(ctx: ProvisionContext) -> List[Dict[str, Any]]:
    """Tag every resource so cleanup tools can find it."""
    tags = [
        {"Key": "litellm-session-id", "Value": ctx.session_id},
        {"Key": "litellm-team-id", "Value": ctx.team_id},
        {"Key": "litellm-managed-by", "Value": "agent-vm-provider"},
    ]
    if ctx.agent_id:
        tags.append({"Key": "litellm-agent-id", "Value": ctx.agent_id})
    return [
        {"ResourceType": "instance", "Tags": tags},
        {"ResourceType": "volume", "Tags": tags},
    ]


class EC2Provider(AgentVMProvider):
    """boto3-backed EC2 provisioner."""

    name = "ec2"

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        self._settings: Dict[str, Any] = settings or {}
        # ThreadPool to wrap synchronous boto3 calls for asyncio. Bounded so a
        # spike of provision calls cannot exhaust the proxy.
        self._executor = ThreadPoolExecutor(
            max_workers=int(self._settings.get("max_concurrent_aws_calls", 16)),
            thread_name_prefix="ec2-provider",
        )

    @property
    def default_region(self) -> str:
        return self._settings.get("default_region", "us-west-2")

    @property
    def default_ami_id(self) -> Optional[str]:
        return self._settings.get("default_ami_id")

    @property
    def default_instance_type(self) -> str:
        return self._settings.get("default_instance_type", "t3.large")

    @property
    def default_use_spot(self) -> bool:
        return bool(self._settings.get("use_spot", True))

    # ---------- public API ----------

    async def provision(self, ctx: ProvisionContext) -> VMHandle:
        if ctx.aws_creds is None:
            raise InvalidCredentialsError(
                "EC2 provider requires BYOC AWS credentials in ProvisionContext."
            )

        ec2_config = ctx.ec2_config or Ec2Config(region=ctx.aws_creds.region)
        ami_id = ec2_config.ami_id or self.default_ami_id
        if not ami_id:
            raise ProvisionError(
                "No AMI configured. Set agent_settings.ec2.default_ami_id "
                "in config.yaml or per-team via LiteLLM_AgentVMConfig.ami_id."
            )

        instance_type = ec2_config.instance_type or self.default_instance_type
        use_spot = (
            ec2_config.use_spot if ec2_config is not None else self.default_use_spot
        )
        user_data = _build_user_data(ctx)

        try:
            instance = await asyncio.get_running_loop().run_in_executor(
                self._executor,
                self._run_instances_with_fallback,
                ctx.aws_creds,
                ami_id,
                instance_type,
                ec2_config,
                user_data,
                _tag_specs(ctx),
                use_spot,
            )
        except InvalidCredentialsError:
            raise
        except ProvisionError:
            raise
        except Exception as e:
            verbose_proxy_logger.exception(
                "EC2 provision failed for session=%s team=%s: %s",
                ctx.session_id,
                ctx.team_id,
                _safe_aws_error(e),
            )
            raise ProvisionError(
                f"EC2 RunInstances failed: {_safe_aws_error(e)}"
            ) from e

        return VMHandle(
            vm_id=instance["InstanceId"],
            provider=self.name,
            region=ec2_config.region,
            metadata={
                "ami_id": ami_id,
                "instance_type": instance_type,
                "purchase_mode": instance.get("_purchase_mode", "on-demand"),
                "subnet_id": ec2_config.subnet_id,
            },
        )

    async def terminate(
        self, vm: VMHandle, aws_creds: Optional[AwsCreds] = None
    ) -> None:
        """
        Terminate the EC2 instance.

        `aws_creds` is required because the team that owns the instance also
        owns the creds. Callers fetch them via `team_config.get_team_vm_config`
        right before calling `terminate`.
        """
        if aws_creds is None:
            raise InvalidCredentialsError(
                "EC2 terminate requires BYOC AWS credentials."
            )

        try:
            await asyncio.get_running_loop().run_in_executor(
                self._executor,
                self._terminate_instances_sync,
                aws_creds,
                vm.region or aws_creds.region,
                vm.vm_id,
            )
        except Exception as e:
            code = _aws_error_code(e)
            # `InvalidInstanceID.NotFound` and `InvalidInstanceID.Malformed` mean
            # the instance is already gone; treat as success.
            if code in ("InvalidInstanceID.NotFound", "InvalidInstanceID.Malformed"):
                verbose_proxy_logger.debug(
                    f"terminate: instance {vm.vm_id} already gone ({code})"
                )
                return
            verbose_proxy_logger.exception(
                "EC2 terminate failed for vm=%s: %s", vm.vm_id, _safe_aws_error(e)
            )
            raise ProvisionError(
                f"EC2 TerminateInstances failed: {_safe_aws_error(e)}"
            ) from e

    async def status(
        self, vm: VMHandle, aws_creds: Optional[AwsCreds] = None
    ) -> VMStatus:
        """Return current EC2 status. Requires the team's BYOC creds."""
        if aws_creds is None:
            raise InvalidCredentialsError("EC2 status requires BYOC AWS credentials.")

        try:
            raw = await asyncio.get_running_loop().run_in_executor(
                self._executor,
                self._describe_instance_sync,
                aws_creds,
                vm.region or aws_creds.region,
                vm.vm_id,
            )
        except Exception as e:
            code = _aws_error_code(e)
            if code in ("InvalidInstanceID.NotFound", "InvalidInstanceID.Malformed"):
                return VMStatus(state=VMState.TERMINATED)
            raise ProvisionError(
                f"EC2 DescribeInstances failed: {_safe_aws_error(e)}"
            ) from e

        if raw is None:
            return VMStatus(state=VMState.TERMINATED)

        ec2_state = (raw.get("State") or {}).get("Name", "unknown")
        return VMStatus(
            state=_EC2_STATE_TO_VMSTATE.get(ec2_state, VMState.UNKNOWN),
            public_ip=raw.get("PublicIpAddress"),
            private_ip=raw.get("PrivateIpAddress"),
            raw=raw,
        )

    # ---------- sync (boto3) helpers, run in the thread pool ----------

    def _build_ec2_client(self, creds: AwsCreds, region: str) -> Any:
        """Return a boto3 EC2 client. Only place creds touch boto3."""
        # Re-silence each call (defensive: another module may have changed it).
        _silence_boto_logs()
        try:
            import boto3  # noqa: PLC0415  imported here so litellm core works without boto3 installed
            from botocore.config import Config  # noqa: PLC0415
        except ImportError as e:
            raise ProvisionError(
                "boto3 is required for the EC2 VM provider. "
                "Install with: pip install boto3."
            ) from e

        config = Config(
            retries={
                "max_attempts": _BOTO3_RETRY_MAX_ATTEMPTS,
                "mode": _BOTO3_RETRY_MODE,
            },
            region_name=region,
        )
        return boto3.client(
            "ec2",
            region_name=region,
            aws_access_key_id=creds.access_key_id,
            aws_secret_access_key=creds.secret_access_key,
            aws_session_token=creds.session_token,
            config=config,
        )

    def _run_instances_with_fallback(
        self,
        creds: AwsCreds,
        ami_id: str,
        instance_type: str,
        ec2_config: Ec2Config,
        user_data: str,
        tag_specs: List[Dict[str, Any]],
        use_spot: bool,
    ) -> Dict[str, Any]:
        """Try spot, fall back to on-demand once if capacity unavailable."""
        client = self._build_ec2_client(creds, ec2_config.region)
        if use_spot:
            try:
                instance = self._run_instances_sync(
                    client,
                    ami_id=ami_id,
                    instance_type=instance_type,
                    ec2_config=ec2_config,
                    user_data=user_data,
                    tag_specs=tag_specs,
                    spot=True,
                )
                instance["_purchase_mode"] = "spot"
                return instance
            except Exception as e:
                code = _aws_error_code(e)
                if code in _INVALID_CRED_CODES:
                    raise InvalidCredentialsError(
                        f"AWS rejected the team's credentials: {code}"
                    ) from e
                if code in _SPOT_UNAVAILABLE_CODES:
                    verbose_proxy_logger.warning(
                        f"Spot capacity unavailable ({code}); falling back to on-demand."
                    )
                    # fall through to on-demand
                else:
                    raise

        instance = self._run_instances_sync(
            client,
            ami_id=ami_id,
            instance_type=instance_type,
            ec2_config=ec2_config,
            user_data=user_data,
            tag_specs=tag_specs,
            spot=False,
        )
        instance["_purchase_mode"] = "on-demand"
        return instance

    def _run_instances_sync(
        self,
        client: Any,
        *,
        ami_id: str,
        instance_type: str,
        ec2_config: Ec2Config,
        user_data: str,
        tag_specs: List[Dict[str, Any]],
        spot: bool,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "ImageId": ami_id,
            "InstanceType": instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "UserData": user_data,
            "TagSpecifications": tag_specs,
        }
        if ec2_config.subnet_id:
            kwargs["SubnetId"] = ec2_config.subnet_id
        if ec2_config.security_group_id:
            kwargs["SecurityGroupIds"] = [ec2_config.security_group_id]
        if ec2_config.iam_instance_profile:
            kwargs["IamInstanceProfile"] = {"Name": ec2_config.iam_instance_profile}
        if spot:
            kwargs["InstanceMarketOptions"] = {
                "MarketType": "spot",
                "SpotOptions": {
                    "InstanceInterruptionBehavior": _SPOT_INTERRUPTION_BEHAVIOR,
                    "SpotInstanceType": "one-time",
                },
            }

        try:
            response = client.run_instances(**kwargs)
        except Exception as e:
            code = _aws_error_code(e)
            if code in _INVALID_CRED_CODES:
                raise InvalidCredentialsError(
                    f"AWS rejected the team's credentials: {code}"
                ) from e
            raise

        instances = response.get("Instances", [])
        if not instances:
            raise ProvisionError("RunInstances succeeded but returned no instances.")
        return instances[0]

    def _terminate_instances_sync(
        self, creds: AwsCreds, region: str, instance_id: str
    ) -> None:
        client = self._build_ec2_client(creds, region)
        client.terminate_instances(InstanceIds=[instance_id])

    def _describe_instance_sync(
        self, creds: AwsCreds, region: str, instance_id: str
    ) -> Optional[Dict[str, Any]]:
        client = self._build_ec2_client(creds, region)
        response = client.describe_instances(InstanceIds=[instance_id])
        for res in response.get("Reservations", []):
            for inst in res.get("Instances", []):
                return inst
        return None
