"""
Mocked tests for `EC2Provider`.

These tests use a fake boto3 client (no AWS calls). The real-cloud tests
that use BYOC creds are in `test_ec2_provider_real.py` and are gated
behind `pytest --slow` so they don't run in the unit suite.

Coverage:
- Validation #5: spot → on-demand fallback when `InsufficientInstanceCapacity`
- Validation #8: terminate is idempotent (covers cascade-terminate cleanup)
- Validation #11: invalid AWS creds raise InvalidCredentialsError fast (no instance)
- Validation #13: AWS keys never appear in any log record or repr
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from litellm.proxy.agent_session_endpoints.vm_providers import (
    AwsCreds,
    EC2Provider,
    Ec2Config,
    InvalidCredentialsError,
    ProvisionContext,
    ProvisionError,
    Repo,
    VMHandle,
    VMState,
)


_FAKE_KEY = "AKIAFAKEEXAMPLEKEY12"
_FAKE_SECRET = "FAKEsecretFAKEsecretFAKEsecretFAKEsecre"  # 40 chars
_FAKE_KEY_LEAK_CANARY = "AKIATESTLEAKCANARY00"


class _FakeClientError(Exception):
    """Stand-in for botocore.exceptions.ClientError."""

    def __init__(self, code: str, message: str = "fake") -> None:
        self.response = {"Error": {"Code": code, "Message": message}}
        super().__init__(f"{code}: {message}")


class _FakeEc2Client:
    """In-memory boto3 EC2 client double."""

    def __init__(
        self,
        *,
        run_responses: Optional[List[Any]] = None,
        terminate_should_raise: Optional[Exception] = None,
        describe_response: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Each call to run_instances pops the next response. The response can be
        # an instance dict OR an Exception subclass to raise.
        self.run_responses: List[Any] = list(run_responses or [])
        self.terminate_should_raise = terminate_should_raise
        self.describe_response = describe_response
        self.run_calls: List[Dict[str, Any]] = []
        self.terminate_calls: List[Dict[str, Any]] = []
        self.describe_calls: List[Dict[str, Any]] = []

    def run_instances(self, **kwargs):
        self.run_calls.append(kwargs)
        if not self.run_responses:
            raise AssertionError("run_instances called more times than expected")
        nxt = self.run_responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return {"Instances": [nxt]}

    def terminate_instances(self, **kwargs):
        self.terminate_calls.append(kwargs)
        if self.terminate_should_raise is not None:
            raise self.terminate_should_raise

    def describe_instances(self, **kwargs):
        self.describe_calls.append(kwargs)
        if isinstance(self.describe_response, Exception):
            raise self.describe_response
        if self.describe_response is None:
            return {"Reservations": []}
        return {"Reservations": [{"Instances": [self.describe_response]}]}


def _patch_provider_client(provider: EC2Provider, fake: _FakeEc2Client):
    """Patch `_build_ec2_client` on this provider instance."""
    return patch.object(provider, "_build_ec2_client", lambda creds, region: fake)


def _ctx(team_id: str = "team-1") -> ProvisionContext:
    return ProvisionContext(
        session_id="sess-1",
        team_id=team_id,
        agent_id="agent-1",
        repos=[Repo(url="https://example.com/x.git")],
        env_vars={"FOO": "bar"},
        aws_creds=AwsCreds(
            access_key_id=_FAKE_KEY,
            secret_access_key=_FAKE_SECRET,
            region="us-west-2",
        ),
        ec2_config=Ec2Config(
            region="us-west-2",
            subnet_id="subnet-1",
            security_group_id="sg-1",
            iam_instance_profile="litellm-ec2-poc",
            instance_type="t3.large",
            use_spot=True,
            ami_id="ami-deadbeef",
        ),
        daemon_jwt="fake.jwt.value",
        daemon_base_url="https://proxy.example/",
        mode="session",
    )


# ---------- Validation #11: invalid creds fail-fast ----------


@pytest.mark.asyncio
async def test_provision_without_creds_raises_invalid_credentials_error():
    provider = EC2Provider({"default_ami_id": "ami-deadbeef"})
    ctx = _ctx()
    ctx.aws_creds = None
    with pytest.raises(InvalidCredentialsError):
        await provider.provision(ctx)


@pytest.mark.asyncio
async def test_provision_invalid_creds_aws_response_raises_invalid_credentials_error():
    """AWS rejects creds with `InvalidClientTokenId` → 400 InvalidCredentialsError, no instance."""
    provider = EC2Provider({"default_ami_id": "ami-deadbeef"})
    fake = _FakeEc2Client(
        run_responses=[_FakeClientError("InvalidClientTokenId", "bad creds")]
    )
    with _patch_provider_client(provider, fake):
        with pytest.raises(InvalidCredentialsError):
            await provider.provision(_ctx())
    # No retry / no on-demand fallback for cred errors — we must fail-fast.
    assert len(fake.run_calls) == 1


@pytest.mark.asyncio
async def test_provision_signature_does_not_match_raises_invalid_credentials():
    provider = EC2Provider({"default_ami_id": "ami-deadbeef"})
    fake = _FakeEc2Client(
        run_responses=[_FakeClientError("SignatureDoesNotMatch", "wrong secret")]
    )
    with _patch_provider_client(provider, fake):
        with pytest.raises(InvalidCredentialsError):
            await provider.provision(_ctx())
    assert len(fake.run_calls) == 1


# ---------- Validation #5: spot → on-demand fallback ----------


@pytest.mark.asyncio
async def test_provision_spot_fallback_to_on_demand():
    """Spot raises `InsufficientInstanceCapacity` → provider retries on-demand once."""
    provider = EC2Provider({"default_ami_id": "ami-deadbeef"})
    fake = _FakeEc2Client(
        run_responses=[
            _FakeClientError("InsufficientInstanceCapacity", "no spot"),
            {"InstanceId": "i-on-demand-1"},
        ]
    )
    with _patch_provider_client(provider, fake):
        handle = await provider.provision(_ctx())

    assert handle.vm_id == "i-on-demand-1"
    assert handle.metadata["purchase_mode"] == "on-demand"
    # First call had spot market options; second did not.
    assert "InstanceMarketOptions" in fake.run_calls[0]
    assert "InstanceMarketOptions" not in fake.run_calls[1]


@pytest.mark.asyncio
async def test_provision_spot_first_succeeds_no_fallback():
    provider = EC2Provider({"default_ami_id": "ami-deadbeef"})
    fake = _FakeEc2Client(run_responses=[{"InstanceId": "i-spot-1"}])
    with _patch_provider_client(provider, fake):
        handle = await provider.provision(_ctx())
    assert handle.metadata["purchase_mode"] == "spot"
    assert len(fake.run_calls) == 1


@pytest.mark.asyncio
async def test_provision_no_spot_when_use_spot_false():
    provider = EC2Provider({"default_ami_id": "ami-deadbeef"})
    fake = _FakeEc2Client(run_responses=[{"InstanceId": "i-1"}])
    ctx = _ctx()
    ctx.ec2_config.use_spot = False  # type: ignore[union-attr]
    with _patch_provider_client(provider, fake):
        handle = await provider.provision(ctx)
    assert handle.metadata["purchase_mode"] == "on-demand"
    assert "InstanceMarketOptions" not in fake.run_calls[0]


# ---------- AMI required ----------


@pytest.mark.asyncio
async def test_provision_no_ami_raises_provision_error():
    provider = EC2Provider({})  # no default_ami_id
    ctx = _ctx()
    ctx.ec2_config.ami_id = None  # type: ignore[union-attr]
    with pytest.raises(ProvisionError) as exc_info:
        await provider.provision(ctx)
    assert "AMI" in str(exc_info.value)


# ---------- Tags + IAM passthrough ----------


@pytest.mark.asyncio
async def test_provision_tags_instance_with_session_team_agent_ids():
    provider = EC2Provider({"default_ami_id": "ami-deadbeef"})
    fake = _FakeEc2Client(run_responses=[{"InstanceId": "i-tagged"}])
    with _patch_provider_client(provider, fake):
        await provider.provision(_ctx(team_id="team-tag-test"))

    tag_specs = fake.run_calls[0]["TagSpecifications"]
    instance_tags = next(
        s["Tags"] for s in tag_specs if s["ResourceType"] == "instance"
    )
    keys_to_values = {t["Key"]: t["Value"] for t in instance_tags}
    assert keys_to_values["litellm-session-id"] == "sess-1"
    assert keys_to_values["litellm-team-id"] == "team-tag-test"
    assert keys_to_values["litellm-agent-id"] == "agent-1"


@pytest.mark.asyncio
async def test_provision_passes_iam_instance_profile():
    provider = EC2Provider({"default_ami_id": "ami-deadbeef"})
    fake = _FakeEc2Client(run_responses=[{"InstanceId": "i-iam"}])
    with _patch_provider_client(provider, fake):
        await provider.provision(_ctx())
    assert fake.run_calls[0]["IamInstanceProfile"] == {"Name": "litellm-ec2-poc"}


# ---------- Validation #8 piece: terminate idempotent ----------


@pytest.mark.asyncio
async def test_terminate_calls_aws():
    provider = EC2Provider({})
    fake = _FakeEc2Client()
    handle = VMHandle(vm_id="i-1", provider="ec2", region="us-west-2")
    with _patch_provider_client(provider, fake):
        await provider.terminate(
            handle,
            aws_creds=AwsCreds(
                access_key_id=_FAKE_KEY,
                secret_access_key=_FAKE_SECRET,
                region="us-west-2",
            ),
        )
    assert fake.terminate_calls == [{"InstanceIds": ["i-1"]}]


@pytest.mark.asyncio
async def test_terminate_already_gone_is_noop():
    provider = EC2Provider({})
    fake = _FakeEc2Client(
        terminate_should_raise=_FakeClientError("InvalidInstanceID.NotFound")
    )
    handle = VMHandle(vm_id="i-already-gone", provider="ec2", region="us-west-2")
    with _patch_provider_client(provider, fake):
        # Must not raise.
        await provider.terminate(
            handle,
            aws_creds=AwsCreds(
                access_key_id=_FAKE_KEY,
                secret_access_key=_FAKE_SECRET,
                region="us-west-2",
            ),
        )


@pytest.mark.asyncio
async def test_terminate_without_creds_raises():
    provider = EC2Provider({})
    handle = VMHandle(vm_id="i-1", provider="ec2", region="us-west-2")
    with pytest.raises(InvalidCredentialsError):
        await provider.terminate(handle)


# ---------- Status ----------


@pytest.mark.asyncio
async def test_status_running():
    provider = EC2Provider({})
    fake = _FakeEc2Client(
        describe_response={
            "InstanceId": "i-1",
            "State": {"Name": "running"},
            "PublicIpAddress": "1.2.3.4",
            "PrivateIpAddress": "10.0.0.1",
        }
    )
    handle = VMHandle(vm_id="i-1", provider="ec2", region="us-west-2")
    with _patch_provider_client(provider, fake):
        status = await provider.status(
            handle,
            aws_creds=AwsCreds(
                access_key_id=_FAKE_KEY,
                secret_access_key=_FAKE_SECRET,
                region="us-west-2",
            ),
        )
    assert status.state == VMState.RUNNING
    assert status.public_ip == "1.2.3.4"


@pytest.mark.asyncio
async def test_status_terminated_when_instance_not_found():
    provider = EC2Provider({})
    fake = _FakeEc2Client(
        describe_response=_FakeClientError("InvalidInstanceID.NotFound")
    )
    handle = VMHandle(vm_id="i-gone", provider="ec2", region="us-west-2")
    with _patch_provider_client(provider, fake):
        status = await provider.status(
            handle,
            aws_creds=AwsCreds(
                access_key_id=_FAKE_KEY,
                secret_access_key=_FAKE_SECRET,
                region="us-west-2",
            ),
        )
    assert status.state == VMState.TERMINATED


# ---------- Validation #13: creds never leak ----------


def test_aws_creds_repr_redacts():
    creds = AwsCreds(
        access_key_id=_FAKE_KEY_LEAK_CANARY,
        secret_access_key="topsecret-secret-secret-secret-secret-1",
        session_token="some-token",
        region="us-west-2",
    )
    text = repr(creds)
    # Neither the access key nor the secret may appear.
    assert _FAKE_KEY_LEAK_CANARY not in text
    assert "topsecret" not in text
    assert "REDACTED" in text
    assert str(creds) == repr(creds)


@pytest.mark.asyncio
async def test_aws_creds_never_logged_during_provision(caplog):
    """Even with DEBUG logging, the access key never lands in proxy logs."""
    caplog.set_level(logging.DEBUG)
    provider = EC2Provider({"default_ami_id": "ami-deadbeef"})
    fake = _FakeEc2Client(run_responses=[{"InstanceId": "i-leakcheck"}])
    ctx = _ctx()
    ctx.aws_creds = AwsCreds(
        access_key_id=_FAKE_KEY_LEAK_CANARY,
        secret_access_key="leak-canary-secret",
        region="us-west-2",
    )
    with _patch_provider_client(provider, fake):
        await provider.provision(ctx)

    full_log = "\n".join(rec.getMessage() for rec in caplog.records)
    assert _FAKE_KEY_LEAK_CANARY not in full_log
    assert "leak-canary-secret" not in full_log


@pytest.mark.asyncio
async def test_aws_creds_never_in_exception_message():
    """A boto3 error message must not echo the access key."""
    provider = EC2Provider({"default_ami_id": "ami-deadbeef"})
    fake = _FakeEc2Client(
        run_responses=[_FakeClientError("ValidationError", "bad request")]
    )
    ctx = _ctx()
    ctx.aws_creds = AwsCreds(
        access_key_id=_FAKE_KEY_LEAK_CANARY,
        secret_access_key="leak-canary-secret",
        region="us-west-2",
    )
    with _patch_provider_client(provider, fake):
        with pytest.raises(ProvisionError) as exc_info:
            await provider.provision(ctx)
    assert _FAKE_KEY_LEAK_CANARY not in str(exc_info.value)


# ---------- User-data shape ----------


@pytest.mark.asyncio
async def test_user_data_includes_session_id_and_jwt():
    """The provider builds user-data with the right env (the daemon reads them)."""
    provider = EC2Provider({"default_ami_id": "ami-deadbeef"})
    fake = _FakeEc2Client(run_responses=[{"InstanceId": "i-user-data"}])
    with _patch_provider_client(provider, fake):
        await provider.provision(_ctx())

    user_data = fake.run_calls[0]["UserData"]
    assert "LITELLM_SESSION_ID=sess-1" in user_data
    assert "LITELLM_TEAM_ID=team-1" in user_data
    assert "LITELLM_AGENT_ID=agent-1" in user_data
    assert "LITELLM_DAEMON_JWT=fake.jwt.value" in user_data
    assert "LITELLM_AGENT_MODE=session" in user_data
