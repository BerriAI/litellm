"""
Real-cloud tests for `EC2Provider`.

Skipped by default. Enable with `pytest -m slow` and the BYOC env vars set:

    LITELLM_AGENT_AWS_ACCESS_KEY_ID
    LITELLM_AGENT_AWS_SECRET_ACCESS_KEY
    LITELLM_AGENT_AWS_REGION                  (default us-west-2)
    LITELLM_TEST_SUBNET_ID
    LITELLM_TEST_SECURITY_GROUP_ID
    LITELLM_TEST_IAM_INSTANCE_PROFILE
    LITELLM_TEST_AMI_ID

These are the resources B0 captured for the BYOC PoC account
(see LIT-2888 deliverables comment).

Each test is wrapped in a try/finally that calls TerminateInstances. A 60-min
process-wide watchdog is also installed so a hung test cannot leak an
instance overnight.

Cost per run: ~$0.03 per session (one t3.large for under a minute).
"""

from __future__ import annotations

import os
import threading
import time
from typing import List, Optional

import pytest

from litellm.proxy.agent_session_endpoints.vm_providers import (
    AwsCreds,
    EC2Provider,
    Ec2Config,
    ProvisionContext,
    Repo,
    VMHandle,
    VMState,
)


# Test markers — `slow` is the standard real-cloud gate.
pytestmark = [pytest.mark.slow]


def _have_creds() -> bool:
    return bool(
        os.getenv("LITELLM_AGENT_AWS_ACCESS_KEY_ID")
        and os.getenv("LITELLM_AGENT_AWS_SECRET_ACCESS_KEY")
        and os.getenv("LITELLM_TEST_AMI_ID")
    )


_skip_no_creds = pytest.mark.skipif(
    not _have_creds(),
    reason="real-cloud test requires LITELLM_AGENT_AWS_* + LITELLM_TEST_* env vars",
)


def _ec2_config_from_env() -> Ec2Config:
    return Ec2Config(
        region=os.getenv("LITELLM_AGENT_AWS_REGION", "us-west-2"),
        subnet_id=os.getenv("LITELLM_TEST_SUBNET_ID"),
        security_group_id=os.getenv("LITELLM_TEST_SECURITY_GROUP_ID"),
        iam_instance_profile=os.getenv("LITELLM_TEST_IAM_INSTANCE_PROFILE"),
        instance_type="t3.large",
        use_spot=False,  # real tests prefer determinism over savings
        ami_id=os.getenv("LITELLM_TEST_AMI_ID"),
    )


def _aws_creds_from_env() -> AwsCreds:
    return AwsCreds(
        access_key_id=os.environ["LITELLM_AGENT_AWS_ACCESS_KEY_ID"],
        secret_access_key=os.environ["LITELLM_AGENT_AWS_SECRET_ACCESS_KEY"],
        session_token=os.getenv("LITELLM_AGENT_AWS_SESSION_TOKEN"),
        region=os.getenv("LITELLM_AGENT_AWS_REGION", "us-west-2"),
    )


def _install_watchdog(
    provider: EC2Provider, handles: List[VMHandle]
) -> threading.Timer:
    """60-min hard watchdog: terminate everything we tracked, no matter what."""

    def _kill_all() -> None:
        import asyncio  # noqa: PLC0415

        async def _go():
            for h in handles:
                try:
                    await provider.terminate(h, aws_creds=_aws_creds_from_env())
                except Exception:
                    pass

        try:
            asyncio.run(_go())
        except Exception:
            pass

    timer = threading.Timer(60 * 60, _kill_all)
    timer.daemon = True
    timer.start()
    return timer


@_skip_no_creds
@pytest.mark.asyncio
async def test_real_boot():
    """Validation #3: instance reaches `running` within 90s, then we terminate."""
    handles: List[VMHandle] = []
    provider = EC2Provider({"default_ami_id": os.environ["LITELLM_TEST_AMI_ID"]})
    watchdog = _install_watchdog(provider, handles)

    ctx = ProvisionContext(
        session_id=f"sess-test-real-{int(time.time())}",
        team_id="team-test-real",
        agent_id="agent-test-real",
        repos=[Repo(url="https://github.com/octocat/Hello-World")],
        env_vars={"FOO": "bar"},
        aws_creds=_aws_creds_from_env(),
        ec2_config=_ec2_config_from_env(),
        daemon_jwt="not-a-real-jwt-test-only",
        daemon_base_url="https://example.invalid/",
        mode="session",
    )

    handle: Optional[VMHandle] = None
    try:
        handle = await provider.provision(ctx)
        handles.append(handle)
        assert handle.vm_id.startswith("i-")

        # Poll for `running` (max 90s).
        deadline = time.time() + 90
        last: Optional[VMState] = None
        while time.time() < deadline:
            status = await provider.status(handle, aws_creds=ctx.aws_creds)
            last = status.state
            if status.state == VMState.RUNNING:
                break
            time.sleep(2)
        assert last == VMState.RUNNING, f"never reached running, last={last}"
    finally:
        if handle is not None:
            await provider.terminate(handle, aws_creds=ctx.aws_creds)
        watchdog.cancel()


@_skip_no_creds
@pytest.mark.asyncio
async def test_byoc_invalid_creds_fail_fast():
    """Validation #11: bad creds → InvalidCredentialsError within ~5s, no instance launched."""
    from litellm.proxy.agent_session_endpoints.vm_providers import (
        InvalidCredentialsError,
    )

    provider = EC2Provider({"default_ami_id": os.environ["LITELLM_TEST_AMI_ID"]})
    bad_creds = AwsCreds(
        access_key_id="AKIAINVALIDFAKEKEY00",
        secret_access_key="invalid-secret-do-not-leak",
        region=os.getenv("LITELLM_AGENT_AWS_REGION", "us-west-2"),
    )
    ctx = ProvisionContext(
        session_id=f"sess-bad-{int(time.time())}",
        team_id="team-bad",
        aws_creds=bad_creds,
        ec2_config=_ec2_config_from_env(),
    )
    t0 = time.time()
    with pytest.raises(InvalidCredentialsError):
        await provider.provision(ctx)
    elapsed = time.time() - t0
    assert elapsed < 10, f"fail-fast took too long: {elapsed:.1f}s"


@_skip_no_creds
@pytest.mark.asyncio
async def test_terminate_idempotent_real():
    """Validation #8 piece: terminate the same instance twice — second is a no-op."""
    handles: List[VMHandle] = []
    provider = EC2Provider({"default_ami_id": os.environ["LITELLM_TEST_AMI_ID"]})
    watchdog = _install_watchdog(provider, handles)

    ctx = ProvisionContext(
        session_id=f"sess-term-{int(time.time())}",
        team_id="team-term",
        aws_creds=_aws_creds_from_env(),
        ec2_config=_ec2_config_from_env(),
        daemon_jwt="x",
        daemon_base_url="https://example.invalid/",
    )
    handle: Optional[VMHandle] = None
    try:
        handle = await provider.provision(ctx)
        handles.append(handle)
        await provider.terminate(handle, aws_creds=ctx.aws_creds)
        # Second call must not raise.
        await provider.terminate(handle, aws_creds=ctx.aws_creds)
    finally:
        if handle is not None:
            try:
                await provider.terminate(handle, aws_creds=ctx.aws_creds)
            except Exception:
                pass
        watchdog.cancel()


@_skip_no_creds
@pytest.mark.asyncio
async def test_cold_boot_p50_under_30s():
    """Validation #14 — cold-boot P50 (RunInstances → daemon-ready) must be ≤ 30s.

    The Packer-baked AMI eliminates the apt-get update step that dominated the B0
    spike's 31.8s user-data phase. With pre-installed node/python/git/gh/uv/bun,
    cold-boot should land in the 18-25s range. P50 ≤ 30s gives ~5s headroom for
    AWS variability.

    Daemon-ready is approximated by polling `provider.status()` for `VMState.RUNNING`
    — once Epic C (LIT-2879) ships the real daemon callback, swap this for the
    session-row `status == 'ready'` check.
    """
    import statistics  # noqa: PLC0415

    handles: List[VMHandle] = []
    provider = EC2Provider({"default_ami_id": os.environ["LITELLM_TEST_AMI_ID"]})
    watchdog = _install_watchdog(provider, handles)

    latencies: List[float] = []
    try:
        for i in range(5):
            ctx = ProvisionContext(
                session_id=f"sess-coldboot-{i}-{int(time.time())}",
                team_id="team-coldboot",
                agent_id="agent-coldboot",
                repos=[Repo(url="https://github.com/octocat/Hello-World")],
                env_vars={"FOO": "bar"},
                aws_creds=_aws_creds_from_env(),
                ec2_config=_ec2_config_from_env(),
                daemon_jwt="not-a-real-jwt-test-only",
                daemon_base_url="https://example.invalid/",
                mode="session",
            )

            handle: Optional[VMHandle] = None
            try:
                t0 = time.perf_counter()
                handle = await provider.provision(ctx)
                handles.append(handle)

                # Poll for daemon-ready (approximated via VMState.RUNNING).
                # 60s timeout gives 2x headroom over the 30s P50 gate.
                deadline = time.time() + 60
                last: Optional[VMState] = None
                while time.time() < deadline:
                    status = await provider.status(handle, aws_creds=ctx.aws_creds)
                    last = status.state
                    if status.state == VMState.RUNNING:
                        break
                    time.sleep(1)
                assert (
                    last == VMState.RUNNING
                ), f"iteration {i}: never reached running, last={last}"
                latencies.append(time.perf_counter() - t0)
            finally:
                if handle is not None:
                    await provider.terminate(handle, aws_creds=ctx.aws_creds)

        p50 = statistics.median(latencies)
        # P95 with n=5 → use the max sample as an approximation.
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        print(f"Cold-boot latencies (s): {latencies}")
        print(f"P50 = {p50:.1f}s, P95 = {p95:.1f}s")
        assert p50 <= 30.0, f"Cold-boot P50 {p50:.1f}s exceeds 30s gate"
    finally:
        watchdog.cancel()
