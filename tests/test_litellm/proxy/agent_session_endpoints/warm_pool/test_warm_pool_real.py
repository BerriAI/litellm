"""Real-cloud tests for the warm-pool attach path (LIT-2890 validation #2).

These tests launch real EC2 instances and issue real SSM RunCommand calls to
measure the actual session-create latency. Marked ``@pytest.mark.slow`` and
gated on ``LITELLM_RUN_WARMPOOL_REAL=1`` so they don't run in CI.

**Cost:** ~$0.50 per full P95 run (50 sequential creates, t3.large, spot).
Every instance is tagged ``litellm-test-warmpool`` and the teardown fixture
terminates everything tagged at end-of-run.

**The gating test** is ``test_attach_latency_p95``: it asserts P95 < 3000ms
across 50 sequential creates. PR cannot merge if this fails.

**Why these tests don't depend on the full proxy stack:**
The latency that matters is the ``attach.attach_warm_vm`` path — claim a
warm row, build the hydrate payload, push it via SSM. We instantiate the
attach module directly with an in-memory prisma and a real
``SSMHydrateTransport``. The DB write + JWT mint cost a constant ~5ms in
production; the variable cost is the SSM round-trip, which IS exercised
end-to-end here.

How to run::

    LITELLM_RUN_WARMPOOL_REAL=1 \\
    AWS_PROFILE=litellm-poc \\
    AWS_REGION=us-west-2 \\
    LITELLM_TEST_WARMPOOL_AMI_ID=ami-074a518157fe137b4 \\
    LITELLM_TEST_WARMPOOL_SUBNET_ID=subnet-... \\
    LITELLM_TEST_WARMPOOL_SECURITY_GROUP=sg-... \\
    LITELLM_TEST_WARMPOOL_IAM_PROFILE=AmazonSSMManagedInstanceProfile \\
    pytest tests/test_litellm/proxy/agent_session_endpoints/warm_pool/test_warm_pool_real.py -v -m slow

If ``LITELLM_TEST_WARMPOOL_AMI_ID`` is unset, the test falls back to the
latest Amazon Linux 2023 AMI (which already has the SSM agent installed)
so it can validate the latency gate without B1's daemon AMI.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest


_REAL_CLOUD_GATE = os.environ.get("LITELLM_RUN_WARMPOOL_REAL") == "1"

pytestmark = pytest.mark.skipif(
    not _REAL_CLOUD_GATE,
    reason="real-cloud warm-pool tests gated by LITELLM_RUN_WARMPOOL_REAL=1",
)


TAG_KEY = "litellm-test-warmpool"
TAG_VALUE = "true"
WARM_POOL_SIZE = 5
P95_GATE_MS = 3000


# ---------------------------------------------------------------------------
# AWS helpers
# ---------------------------------------------------------------------------


def _aws_region() -> str:
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-west-2"
    )


def _ec2_client() -> Any:
    import boto3  # noqa: PLC0415

    return boto3.client("ec2", region_name=_aws_region())


def _ssm_client() -> Any:
    import boto3  # noqa: PLC0415

    return boto3.client("ssm", region_name=_aws_region())


def _resolve_ami_id() -> str:
    """Return the AMI to launch warm instances from.

    Preference order:
      1. ``LITELLM_TEST_WARMPOOL_AMI_ID`` (B1's daemon AMI when available)
      2. Latest Amazon Linux 2023 AMI (Maintained by AWS, has SSM agent)
    """
    explicit = os.environ.get("LITELLM_TEST_WARMPOOL_AMI_ID")
    if explicit:
        return explicit
    ec2 = _ec2_client()
    response = ec2.describe_images(
        Owners=["amazon"],
        Filters=[
            {"Name": "name", "Values": ["al2023-ami-2023.*-x86_64"]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": ["x86_64"]},
        ],
    )
    images = sorted(
        response.get("Images", []),
        key=lambda i: i.get("CreationDate", ""),
        reverse=True,
    )
    if not images:
        raise RuntimeError(
            "No AL2023 AMI found and LITELLM_TEST_WARMPOOL_AMI_ID is unset"
        )
    return images[0]["ImageId"]


def _launch_warm_instance(ami_id: str) -> str:
    """Launch one ``litellm-test-warmpool``-tagged instance, return its id."""
    ec2 = _ec2_client()
    kwargs: Dict[str, Any] = {
        "ImageId": ami_id,
        "InstanceType": "t3.large",
        "MinCount": 1,
        "MaxCount": 1,
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": TAG_KEY, "Value": TAG_VALUE},
                    {"Key": "Name", "Value": "litellm-test-warmpool"},
                ],
            }
        ],
        "InstanceMarketOptions": {
            "MarketType": "spot",
            "SpotOptions": {
                "InstanceInterruptionBehavior": "terminate",
                "SpotInstanceType": "one-time",
            },
        },
    }
    if os.environ.get("LITELLM_TEST_WARMPOOL_SUBNET_ID"):
        kwargs["SubnetId"] = os.environ["LITELLM_TEST_WARMPOOL_SUBNET_ID"]
    if os.environ.get("LITELLM_TEST_WARMPOOL_SECURITY_GROUP"):
        kwargs["SecurityGroupIds"] = [
            os.environ["LITELLM_TEST_WARMPOOL_SECURITY_GROUP"]
        ]
    if os.environ.get("LITELLM_TEST_WARMPOOL_IAM_PROFILE"):
        kwargs["IamInstanceProfile"] = {
            "Name": os.environ["LITELLM_TEST_WARMPOOL_IAM_PROFILE"]
        }

    response = ec2.run_instances(**kwargs)
    instance_id = response["Instances"][0]["InstanceId"]
    return instance_id


def _wait_until_ssm_online(instance_ids: List[str], timeout_seconds: int = 600) -> None:
    """Block until every instance shows ``Online`` in SSM (or timeout)."""
    ssm = _ssm_client()
    deadline = time.monotonic() + timeout_seconds
    pending = set(instance_ids)
    while pending and time.monotonic() < deadline:
        response = ssm.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": list(pending)}]
        )
        for info in response.get("InstanceInformationList", []):
            if info.get("PingStatus") == "Online":
                pending.discard(info["InstanceId"])
        if pending:
            time.sleep(5)
    if pending:
        raise RuntimeError(f"timed out waiting for SSM Online: {sorted(pending)}")


def _terminate_tagged_instances() -> None:
    """Teardown: terminate every ``litellm-test-warmpool``-tagged instance."""
    ec2 = _ec2_client()
    response = ec2.describe_instances(
        Filters=[
            {"Name": f"tag:{TAG_KEY}", "Values": [TAG_VALUE]},
            {
                "Name": "instance-state-name",
                "Values": ["pending", "running", "stopping", "stopped"],
            },
        ]
    )
    ids = [
        inst["InstanceId"]
        for res in response.get("Reservations", [])
        for inst in res.get("Instances", [])
    ]
    if ids:
        ec2.terminate_instances(InstanceIds=ids)


# ---------------------------------------------------------------------------
# In-memory prisma stand-in (subset needed by attach_warm_vm)
# ---------------------------------------------------------------------------


class _Row:
    def __init__(self, **fields: Any) -> None:
        for k, v in fields.items():
            setattr(self, k, v)


class _AgentVMTable:
    def __init__(self) -> None:
        self.rows: List[_Row] = []
        self._lock = asyncio.Lock()

    async def find_many(self, where=None, order=None, take=None):
        out: List[_Row] = []
        for r in self.rows:
            ok = all(getattr(r, k, None) == v for k, v in (where or {}).items())
            if ok:
                out.append(r)
        if order:
            for entry in order if isinstance(order, list) else [order]:
                for k, direction in entry.items():
                    out.sort(
                        key=lambda r: getattr(r, k, None) or 0,
                        reverse=(direction == "desc"),
                    )
        if take is not None:
            out = out[:take]
        return out

    async def find_unique(self, where: Dict[str, Any]):
        for r in self.rows:
            if getattr(r, "id", None) == where.get("id"):
                return r
        return None

    async def update_many(self, where: Dict[str, Any], data: Dict[str, Any]):
        async with self._lock:
            count = 0
            for r in self.rows:
                if all(getattr(r, k, None) == v for k, v in where.items()):
                    for k, v in data.items():
                        setattr(r, k, v)
                    count += 1
            return _Row(count=count)


class _Empty:
    async def find_many(self, where=None):
        return []

    async def find_unique(self, where=None):
        return None


class _DB:
    def __init__(self) -> None:
        self.litellm_agentvm = _AgentVMTable()
        self.litellm_agentsecret = _Empty()
        self.litellm_agentvmconfig = _Empty()


class _Prisma:
    def __init__(self) -> None:
        self.db = _DB()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def warm_instance_ids():
    """Pre-warm ``WARM_POOL_SIZE`` real EC2 instances. Always teardown.

    Returns the list of instance IDs once SSM is Online for every one.
    """
    try:
        ami_id = _resolve_ami_id()
    except Exception as exc:
        pytest.skip(f"could not resolve AMI: {exc}")

    instance_ids: List[str] = []
    try:
        for _ in range(WARM_POOL_SIZE):
            instance_ids.append(_launch_warm_instance(ami_id))
        _wait_until_ssm_online(instance_ids)
        yield instance_ids
    finally:
        try:
            _terminate_tagged_instances()
        except Exception as exc:
            print(f"WARN: teardown terminate_instances failed: {exc}")


@pytest.fixture
def prisma_with_warm_pool(warm_instance_ids):
    prisma = _Prisma()
    now = datetime.now(timezone.utc)
    for i, vm_id in enumerate(warm_instance_ids):
        prisma.db.litellm_agentvm.rows.append(
            _Row(
                id=vm_id,
                team_id="team-real",
                pool_id="team-real",
                state="warm",
                provider="ec2",
                region=_aws_region(),
                warmed_at=now - timedelta(seconds=WARM_POOL_SIZE - i),
                metadata={},
            )
        )
    return prisma


@pytest.fixture
def real_aws_creds():
    from litellm.proxy.agent_session_endpoints.vm_providers.base import AwsCreds

    # boto3 picks up creds from the default chain; we just need to ensure the
    # AwsCreds type carries something. The SSMHydrateTransport reaches into
    # boto3 directly so the fields here are placeholders for the function
    # signature; the real auth happens via the AWS_PROFILE env var.
    return AwsCreds(
        access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "from-profile"),
        secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "from-profile"),
        session_token=os.environ.get("AWS_SESSION_TOKEN"),
        region=_aws_region(),
    )


@pytest.fixture
def stub_team_creds(monkeypatch, real_aws_creds):
    """Monkeypatch ``get_team_vm_config`` so we don't need an encrypted DB row."""
    from litellm.proxy.agent_session_endpoints.vm_providers.base import Ec2Config
    from litellm.proxy.agent_session_endpoints.vm_providers.team_config import (
        TeamVMConfig,
    )

    async def fake(team_id, prisma_client, default_region="us-west-2"):
        return TeamVMConfig(
            aws_creds=real_aws_creds,
            ec2_config=Ec2Config(region=_aws_region()),
        )

    monkeypatch.setattr(
        "litellm.proxy.agent_session_endpoints.warm_pool.attach.get_team_vm_config",
        fake,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.asyncio
async def test_pool_fills_and_ssm_online(warm_instance_ids):
    """Sanity: ``WARM_POOL_SIZE`` instances launched and registered with SSM."""
    assert len(warm_instance_ids) == WARM_POOL_SIZE
    assert all(iid.startswith("i-") for iid in warm_instance_ids)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_attach_latency_p95(
    prisma_with_warm_pool, stub_team_creds, real_aws_creds
):
    """**Headline gate:** P95 of POST-equivalent attach across 50 sequential
    creates < 3000ms.

    Each iteration:
      1. Adds a fresh "warm" row simulating the maintenance loop refilling
         (we reuse the same EC2 instance across iterations because launching
         50 real instances would cost too much; the SSM RunCommand path is
         what we're measuring, not the EC2 boot path)
      2. Calls ``attach_warm_vm`` end-to-end, including real SSM push
      3. Records wall-clock latency
      4. Resets the row state to ``warm`` for the next iteration

    This isolates the latency we care about: claim + payload-build + SSM-push.
    The proxy DB write + JWT mint is NOT in this path here, but those are
    constant ~5ms operations measured separately; the variable cost — and the
    cost B0 measured at 1700ms — is the SSM round-trip.
    """
    from litellm.proxy.agent_session_endpoints.warm_pool.attach import (
        attach_warm_vm,
    )
    from litellm.proxy.agent_session_endpoints.warm_pool.transports.ssm import (
        SSMHydrateTransport,
    )

    # Real SSM transport — same as production.
    transport = SSMHydrateTransport()
    latencies_ms: List[float] = []

    for iteration in range(50):
        # Reset every row to warm (so we always have something to claim).
        for row in prisma_with_warm_pool.db.litellm_agentvm.rows:
            row.state = "warm"
            row.attached_session_id = None

        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        expires = datetime.now(timezone.utc) + timedelta(hours=1)

        t0 = time.perf_counter()
        await attach_warm_vm(
            prisma_client=prisma_with_warm_pool,
            team_id="team-real",
            session_id=session_id,
            agent_id=agent_id,
            jwt="test-jwt-token",
            jwt_expires_at=expires,
            repos=[{"url": "https://github.com/example/repo"}],
            env_vars={"FOO": "bar"},
            transport=transport,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)
        print(f"iter {iteration:02d}: {elapsed_ms:.0f}ms")

    p50 = statistics.median(latencies_ms)
    sorted_ms = sorted(latencies_ms)
    p95 = sorted_ms[int(len(sorted_ms) * 0.95)]
    p99 = sorted_ms[int(len(sorted_ms) * 0.99)]

    print()
    print(f"WARM-POOL ATTACH LATENCY (n={len(latencies_ms)})")
    print(f"  P50 = {p50:.0f}ms")
    print(f"  P95 = {p95:.0f}ms  (gate: <{P95_GATE_MS}ms)")
    print(f"  P99 = {p99:.0f}ms")

    assert p95 < P95_GATE_MS, (
        f"P95 {p95:.0f}ms exceeds {P95_GATE_MS}ms gate "
        f"(P50={p50:.0f}, P99={p99:.0f})"
    )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_attach_no_double_claim_real(
    prisma_with_warm_pool, stub_team_creds, real_aws_creds
):
    """5 concurrent attaches against pool size 5 -> all 5 succeed, no double-claim."""
    from litellm.proxy.agent_session_endpoints.warm_pool.attach import (
        attach_warm_vm,
    )
    from litellm.proxy.agent_session_endpoints.warm_pool.transports.ssm import (
        SSMHydrateTransport,
    )

    transport = SSMHydrateTransport()

    async def one(sid: str):
        return await attach_warm_vm(
            prisma_client=prisma_with_warm_pool,
            team_id="team-real",
            session_id=sid,
            agent_id="agent-x",
            jwt="jwt",
            jwt_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            repos=[],
            env_vars=None,
            transport=transport,
        )

    results = await asyncio.gather(*(one(f"sess_{i}") for i in range(WARM_POOL_SIZE)))
    vm_ids = [r.vm_id for r in results]
    # No two attaches got the same vm_id.
    assert len(set(vm_ids)) == len(vm_ids)
