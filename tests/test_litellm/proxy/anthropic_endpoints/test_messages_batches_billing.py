"""Unit tests for /v1/messages/batches spend-tracking registration.

Covers: unified-id round-trip with the CheckBatchCost decoders, the
attribution stash read by CheckBatchCost._get_job_attribution, the
ManagedObjectTable upsert shape, deployment-id resolution for the upstream
leg, and the ANTHROPIC_BATCHES_REQUIRE_BILLING refusal path on the Bedrock
create leg.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm.proxy.anthropic_endpoints.messages_batches as mb
from litellm.proxy._types import UserAPIKeyAuth

JOB_ARN = "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/abc123xyz"


def _auth(**overrides):
    defaults = dict(
        api_key="a" * 64,  # UserAPIKeyAuth carries the hash
        user_id="user-1",
        team_id="team-1",
        end_user_id="end-user-1",
        key_alias="alias-1",
    )
    defaults.update(overrides)
    return UserAPIKeyAuth(**defaults)


class _PrismaStub:
    def __init__(self):
        self.upsert = AsyncMock()
        self.db = SimpleNamespace(litellm_managedobjecttable=SimpleNamespace(upsert=self.upsert))


# ── unified id round-trips through the CheckBatchCost decoders ───────────────


def _decode_like_check_batch_cost(unified: str):
    """Mirror CheckBatchCost._resolve_job_routing's managed-id pipeline."""
    from litellm.proxy.openai_files_endpoints.common_utils import (
        _is_base64_encoded_unified_file_id,
        get_batch_id_from_unified_batch_id,
        get_model_id_from_unified_batch_id,
    )

    decoded = _is_base64_encoded_unified_file_id(unified)
    assert decoded, f"unified id did not decode as a managed id: {unified!r}"
    return get_model_id_from_unified_batch_id(decoded), get_batch_id_from_unified_batch_id(decoded)


def test_unified_batch_id_round_trip():
    unified = mb._unified_batch_object_id("deployment-id-123", JOB_ARN)
    assert _decode_like_check_batch_cost(unified) == ("deployment-id-123", JOB_ARN)


def test_unified_batch_id_round_trip_msgbatch():
    unified = mb._unified_batch_object_id("anthropic/claude-opus-4-6", "msgbatch_01ABC")
    assert _decode_like_check_batch_cost(unified) == ("anthropic/claude-opus-4-6", "msgbatch_01ABC")


# ── CheckBatchCost._get_job_attribution reads the stash ──────────────────────


def test_get_job_attribution_from_json_string():
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    attribution = {"user_api_key": "b" * 64, "user_api_key_team_id": "team-9"}
    job = SimpleNamespace(file_object=json.dumps({"id": "x", "litellm_attribution": attribution}))
    assert CheckBatchCost._get_job_attribution(job) == attribution


def test_get_job_attribution_from_dict_and_missing():
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    job = SimpleNamespace(file_object={"litellm_attribution": {"user_api_key": "k"}})
    assert CheckBatchCost._get_job_attribution(job) == {"user_api_key": "k"}
    assert CheckBatchCost._get_job_attribution(SimpleNamespace(file_object=None)) == {}
    assert CheckBatchCost._get_job_attribution(SimpleNamespace(file_object=json.dumps({"id": "x"}))) == {}
    assert CheckBatchCost._get_job_attribution(SimpleNamespace(file_object="not json")) == {}


def test_get_job_attribution_double_encoded():
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    inner = json.dumps({"litellm_attribution": {"user_api_key": "c" * 64}})
    job = SimpleNamespace(file_object=json.dumps(inner))
    assert CheckBatchCost._get_job_attribution(job) == {"user_api_key": "c" * 64}


def test_get_job_file_object_model_stash():
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    job = SimpleNamespace(
        file_object=json.dumps(
            {"model": "claude-opus-4-6", "litellm_attribution": {"user_api_key": "d" * 64}}
        )
    )
    file_object = CheckBatchCost._get_job_file_object(job)
    # the tracker prefers the stashed model over the (possibly borrowed)
    # deployment's model_name only when the attribution stash is present
    assert file_object.get("model") == "claude-opus-4-6"
    assert file_object.get("litellm_attribution")
    assert CheckBatchCost._get_job_file_object(SimpleNamespace(file_object="nope")) == {}


# ── _record_batch_for_billing ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_batch_upsert_shape(monkeypatch):
    prisma = _PrismaStub()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)

    ok = await mb._record_batch_for_billing(
        provider_batch_id=JOB_ARN,
        router_model_id="deployment-id-123",
        client_batch_id="msgbatch_bedrock_abc123xyz_deadbeef",
        model_name="claude-opus-4-6-batch",
        total_records=100,
        user_api_key_dict=_auth(),
    )
    assert ok is True
    prisma.upsert.assert_awaited_once()
    kwargs = prisma.upsert.await_args.kwargs
    create = kwargs["data"]["create"]
    assert kwargs["where"] == {"unified_object_id": mb._unified_batch_object_id("deployment-id-123", JOB_ARN)}
    assert create["model_object_id"] == JOB_ARN
    assert create["file_purpose"] == "batch"
    assert create["status"] == "validating"
    assert create["created_by"] == "user-1"
    assert create["team_id"] == "team-1"
    file_object = json.loads(create["file_object"])
    assert file_object["id"] == "msgbatch_bedrock_abc123xyz_deadbeef"
    assert file_object["litellm_attribution"] == {
        "user_api_key": "a" * 64,
        "user_api_key_user_id": "user-1",
        "user_api_key_team_id": "team-1",
        "user_api_key_end_user_id": "end-user-1",
        "user_api_key_alias": "alias-1",
    }


@pytest.mark.asyncio
async def test_record_batch_no_db_or_model_id(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    assert (
        await mb._record_batch_for_billing(
            provider_batch_id=JOB_ARN,
            router_model_id="deployment-id-123",
            client_batch_id="x",
            model_name="m",
            total_records=1,
            user_api_key_dict=_auth(),
        )
        is False
    )

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _PrismaStub())
    assert (
        await mb._record_batch_for_billing(
            provider_batch_id=JOB_ARN,
            router_model_id=None,
            client_batch_id="x",
            model_name="m",
            total_records=1,
            user_api_key_dict=_auth(),
        )
        is False
    )


@pytest.mark.asyncio
async def test_record_batch_upsert_failure_returns_false(monkeypatch):
    prisma = _PrismaStub()
    prisma.upsert.side_effect = RuntimeError("db down")
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)
    ok = await mb._record_batch_for_billing(
        provider_batch_id=JOB_ARN,
        router_model_id="deployment-id-123",
        client_batch_id="x",
        model_name="m",
        total_records=1,
        user_api_key_dict=_auth(),
    )
    assert ok is False


# ── anthropic deployment id resolution (upstream leg) ────────────────────────


def test_find_anthropic_deployment_model_id(monkeypatch):
    deployments = [
        {  # bedrock deployment for the same model group — must be skipped
            "model_name": "claude-opus-4-6",
            "litellm_params": {"model": "bedrock/us.anthropic.claude-opus-4-6-v1"},
            "model_info": {"id": "bedrock-dep-id"},
        },
        {
            "model_name": "claude-opus-4-6",
            "litellm_params": {"model": "anthropic/claude-opus-4-6"},
            "model_info": {"id": "anthropic-dep-id"},
        },
        {  # this stack's spelling: bare model + custom_llm_provider (live config)
            "model_name": "claude-opus-4-8",
            "litellm_params": {"model": "claude-opus-4-8", "custom_llm_provider": "anthropic"},
            "model_info": {"id": "anthropic-claude-opus-4-8"},
        },
    ]
    router = SimpleNamespace(get_model_list=lambda: deployments)
    monkeypatch.setattr(mb, "_get_llm_router", lambda: router)
    assert mb._find_anthropic_deployment_model_id("claude-opus-4-6") == "anthropic-dep-id"
    assert mb._find_anthropic_deployment_model_id("claude-opus-4-8") == "anthropic-claude-opus-4-8"
    # model without its own anthropic deployment -> borrow ANY anthropic
    # deployment (shared workspace credentials; rows price by their own model)
    assert mb._find_anthropic_deployment_model_id("claude-haiku-4-5") == "anthropic-dep-id"
    # no anthropic deployments at all -> None (the "anthropic/<model>" string
    # is unrouteable — registering it would satisfy REQUIRE_BILLING with a row
    # the poller can never price)
    monkeypatch.setattr(mb, "_get_llm_router", lambda: SimpleNamespace(get_model_list=lambda: []))
    assert mb._find_anthropic_deployment_model_id("claude-haiku-4-5") is None


def test_s3_prefixes_under_managed_namespace():
    """The poller's output download rejects reads outside the managed
    prefixes — creation must stage under them."""
    from litellm.litellm_core_utils.cloud_storage_security import BEDROCK_MANAGED_S3_PREFIXES

    assert mb._S3_INPUT_PREFIX.startswith(BEDROCK_MANAGED_S3_PREFIXES)
    assert mb._S3_OUTPUT_PREFIX.startswith(BEDROCK_MANAGED_S3_PREFIXES)


def test_billing_required_env(monkeypatch):
    monkeypatch.delenv(mb._REQUIRE_BILLING_ENV, raising=False)
    assert mb._billing_required() is False
    monkeypatch.setenv(mb._REQUIRE_BILLING_ENV, "true")
    assert mb._billing_required() is True
    monkeypatch.setenv(mb._REQUIRE_BILLING_ENV, "false")
    assert mb._billing_required() is False


# ── Bedrock create leg: billing-required refusal stops the job ───────────────


def _bedrock_deployment():
    return {
        "model_name": "claude-opus-4-6-batch",
        "litellm_params": {
            "model": "bedrock/us.anthropic.claude-opus-4-6-v1",
            "aws_batch_role_arn": "arn:aws:iam::123456789012:role/batch-role",
            "s3_bucket_name": "test-bucket",
            "s3_region_name": "us-west-2",
        },
        "model_info": {"id": "bedrock-dep-id", "mode": "batch"},
    }


def _create_body(n=100):
    return {
        "requests": [
            {
                "custom_id": f"r-{i}",
                "params": {"model": "claude-opus-4-6", "max_tokens": 8, "messages": []},
            }
            for i in range(n)
        ]
    }


@pytest.mark.asyncio
async def test_bedrock_create_records_billing_row(monkeypatch):
    prisma = _PrismaStub()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)
    monkeypatch.setattr(mb, "_find_bedrock_batch_deployment", lambda model: _bedrock_deployment())

    async def _fake_read_body(request):
        return _create_body()

    async def _fake_check_model_access(models, user_api_key_dict):
        return None

    aws_calls = []

    async def _fake_aws_call(method, url, body, service, aws_params, region):
        aws_calls.append((method, url))
        if service == "s3":
            return SimpleNamespace(status_code=200, text="")
        return SimpleNamespace(status_code=200, text="", json=lambda: {"jobArn": JOB_ARN})

    monkeypatch.setattr(mb, "_read_request_body", _fake_read_body)
    monkeypatch.setattr(mb, "_check_model_access", _fake_check_model_access)
    monkeypatch.setattr(mb, "_aws_call", _fake_aws_call)

    response = await mb.create_message_batch(MagicMock(), MagicMock(), _auth())
    payload = json.loads(response.body)
    assert payload["id"].startswith("msgbatch_bedrock_abc123xyz_")
    prisma.upsert.assert_awaited_once()
    assert prisma.upsert.await_args.kwargs["data"]["create"]["model_object_id"] == JOB_ARN


@pytest.mark.asyncio
async def test_bedrock_create_refused_when_billing_unavailable(monkeypatch):
    monkeypatch.setenv(mb._REQUIRE_BILLING_ENV, "true")
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)  # billing row cannot be stored
    monkeypatch.setattr(mb, "_find_bedrock_batch_deployment", lambda model: _bedrock_deployment())

    async def _fake_read_body(request):
        return _create_body()

    async def _fake_check_model_access(models, user_api_key_dict):
        return None

    aws_calls = []

    async def _fake_aws_call(method, url, body, service, aws_params, region):
        aws_calls.append((method, url))
        if service == "s3":
            return SimpleNamespace(status_code=200, text="")
        return SimpleNamespace(status_code=200, text="", json=lambda: {"jobArn": JOB_ARN})

    monkeypatch.setattr(mb, "_read_request_body", _fake_read_body)
    monkeypatch.setattr(mb, "_check_model_access", _fake_check_model_access)
    monkeypatch.setattr(mb, "_aws_call", _fake_aws_call)

    response = await mb.create_message_batch(MagicMock(), MagicMock(), _auth())
    assert response.status_code == 503
    # the preflight refuses BEFORE any provider-side work: no S3 staging, no
    # job creation, nothing to stop (codex P1 — the old flow created the job
    # first and only best-effort-stopped it)
    assert aws_calls == [], aws_calls


@pytest.mark.asyncio
async def test_bedrock_create_stops_job_when_row_write_fails(monkeypatch):
    """Preflight passes (DB up) but the row write itself fails after the job
    exists -> the job is stopped and the stop is verified."""
    monkeypatch.setenv(mb._REQUIRE_BILLING_ENV, "true")
    prisma = _PrismaStub()
    prisma.upsert.side_effect = RuntimeError("db write failed")
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)
    monkeypatch.setattr(mb, "_find_bedrock_batch_deployment", lambda model: _bedrock_deployment())

    async def _fake_read_body(request):
        return _create_body()

    async def _fake_check_model_access(models, user_api_key_dict):
        return None

    aws_calls = []

    async def _fake_aws_call(method, url, body, service, aws_params, region):
        aws_calls.append((method, url))
        if service == "s3":
            return SimpleNamespace(status_code=200, text="")
        return SimpleNamespace(status_code=200, text="", json=lambda: {"jobArn": JOB_ARN})

    monkeypatch.setattr(mb, "_read_request_body", _fake_read_body)
    monkeypatch.setattr(mb, "_check_model_access", _fake_check_model_access)
    monkeypatch.setattr(mb, "_aws_call", _fake_aws_call)

    response = await mb.create_message_batch(MagicMock(), MagicMock(), _auth())
    assert response.status_code == 503
    assert any(url.endswith("/stop") for _method, url in aws_calls), aws_calls


# ── ownership checks (codex P1 fixes) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_bedrock_owner_check_rejects_untagged_ids(monkeypatch):
    """Stripping the _owner8 suffix off a leaked id must not bypass ownership
    (legacy tag path — no DB configured)."""
    from fastapi import HTTPException

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    untagged = "msgbatch_bedrock_abc123xyz"
    with pytest.raises(HTTPException) as exc:
        await mb._check_bedrock_batch_owner(untagged, _auth())
    assert exc.value.status_code == 404
    # admins may still access untagged (pre-owner-tag) ids
    admin = _auth()
    admin.user_role = "proxy_admin"
    await mb._check_bedrock_batch_owner(untagged, admin)


@pytest.mark.asyncio
async def test_bedrock_owner_check_is_row_based(monkeypatch):
    """A spliced id (victim job id + the ATTACKER'S OWN valid tag) must fail:
    ownership binds through the billing row, not the caller-supplied suffix
    (codex P1 round 2)."""
    from fastapi import HTTPException

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _PrismaStub())
    attacker = _auth(api_key="e" * 64, user_id="attacker", team_id="attacker-team")
    victim_row = _billing_row(attribution={"user_api_key": "a" * 64, "user_api_key_user_id": "user-1"})
    spliced_id = f"msgbatch_bedrock_victimjob_{mb._owner_tag(attacker)}"

    async def _victim_row(_batch_id):
        return victim_row

    monkeypatch.setattr(mb, "_get_billing_row", _victim_row)
    with pytest.raises(HTTPException) as exc:
        await mb._check_bedrock_batch_owner(spliced_id, attacker)
    assert exc.value.status_code == 404
    # the real owner still passes on the same row
    await mb._check_bedrock_batch_owner(spliced_id, _auth())

    # row-less id: billing-required -> 404; optional -> legacy tag decides
    async def _no_row(_batch_id):
        return None

    monkeypatch.setattr(mb, "_get_billing_row", _no_row)
    monkeypatch.setenv(mb._REQUIRE_BILLING_ENV, "true")
    with pytest.raises(HTTPException):
        await mb._check_bedrock_batch_owner(spliced_id, attacker)
    monkeypatch.setenv(mb._REQUIRE_BILLING_ENV, "false")
    await mb._check_bedrock_batch_owner(spliced_id, attacker)  # tag matches attacker


def _billing_row(attribution=None, batch_processed=False, team_id=None, created_by=None, status="validating"):
    file_object = {"id": "x", "litellm_attribution": attribution or {}}
    return SimpleNamespace(
        file_object=json.dumps(file_object),
        batch_processed=batch_processed,
        team_id=team_id,
        created_by=created_by,
        status=status,
    )


@pytest.mark.asyncio
async def test_upstream_owner_check(monkeypatch):
    from fastapi import HTTPException

    caller = _auth()
    row_mine = _billing_row(attribution={"user_api_key": caller.api_key})
    row_foreign = _billing_row(attribution={"user_api_key": "f" * 64, "user_api_key_team_id": "other-team"})

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _PrismaStub())

    async def _row(_batch_id):
        return row_mine

    monkeypatch.setattr(mb, "_get_billing_row", _row)
    await mb._check_upstream_batch_owner("msgbatch_01X", caller)  # no raise

    async def _foreign(_batch_id):
        return row_foreign

    monkeypatch.setattr(mb, "_get_billing_row", _foreign)
    with pytest.raises(HTTPException) as exc:
        await mb._check_upstream_batch_owner("msgbatch_01X", caller)
    assert exc.value.status_code == 404

    async def _missing(_batch_id):
        return None

    monkeypatch.setattr(mb, "_get_billing_row", _missing)
    monkeypatch.setenv(mb._REQUIRE_BILLING_ENV, "true")
    with pytest.raises(HTTPException):
        await mb._check_upstream_batch_owner("msgbatch_01X", caller)
    # billing optional: a registration failure is survivable by design, so a
    # row-less batch stays usable (owner would otherwise be locked out)
    monkeypatch.setenv(mb._REQUIRE_BILLING_ENV, "false")
    await mb._check_upstream_batch_owner("msgbatch_01X", caller)

    # admin bypasses; no-DB deployments keep the historical shared posture
    admin = _auth()
    admin.user_role = "proxy_admin"
    await mb._check_upstream_batch_owner("msgbatch_01X", admin)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    await mb._check_upstream_batch_owner("msgbatch_01X", caller)


@pytest.mark.asyncio
async def test_unbilled_delete_block(monkeypatch):
    caller = _auth()

    async def _unbilled(_batch_id, **_kwargs):
        return _billing_row(batch_processed=False)

    monkeypatch.setattr(mb, "_get_billing_row", _unbilled)
    blocked = await mb._unbilled_delete_block("msgbatch_01X", caller)
    assert blocked is not None and blocked.status_code == 400

    # batch_processed=True while the claim is merely held ('pricing') is NOT
    # finalization — deletion mid-pricing would erase the priced output
    async def _pricing(_batch_id, **_kwargs):
        return _billing_row(batch_processed=True, status="pricing")

    monkeypatch.setattr(mb, "_get_billing_row", _pricing)
    blocked = await mb._unbilled_delete_block("msgbatch_01X", caller)
    assert blocked is not None and blocked.status_code == 400

    async def _billed(_batch_id, **_kwargs):
        return _billing_row(batch_processed=True, status="complete")

    monkeypatch.setattr(mb, "_get_billing_row", _billed)
    assert await mb._unbilled_delete_block("msgbatch_01X", caller) is None

    async def _none(_batch_id, **_kwargs):
        return None

    monkeypatch.setattr(mb, "_get_billing_row", _none)
    assert await mb._unbilled_delete_block("msgbatch_01X", caller) is None

    # DB lookup errors fail CLOSED when billing is required
    async def _boom(_batch_id, **kwargs):
        if kwargs.get("raise_on_error"):
            raise RuntimeError("db down")
        return None

    monkeypatch.setenv(mb._REQUIRE_BILLING_ENV, "true")
    monkeypatch.setattr(mb, "_get_billing_row", _boom)
    blocked = await mb._unbilled_delete_block("msgbatch_01X", caller)
    assert blocked is not None and blocked.status_code == 500

    # admins bypass the gate entirely
    admin = _auth()
    admin.user_role = "proxy_admin"
    assert await mb._unbilled_delete_block("msgbatch_01X", admin) is None


def test_billing_preflight(monkeypatch):
    monkeypatch.delenv(mb._REQUIRE_BILLING_ENV, raising=False)
    assert mb._billing_preflight(None) is None  # billing not required

    monkeypatch.setenv(mb._REQUIRE_BILLING_ENV, "true")
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", _PrismaStub())
    assert mb._billing_preflight("dep-id") is None
    refused = mb._billing_preflight(None)  # unrouteable
    assert refused is not None and refused.status_code == 503
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    refused = mb._billing_preflight("dep-id")  # no DB
    assert refused is not None and refused.status_code == 503


@pytest.mark.asyncio
async def test_record_batch_stashes_mixed_models_flag(monkeypatch):
    prisma = _PrismaStub()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)
    await mb._record_batch_for_billing(
        provider_batch_id="msgbatch_01MIX",
        router_model_id="anthropic-dep-id",
        client_batch_id="msgbatch_01MIX",
        model_name="claude-opus-4-6",
        total_records=2,
        user_api_key_dict=_auth(),
        mixed_models=True,
    )
    file_object = json.loads(prisma.upsert.await_args.kwargs["data"]["create"]["file_object"])
    assert file_object["mixed_models"] is True


# ── poller round-2 behaviors (codex P1/P2 round 2) ───────────────────────────


def test_finalized_file_object_preserves_attribution():
    """Finalization must keep litellm_attribution — the upstream ownership
    check matches on it, so dropping it would 404 key-only callers on their
    own batch after billing."""
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    job = SimpleNamespace(
        file_object=json.dumps(
            {
                "id": "msgbatch_01X",
                "model": "claude-opus-4-6",
                "mixed_models": False,
                "litellm_attribution": {"user_api_key": "a" * 64},
            }
        )
    )
    response = MagicMock()
    response.model_dump_json.return_value = json.dumps({"id": "msgbatch_01X", "status": "completed"})
    finalized = json.loads(CheckBatchCost._finalized_file_object(job, response))
    assert finalized["litellm_attribution"] == {"user_api_key": "a" * 64}
    assert finalized["model"] == "claude-opus-4-6"
    assert finalized["status"] == "completed"


@pytest.mark.asyncio
async def test_reclaim_abandoned_pricing_claims():
    """A worker that dies after claiming leaves batch_processed=True,
    status='pricing' — invisible to the primary query forever. The reclaim
    sweep conditionally requeues stale claims."""
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    instance = CheckBatchCost.__new__(CheckBatchCost)
    instance._has_batch_processed_column = True
    update_many = AsyncMock(return_value=2)
    instance.prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=SimpleNamespace(update_many=update_many))
    )
    await instance._reclaim_abandoned_pricing_claims()
    where = update_many.await_args.kwargs["where"]
    assert where["status"] == "pricing"
    assert where["batch_processed"] is True
    assert "updated_at" in where
    assert update_many.await_args.kwargs["data"] == {"batch_processed": False, "status": "validating"}


def test_spend_logs_id_honors_batch_cost_prefix():
    """aretrieve_batch spend ids hash the response (user-poll dedup), but the
    poller's deterministic prefixed call id must win — the response hash
    changes every retry via freshly-minted managed file ids (codex P1 r4)."""
    from litellm.proxy.spend_tracking.spend_tracking_utils import (
        BATCH_COST_CALL_ID_PREFIX,
        get_spend_logs_id,
    )

    deterministic = BATCH_COST_CALL_ID_PREFIX + "abc-123"
    assert (
        get_spend_logs_id("aretrieve_batch", {"id": "x"}, {"litellm_call_id": deterministic})
        == deterministic
    )
    # random user-poll call ids keep the response-hash dedup
    hashed = get_spend_logs_id("aretrieve_batch", {"id": "x"}, {"litellm_call_id": "9f2c0b7e"})
    assert hashed != "9f2c0b7e"
    assert hashed == get_spend_logs_id("aretrieve_batch", {"id": "x"}, {"litellm_call_id": "other"})
    # non-batch call types keep response id / call id precedence
    assert get_spend_logs_id("acompletion", {"id": "resp-1"}, {"litellm_call_id": "c"}) == "resp-1"


@pytest.mark.asyncio
async def test_spend_already_recorded_lookup():
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    instance = CheckBatchCost.__new__(CheckBatchCost)
    find_unique = AsyncMock(return_value=SimpleNamespace(request_id="x"))
    instance.prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_spendlogs=SimpleNamespace(find_unique=find_unique))
    )
    assert await instance._spend_already_recorded("msgbatch_01X") is True
    assert find_unique.await_args.kwargs["where"] == {
        "request_id": CheckBatchCost._batch_cost_call_id("msgbatch_01X")
    }
    find_unique.return_value = None
    assert await instance._spend_already_recorded("msgbatch_01X") is False
    # lookup errors fall through to False (claim + deterministic id still dedup the log)
    find_unique.side_effect = RuntimeError("db down")
    assert await instance._spend_already_recorded("msgbatch_01X") is False


# ── poller round-5 behaviors (codex P1 round 5) ──────────────────────────────


@pytest.mark.asyncio
async def test_spend_already_recorded_row_marker():
    """The row-local marker must dedup even when no SpendLogs row exists —
    disable_spend_logs deployments never write one, but the key/team/daily
    counters still increment (codex P1 round 5)."""
    from litellm_enterprise.proxy.common_utils.check_batch_cost import (
        SPEND_RECORDED_MARKER_KEY,
        CheckBatchCost,
    )

    instance = CheckBatchCost.__new__(CheckBatchCost)
    find_unique = AsyncMock(return_value=None)
    instance.prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_spendlogs=SimpleNamespace(find_unique=find_unique))
    )
    marked_job = SimpleNamespace(
        file_object=json.dumps({"id": "msgbatch_01X", SPEND_RECORDED_MARKER_KEY: True})
    )
    assert await instance._spend_already_recorded("msgbatch_01X", marked_job) is True
    find_unique.assert_not_awaited()  # marker short-circuits the DB lookup
    unmarked_job = SimpleNamespace(file_object=json.dumps({"id": "msgbatch_01X"}))
    assert await instance._spend_already_recorded("msgbatch_01X", unmarked_job) is False
    find_unique.assert_awaited()


@pytest.mark.asyncio
async def test_mark_spend_recorded_fenced_write():
    """The marker write is fenced to the held claim (like release/finalize)
    and merges into the existing file_object stash; failures are swallowed —
    spend already ran, so a marker error must not block finalization."""
    from litellm_enterprise.proxy.common_utils.check_batch_cost import (
        SPEND_RECORDED_MARKER_KEY,
        CheckBatchCost,
    )

    instance = CheckBatchCost.__new__(CheckBatchCost)
    instance._has_batch_processed_column = True
    update_many = AsyncMock(return_value=1)
    instance.prisma_client = SimpleNamespace(
        db=SimpleNamespace(litellm_managedobjecttable=SimpleNamespace(update_many=update_many))
    )
    job = SimpleNamespace(
        id="job-1",
        file_object=json.dumps(
            {"id": "msgbatch_01X", "litellm_attribution": {"user_api_key": "a" * 64}}
        ),
    )
    await instance._mark_spend_recorded(job)
    assert update_many.await_args.kwargs["where"] == {
        "id": "job-1",
        "status": "pricing",
        "batch_processed": True,
    }
    stamped = json.loads(update_many.await_args.kwargs["data"]["file_object"])
    assert stamped[SPEND_RECORDED_MARKER_KEY] is True
    assert stamped["litellm_attribution"] == {"user_api_key": "a" * 64}
    # write errors are swallowed (best effort)
    update_many.side_effect = RuntimeError("db down")
    await instance._mark_spend_recorded(job)


def test_finalized_file_object_stamps_spend_marker():
    """spend_recorded=True stamps the marker into the finalized row, and a
    marker already in the stash survives finalization by default."""
    from litellm_enterprise.proxy.common_utils.check_batch_cost import (
        SPEND_RECORDED_MARKER_KEY,
        CheckBatchCost,
    )

    response = MagicMock()
    response.model_dump_json.return_value = json.dumps(
        {"id": "msgbatch_01X", "status": "completed"}
    )
    plain_job = SimpleNamespace(file_object=json.dumps({"id": "msgbatch_01X"}))
    finalized = json.loads(
        CheckBatchCost._finalized_file_object(plain_job, response, spend_recorded=True)
    )
    assert finalized[SPEND_RECORDED_MARKER_KEY] is True
    # no spend emitted → no marker invented
    untracked = json.loads(CheckBatchCost._finalized_file_object(plain_job, response))
    assert SPEND_RECORDED_MARKER_KEY not in untracked
    # marker in the stash is preserved even without the flag
    marked_job = SimpleNamespace(
        file_object=json.dumps({"id": "msgbatch_01X", SPEND_RECORDED_MARKER_KEY: True})
    )
    preserved = json.loads(CheckBatchCost._finalized_file_object(marked_job, response))
    assert preserved[SPEND_RECORDED_MARKER_KEY] is True
