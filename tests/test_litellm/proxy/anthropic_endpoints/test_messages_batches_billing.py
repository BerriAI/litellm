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
    # no anthropic deployments at all -> passthrough-handler string convention
    monkeypatch.setattr(mb, "_get_llm_router", lambda: SimpleNamespace(get_model_list=lambda: []))
    assert mb._find_anthropic_deployment_model_id("claude-haiku-4-5") == "anthropic/claude-haiku-4-5"


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
    # the just-created job must have been stopped
    assert any(url.endswith("/stop") for _method, url in aws_calls), aws_calls
