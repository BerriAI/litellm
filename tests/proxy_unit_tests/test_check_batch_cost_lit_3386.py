"""Regression tests for LIT-3386: CheckBatchCost background poller must route
managed-file-id normalization through ensure_batch_response_managed_file_ids
(the same helper update_batch_in_database uses on the retrieve/cancel HTTP
path) so that:

  1. input_file_id is resolved to an existing managed unified ID when a
     managed_file row exists for the raw provider ID (the legacy manual
     block in CheckBatchCost only handled output_file_id and error_file_id,
     leaving input_file_id raw).
  2. output_file_id / error_file_id continue to be registered with the
     batch creator's UserAPIKeyAuth as the auth context (created_by =
     real user, not default_user_id).
  3. Failures in ensure_batch_response_managed_file_ids do NOT abort the
     poller — they should log a warning and continue (the cost-tracking
     path and DB write must still happen).
"""
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


UNIFIED_JOB_ID_RAW = (
    "litellm_proxy;model_id:model-abc123;llm_batch_id:batch_xyz789;"
    "llm_output_file_id:placeholder"
)
UNIFIED_JOB_ID_B64 = (
    base64.urlsafe_b64encode(UNIFIED_JOB_ID_RAW.encode()).decode().rstrip("=")
)

RAW_INPUT = "file-bedrock-input-bbb111"
RAW_OUTPUT = "file-bedrock-output-aaa222"
RAW_ERROR = "file-bedrock-error-ccc333"
EXISTING_MANAGED_INPUT_RAW = (
    "litellm_proxy:application/jsonl;unified_id,aaa11111-bbbb-2222;"
    "target_model_names,claude-sonnet"
)
EXISTING_MANAGED_INPUT = (
    base64.urlsafe_b64encode(EXISTING_MANAGED_INPUT_RAW.encode()).decode().rstrip("=")
)


def _make_response():
    from litellm.types.utils import LiteLLMBatch

    return LiteLLMBatch(
        id="batch_xyz789",
        completion_window="24h",
        created_at=1700000000,
        endpoint="/v1/chat/completions",
        input_file_id=RAW_INPUT,
        object="batch",
        status="completed",
        completed_at=1700001000,
        output_file_id=RAW_OUTPUT,
        error_file_id=RAW_ERROR,
    )


def _build_environment(input_has_existing_managed_row: bool = True):
    prisma = MagicMock()
    prisma.db = MagicMock()
    prisma.db.litellm_managedobjecttable = MagicMock()
    prisma.db.litellm_usertable = MagicMock()
    prisma.db.litellm_managedfiletable = MagicMock()
    prisma.db.litellm_managedobjecttable.update_many = AsyncMock(return_value=0)
    prisma.db.litellm_managedobjecttable.update = AsyncMock()
    prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)

    async def find_first(where, **kw):
        ids = (where or {}).get("flat_model_file_ids", {})
        if (
            input_has_existing_managed_row
            and isinstance(ids, dict)
            and ids.get("has") == RAW_INPUT
        ):
            row = MagicMock()
            row.unified_file_id = EXISTING_MANAGED_INPUT
            return row
        return None

    prisma.db.litellm_managedfiletable.find_first = AsyncMock(side_effect=find_first)

    job = MagicMock()
    job.id = "job-lit-3386-1"
    job.unified_object_id = UNIFIED_JOB_ID_B64
    job.created_by = "shin-real-user-id"
    job.team_id = "team-lit-3386"
    prisma.db.litellm_managedobjecttable.find_many = AsyncMock(return_value=[job])

    router = MagicMock()
    response = _make_response()
    router.aretrieve_batch = AsyncMock(return_value=response)
    router.get_deployment_credentials_with_provider = MagicMock(
        return_value={"api_key": "sk-test"}
    )
    deployment = MagicMock()
    deployment.litellm_params.custom_llm_provider = "bedrock"
    deployment.litellm_params.model = "bedrock/claude-sonnet"
    deployment.model_name = "claude-sonnet-batch"
    deployment.model_info.model_dump.return_value = {}
    router.get_deployment = MagicMock(return_value=deployment)

    hook = MagicMock()

    def _gen_unified_id(output_file_id, model_id, model_name):
        unified = f"litellm_proxy::managed_for::{output_file_id}::on::{model_id}"
        return base64.urlsafe_b64encode(unified.encode()).decode().rstrip("=")

    hook.get_unified_output_file_id = MagicMock(side_effect=_gen_unified_id)
    store_calls = []

    async def _store(file_id, file_object, litellm_parent_otel_span,
                     model_mappings, user_api_key_dict):
        store_calls.append(
            {
                "file_id": file_id,
                "model_mappings": model_mappings,
                "user_id": user_api_key_dict.user_id,
                "team_id": user_api_key_dict.team_id,
            }
        )

    hook.store_unified_file_id = AsyncMock(side_effect=_store)

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.get_proxy_hook = MagicMock(return_value=hook)

    return {
        "prisma": prisma,
        "router": router,
        "hook": hook,
        "proxy_logging_obj": proxy_logging_obj,
        "response": response,
        "store_calls": store_calls,
    }


async def _run_check_batch_cost(env):
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    file_content = MagicMock()
    file_content.content = b'{"id": "req-1"}'

    cbc = CheckBatchCost(
        proxy_logging_obj=env["proxy_logging_obj"],
        prisma_client=env["prisma"],
        llm_router=env["router"],
    )

    with (
        patch("litellm.files.main.afile_content",
              new_callable=AsyncMock, return_value=file_content),
        patch("litellm.batches.batch_utils._get_file_content_as_dictionary",
              return_value=[{"id": "req-1"}]),
        patch("litellm.batches.batch_utils.calculate_batch_cost_and_usage",
              new_callable=AsyncMock,
              return_value=(0.01, {"prompt_tokens": 10, "completion_tokens": 5},
                            ["claude-sonnet"])),
        patch("litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider",
              return_value=("claude-sonnet", "bedrock", None, None)),
        patch("litellm.litellm_core_utils.litellm_logging.Logging") as mock_logging_cls,
    ):
        logging_obj = MagicMock()
        logging_obj.async_success_handler = AsyncMock()
        mock_logging_cls.return_value = logging_obj
        await cbc.check_batch_cost()


@pytest.mark.asyncio
async def test_lit_3386_input_file_id_resolves_to_existing_managed_row():
    """When a managed_file row exists for the raw input_file_id (because the
    user uploaded the file via the proxy), CheckBatchCost MUST persist the
    managed unified ID, NOT the raw provider ID, into file_object.input_file_id.

    Pre-fix: the poller's manual block only iterated output_file_id and
    error_file_id, so input_file_id was never normalized and stayed raw —
    causing GET /v1/batches/{id} to leak the provider ID to the user.
    """
    env = _build_environment(input_has_existing_managed_row=True)

    await _run_check_batch_cost(env)

    update_calls = env["prisma"].db.litellm_managedobjecttable.update.call_args_list
    assert len(update_calls) == 1, f"expected one DB write, got {len(update_calls)}"
    persisted = json.loads(update_calls[0][1]["data"]["file_object"])
    assert persisted["input_file_id"] == EXISTING_MANAGED_INPUT, (
        f"input_file_id was not normalized to existing managed unified id; "
        f"persisted={persisted['input_file_id']!r}"
    )
    assert persisted["input_file_id"] != RAW_INPUT, (
        "regression: input_file_id is still the raw provider id"
    )


@pytest.mark.asyncio
async def test_lit_3386_invokes_ensure_batch_response_managed_file_ids():
    """The poller must route normalization through
    ensure_batch_response_managed_file_ids — the same helper
    update_batch_in_database uses on the retrieve/cancel HTTP path — so the
    HTTP path and the background poller stay in sync.
    """
    env = _build_environment(input_has_existing_managed_row=True)

    import litellm.proxy.openai_files_endpoints.common_utils as _cu
    real_fn = _cu.ensure_batch_response_managed_file_ids
    call_args_list = []

    async def _wrapped(*a, **kw):
        call_args_list.append({
            "user_id": getattr(kw.get("user_api_key_dict"), "user_id", None),
        })
        return await real_fn(*a, **kw)

    with patch.object(_cu, "ensure_batch_response_managed_file_ids",
                      side_effect=_wrapped):
        await _run_check_batch_cost(env)

    assert len(call_args_list) == 1, (
        f"ensure_batch_response_managed_file_ids must be called exactly once "
        f"per finished batch; got {len(call_args_list)} call(s)"
    )
    assert call_args_list[0]["user_id"] == "shin-real-user-id", (
        "auth context user_id must be the batch creator (job.created_by), "
        "not default_user_id"
    )


@pytest.mark.asyncio
async def test_lit_3386_output_and_error_use_creator_auth():
    """store_unified_file_id (the call that writes the managed_file row to DB)
    must be invoked with UserAPIKeyAuth whose user_id is the batch creator —
    so the new managed-file row is owned by the real user.
    """
    env = _build_environment(input_has_existing_managed_row=False)

    await _run_check_batch_cost(env)

    store_calls = env["store_calls"]
    assert len(store_calls) == 2, (
        f"expected exactly 2 store_unified_file_id calls (output + error), "
        f"got {len(store_calls)}"
    )
    for c in store_calls:
        assert c["user_id"] == "shin-real-user-id", (
            f"store_unified_file_id was called with wrong auth context: {c['user_id']!r}"
        )
        assert c["team_id"] == "team-lit-3386", (
            f"store_unified_file_id was called with wrong team_id: {c['team_id']!r}"
        )

    update_calls = env["prisma"].db.litellm_managedobjecttable.update.call_args_list
    persisted = json.loads(update_calls[0][1]["data"]["file_object"])
    assert persisted["output_file_id"] != RAW_OUTPUT
    assert persisted["error_file_id"] != RAW_ERROR


@pytest.mark.asyncio
async def test_lit_3386_helper_failure_does_not_abort_poller():
    """If ensure_batch_response_managed_file_ids raises, the poller must
    still complete the cost-tracking pipeline and write the job row to the
    DB. The helper's failures should be logged + swallowed, never aborting
    the poll cycle for other batches.
    """
    env = _build_environment(input_has_existing_managed_row=True)

    import litellm.proxy.openai_files_endpoints.common_utils as _cu

    async def _boom(*a, **kw):
        raise RuntimeError("simulated managed-files-hook failure")

    with patch.object(_cu, "ensure_batch_response_managed_file_ids",
                      side_effect=_boom):
        await _run_check_batch_cost(env)

    # The DB update should still happen — the poller must not abort on
    # helper failures (the existing cost-tracking pipeline still needs to
    # mark the batch processed).
    update_calls = env["prisma"].db.litellm_managedobjecttable.update.call_args_list
    assert len(update_calls) == 1, (
        f"poller aborted on managed-files helper failure; "
        f"expected 1 update call, got {len(update_calls)}"
    )
