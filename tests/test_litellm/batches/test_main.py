"""
Provider-dispatch contract tests for litellm/batches/main.py

main.py is the SDK layer beneath the proxy batch endpoints: each of
create/retrieve/list/cancel_batch is a switch on `custom_llm_provider` (and, for
create/retrieve, on whether a provider-config + model is present) that hands off
to exactly one provider handler. These tests lock that dispatch:

  1. DISPATCH   - exactly which provider seam fired (openai_batches_instance vs
                  azure vs vertex vs anthropic vs base_llm_http_handler vs the
                  Bedrock ARN handlers), with every sibling seam asserted NOT
                  called. A reordered/negated branch flips this.
  2. PAYLOAD    - the request object (CreateBatchRequest/RetrieveBatchRequest/...)
                  and the _is_async flag forwarded to the handler.
  3. RESULT     - the handler's return value is what the function returns.
  4. DELEGATION - the async wrappers (a*_batch) forward to the sync function in an
                  executor with the right "_is_async" flag, and pass the result
                  back untouched.

Only the provider handler instances are mocked (true network boundaries). The
real public functions run (including the @client decorator) so dispatch reflects
production. Provider env vars are not required: missing creds resolve to None and
flow through harmlessly because the handler is mocked.
"""

import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
import litellm.batches.main as bm


# --------------------------------------------------------------------------- #
# Seam harness - one mock per provider handler instance + the Bedrock ARN
# handler. Each handler method auto-returns a unique sentinel (its
# return_value), so "result is seam.<method>.return_value" verifies dispatch.
# --------------------------------------------------------------------------- #


@dataclass
class Seams:
    openai: MagicMock
    azure: MagicMock
    vertex: MagicMock
    anthropic: MagicMock
    base_http: MagicMock
    bedrock_arn: MagicMock


@pytest.fixture
def seams():
    openai_i = MagicMock(name="openai_batches_instance")
    azure_i = MagicMock(name="azure_batches_instance")
    vertex_i = MagicMock(name="vertex_ai_batches_instance")
    anthropic_i = MagicMock(name="anthropic_batches_instance")
    base_http = MagicMock(name="base_llm_http_handler")
    bedrock_arn = MagicMock(name="BedrockBatchesHandler")

    with ExitStack() as stack:
        stack.enter_context(patch.object(bm, "openai_batches_instance", openai_i))
        stack.enter_context(patch.object(bm, "azure_batches_instance", azure_i))
        stack.enter_context(patch.object(bm, "vertex_ai_batches_instance", vertex_i))
        stack.enter_context(
            patch.object(bm, "anthropic_batches_instance", anthropic_i)
        )
        stack.enter_context(patch.object(bm, "base_llm_http_handler", base_http))
        stack.enter_context(patch.object(bm, "BedrockBatchesHandler", bedrock_arn))
        yield Seams(
            openai=openai_i,
            azure=azure_i,
            vertex=vertex_i,
            anthropic=anthropic_i,
            base_http=base_http,
            bedrock_arn=bedrock_arn,
        )


# Every <op> handler method across all provider instances - used to assert
# "no sibling seam fired" exhaustively.
def _all_seam_methods(seams: Seams, op: str):
    return [
        getattr(seams.openai, op),
        getattr(seams.azure, op),
        getattr(seams.vertex, op),
        getattr(seams.anthropic, op),
        getattr(seams.base_http, op),
    ]


def _assert_only(fired, seams: Seams, op: str):
    """Assert `fired` was called exactly once and every other op seam was not."""
    assert fired.call_count == 1
    for m in _all_seam_methods(seams, op):
        if m is not fired:
            m.assert_not_called()


CREATE_KW: Dict[str, Any] = dict(
    completion_window="24h",
    endpoint="/v1/chat/completions",
    input_file_id="file-abc",
)


# =========================================================================== #
# create_batch
# =========================================================================== #


def test_create__openai_dispatch_and_payload(seams):
    result = bm.create_batch(**CREATE_KW, custom_llm_provider="openai")

    # DISPATCH + RESULT
    assert result is seams.openai.create_batch.return_value
    _assert_only(seams.openai.create_batch, seams, "create_batch")
    seams.bedrock_arn._handle_async_invoke_status.assert_not_called()

    # PAYLOAD - request object built from the call, sync flag off.
    kw = seams.openai.create_batch.call_args.kwargs
    assert kw["create_batch_data"] == {
        "completion_window": "24h",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-abc",
        "metadata": None,
        "extra_headers": None,
        "extra_body": None,
    }
    assert kw["_is_async"] is False
    assert kw["timeout"] == 600.0


def test_create__hosted_vllm_routes_to_openai_instance(seams):
    """hosted_vllm is in OPENAI_COMPATIBLE_BATCH_AND_FILES_PROVIDERS, so it shares
    the openai handler. Locks that set membership."""
    result = bm.create_batch(**CREATE_KW, custom_llm_provider="hosted_vllm")

    assert result is seams.openai.create_batch.return_value
    _assert_only(seams.openai.create_batch, seams, "create_batch")


def test_create__azure_dispatch(seams):
    result = bm.create_batch(**CREATE_KW, custom_llm_provider="azure")

    assert result is seams.azure.create_batch.return_value
    _assert_only(seams.azure.create_batch, seams, "create_batch")


def test_create__vertex_ai_dispatch(seams):
    result = bm.create_batch(**CREATE_KW, custom_llm_provider="vertex_ai")

    assert result is seams.vertex.create_batch.return_value
    _assert_only(seams.vertex.create_batch, seams, "create_batch")


def test_create__provider_config_routes_to_base_http_handler(seams):
    """model + a provider batches config (bedrock-style) routes to the generic
    base_llm_http_handler, NOT the per-provider instance."""
    with patch.object(
        bm.ProviderConfigManager,
        "get_provider_batches_config",
        return_value=MagicMock(name="provider_config"),
    ):
        result = bm.create_batch(
            **CREATE_KW, custom_llm_provider="bedrock", model="bedrock/my-batch-model"
        )

    assert result is seams.base_http.create_batch.return_value
    _assert_only(seams.base_http.create_batch, seams, "create_batch")


def test_create__unsupported_provider_raises_badrequest(seams):
    with pytest.raises(litellm.exceptions.BadRequestError):
        bm.create_batch(**CREATE_KW, custom_llm_provider="cohere")  # type: ignore[arg-type]

    for m in _all_seam_methods(seams, "create_batch"):
        m.assert_not_called()


@pytest.mark.asyncio
async def test_create__async_path_propagates_is_async(seams):
    """Through the real async wrapper, the handler is invoked with _is_async=True.
    (Calling the @client sync create_batch with acreate_batch=True directly is not
    a real code path - logging-obj setup only happens on the async wrapper path.)"""
    await bm.acreate_batch(**CREATE_KW, custom_llm_provider="openai")

    assert seams.openai.create_batch.call_args.kwargs["_is_async"] is True


# =========================================================================== #
# retrieve_batch
# =========================================================================== #


def test_retrieve__openai_dispatch_and_payload(seams):
    result = bm.retrieve_batch(batch_id="batch-1", custom_llm_provider="openai")

    assert result is seams.openai.retrieve_batch.return_value
    _assert_only(seams.openai.retrieve_batch, seams, "retrieve_batch")

    kw = seams.openai.retrieve_batch.call_args.kwargs
    assert kw["retrieve_batch_data"] == {
        "batch_id": "batch-1",
        "extra_headers": None,
        "extra_body": None,
    }
    assert kw["_is_async"] is False


def test_retrieve__hosted_vllm_routes_to_openai_instance(seams):
    result = bm.retrieve_batch(batch_id="batch-1", custom_llm_provider="hosted_vllm")

    assert result is seams.openai.retrieve_batch.return_value
    _assert_only(seams.openai.retrieve_batch, seams, "retrieve_batch")


def test_retrieve__azure_dispatch(seams):
    result = bm.retrieve_batch(batch_id="batch-1", custom_llm_provider="azure")

    assert result is seams.azure.retrieve_batch.return_value
    _assert_only(seams.azure.retrieve_batch, seams, "retrieve_batch")


def test_retrieve__vertex_ai_dispatch(seams):
    result = bm.retrieve_batch(batch_id="batch-1", custom_llm_provider="vertex_ai")

    assert result is seams.vertex.retrieve_batch.return_value
    _assert_only(seams.vertex.retrieve_batch, seams, "retrieve_batch")


def test_retrieve__anthropic_dispatch(seams):
    """anthropic is retrieve-capable (not in create's provider set)."""
    result = bm.retrieve_batch(batch_id="batch-1", custom_llm_provider="anthropic")

    assert result is seams.anthropic.retrieve_batch.return_value
    _assert_only(seams.anthropic.retrieve_batch, seams, "retrieve_batch")


def test_retrieve__provider_config_routes_to_base_http_handler(seams):
    with patch.object(
        bm.ProviderConfigManager,
        "get_provider_batches_config",
        return_value=MagicMock(name="provider_config"),
    ):
        result = bm.retrieve_batch(
            batch_id="batch-1",
            custom_llm_provider="bedrock",
            model="bedrock/my-batch-model",
        )

    assert result is seams.base_http.retrieve_batch.return_value
    _assert_only(seams.base_http.retrieve_batch, seams, "retrieve_batch")


def test_retrieve__bedrock_async_invoke_arn(seams):
    arn = "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123"
    result = bm.retrieve_batch(batch_id=arn, custom_llm_provider="bedrock")

    seams.bedrock_arn._handle_async_invoke_status.assert_called_once()
    assert result is seams.bedrock_arn._handle_async_invoke_status.return_value
    # provider instances untouched.
    for m in _all_seam_methods(seams, "retrieve_batch"):
        m.assert_not_called()


def test_retrieve__bedrock_model_invocation_job_arn(seams):
    arn = "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/xyz789"
    result = bm.retrieve_batch(batch_id=arn, custom_llm_provider="bedrock")

    seams.bedrock_arn._handle_model_invocation_job_status.assert_called_once()
    assert (
        result is seams.bedrock_arn._handle_model_invocation_job_status.return_value
    )
    seams.bedrock_arn._handle_async_invoke_status.assert_not_called()


def test_retrieve__unsupported_provider_raises_badrequest(seams):
    with pytest.raises(litellm.exceptions.BadRequestError):
        bm.retrieve_batch(batch_id="batch-1", custom_llm_provider="cohere")  # type: ignore[arg-type]

    for m in _all_seam_methods(seams, "retrieve_batch"):
        m.assert_not_called()


# =========================================================================== #
# list_batches  (supported: openai, hosted_vllm, azure, vertex_ai)
# =========================================================================== #


def test_list__openai_dispatch_and_payload(seams):
    result = bm.list_batches(custom_llm_provider="openai", after="cur", limit=5)

    assert result is seams.openai.list_batches.return_value
    _assert_only(seams.openai.list_batches, seams, "list_batches")

    kw = seams.openai.list_batches.call_args.kwargs
    assert kw["after"] == "cur"
    assert kw["limit"] == 5
    assert kw["_is_async"] is False


def test_list__hosted_vllm_routes_to_openai_instance(seams):
    result = bm.list_batches(custom_llm_provider="hosted_vllm")

    assert result is seams.openai.list_batches.return_value
    _assert_only(seams.openai.list_batches, seams, "list_batches")


def test_list__azure_dispatch(seams):
    result = bm.list_batches(custom_llm_provider="azure")

    assert result is seams.azure.list_batches.return_value
    _assert_only(seams.azure.list_batches, seams, "list_batches")


def test_list__vertex_ai_dispatch(seams):
    result = bm.list_batches(custom_llm_provider="vertex_ai")

    assert result is seams.vertex.list_batches.return_value
    _assert_only(seams.vertex.list_batches, seams, "list_batches")


def test_list__unsupported_provider_raises_badrequest(seams):
    # anthropic supports retrieve but NOT list - good negative case.
    with pytest.raises(litellm.exceptions.BadRequestError):
        bm.list_batches(custom_llm_provider="anthropic")  # type: ignore[arg-type]

    for m in _all_seam_methods(seams, "list_batches"):
        m.assert_not_called()


# =========================================================================== #
# cancel_batch  (supported: openai, hosted_vllm, azure, vertex_ai; no @client)
# =========================================================================== #


def test_cancel__openai_dispatch_and_payload(seams):
    result = bm.cancel_batch(batch_id="batch-1", custom_llm_provider="openai")

    assert result is seams.openai.cancel_batch.return_value
    _assert_only(seams.openai.cancel_batch, seams, "cancel_batch")

    kw = seams.openai.cancel_batch.call_args.kwargs
    assert kw["cancel_batch_data"] == {
        "batch_id": "batch-1",
        "extra_headers": None,
        "extra_body": None,
    }
    assert kw["_is_async"] is False


def test_cancel__azure_dispatch(seams):
    result = bm.cancel_batch(batch_id="batch-1", custom_llm_provider="azure")

    assert result is seams.azure.cancel_batch.return_value
    _assert_only(seams.azure.cancel_batch, seams, "cancel_batch")


def test_cancel__vertex_ai_dispatch(seams):
    result = bm.cancel_batch(batch_id="batch-1", custom_llm_provider="vertex_ai")

    assert result is seams.vertex.cancel_batch.return_value
    _assert_only(seams.vertex.cancel_batch, seams, "cancel_batch")


def test_cancel__unsupported_provider_raises_badrequest(seams):
    with pytest.raises(litellm.exceptions.BadRequestError):
        bm.cancel_batch(batch_id="batch-1", custom_llm_provider="cohere")

    for m in _all_seam_methods(seams, "cancel_batch"):
        m.assert_not_called()


def test_cancel__async_flag_propagates_is_async(seams):
    bm.cancel_batch(
        batch_id="batch-1", custom_llm_provider="openai", acancel_batch=True
    )

    assert seams.openai.cancel_batch.call_args.kwargs["_is_async"] is True


# =========================================================================== #
# Async wrappers - delegate to the sync function in an executor, set the right
# "_is_async" flag, and return the result untouched.
# =========================================================================== #


@pytest.mark.asyncio
async def test_acreate_batch_delegates_to_create_batch():
    with patch.object(bm, "create_batch", MagicMock(return_value="SENTINEL")) as m:
        result = await bm.acreate_batch(**CREATE_KW, custom_llm_provider="openai")

    assert result == "SENTINEL"
    assert m.call_count == 1
    assert m.call_args.kwargs.get("acreate_batch") is True
    # positional handoff: (completion_window, endpoint, input_file_id, provider, ...)
    assert m.call_args.args[0] == "24h"
    assert m.call_args.args[2] == "file-abc"
    assert m.call_args.args[3] == "openai"


@pytest.mark.asyncio
async def test_aretrieve_batch_delegates_to_retrieve_batch():
    with patch.object(bm, "retrieve_batch", MagicMock(return_value="SENTINEL")) as m:
        result = await bm.aretrieve_batch(
            batch_id="batch-1", custom_llm_provider="azure"
        )

    assert result == "SENTINEL"
    assert m.call_count == 1
    assert m.call_args.kwargs.get("aretrieve_batch") is True
    assert m.call_args.args[0] == "batch-1"
    assert m.call_args.args[1] == "azure"


@pytest.mark.asyncio
async def test_alist_batches_delegates_to_list_batches():
    with patch.object(bm, "list_batches", MagicMock(return_value="SENTINEL")) as m:
        result = await bm.alist_batches(
            after="cur", limit=3, custom_llm_provider="vertex_ai"
        )

    assert result == "SENTINEL"
    assert m.call_count == 1
    assert m.call_args.kwargs.get("alist_batches") is True
    assert m.call_args.args[0] == "cur"
    assert m.call_args.args[1] == 3
    assert m.call_args.args[2] == "vertex_ai"


@pytest.mark.asyncio
async def test_acancel_batch_delegates_to_cancel_batch():
    with patch.object(bm, "cancel_batch", MagicMock(return_value="SENTINEL")) as m:
        result = await bm.acancel_batch(
            batch_id="batch-1", custom_llm_provider="openai"
        )

    assert result == "SENTINEL"
    assert m.call_count == 1
    assert m.call_args.kwargs.get("acancel_batch") is True
    assert m.call_args.args[0] == "batch-1"


# =========================================================================== #
# Credential passthrough - when the caller supplies credentials in kwargs, they
# must reach the provider handler. Explicit kwargs win over litellm.* globals and
# env vars (they are first in each `optional_params.x or litellm.x or env` chain),
# so these assertions are deterministic regardless of the test environment.
#
# The credential-resolution blocks are copy-pasted per provider in EACH of
# create/retrieve/list/cancel, so a regression can land in any one independently;
# every function is checked.
# =========================================================================== #


# Distinct values so a cross-wired field (e.g. api_key forwarded as api_base) is
# impossible to miss.
OPENAI_CREDS: Dict[str, Any] = dict(
    api_key="sk-user-openai",
    api_base="https://openai.user.test",
    organization="org-user-123",
    max_retries=7,
)
AZURE_CREDS: Dict[str, Any] = dict(
    api_key="sk-user-azure",
    api_base="https://azure.user.test",
    api_version="2024-12-99",
)
VERTEX_CREDS: Dict[str, Any] = dict(
    vertex_project="proj-user",
    vertex_location="loc-user",
    vertex_credentials="cred-user",
    api_base="https://vertex.user.test",
)


def _sent(mock_method, *keys):
    """Subset of the call kwargs limited to `keys`, for exact comparison."""
    kw = mock_method.call_args.kwargs
    return {k: kw.get(k) for k in keys}


# ---- create_batch ---------------------------------------------------------- #


def test_create__openai_credentials_passthrough(seams):
    bm.create_batch(**CREATE_KW, custom_llm_provider="openai", **OPENAI_CREDS)

    assert _sent(
        seams.openai.create_batch, "api_key", "api_base", "organization", "max_retries"
    ) == {
        "api_key": "sk-user-openai",
        "api_base": "https://openai.user.test",
        "organization": "org-user-123",
        "max_retries": 7,
    }


def test_create__azure_credentials_passthrough(seams):
    bm.create_batch(**CREATE_KW, custom_llm_provider="azure", **AZURE_CREDS)

    assert _sent(
        seams.azure.create_batch, "api_key", "api_base", "api_version"
    ) == {
        "api_key": "sk-user-azure",
        "api_base": "https://azure.user.test",
        "api_version": "2024-12-99",
    }


def test_create__vertex_credentials_passthrough(seams):
    bm.create_batch(**CREATE_KW, custom_llm_provider="vertex_ai", **VERTEX_CREDS)

    assert _sent(
        seams.vertex.create_batch,
        "vertex_project",
        "vertex_location",
        "vertex_credentials",
        "api_base",
    ) == {
        "vertex_project": "proj-user",
        "vertex_location": "loc-user",
        "vertex_credentials": "cred-user",
        "api_base": "https://vertex.user.test",
    }


def test_create__provider_config_credentials_passthrough(seams):
    with patch.object(
        bm.ProviderConfigManager,
        "get_provider_batches_config",
        return_value=MagicMock(name="provider_config"),
    ):
        bm.create_batch(
            **CREATE_KW,
            custom_llm_provider="bedrock",
            model="bedrock/my-batch-model",
            api_key="sk-user-bedrock",
            api_base="https://bedrock.user.test",
        )

    assert _sent(seams.base_http.create_batch, "api_key", "api_base") == {
        "api_key": "sk-user-bedrock",
        "api_base": "https://bedrock.user.test",
    }


# ---- retrieve_batch -------------------------------------------------------- #


def test_retrieve__openai_credentials_passthrough(seams):
    bm.retrieve_batch(batch_id="b1", custom_llm_provider="openai", **OPENAI_CREDS)

    assert _sent(
        seams.openai.retrieve_batch, "api_key", "api_base", "organization"
    ) == {
        "api_key": "sk-user-openai",
        "api_base": "https://openai.user.test",
        "organization": "org-user-123",
    }


def test_retrieve__azure_credentials_passthrough(seams):
    bm.retrieve_batch(batch_id="b1", custom_llm_provider="azure", **AZURE_CREDS)

    assert _sent(
        seams.azure.retrieve_batch, "api_key", "api_base", "api_version"
    ) == {
        "api_key": "sk-user-azure",
        "api_base": "https://azure.user.test",
        "api_version": "2024-12-99",
    }


def test_retrieve__vertex_credentials_passthrough(seams):
    bm.retrieve_batch(batch_id="b1", custom_llm_provider="vertex_ai", **VERTEX_CREDS)

    assert _sent(
        seams.vertex.retrieve_batch,
        "vertex_project",
        "vertex_location",
        "vertex_credentials",
    ) == {
        "vertex_project": "proj-user",
        "vertex_location": "loc-user",
        "vertex_credentials": "cred-user",
    }


def test_retrieve__anthropic_credentials_passthrough(seams):
    bm.retrieve_batch(
        batch_id="b1",
        custom_llm_provider="anthropic",
        api_key="sk-user-anthropic",
        api_base="https://anthropic.user.test",
    )

    assert _sent(seams.anthropic.retrieve_batch, "api_key", "api_base") == {
        "api_key": "sk-user-anthropic",
        "api_base": "https://anthropic.user.test",
    }


def test_retrieve__provider_config_credentials_passthrough(seams):
    with patch.object(
        bm.ProviderConfigManager,
        "get_provider_batches_config",
        return_value=MagicMock(name="provider_config"),
    ):
        bm.retrieve_batch(
            batch_id="b1",
            custom_llm_provider="bedrock",
            model="bedrock/my-batch-model",
            api_key="sk-user-bedrock",
            api_base="https://bedrock.user.test",
        )

    assert _sent(seams.base_http.retrieve_batch, "api_key", "api_base") == {
        "api_key": "sk-user-bedrock",
        "api_base": "https://bedrock.user.test",
    }


# ---- list_batches ---------------------------------------------------------- #


def test_list__openai_credentials_passthrough(seams):
    bm.list_batches(custom_llm_provider="openai", **OPENAI_CREDS)

    assert _sent(
        seams.openai.list_batches, "api_key", "api_base", "organization"
    ) == {
        "api_key": "sk-user-openai",
        "api_base": "https://openai.user.test",
        "organization": "org-user-123",
    }


def test_list__azure_credentials_passthrough(seams):
    bm.list_batches(custom_llm_provider="azure", **AZURE_CREDS)

    assert _sent(
        seams.azure.list_batches, "api_key", "api_base", "api_version"
    ) == {
        "api_key": "sk-user-azure",
        "api_base": "https://azure.user.test",
        "api_version": "2024-12-99",
    }


def test_list__vertex_credentials_passthrough(seams):
    bm.list_batches(custom_llm_provider="vertex_ai", **VERTEX_CREDS)

    assert _sent(
        seams.vertex.list_batches,
        "vertex_project",
        "vertex_location",
        "vertex_credentials",
    ) == {
        "vertex_project": "proj-user",
        "vertex_location": "loc-user",
        "vertex_credentials": "cred-user",
    }


# ---- cancel_batch ---------------------------------------------------------- #


def test_cancel__openai_credentials_passthrough(seams):
    bm.cancel_batch(batch_id="b1", custom_llm_provider="openai", **OPENAI_CREDS)

    assert _sent(
        seams.openai.cancel_batch, "api_key", "api_base", "organization"
    ) == {
        "api_key": "sk-user-openai",
        "api_base": "https://openai.user.test",
        "organization": "org-user-123",
    }


def test_cancel__azure_credentials_passthrough(seams):
    bm.cancel_batch(batch_id="b1", custom_llm_provider="azure", **AZURE_CREDS)

    assert _sent(
        seams.azure.cancel_batch, "api_key", "api_base", "api_version"
    ) == {
        "api_key": "sk-user-azure",
        "api_base": "https://azure.user.test",
        "api_version": "2024-12-99",
    }


def test_cancel__vertex_credentials_passthrough(seams):
    bm.cancel_batch(batch_id="b1", custom_llm_provider="vertex_ai", **VERTEX_CREDS)

    assert _sent(
        seams.vertex.cancel_batch,
        "vertex_project",
        "vertex_location",
        "vertex_credentials",
    ) == {
        "vertex_project": "proj-user",
        "vertex_location": "loc-user",
        "vertex_credentials": "cred-user",
    }


# =========================================================================== #
# _resolve_timeout - pure helper (used by create_batch).
# =========================================================================== #


def _params(**kw):
    from litellm.types.router import GenericLiteLLMParams

    return GenericLiteLLMParams(**kw)


def test_resolve_timeout__explicit_numeric():
    assert bm._resolve_timeout(_params(timeout=30), {}, "openai") == 30.0


def test_resolve_timeout__default_when_unset():
    assert bm._resolve_timeout(_params(), {}, "openai") == 600.0


def test_resolve_timeout__request_timeout_kwarg_fallback():
    assert bm._resolve_timeout(_params(), {"request_timeout": 45}, "openai") == 45.0


def test_resolve_timeout__httpx_timeout_returns_float_read():
    import httpx

    t = httpx.Timeout(99.0, connect=5.0)
    resolved = bm._resolve_timeout(_params(timeout=t), {}, "openai")
    assert isinstance(resolved, float)
    assert resolved == 99.0
