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

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Dict
from types import MappingProxyType
from unittest.mock import MagicMock, patch

import pytest


import litellm
import litellm.batches.main as bm
import asyncio
import datetime
import json
from collections.abc import Mapping
from typing import Final
import httpx
import respx
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict
from litellm.integrations.custom_logger import CustomLogger


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
        stack.enter_context(patch.object(bm, "anthropic_batches_instance", anthropic_i))
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
    seams.bedrock_arn.handle_async_invoke_status.assert_not_called()

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


def test_create__vertex_ai_forwards_custom_endpoint(seams):
    """The vertex handler owns the custom_endpoint batch rejection (LIT-6899), so the dispatcher
    must forward the flag for the handler to act on."""
    bm.create_batch(**CREATE_KW, custom_llm_provider="vertex_ai", custom_endpoint=True)

    assert seams.vertex.create_batch.call_args.kwargs["custom_endpoint"] is True


def test_create__provider_config_routes_to_base_http_handler(seams):
    """model + a provider batches config (bedrock-style) routes to the generic
    base_llm_http_handler, NOT the per-provider instance."""
    with patch.object(
        bm.ProviderConfigManager,
        "get_provider_batches_config",
        return_value=MagicMock(name="provider_config"),
    ):
        result = bm.create_batch(**CREATE_KW, custom_llm_provider="bedrock", model="bedrock/my-batch-model")

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

    seams.bedrock_arn.handle_async_invoke_status.assert_called_once()
    assert result is seams.bedrock_arn.handle_async_invoke_status.return_value
    # provider instances untouched.
    for m in _all_seam_methods(seams, "retrieve_batch"):
        m.assert_not_called()


def test_retrieve__bedrock_model_invocation_job_arn(seams):
    arn = "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/xyz789"
    result = bm.retrieve_batch(batch_id=arn, custom_llm_provider="bedrock")

    seams.bedrock_arn.handle_model_invocation_job_status.assert_called_once()
    assert result is seams.bedrock_arn.handle_model_invocation_job_status.return_value
    seams.bedrock_arn.handle_async_invoke_status.assert_not_called()


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
    bm.cancel_batch(batch_id="batch-1", custom_llm_provider="openai", acancel_batch=True)

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
        result = await bm.aretrieve_batch(batch_id="batch-1", custom_llm_provider="azure")

    assert result == "SENTINEL"
    assert m.call_count == 1
    assert m.call_args.kwargs.get("aretrieve_batch") is True
    assert m.call_args.args[0] == "batch-1"
    assert m.call_args.args[1] == "azure"


@pytest.mark.asyncio
async def test_alist_batches_delegates_to_list_batches():
    with patch.object(bm, "list_batches", MagicMock(return_value="SENTINEL")) as m:
        result = await bm.alist_batches(after="cur", limit=3, custom_llm_provider="vertex_ai")

    assert result == "SENTINEL"
    assert m.call_count == 1
    assert m.call_args.kwargs.get("alist_batches") is True
    assert m.call_args.args[0] == "cur"
    assert m.call_args.args[1] == 3
    assert m.call_args.args[2] == "vertex_ai"


@pytest.mark.asyncio
async def test_acancel_batch_delegates_to_cancel_batch():
    with patch.object(bm, "cancel_batch", MagicMock(return_value="SENTINEL")) as m:
        result = await bm.acancel_batch(batch_id="batch-1", custom_llm_provider="openai")

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

    assert _sent(seams.openai.create_batch, "api_key", "api_base", "organization", "max_retries") == {
        "api_key": "sk-user-openai",
        "api_base": "https://openai.user.test",
        "organization": "org-user-123",
        "max_retries": 7,
    }


def test_create__azure_credentials_passthrough(seams):
    bm.create_batch(**CREATE_KW, custom_llm_provider="azure", **AZURE_CREDS)

    assert _sent(seams.azure.create_batch, "api_key", "api_base", "api_version") == {
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

    assert _sent(seams.openai.retrieve_batch, "api_key", "api_base", "organization") == {
        "api_key": "sk-user-openai",
        "api_base": "https://openai.user.test",
        "organization": "org-user-123",
    }


def test_retrieve__azure_credentials_passthrough(seams):
    bm.retrieve_batch(batch_id="b1", custom_llm_provider="azure", **AZURE_CREDS)

    assert _sent(seams.azure.retrieve_batch, "api_key", "api_base", "api_version") == {
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

    assert _sent(seams.openai.list_batches, "api_key", "api_base", "organization") == {
        "api_key": "sk-user-openai",
        "api_base": "https://openai.user.test",
        "organization": "org-user-123",
    }


def test_list__azure_credentials_passthrough(seams):
    bm.list_batches(custom_llm_provider="azure", **AZURE_CREDS)

    assert _sent(seams.azure.list_batches, "api_key", "api_base", "api_version") == {
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

    assert _sent(seams.openai.cancel_batch, "api_key", "api_base", "organization") == {
        "api_key": "sk-user-openai",
        "api_base": "https://openai.user.test",
        "organization": "org-user-123",
    }


def test_cancel__azure_credentials_passthrough(seams):
    bm.cancel_batch(batch_id="b1", custom_llm_provider="azure", **AZURE_CREDS)

    assert _sent(seams.azure.cancel_batch, "api_key", "api_base", "api_version") == {
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


def test_retrieve__forwards_trusted_model_credentials_into_litellm_params(seams):
    """The batch's cost is computed by reading its output file after the retrieve, and
    Bedrock resolves that bucket only from this immutable snapshot. get_litellm_params has
    a fixed signature that drops it, so without re-adding it here the snapshot never
    reaches the logging object and cost accounting fails on a bucket that is configured."""
    snapshot = MappingProxyType({"s3_bucket_name": "configured-bucket"})
    logging_obj = MagicMock()

    bm.retrieve_batch(
        batch_id="batch-1",
        custom_llm_provider="openai",
        litellm_logging_obj=logging_obj,
        _litellm_internal_model_credentials=snapshot,
    )

    litellm_params = logging_obj.update_from_kwargs.call_args.kwargs["litellm_params"]
    assert litellm_params["_litellm_internal_model_credentials"] is snapshot


def test_retrieve__omits_trusted_model_credentials_when_not_supplied(seams):
    """A retrieve with no snapshot must not invent an empty one, which would read as a
    configured bucket of nothing."""
    logging_obj = MagicMock()

    bm.retrieve_batch(batch_id="batch-1", custom_llm_provider="openai", litellm_logging_obj=logging_obj)

    litellm_params = logging_obj.update_from_kwargs.call_args.kwargs["litellm_params"]
    assert "_litellm_internal_model_credentials" not in litellm_params


# =========================================================================== #
# mistral - a provider-config provider, like bedrock, so it requires `model`
# =========================================================================== #


def test_create__mistral_ocr_routes_to_base_http_handler_with_mistral_config(seams):
    result = bm.create_batch(
        completion_window="24h",
        endpoint="/v1/ocr",
        input_file_id="file-abc",
        custom_llm_provider="mistral",
        model="mistral/mistral-ocr-latest",
    )

    assert result is seams.base_http.create_batch.return_value
    _assert_only(seams.base_http.create_batch, seams, "create_batch")
    forwarded = seams.base_http.create_batch.call_args.kwargs
    assert type(forwarded["provider_config"]).__name__ == "MistralBatchesConfig"
    assert forwarded["model"] == "mistral-ocr-latest"
    assert forwarded["create_batch_data"]["endpoint"] == "/v1/ocr"


def test_create__mistral_without_model_raises_badrequest(seams):
    with pytest.raises(litellm.exceptions.BadRequestError):
        bm.create_batch(**CREATE_KW, custom_llm_provider="mistral")

    for m in _all_seam_methods(seams, "create_batch"):
        m.assert_not_called()


def test_retrieve__mistral_routes_to_base_http_handler_with_mistral_config(seams):
    result = bm.retrieve_batch(batch_id="job-1", custom_llm_provider="mistral", model="mistral/mistral-ocr-latest")

    assert result is seams.base_http.retrieve_batch.return_value
    _assert_only(seams.base_http.retrieve_batch, seams, "retrieve_batch")
    forwarded = seams.base_http.retrieve_batch.call_args.kwargs
    assert type(forwarded["provider_config"]).__name__ == "MistralBatchesConfig"
    assert forwarded["batch_id"] == "job-1"


@pytest.mark.asyncio()
async def test_batch_logging_azure_credentials_regression():
    """
    Regression test: LoggingWorker Missing Azure Credentials When Fetching Batch Output

    This test ensures that Azure credentials are properly passed when fetching batch
    output files during logging, preventing "Missing credentials" errors.

    Bug: The LoggingWorker failed when processing completed Azure batches because
    it attempted to fetch batch output file content without Azure credentials.

    Fix: Pass litellm_params (containing credentials) from the logging object
    through to the file content retrieval functions.
    """
    from unittest.mock import AsyncMock, MagicMock, patch
    from litellm.batches.batch_utils import (
        extract_file_access_credentials,
        _fetch_batch_output_file_content,
        handle_completed_batch,
    )
    from litellm.types.llms.openai import Batch, HttpxBinaryResponseContent
    import httpx

    print("\n=== Regression Test: Azure Batch Logging Credentials ===")

    mock_batch = Batch(
        id="batch-azure-test",
        object="batch",
        endpoint="/v1/chat/completions",
        errors=None,
        input_file_id="file-input-azure",
        completion_window="24h",
        status="completed",
        output_file_id="file-output-azure",
        error_file_id=None,
        created_at=1234567890,
        in_progress_at=1234567900,
        expires_at=1234654290,
        finalizing_at=1234568000,
        completed_at=1234568100,
        failed_at=None,
        expired_at=None,
        cancelling_at=None,
        cancelled_at=None,
        request_counts=None,
        metadata=None,
    )

    azure_credentials = {
        "api_key": "test-azure-key-regression",
        "api_base": "https://test-regression.openai.azure.com",
        "api_version": "2024-02-15-preview",
        "organization": "test-org",
        "timeout": 600,
    }

    batch_output = b'{"id": "batch_req_1", "custom_id": "request-1", "response": {"status_code": 200, "body": {"id": "chatcmpl-azure", "object": "chat.completion", "model": "gpt-4", "usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}}}}\n'

    print("\n1. Testing credential extraction...")

    extracted_creds = extract_file_access_credentials(azure_credentials)
    assert "api_key" in extracted_creds, "api_key should be extracted"
    assert (
        extracted_creds["api_key"] == "test-azure-key-regression"
    ), "Incorrect api_key"
    assert "api_base" in extracted_creds, "api_base should be extracted"
    assert "api_version" in extracted_creds, "api_version should be extracted"
    assert "timeout" in extracted_creds, "timeout should be extracted"

    print("   ✓ Credentials extracted correctly")
    print(f"   ✓ Extracted keys: {list(extracted_creds.keys())}")

    print("\n2. Testing credentials passed to afile_content...")

    credentials_received = {"value": False, "params": None}

    async def mock_afile_content_tracker(**kwargs):
        if "api_key" in kwargs and "api_base" in kwargs and "api_version" in kwargs:
            credentials_received["value"] = True
            credentials_received["params"] = {
                "api_key": kwargs.get("api_key"),
                "api_base": kwargs.get("api_base"),
                "api_version": kwargs.get("api_version"),
            }
        mock_response = httpx.Response(
            status_code=200,
            content=batch_output,
            headers={"content-type": "application/octet-stream"},
        )
        return HttpxBinaryResponseContent(response=mock_response)

    with patch(
        "litellm.files.main.afile_content", side_effect=mock_afile_content_tracker
    ):
        result = await _fetch_batch_output_file_content(
            batch=mock_batch,
            custom_llm_provider="azure",
            litellm_params=azure_credentials,
        )

        assert credentials_received[
            "value"
        ], "REGRESSION: Azure credentials not passed to afile_content! This causes 'Missing credentials' error."
        assert (
            credentials_received["params"]["api_key"] == "test-azure-key-regression"
        ), "REGRESSION: Incorrect api_key"
        assert (
            credentials_received["params"]["api_base"]
            == "https://test-regression.openai.azure.com"
        ), "REGRESSION: Incorrect api_base"

        print("   ✓ Credentials passed to afile_content")
        print(f"   ✓ api_key: {credentials_received['params']['api_key']}")
        print(f"   ✓ api_base: {credentials_received['params']['api_base']}")

    print("\n3. Testing full logging flow...")

    credentials_received["value"] = False
    credentials_received["params"] = None

    with patch(
        "litellm.files.main.afile_content", side_effect=mock_afile_content_tracker
    ):
        result = await handle_completed_batch(
            batch=mock_batch,
            custom_llm_provider="azure",
            litellm_params=azure_credentials,
        )

        assert credentials_received[
            "value"
        ], "REGRESSION: Credentials not passed through _handle_completed_batch"

        assert result.cost > 0, "Cost should be calculated"
        assert result.usage.total_tokens == 40, "Usage should be calculated correctly"

        print("   ✓ Credentials passed through full flow")
        print(f"   ✓ Cost: {result.cost}")
        print(f"   ✓ Usage: {result.usage.total_tokens} tokens")
        print(f"   ✓ Models: {result.models}")

    print("\n4. Testing 'Missing credentials' error prevention...")

    with patch("litellm.files.main.afile_content") as mock_afile_content_fail:
        mock_afile_content_fail.side_effect = Exception(
            "Missing credentials. Please pass one of `api_key`, `azure_ad_token`, "
            "`azure_ad_token_provider`, or the `AZURE_OPENAI_API_KEY` or "
            "`AZURE_OPENAI_AD_TOKEN` environment variables."
        )

        with patch(
            "litellm.files.main.afile_content", side_effect=mock_afile_content_tracker
        ):
            try:
                result = await handle_completed_batch(
                    batch=mock_batch,
                    custom_llm_provider="azure",
                    litellm_params=azure_credentials,
                )
                print("   ✓ No 'Missing credentials' error with fix")
            except Exception as e:
                if "Missing credentials" in str(e):
                    pytest.fail(
                        f"REGRESSION: 'Missing credentials' error occurred! "
                        f"Credentials not being passed. Error: {str(e)}"
                    )
                raise

    print("\n5. Testing backwards compatibility...")

    with patch("litellm.files.main.afile_content") as mock_afile_content:
        mock_response = httpx.Response(
            status_code=200,
            content=batch_output,
            headers={"content-type": "application/octet-stream"},
        )
        mock_afile_content.return_value = HttpxBinaryResponseContent(
            response=mock_response
        )

        result = await _fetch_batch_output_file_content(
            batch=mock_batch,
            custom_llm_provider="openai",
            litellm_params=None,
        )

        assert len(result) > 0, "Should return file content"
        print("   ✓ Backwards compatibility maintained")
        print("   ✓ Works without litellm_params for OpenAI")

    print("\n=== Regression Test Passed ===")
    print("✓ Azure credentials properly passed from logging to file retrieval")
    print("✓ 'Missing credentials' error prevented")
    print("✓ Batch output files can be fetched with Azure credentials")
    print("✓ Cost and usage tracking works for Azure batches")
    print("✓ Backwards compatibility maintained\n")


_OPENAI_FILE_JSON: Final = MappingProxyType(
    {
        "id": "file-abc123",
        "object": "file",
        "purpose": "batch",
        "filename": "batch.jsonl",
        "bytes": 416,
        "created_at": 1739598666,
        "status": "processed",
    }
)


_OPENAI_BATCH_JSON: Final = MappingProxyType(
    {
        "id": "batch_abc123",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-abc123",
        "status": "validating",
        "completion_window": "24h",
        "created_at": 1739598666,
    }
)


class _KeyAliasMetadata(TypedDict):
    user_api_key_alias: ReadOnly[str | None]
    user_api_key_team_alias: ReadOnly[str | None]


class _LoggedCall(TypedDict):
    call_type: ReadOnly[str]
    metadata: ReadOnly[_KeyAliasMetadata]


_LOGGED_CALL: Final = TypeAdapter(_LoggedCall)


class _SuccessPayloadRecorder(CustomLogger):
    def __init__(self, call_type: str) -> None:
        super().__init__()
        self._call_type: Final = call_type
        self.logged: Final = asyncio.Event()
        self.payload: _LoggedCall | None = None

    async def async_log_success_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        payload: Final = _LOGGED_CALL.validate_python(kwargs["standard_logging_object"])
        if payload["call_type"] != self._call_type:
            return
        self.payload = payload
        self.logged.set()


@pytest.mark.asyncio
async def test_acreate_batch_full_crud_and_logging_metadata(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.logging_callback_manager._reset_all_callbacks()
    recorder: Final = _SuccessPayloadRecorder("acreate_batch")
    monkeypatch.setattr(litellm, "callbacks", [recorder])

    upload_route: Final = respx_mock.post("https://api.openai.com/v1/files").mock(
        return_value=httpx.Response(200, json=dict(_OPENAI_FILE_JSON))
    )
    create_route: Final = respx_mock.post("https://api.openai.com/v1/batches").mock(
        return_value=httpx.Response(200, json=dict(_OPENAI_BATCH_JSON))
    )
    retrieve_route: Final = respx_mock.get("https://api.openai.com/v1/batches/batch_abc123").mock(
        return_value=httpx.Response(200, json=dict(_OPENAI_BATCH_JSON))
    )
    list_batches_route: Final = respx_mock.get("https://api.openai.com/v1/batches").mock(
        return_value=httpx.Response(200, json={"object": "list", "data": [dict(_OPENAI_BATCH_JSON)]})
    )
    respx_mock.get("https://api.openai.com/v1/files/file-abc123/content").mock(
        return_value=httpx.Response(200, content=b'{"custom_id": "request-1"}\n')
    )
    respx_mock.get("https://api.openai.com/v1/files/file-abc123").mock(
        return_value=httpx.Response(200, json=dict(_OPENAI_FILE_JSON))
    )
    respx_mock.delete("https://api.openai.com/v1/files/file-abc123").mock(
        return_value=httpx.Response(200, json={"id": "file-abc123", "object": "file", "deleted": True})
    )
    list_files_route: Final = respx_mock.get("https://api.openai.com/v1/files").mock(
        return_value=httpx.Response(200, json={"object": "list", "data": [dict(_OPENAI_FILE_JSON)]})
    )
    cancel_route: Final = respx_mock.post("https://api.openai.com/v1/batches/batch_abc123/cancel").mock(
        return_value=httpx.Response(200, json={**_OPENAI_BATCH_JSON, "status": "cancelling"})
    )

    batch_file: Final = (
        "batch.jsonl",
        b'{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", '
        b'"body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}}\n',
        "application/jsonl",
    )
    file_obj: Final = await litellm.acreate_file(
        file=batch_file, purpose="batch", custom_llm_provider="openai", api_key="fake-key"
    )
    assert file_obj.id == "file-abc123"
    upload_body: Final = upload_route.calls.last.request.content
    assert b'name="purpose"\r\n\r\nbatch' in upload_body
    assert batch_file[1] in upload_body

    extra_metadata_field: Final = {
        "user_api_key_alias": "special_api_key_alias",
        "user_api_key_team_alias": "special_team_alias",
    }
    create_batch_response: Final = await litellm.acreate_batch(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=file_obj.id,
        custom_llm_provider="openai",
        api_key="fake-key",
        metadata={"key1": "value1", "key2": "value2"},
        litellm_metadata=extra_metadata_field,
    )

    assert json.loads(create_route.calls.last.request.content) == {
        "completion_window": "24h",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-abc123",
        "metadata": {"key1": "value1", "key2": "value2"},
    }
    assert create_batch_response.id == "batch_abc123"
    assert create_batch_response.endpoint == "/v1/chat/completions"
    assert create_batch_response.input_file_id == file_obj.id

    await asyncio.wait_for(recorder.logged.wait(), timeout=10)
    assert recorder.payload is not None
    standard_logging_object: Final = recorder.payload
    assert standard_logging_object["metadata"]["user_api_key_alias"] == extra_metadata_field["user_api_key_alias"]
    assert (
        standard_logging_object["metadata"]["user_api_key_team_alias"]
        == extra_metadata_field["user_api_key_team_alias"]
    )

    retrieved_batch: Final = await litellm.aretrieve_batch(
        batch_id=create_batch_response.id, custom_llm_provider="openai", api_key="fake-key"
    )
    assert retrieve_route.called
    assert retrieved_batch.id == create_batch_response.id

    list_batches: Final = await litellm.alist_batches(custom_llm_provider="openai", limit=2, api_key="fake-key")
    assert list_batches_route.calls.last.request.url.params["limit"] == "2"
    assert [batch.id for batch in list_batches.data] == ["batch_abc123"]

    file_content: Final = await litellm.afile_content(
        file_id=file_obj.id, custom_llm_provider="openai", api_key="fake-key"
    )
    assert file_content.content == b'{"custom_id": "request-1"}\n'

    retrieved_file: Final = await litellm.afile_retrieve(
        file_id=file_obj.id, custom_llm_provider="openai", api_key="fake-key"
    )
    assert retrieved_file.id == file_obj.id

    delete_file_response: Final = await litellm.afile_delete(
        file_id=file_obj.id, custom_llm_provider="openai", api_key="fake-key"
    )
    assert delete_file_response.id == file_obj.id

    all_files_list: Final = await litellm.afile_list(custom_llm_provider="openai", api_key="fake-key")
    assert list_files_route.called
    assert [file.id for file in all_files_list.data] == ["file-abc123"]

    cancel_batch_response: Final = await litellm.acancel_batch(
        batch_id=create_batch_response.id, custom_llm_provider="openai", api_key="fake-key"
    )
    assert cancel_route.called
    assert cancel_batch_response.id == create_batch_response.id


@pytest.mark.asyncio
async def test_delete_batch_output_file(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    batch_with_output: Final = {
        **_OPENAI_BATCH_JSON,
        "status": "completed",
        "output_file_id": "file-output123",
    }
    respx_mock.get("https://api.openai.com/v1/batches/batch_abc123").mock(
        return_value=httpx.Response(200, json=batch_with_output)
    )
    delete_route: Final = respx_mock.delete("https://api.openai.com/v1/files/file-output123").mock(
        return_value=httpx.Response(200, json={"id": "file-output123", "object": "file", "deleted": True})
    )

    batch: Final = await litellm.aretrieve_batch(
        batch_id="batch_abc123", custom_llm_provider="openai", api_key="fake-key"
    )
    assert batch.output_file_id == "file-output123"

    delete_response: Final = await litellm.afile_delete(
        file_id=batch.output_file_id, custom_llm_provider="openai", api_key="fake-key"
    )
    assert delete_route.call_count == 1
    assert delete_response.id == "file-output123"
    assert delete_response.deleted is True
