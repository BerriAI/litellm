"""
Routing-contract tests for litellm/proxy/batches_endpoints/endpoints.py

These are not happy-path smoke tests. Each row of the matrix locks the full
contract of a single routing branch so that *any* behavior change in this layer
fails loudly:

  1. DISPATCH      - exactly which downstream seam fired (litellm.acreate_batch
                     vs llm_router.acreate_batch), and every sibling seam is
                     asserted NOT called. A reordered/negated branch flips this.
  2. CREDENTIALS   - the credential resolver receives the model derived from the
                     request, not a hardcoded value. The router's
                     get_deployment_credentials_with_provider is input-locked.
  3. SEAM PAYLOAD  - the *entire* kwargs dict forwarded to the provider call is
                     exact-matched. Because LiteLLMBatchCreateRequest is a
                     TypedDict (zero runtime filtering), nothing else stops a
                     newly-added param from silently reaching every provider.
                     This exact-match is that missing guard: a new key fails the
                     test and forces a reviewer to ask "does this work for all
                     providers, or just openai".
  4. OUTPUT SHAPE  - the id encode/decode round-trip clients depend on.

Only true I/O boundaries are mocked (provider call, router, proxy logging,
request parsing, pre-call enrichment). The pure encode/decode/credential-merge
helpers run for real so the payload assertions reflect production exactly.

The object mocks are spec'd to their real classes, so a brand-new method call
added to this layer raises instead of silently passing - the inventory of seams
cannot drift without a test failure.
"""

import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
import litellm.proxy.batches_endpoints.endpoints as endpoints
import litellm.proxy.proxy_server as proxy_server
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.openai_files_endpoints.common_utils import (
    encode_file_id_with_model,
)
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.types.llms.openai import BatchJobStatus
from litellm.types.utils import LiteLLMBatch

from fastapi import Response

# --------------------------------------------------------------------------- #
# Fixtures: distinguishable credentials per model so a wrong/hardcoded model_id
# produces wrong creds (or KeyError) and is impossible to hide.
# --------------------------------------------------------------------------- #

CREDS: Dict[str, Dict[str, str]] = {
    "azure/gpt-4o": {
        "custom_llm_provider": "azure",
        "api_key": "sk-azure",
        "api_base": "https://azure.test",
        "model": "azure/gpt-4o-deployment",
    },
    "vertex-model": {
        "custom_llm_provider": "vertex_ai",
        "api_key": "sk-vertex",
        "api_base": "https://vertex.test",
        "model": "vertex_ai/gemini-2.0",
    },
}

# A real model-encoded file id: decodes to "azure/gpt-4o", strips to "file-original123".
AZURE_FILE_ID = encode_file_id_with_model(
    "file-original123", "azure/gpt-4o", id_type="file"
)


def make_batch(
    *,
    id: str = "batch-provider-id",
    output_file_id: Optional[str] = None,
    error_file_id: Optional[str] = None,
    input_file_id: Optional[str] = None,
    status: BatchJobStatus = "validating",
) -> LiteLLMBatch:
    batch = LiteLLMBatch(
        id=id,
        completion_window="24h",
        created_at=1234567890,
        endpoint="/v1/chat/completions",
        input_file_id=input_file_id or "file-provider-input",
        object="batch",
        status=status,
    )
    if output_file_id is not None:
        batch.output_file_id = output_file_id
    if error_file_id is not None:
        batch.error_file_id = error_file_id
    batch._hidden_params = {}
    return batch


class FakeRequest:
    """Minimal stand-in. The request is only read via .headers/.query_params on
    the model-param fallback path; everything else that touches it is mocked."""

    def __init__(
        self,
        headers: Optional[Dict[str, str]] = None,
        query: Optional[Dict[str, str]] = None,
    ):
        self.headers = headers or {}
        self.query_params = query or {}


@dataclass
class Harness:
    """Holds every mocked seam so a test can configure inputs and assert calls."""

    body: Dict[str, Any]
    read_body: AsyncMock
    pre_call: AsyncMock
    get_headers: MagicMock
    provider_from_headers: MagicMock
    is_known_model: MagicMock
    litellm_acreate: AsyncMock
    router: MagicMock
    logging: MagicMock
    creds_resolver: MagicMock

    @property
    def router_acreate(self) -> AsyncMock:
        return self.router.acreate_batch

    def acreate_kwargs(self) -> Dict[str, Any]:
        """Exact kwargs forwarded to litellm.acreate_batch."""
        assert self.litellm_acreate.call_count == 1
        return dict(self.litellm_acreate.call_args.kwargs)

    def router_kwargs(self) -> Dict[str, Any]:
        assert self.router_acreate.call_count == 1
        return dict(self.router_acreate.call_args.kwargs)


def _creds_lookup(*, model_id: str) -> Dict[str, str]:
    # KeyError on an unknown/hardcoded model_id - the bug cannot hide.
    return dict(CREDS[model_id])


@pytest.fixture
def harness():
    """Seam harness. Patches only true I/O boundaries; pure encode/decode/merge
    helpers run for real. Object mocks are spec'd so unknown method calls raise."""
    body_holder: Dict[str, Any] = {}
    logging = MagicMock(spec=ProxyLogging)
    logging.post_call_success_hook = AsyncMock(side_effect=lambda **kw: kw["response"])
    logging.post_call_failure_hook = AsyncMock()
    logging.update_request_status = AsyncMock()
    logging.get_proxy_hook = MagicMock(return_value=None)

    router = MagicMock(spec=Router)
    router.acreate_batch = AsyncMock(return_value=make_batch())
    router.get_deployment_credentials_with_provider = MagicMock(
        side_effect=_creds_lookup
    )

    read_body = AsyncMock(side_effect=lambda request: body_holder["body"])
    pre_call = AsyncMock(side_effect=lambda **kw: (body_holder["body"], MagicMock()))
    get_headers = MagicMock(return_value={})
    provider_from_headers = MagicMock(return_value=None)
    is_known_model = MagicMock(return_value=False)
    litellm_acreate = AsyncMock(return_value=make_batch())

    with ExitStack() as stack:
        stack.enter_context(patch.object(endpoints, "_read_request_body", read_body))
        stack.enter_context(
            patch.object(
                ProxyBaseLLMRequestProcessing,
                "common_processing_pre_call_logic",
                pre_call,
            )
        )
        stack.enter_context(
            patch.object(
                ProxyBaseLLMRequestProcessing, "get_custom_headers", get_headers
            )
        )
        stack.enter_context(
            patch.object(
                endpoints,
                "get_custom_llm_provider_from_request_headers",
                provider_from_headers,
            )
        )
        stack.enter_context(patch.object(endpoints, "is_known_model", is_known_model))
        stack.enter_context(patch.object(litellm, "acreate_batch", litellm_acreate))
        stack.enter_context(
            patch.object(litellm, "enable_loadbalancing_on_batch_endpoints", False)
        )
        stack.enter_context(patch.object(proxy_server, "llm_router", router))
        stack.enter_context(patch.object(proxy_server, "proxy_logging_obj", logging))
        stack.enter_context(patch.object(proxy_server, "general_settings", {}))
        stack.enter_context(patch.object(proxy_server, "proxy_config", MagicMock()))
        stack.enter_context(patch.object(proxy_server, "version", "test-version"))

        h = Harness(
            body=body_holder,
            read_body=read_body,
            pre_call=pre_call,
            get_headers=get_headers,
            provider_from_headers=provider_from_headers,
            is_known_model=is_known_model,
            litellm_acreate=litellm_acreate,
            router=router,
            logging=logging,
            creds_resolver=router.get_deployment_credentials_with_provider,
        )
        yield h


def set_body(harness: Harness, body: Dict[str, Any]) -> None:
    harness.body["body"] = body


async def call_create(
    harness: Harness,
    *,
    provider: Optional[str] = None,
    user: Optional[UserAPIKeyAuth] = None,
    headers: Optional[Dict[str, str]] = None,
    query: Optional[Dict[str, str]] = None,
):
    return await endpoints.create_batch(
        request=FakeRequest(headers=headers, query=query),
        fastapi_response=Response(),
        provider=provider,
        user_api_key_dict=user or UserAPIKeyAuth(api_key="sk-test"),
    )


# =========================================================================== #
# SCENARIO 1 - input_file_id encoded with model. The full showcase: every
# assertion type from the design lives here.
# =========================================================================== #


@pytest.mark.asyncio
async def test_create__model_encoded_file_id(harness):
    set_body(
        harness,
        {
            "input_file_id": AZURE_FILE_ID,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )

    resp = await call_create(harness)

    # 1. DISPATCH - model-credential path fired via litellm, router did not.
    assert harness.litellm_acreate.call_count == 1
    harness.router_acreate.assert_not_called()

    # 2. CREDENTIALS - resolved for the model decoded FROM the file id.
    harness.creds_resolver.assert_called_once_with(model_id="azure/gpt-4o")

    # 3. SEAM PAYLOAD - exact, whole dict. A new forwarded key breaks this.
    assert harness.acreate_kwargs() == {
        "custom_llm_provider": "azure",
        "input_file_id": "file-original123",  # encoding stripped by this layer
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": None,  # sanitize_openai_provider_metadata(None)
        "api_key": "sk-azure",
        "api_base": "https://azure.test",
        "model": "azure/gpt-4o-deployment",
    }

    # 4. OUTPUT SHAPE - ids re-encoded with the model; input_file_id restored.
    assert resp.id == encode_file_id_with_model(
        "batch-provider-id", "azure/gpt-4o", id_type="batch"
    )
    assert resp.input_file_id == AZURE_FILE_ID


@pytest.mark.asyncio
async def test_create__model_encoded_file_id__encodes_output_and_error_ids(harness):
    set_body(
        harness,
        {
            "input_file_id": AZURE_FILE_ID,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    harness.litellm_acreate.return_value = make_batch(
        id="batch-xyz",
        output_file_id="file-out-raw",
        error_file_id="file-err-raw",
    )

    resp = await call_create(harness)

    assert resp.output_file_id == encode_file_id_with_model(
        "file-out-raw", "azure/gpt-4o"
    )
    assert resp.error_file_id == encode_file_id_with_model(
        "file-err-raw", "azure/gpt-4o"
    )


@pytest.mark.asyncio
async def test_create__model_encoded_file_id__resolver_gets_decoded_model(harness):
    """Regression guard: model_id for credential resolution must be derived from
    the file id. A hardcode would call the resolver with the wrong model."""
    set_body(
        harness,
        {
            "input_file_id": AZURE_FILE_ID,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )

    await call_create(harness)

    harness.creds_resolver.assert_called_once_with(model_id="azure/gpt-4o")


# =========================================================================== #
# SCENARIO 2 - model from body / header / query. Locks source precedence.
# =========================================================================== #


@pytest.mark.asyncio
async def test_create__model_from_body(harness):
    set_body(
        harness,
        {
            "input_file_id": "file-plain",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "model": "vertex-model",
        },
    )

    resp = await call_create(harness)

    assert harness.litellm_acreate.call_count == 1
    harness.router_acreate.assert_not_called()
    harness.creds_resolver.assert_called_once_with(model_id="vertex-model")
    payload = harness.acreate_kwargs()
    assert payload["custom_llm_provider"] == "vertex_ai"
    assert payload["input_file_id"] == "file-plain"
    assert resp.id == encode_file_id_with_model(
        "batch-provider-id", "vertex-model", id_type="batch"
    )


@pytest.mark.asyncio
async def test_create__model_from_header(harness):
    set_body(
        harness,
        {
            "input_file_id": "file-plain",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )

    await call_create(harness, headers={"x-litellm-model": "vertex-model"})

    harness.creds_resolver.assert_called_once_with(model_id="vertex-model")
    harness.router_acreate.assert_not_called()


@pytest.mark.asyncio
async def test_create__model_from_query(harness):
    set_body(
        harness,
        {
            "input_file_id": "file-plain",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )

    await call_create(harness, query={"model": "vertex-model"})

    harness.creds_resolver.assert_called_once_with(model_id="vertex-model")
    harness.router_acreate.assert_not_called()


@pytest.mark.asyncio
async def test_create__body_model_beats_header_and_query(harness):
    """Precedence row: body > header > query (data.get('model') first)."""
    set_body(
        harness,
        {
            "input_file_id": "file-plain",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "model": "azure/gpt-4o",
        },
    )

    await call_create(
        harness,
        headers={"x-litellm-model": "vertex-model"},
        query={"model": "vertex-model"},
    )

    harness.creds_resolver.assert_called_once_with(model_id="azure/gpt-4o")


# =========================================================================== #
# SCENARIO 3 - fallback to custom_llm_provider (env-var creds). MUST NOT touch
# the credential resolver.
# =========================================================================== #


@pytest.mark.asyncio
async def test_create__fallback_default_openai(harness):
    set_body(
        harness,
        {
            "input_file_id": "file-plain",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )

    await call_create(harness)

    assert harness.litellm_acreate.call_count == 1
    harness.router_acreate.assert_not_called()
    harness.creds_resolver.assert_not_called()  # inverse-bug guard
    assert harness.acreate_kwargs()["custom_llm_provider"] == "openai"


@pytest.mark.asyncio
async def test_create__fallback_provider_path_param(harness):
    set_body(
        harness,
        {
            "input_file_id": "file-plain",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )

    await call_create(harness, provider="anthropic")

    harness.creds_resolver.assert_not_called()
    assert harness.acreate_kwargs()["custom_llm_provider"] == "anthropic"


@pytest.mark.asyncio
async def test_create__fallback_body_custom_llm_provider(harness):
    set_body(
        harness,
        {
            "input_file_id": "file-plain",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "custom_llm_provider": "bedrock",
        },
    )

    await call_create(harness)

    payload = harness.acreate_kwargs()
    assert payload["custom_llm_provider"] == "bedrock"


# =========================================================================== #
# Unified file id routing (-> llm_router). Helpers mocked only here because a
# real unified id is opaque base64; the routing contract is what we lock.
# =========================================================================== #


@pytest.mark.asyncio
async def test_create__unified_file_id_single_model(harness):
    set_body(
        harness,
        {
            "input_file_id": "litellm_proxy_unified_id",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    with patch.object(
        endpoints, "_is_base64_encoded_unified_file_id", return_value="unified-xyz"
    ), patch.object(
        endpoints, "get_models_from_unified_file_id", return_value=["gpt-4o-mini"]
    ):
        resp = await call_create(harness)

    # DISPATCH - router fired, direct litellm did not.
    assert harness.router_acreate.call_count == 1
    harness.litellm_acreate.assert_not_called()
    # model injected from the unified id, input_file_id restored, hidden param set
    assert harness.router_kwargs()["model"] == "gpt-4o-mini"
    assert resp.input_file_id == "litellm_proxy_unified_id"
    assert resp._hidden_params["unified_file_id"] == "unified-xyz"


@pytest.mark.asyncio
@pytest.mark.parametrize("models", [[], ["m1", "m2"]])
async def test_create__unified_file_id_not_exactly_one_model_400(harness, models):
    set_body(
        harness,
        {
            "input_file_id": "litellm_proxy_unified_id",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    with patch.object(
        endpoints, "_is_base64_encoded_unified_file_id", return_value="unified-xyz"
    ), patch.object(
        endpoints, "get_models_from_unified_file_id", return_value=models
    ):
        with pytest.raises(ProxyException) as exc:
            await call_create(harness)

    assert exc.value.code == "400"
    harness.router_acreate.assert_not_called()
    harness.litellm_acreate.assert_not_called()


@pytest.mark.asyncio
async def test_create__unified_file_id_resolves_real_storage_url(harness):
    """A base64 unified_file_id is a LiteLLM-internal token, not a real
    provider-side file reference (e.g. Vertex AI's batch transformation parses
    a `publishers/` segment out of the file URI and crashes on the opaque
    base64 string). The real backend location (`storage_url`) must be looked
    up from LiteLLM_ManagedFileTable and substituted before dispatch - the
    unified id itself must never reach the router/provider call."""
    set_body(
        harness,
        {
            "input_file_id": "litellm_proxy_unified_id",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )

    fake_db_file = MagicMock(
        storage_url="gs://bucket/litellm-vertex-files/publishers/google/models/gemini-2.0/abc"
    )
    find_first = AsyncMock(return_value=fake_db_file)
    fake_repo_instance = MagicMock()
    fake_repo_instance.table.find_first = find_first
    fake_repo_cls = MagicMock(return_value=fake_repo_instance)

    fake_prisma_client = MagicMock()

    with patch.object(
        endpoints, "_is_base64_encoded_unified_file_id", return_value="unified-xyz"
    ), patch.object(
        endpoints, "get_models_from_unified_file_id", return_value=["gemini-2.0"]
    ), patch.object(
        proxy_server, "prisma_client", fake_prisma_client
    ), patch(
        "litellm.repositories.table_repositories.ManagedFileRepository",
        fake_repo_cls,
    ):
        resp = await call_create(harness)

    # The real storage_url - not the opaque unified id - must be what's
    # forwarded to the router/provider.
    assert harness.router_kwargs()["input_file_id"] == fake_db_file.storage_url
    find_first.assert_awaited_once_with(where={"unified_file_id": "unified-xyz"})
    # The unified id is still what's returned to the client.
    assert resp.input_file_id == "litellm_proxy_unified_id"
    assert resp._hidden_params["unified_file_id"] == "unified-xyz"


@pytest.mark.asyncio
async def test_create__unified_file_id_no_managed_file_record_falls_back_to_raw_id(
    harness,
):
    """If there's no LiteLLM_ManagedFileTable row (or it has no storage_url),
    fall back to the previous behavior instead of raising - callers/providers
    that don't need the resolved path (or legacy data) keep working."""
    set_body(
        harness,
        {
            "input_file_id": "litellm_proxy_unified_id",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )

    find_first = AsyncMock(return_value=None)
    fake_repo_instance = MagicMock()
    fake_repo_instance.table.find_first = find_first
    fake_repo_cls = MagicMock(return_value=fake_repo_instance)

    with patch.object(
        endpoints, "_is_base64_encoded_unified_file_id", return_value="unified-xyz"
    ), patch.object(
        endpoints, "get_models_from_unified_file_id", return_value=["gemini-2.0"]
    ), patch.object(
        proxy_server, "prisma_client", MagicMock()
    ), patch(
        "litellm.repositories.table_repositories.ManagedFileRepository",
        fake_repo_cls,
    ):
        await call_create(harness)

    assert harness.router_kwargs()["input_file_id"] == "litellm_proxy_unified_id"


@pytest.mark.asyncio
async def test_create__model_encoded_beats_unified(harness):
    """Precedence row: a file id that is BOTH model-encoded and (pretend) unified
    must take the model-encoded branch (checked first)."""
    set_body(
        harness,
        {
            "input_file_id": AZURE_FILE_ID,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    with patch.object(
        endpoints, "_is_base64_encoded_unified_file_id", return_value="unified-xyz"
    ), patch.object(
        endpoints, "get_models_from_unified_file_id", return_value=["something-else"]
    ):
        await call_create(harness)

    assert harness.litellm_acreate.call_count == 1
    harness.router_acreate.assert_not_called()
    harness.creds_resolver.assert_called_once_with(model_id="azure/gpt-4o")


# =========================================================================== #
# Loadbalancing branch (-> llm_router) and its precedence vs model-encoded.
# =========================================================================== #


@pytest.mark.asyncio
async def test_create__loadbalancing_routes_to_router(harness):
    set_body(
        harness,
        {
            "input_file_id": "file-plain",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "model": "lb-model",
        },
    )
    harness.is_known_model.return_value = True
    with patch.object(litellm, "enable_loadbalancing_on_batch_endpoints", True):
        await call_create(harness)

    harness.is_known_model.assert_called_once_with(
        model="lb-model", llm_router=harness.router
    )
    assert harness.router_acreate.call_count == 1
    harness.litellm_acreate.assert_not_called()
    harness.creds_resolver.assert_not_called()


@pytest.mark.asyncio
async def test_create__model_encoded_beats_loadbalancing(harness):
    set_body(
        harness,
        {
            "input_file_id": AZURE_FILE_ID,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "model": "lb-model",
        },
    )
    harness.is_known_model.return_value = True
    with patch.object(litellm, "enable_loadbalancing_on_batch_endpoints", True):
        await call_create(harness)

    assert harness.litellm_acreate.call_count == 1
    harness.router_acreate.assert_not_called()
    harness.creds_resolver.assert_called_once_with(model_id="azure/gpt-4o")


# =========================================================================== #
# Team-level batch expiry enforcement (independent of routing).
# =========================================================================== #


def _user_with_expiry(expiry: Any) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-test",
        team_metadata={"enforced_batch_output_expires_after": expiry},
    )


@pytest.mark.asyncio
async def test_create__team_expiry_injected(harness):
    set_body(
        harness,
        {
            "input_file_id": "file-plain",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )

    await call_create(
        harness, user=_user_with_expiry({"anchor": "created_at", "seconds": 3600})
    )

    assert harness.acreate_kwargs()["output_expires_after"] == {
        "anchor": "created_at",
        "seconds": 3600,
    }


@pytest.mark.asyncio
async def test_create__no_team_expiry_not_injected(harness):
    set_body(
        harness,
        {
            "input_file_id": "file-plain",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )

    await call_create(harness, user=UserAPIKeyAuth(api_key="sk-test"))

    assert "output_expires_after" not in harness.acreate_kwargs()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expiry",
    [
        {"seconds": 3600},  # missing anchor
        {"anchor": "created_at"},  # missing seconds
        {"anchor": "completed_at", "seconds": 3600},  # wrong anchor
    ],
)
async def test_create__team_expiry_malformed_500(harness, expiry):
    set_body(
        harness,
        {
            "input_file_id": "file-plain",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )

    with pytest.raises(ProxyException) as exc:
        await call_create(harness, user=_user_with_expiry(expiry))

    assert exc.value.code == "500"
    harness.litellm_acreate.assert_not_called()
    harness.router_acreate.assert_not_called()


# =========================================================================== #
# Cross-cutting: enrichment route_type, metadata sanitization, failure hook.
# =========================================================================== #


@pytest.mark.asyncio
async def test_create__uses_acreate_batch_route_type(harness):
    set_body(
        harness,
        {
            "input_file_id": "file-plain",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )

    await call_create(harness)

    assert harness.pre_call.call_args.kwargs["route_type"] == "acreate_batch"


@pytest.mark.asyncio
async def test_create__metadata_sanitized_before_forwarding(harness):
    set_body(
        harness,
        {
            "input_file_id": "file-plain",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "metadata": {"user_key": "user_val", "spend_logs_metadata": {"x": 1}},
        },
    )

    await call_create(harness)

    # provider-internal-only key dropped, string key kept (real sanitize runs)
    assert harness.acreate_kwargs()["metadata"] == {"user_key": "user_val"}


@pytest.mark.asyncio
async def test_create__exception_calls_failure_hook(harness):
    set_body(
        harness,
        {
            "input_file_id": "file-plain",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    harness.litellm_acreate.side_effect = ValueError("provider boom")

    with pytest.raises(Exception):
        await call_create(harness)

    harness.logging.post_call_failure_hook.assert_called_once()
    assert (
        harness.logging.post_call_failure_hook.call_args.kwargs[
            "original_exception"
        ].args[0]
        == "provider boom"
    )


# =========================================================================== #
#                                                                             #
#   GET /v1/batches/{batch_id}  -  retrieve_batch routing-contract tests      #
#                                                                             #
#   Same discipline as create_batch above. retrieve_batch has more seams: it  #
#   first consults the ManagedObjectTable (get_batch_from_database) and may    #
#   short-circuit on a terminal-status row WITHOUT ever calling a provider,    #
#   then on a miss/non-terminal row routes to one of three downstream seams    #
#   (litellm.aretrieve_batch via model creds, llm_router.aretrieve_batch, or   #
#   litellm.aretrieve_batch via env-var provider) and writes the fresh state   #
#   back via update_batch_in_database. Every test below locks exactly which    #
#   of those seams fired and asserts the siblings did NOT, so a reordered or   #
#   negated branch - or a dropped DB short-circuit / write-back - fails loud.  #
#                                                                             #
#   The DB seams (get_batch_from_database, update_batch_in_database,           #
#   resolve_*_to_unified) are true prisma I/O boundaries and are mocked. The   #
#   encode/decode/credential-merge helpers run for real, so payload and id     #
#   round-trip assertions reflect production exactly.                          #
# =========================================================================== #


# A real model-encoded BATCH id: decodes to "azure/gpt-4o", strips to
# "batch_orig123". Distinct from AZURE_FILE_ID so retrieve tests can't pass by
# accidentally reusing the create fixture's value.
AZURE_BATCH_ID = encode_file_id_with_model(
    "batch_orig123", "azure/gpt-4o", id_type="batch"
)

# A realistic decoded unified batch id (what _is_base64_encoded_unified_file_id
# returns). model_id / llm_batch_id are parsed out of this by the real helpers.
UNIFIED_BATCH_ID = "litellm_proxy;model_id:gpt-4o-mini;llm_batch_id:batch-raw-xyz"


@dataclass
class RetrieveHarness:
    """Seams for retrieve_batch. `data['data']` is the dict pre-call enrichment
    returns; the routing branches mutate it, so it is reset per call."""

    data: Dict[str, Any]
    pre_call: AsyncMock
    get_headers: MagicMock
    provider_from_headers: MagicMock
    provider_from_query: MagicMock
    litellm_aretrieve: AsyncMock
    router: MagicMock
    logging: MagicMock
    creds_resolver: MagicMock
    get_batch_from_db: AsyncMock
    update_batch_in_db: AsyncMock
    resolve_input: AsyncMock
    resolve_output: AsyncMock

    @property
    def router_aretrieve(self) -> AsyncMock:
        return self.router.aretrieve_batch

    def aretrieve_kwargs(self) -> Dict[str, Any]:
        """Exact kwargs forwarded to litellm.aretrieve_batch."""
        assert self.litellm_aretrieve.call_count == 1
        return dict(self.litellm_aretrieve.call_args.kwargs)

    def router_kwargs(self) -> Dict[str, Any]:
        assert self.router_aretrieve.call_count == 1
        return dict(self.router_aretrieve.call_args.kwargs)


@pytest.fixture
def retrieve_harness():
    data_holder: Dict[str, Any] = {"data": {}}
    logging = MagicMock(spec=ProxyLogging)
    logging.post_call_success_hook = AsyncMock(side_effect=lambda **kw: kw["response"])
    logging.post_call_failure_hook = AsyncMock()
    logging.update_request_status = AsyncMock()
    logging.get_proxy_hook = MagicMock(return_value=None)

    router = MagicMock(spec=Router)
    router.aretrieve_batch = AsyncMock(return_value=make_batch())
    router.get_deployment_credentials_with_provider = MagicMock(
        side_effect=_creds_lookup
    )

    pre_call = AsyncMock(side_effect=lambda **kw: (data_holder["data"], MagicMock()))
    get_headers = MagicMock(return_value={})
    provider_from_headers = MagicMock(return_value=None)
    provider_from_query = MagicMock(return_value=None)
    litellm_aretrieve = AsyncMock(return_value=make_batch())
    # Default: DB miss -> always fall through to provider routing.
    get_batch_from_db = AsyncMock(return_value=(None, None))
    update_batch_in_db = AsyncMock(return_value=None)
    resolve_input = AsyncMock(return_value=None)
    resolve_output = AsyncMock(return_value=None)

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                ProxyBaseLLMRequestProcessing,
                "common_processing_pre_call_logic",
                pre_call,
            )
        )
        stack.enter_context(
            patch.object(
                ProxyBaseLLMRequestProcessing, "get_custom_headers", get_headers
            )
        )
        stack.enter_context(
            patch.object(
                endpoints,
                "get_custom_llm_provider_from_request_headers",
                provider_from_headers,
            )
        )
        stack.enter_context(
            patch.object(
                endpoints,
                "get_custom_llm_provider_from_request_query",
                provider_from_query,
            )
        )
        stack.enter_context(
            patch.object(endpoints, "get_batch_from_database", get_batch_from_db)
        )
        stack.enter_context(
            patch.object(endpoints, "update_batch_in_database", update_batch_in_db)
        )
        stack.enter_context(
            patch.object(endpoints, "resolve_input_file_id_to_unified", resolve_input)
        )
        stack.enter_context(
            patch.object(
                endpoints, "resolve_output_file_ids_to_unified", resolve_output
            )
        )
        stack.enter_context(patch.object(litellm, "aretrieve_batch", litellm_aretrieve))
        stack.enter_context(
            patch.object(litellm, "enable_loadbalancing_on_batch_endpoints", False)
        )
        stack.enter_context(patch.object(proxy_server, "llm_router", router))
        stack.enter_context(patch.object(proxy_server, "proxy_logging_obj", logging))
        stack.enter_context(patch.object(proxy_server, "general_settings", {}))
        stack.enter_context(patch.object(proxy_server, "proxy_config", MagicMock()))
        stack.enter_context(patch.object(proxy_server, "version", "test-version"))
        stack.enter_context(patch.object(proxy_server, "prisma_client", MagicMock()))

        yield RetrieveHarness(
            data=data_holder,
            pre_call=pre_call,
            get_headers=get_headers,
            provider_from_headers=provider_from_headers,
            provider_from_query=provider_from_query,
            litellm_aretrieve=litellm_aretrieve,
            router=router,
            logging=logging,
            creds_resolver=router.get_deployment_credentials_with_provider,
            get_batch_from_db=get_batch_from_db,
            update_batch_in_db=update_batch_in_db,
            resolve_input=resolve_input,
            resolve_output=resolve_output,
        )


async def call_retrieve(
    harness: RetrieveHarness,
    batch_id: str,
    *,
    provider: Optional[str] = None,
    user: Optional[UserAPIKeyAuth] = None,
    headers: Optional[Dict[str, str]] = None,
    query: Optional[Dict[str, str]] = None,
):
    # Mirror the real flow: data starts as RetrieveBatchRequest(batch_id=...).
    harness.data["data"] = {"batch_id": batch_id}
    return await endpoints.retrieve_batch(
        request=FakeRequest(headers=headers, query=query),
        fastapi_response=Response(),
        user_api_key_dict=user or UserAPIKeyAuth(api_key="sk-test"),
        provider=provider,
        batch_id=batch_id,
    )


# --------------------------------------------------------------------------- #
# SCENARIO 1 - batch id encoded with model. litellm.aretrieve_batch via the
# model's resolved credentials; response ids re-encoded for the client.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_retrieve__model_encoded_id(retrieve_harness):
    resp = await call_retrieve(retrieve_harness, AZURE_BATCH_ID)

    # 1. DISPATCH - model-credential path fired via litellm, router did not.
    assert retrieve_harness.litellm_aretrieve.call_count == 1
    retrieve_harness.router_aretrieve.assert_not_called()

    # 2. CREDENTIALS - resolved for the model decoded FROM the batch id.
    retrieve_harness.creds_resolver.assert_called_once_with(model_id="azure/gpt-4o")

    # 3. SEAM PAYLOAD - exact, whole dict forwarded to the provider call.
    #    Note `model` is the DECODED model, not the deployment from creds: the
    #    endpoint overrides it (provider-config providers like bedrock need it).
    assert retrieve_harness.aretrieve_kwargs() == {
        "custom_llm_provider": "azure",
        "batch_id": "batch_orig123",  # encoding stripped by this layer
        "api_key": "sk-azure",
        "api_base": "https://azure.test",
        "model": "azure/gpt-4o",
    }

    # 4. OUTPUT SHAPE - ids re-encoded with the model for the round-trip.
    assert resp.id == encode_file_id_with_model(
        "batch-provider-id", "azure/gpt-4o", id_type="batch"
    )

    # write-back to the managed-object table happened, tagged as a retrieve.
    assert retrieve_harness.update_batch_in_db.call_count == 1
    assert retrieve_harness.update_batch_in_db.call_args.kwargs["operation"] == "retrieve"


@pytest.mark.asyncio
async def test_retrieve__model_encoded_id__forwards_decoded_model_not_deployment(
    retrieve_harness,
):
    """Regression guard for the line-483 override: the model forwarded to the
    provider must be the decoded model id, never the deployment name that the
    credential merge pulled in. Dropping the override silently 400s bedrock."""
    await call_retrieve(retrieve_harness, AZURE_BATCH_ID)

    assert retrieve_harness.aretrieve_kwargs()["model"] == "azure/gpt-4o"


@pytest.mark.asyncio
async def test_retrieve__model_encoded_id__encodes_output_and_error_ids(
    retrieve_harness,
):
    retrieve_harness.litellm_aretrieve.return_value = make_batch(
        id="batch-xyz",
        output_file_id="file-out-raw",
        error_file_id="file-err-raw",
    )

    resp = await call_retrieve(retrieve_harness, AZURE_BATCH_ID)

    assert resp.output_file_id == encode_file_id_with_model(
        "file-out-raw", "azure/gpt-4o"
    )
    assert resp.error_file_id == encode_file_id_with_model(
        "file-err-raw", "azure/gpt-4o"
    )


@pytest.mark.asyncio
async def test_retrieve__model_encoded_beats_loadbalancing(retrieve_harness):
    """Precedence: model-encoded id is checked before the loadbalancing/unified
    elif, so it wins even with loadbalancing enabled."""
    with patch.object(litellm, "enable_loadbalancing_on_batch_endpoints", True):
        await call_retrieve(retrieve_harness, AZURE_BATCH_ID)

    assert retrieve_harness.litellm_aretrieve.call_count == 1
    retrieve_harness.router_aretrieve.assert_not_called()
    retrieve_harness.creds_resolver.assert_called_once_with(model_id="azure/gpt-4o")


# --------------------------------------------------------------------------- #
# Unified managed batch id -> llm_router.aretrieve_batch. model_id is parsed
# out of the unified id and stamped onto hidden params; raw file ids on the
# response are resolved back to unified ids.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_retrieve__unified_batch_id_routes_to_router(retrieve_harness):
    with patch.object(
        endpoints, "_is_base64_encoded_unified_file_id", return_value=UNIFIED_BATCH_ID
    ):
        resp = await call_retrieve(retrieve_harness, "batch-unified-blob")

    # DISPATCH - router fired, direct litellm did not.
    assert retrieve_harness.router_aretrieve.call_count == 1
    retrieve_harness.litellm_aretrieve.assert_not_called()
    retrieve_harness.creds_resolver.assert_not_called()

    # router receives the (still-encoded) batch id verbatim - this layer does
    # not decode it for the unified path.
    assert retrieve_harness.router_kwargs() == {"batch_id": "batch-unified-blob"}

    # hidden params: unified id passed through, model_id parsed from it.
    assert resp._hidden_params["unified_batch_id"] == UNIFIED_BATCH_ID
    assert resp._hidden_params["model_id"] == "gpt-4o-mini"

    # raw provider file ids on the response are resolved back to unified ids.
    retrieve_harness.resolve_input.assert_called_once()
    retrieve_harness.resolve_output.assert_called_once()


@pytest.mark.asyncio
async def test_retrieve__loadbalancing_raw_id_routes_to_router(retrieve_harness):
    """Loadbalancing on + a plain (non-encoded, non-unified) batch id routes to
    the router. Locks the current dispatch contract of the shared elif."""
    with patch.object(litellm, "enable_loadbalancing_on_batch_endpoints", True):
        resp = await call_retrieve(retrieve_harness, "batch-raw-xyz")

    assert retrieve_harness.router_aretrieve.call_count == 1
    retrieve_harness.litellm_aretrieve.assert_not_called()
    assert retrieve_harness.router_kwargs() == {"batch_id": "batch-raw-xyz"}
    # not a unified id -> hidden param reflects that, no model_id stamped.
    assert resp._hidden_params["unified_batch_id"] is False
    assert "model_id" not in resp._hidden_params
    # not a unified id -> no file-id resolution.
    retrieve_harness.resolve_input.assert_not_called()
    retrieve_harness.resolve_output.assert_not_called()


# --------------------------------------------------------------------------- #
# SCENARIO 3 - fallback to custom_llm_provider (env-var creds). MUST NOT touch
# the credential resolver or the router.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_retrieve__fallback_default_openai(retrieve_harness):
    await call_retrieve(retrieve_harness, "batch-raw-xyz")

    assert retrieve_harness.litellm_aretrieve.call_count == 1
    retrieve_harness.router_aretrieve.assert_not_called()
    retrieve_harness.creds_resolver.assert_not_called()  # inverse-bug guard
    assert retrieve_harness.aretrieve_kwargs() == {
        "custom_llm_provider": "openai",
        "batch_id": "batch-raw-xyz",
    }
    assert retrieve_harness.update_batch_in_db.call_count == 1


@pytest.mark.asyncio
async def test_retrieve__fallback_provider_path_param(retrieve_harness):
    await call_retrieve(retrieve_harness, "batch-raw-xyz", provider="anthropic")

    retrieve_harness.creds_resolver.assert_not_called()
    assert retrieve_harness.aretrieve_kwargs()["custom_llm_provider"] == "anthropic"


@pytest.mark.asyncio
async def test_retrieve__fallback_provider_from_header(retrieve_harness):
    retrieve_harness.provider_from_headers.return_value = "bedrock"

    await call_retrieve(retrieve_harness, "batch-raw-xyz")

    assert retrieve_harness.aretrieve_kwargs()["custom_llm_provider"] == "bedrock"


@pytest.mark.asyncio
async def test_retrieve__fallback_provider_from_query(retrieve_harness):
    retrieve_harness.provider_from_query.return_value = "vertex_ai"

    await call_retrieve(retrieve_harness, "batch-raw-xyz")

    assert retrieve_harness.aretrieve_kwargs()["custom_llm_provider"] == "vertex_ai"


@pytest.mark.asyncio
async def test_retrieve__fallback_provider_precedence_path_over_header(
    retrieve_harness,
):
    """provider path param beats the header-derived provider."""
    retrieve_harness.provider_from_headers.return_value = "bedrock"

    await call_retrieve(retrieve_harness, "batch-raw-xyz", provider="anthropic")

    assert retrieve_harness.aretrieve_kwargs()["custom_llm_provider"] == "anthropic"


# --------------------------------------------------------------------------- #
# ManagedObjectTable short-circuit. A terminal-status DB row is returned
# immediately - no provider call, no write-back. A non-terminal row falls
# through to a provider sync.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", ["completed", "complete", "failed", "cancelled", "expired"]
)
async def test_retrieve__db_terminal_state_short_circuits(retrieve_harness, status):
    # "complete" is the DB-normalized alias of "completed"; it is not a valid
    # constructor literal but reaches the endpoint via a stored row, so set it
    # post-construction to exercise that exact branch.
    db_response = make_batch(id="batch-from-db", status="completed")
    db_response.status = status
    retrieve_harness.get_batch_from_db.return_value = (MagicMock(), db_response)

    resp = await call_retrieve(retrieve_harness, "batch-raw-xyz")

    # No provider seam fired, and no write-back (the row is already terminal).
    retrieve_harness.litellm_aretrieve.assert_not_called()
    retrieve_harness.router_aretrieve.assert_not_called()
    retrieve_harness.update_batch_in_db.assert_not_called()
    # The DB object is what the client gets back.
    assert resp is db_response


@pytest.mark.asyncio
async def test_retrieve__db_terminal_unified_resolves_file_ids(retrieve_harness):
    db_response = make_batch(id="batch-from-db", status="completed")
    retrieve_harness.get_batch_from_db.return_value = (MagicMock(), db_response)

    with patch.object(
        endpoints, "_is_base64_encoded_unified_file_id", return_value=UNIFIED_BATCH_ID
    ):
        await call_retrieve(retrieve_harness, "batch-unified-blob")

    # Terminal short-circuit still resolves raw provider file ids to unified.
    retrieve_harness.resolve_input.assert_called_once()
    retrieve_harness.resolve_output.assert_called_once()
    retrieve_harness.litellm_aretrieve.assert_not_called()
    retrieve_harness.router_aretrieve.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve__db_non_terminal_state_syncs_with_provider(retrieve_harness):
    """A non-terminal DB row must NOT short-circuit; the endpoint syncs with the
    provider to refresh state."""
    db_response = make_batch(id="batch-from-db", status="validating")
    retrieve_harness.get_batch_from_db.return_value = (MagicMock(), db_response)

    await call_retrieve(retrieve_harness, "batch-raw-xyz")

    # Provider sync happened despite the DB hit.
    assert retrieve_harness.litellm_aretrieve.call_count == 1
    assert retrieve_harness.update_batch_in_db.call_count == 1


# --------------------------------------------------------------------------- #
# Cross-cutting: enrichment route_type and failure-hook on provider error.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_retrieve__uses_aretrieve_batch_route_type(retrieve_harness):
    await call_retrieve(retrieve_harness, "batch-raw-xyz")

    assert (
        retrieve_harness.pre_call.call_args.kwargs["route_type"] == "aretrieve_batch"
    )


@pytest.mark.asyncio
async def test_retrieve__exception_calls_failure_hook(retrieve_harness):
    retrieve_harness.litellm_aretrieve.side_effect = ValueError("provider boom")

    with pytest.raises(Exception):
        await call_retrieve(retrieve_harness, "batch-raw-xyz")

    retrieve_harness.logging.post_call_failure_hook.assert_called_once()
    assert (
        retrieve_harness.logging.post_call_failure_hook.call_args.kwargs[
            "original_exception"
        ].args[0]
        == "provider boom"
    )


# =========================================================================== #
#                                                                             #
#   GET /v1/batches  -  list_batches routing-contract tests                    #
#                                                                             #
#   Branch order (first match wins):                                          #
#     1. managed_files hook present  -> managed_files_obj.list_user_batches    #
#     2. model from body/query/header -> litellm.alist_batches + id encode     #
#     3. target_model_names (param or body) -> llm_router.alist_batches        #
#     4. fallback -> litellm.alist_batches via env-var custom_llm_provider     #
#                                                                             #
#   llm_router is required; absence is a 500 before any branch runs.          #
# =========================================================================== #


class FakeListPage:
    """Stand-in for the SyncCursorPage[Batch] that alist_batches returns. The
    endpoint only touches `.data` (to encode ids) and `._hidden_params`."""

    def __init__(self, data: Any):
        self.data = data
        self._hidden_params: Dict[str, Any] = {}


@dataclass
class ListHarness:
    body: Dict[str, Any]
    read_body: AsyncMock
    pre_call: AsyncMock
    get_headers: MagicMock
    provider_from_headers: MagicMock
    provider_from_query: MagicMock
    litellm_alist: AsyncMock
    router: MagicMock
    logging: MagicMock
    creds_resolver: MagicMock

    @property
    def router_alist(self) -> AsyncMock:
        return self.router.alist_batches

    def set_managed_files(self, page: Any) -> AsyncMock:
        """Install a managed_files hook exposing list_user_batches -> page."""
        hook = MagicMock()
        hook.list_user_batches = AsyncMock(return_value=page)
        self.logging.get_proxy_hook = MagicMock(return_value=hook)
        return hook.list_user_batches

    def alist_kwargs(self) -> Dict[str, Any]:
        assert self.litellm_alist.call_count == 1
        return dict(self.litellm_alist.call_args.kwargs)

    def router_kwargs(self) -> Dict[str, Any]:
        assert self.router_alist.call_count == 1
        return dict(self.router_alist.call_args.kwargs)


@pytest.fixture
def list_harness():
    body_holder: Dict[str, Any] = {"body": {}}
    logging = MagicMock(spec=ProxyLogging)
    logging.post_call_success_hook = AsyncMock(side_effect=lambda **kw: kw["response"])
    logging.post_call_failure_hook = AsyncMock()
    logging.update_request_status = AsyncMock()
    # Default: no managed_files hook -> branches 2/3/4 are reachable.
    logging.get_proxy_hook = MagicMock(return_value=None)

    router = MagicMock(spec=Router)
    router.alist_batches = AsyncMock(return_value=FakeListPage([]))
    router.get_deployment_credentials_with_provider = MagicMock(
        side_effect=_creds_lookup
    )

    read_body = AsyncMock(side_effect=lambda request: body_holder["body"])
    pre_call = AsyncMock(side_effect=lambda **kw: (body_holder["body"], MagicMock()))
    get_headers = MagicMock(return_value={})
    provider_from_headers = MagicMock(return_value=None)
    provider_from_query = MagicMock(return_value=None)
    litellm_alist = AsyncMock(return_value=FakeListPage([]))

    with ExitStack() as stack:
        stack.enter_context(patch.object(endpoints, "_read_request_body", read_body))
        stack.enter_context(
            patch.object(
                ProxyBaseLLMRequestProcessing,
                "common_processing_pre_call_logic",
                pre_call,
            )
        )
        stack.enter_context(
            patch.object(
                ProxyBaseLLMRequestProcessing, "get_custom_headers", get_headers
            )
        )
        stack.enter_context(
            patch.object(
                endpoints,
                "get_custom_llm_provider_from_request_headers",
                provider_from_headers,
            )
        )
        stack.enter_context(
            patch.object(
                endpoints,
                "get_custom_llm_provider_from_request_query",
                provider_from_query,
            )
        )
        stack.enter_context(patch.object(litellm, "alist_batches", litellm_alist))
        stack.enter_context(patch.object(proxy_server, "llm_router", router))
        stack.enter_context(patch.object(proxy_server, "proxy_logging_obj", logging))
        stack.enter_context(patch.object(proxy_server, "general_settings", {}))
        stack.enter_context(patch.object(proxy_server, "proxy_config", MagicMock()))
        stack.enter_context(patch.object(proxy_server, "version", "test-version"))

        yield ListHarness(
            body=body_holder,
            read_body=read_body,
            pre_call=pre_call,
            get_headers=get_headers,
            provider_from_headers=provider_from_headers,
            provider_from_query=provider_from_query,
            litellm_alist=litellm_alist,
            router=router,
            logging=logging,
            creds_resolver=router.get_deployment_credentials_with_provider,
        )


async def call_list(
    harness: ListHarness,
    *,
    provider: Optional[str] = None,
    limit: Optional[int] = None,
    after: Optional[str] = None,
    target_model_names: Optional[str] = None,
    user: Optional[UserAPIKeyAuth] = None,
    headers: Optional[Dict[str, str]] = None,
    query: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
):
    harness.body["body"] = body if body is not None else {}
    return await endpoints.list_batches(
        request=FakeRequest(headers=headers, query=query),
        fastapi_response=Response(),
        provider=provider,
        limit=limit,
        after=after,
        user_api_key_dict=user or UserAPIKeyAuth(api_key="sk-test"),
        target_model_names=target_model_names,
    )


# --------------------------------------------------------------------------- #
# Branch 1 - ManagedObjectTable listing. This is the default production path
# (the managed_files hook is registered) and wins over every other branch.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_list__managed_files_path(list_harness):
    page = FakeListPage([make_batch(id="batch-1")])
    list_user_batches = list_harness.set_managed_files(page)

    user = UserAPIKeyAuth(api_key="sk-test")
    resp = await call_list(
        list_harness,
        user=user,
        limit=7,
        after="batch-cursor",
        provider="openai",
        target_model_names="m1,m2",
    )

    # DISPATCH - managed-files seam fired, neither provider seam did.
    list_user_batches.assert_called_once_with(
        user_api_key_dict=user,
        limit=7,
        after="batch-cursor",
        provider="openai",
        target_model_names="m1,m2",
        llm_router=list_harness.router,
    )
    list_harness.litellm_alist.assert_not_called()
    list_harness.router_alist.assert_not_called()
    assert resp is page


@pytest.mark.asyncio
async def test_list__managed_files_beats_model_param(list_harness):
    """Branch 1 is checked before the model branch: a model in the body does not
    divert away from managed-files listing."""
    page = FakeListPage([])
    list_user_batches = list_harness.set_managed_files(page)

    await call_list(list_harness, body={"model": "azure/gpt-4o"})

    list_user_batches.assert_called_once()
    list_harness.litellm_alist.assert_not_called()
    list_harness.router_alist.assert_not_called()
    list_harness.creds_resolver.assert_not_called()


# --------------------------------------------------------------------------- #
# Branch 2 - model from body/query/header. CURRENTLY BROKEN: the endpoint
# forwards custom_llm_provider both explicitly and via **data (it calls
# data.update(credentials) but never pops custom_llm_provider the way
# create/retrieve do through prepare_data_with_credentials), so every call
# raises "multiple values for keyword argument 'custom_llm_provider'".
#
# The strict xfail below encodes the INTENDED contract (litellm seam fires,
# creds resolved for the body model, response ids encoded). It xfails today on
# the duplicate-kwarg TypeError; the day that branch is fixed it will XPASS and
# strict-mode turns the green into a failure, forcing whoever fixes it to drop
# the marker and adopt this as a live regression test.
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(
    strict=True,
    raises=ProxyException,
    reason="list_batches model branch passes custom_llm_provider twice "
    "(explicit kwarg + **data after data.update(credentials)); remove when fixed",
)
@pytest.mark.asyncio
async def test_list__model_from_body_routes_and_encodes(list_harness):
    list_harness.litellm_alist.return_value = FakeListPage(
        [make_batch(id="batch-1"), make_batch(id="batch-2")]
    )

    resp = await call_list(list_harness, body={"model": "azure/gpt-4o"})

    assert list_harness.litellm_alist.call_count == 1
    list_harness.router_alist.assert_not_called()
    list_harness.creds_resolver.assert_called_once_with(model_id="azure/gpt-4o")
    assert resp.data[0].id == encode_file_id_with_model(
        "batch-1", "azure/gpt-4o", id_type="batch"
    )
    assert resp.data[1].id == encode_file_id_with_model(
        "batch-2", "azure/gpt-4o", id_type="batch"
    )


# --------------------------------------------------------------------------- #
# Branch 3 - target_model_names (function param or body) -> llm_router. Routes
# to the FIRST model in the comma list; `model` is stripped from data first.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_list__target_model_names_param_routes_to_router(list_harness):
    await call_list(list_harness, target_model_names="m1,m2", limit=3, after="cur")

    assert list_harness.router_alist.call_count == 1
    list_harness.litellm_alist.assert_not_called()
    list_harness.creds_resolver.assert_not_called()
    # first model only; after/limit forwarded; nothing else (param not in data).
    assert list_harness.router_kwargs() == {
        "model": "m1",
        "after": "cur",
        "limit": 3,
    }


@pytest.mark.asyncio
async def test_list__target_model_names_from_body(list_harness):
    await call_list(list_harness, body={"target_model_names": "m1,m2"})

    assert list_harness.router_alist.call_count == 1
    list_harness.litellm_alist.assert_not_called()
    kwargs = list_harness.router_kwargs()
    assert kwargs["model"] == "m1"
    # body-sourced target_model_names stays in the forwarded data.
    assert kwargs["target_model_names"] == "m1,m2"


@pytest.mark.asyncio
async def test_list__target_model_names_takes_first_only(list_harness):
    """Locks the current behavior: with multiple target models, only the first
    is routed to (silently, unlike create which 400s on >1). A change here -
    intentional or not - must update this test."""
    await call_list(list_harness, target_model_names="alpha,beta,gamma")

    assert list_harness.router_kwargs()["model"] == "alpha"


# --------------------------------------------------------------------------- #
# Branch 4 - fallback to custom_llm_provider (env-var creds). MUST NOT touch
# the credential resolver or the router.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_list__fallback_default_openai(list_harness):
    await call_list(list_harness)

    assert list_harness.litellm_alist.call_count == 1
    list_harness.router_alist.assert_not_called()
    list_harness.creds_resolver.assert_not_called()  # inverse-bug guard
    assert list_harness.alist_kwargs() == {
        "custom_llm_provider": "openai",
        "after": None,
        "limit": None,
    }


@pytest.mark.asyncio
async def test_list__fallback_provider_path_param(list_harness):
    await call_list(list_harness, provider="anthropic")

    list_harness.creds_resolver.assert_not_called()
    assert list_harness.alist_kwargs()["custom_llm_provider"] == "anthropic"


@pytest.mark.asyncio
async def test_list__fallback_provider_from_header(list_harness):
    list_harness.provider_from_headers.return_value = "bedrock"

    await call_list(list_harness)

    assert list_harness.alist_kwargs()["custom_llm_provider"] == "bedrock"


@pytest.mark.asyncio
async def test_list__fallback_provider_from_query(list_harness):
    list_harness.provider_from_query.return_value = "vertex_ai"

    await call_list(list_harness)

    assert list_harness.alist_kwargs()["custom_llm_provider"] == "vertex_ai"


@pytest.mark.asyncio
async def test_list__fallback_after_and_limit_forwarded(list_harness):
    await call_list(list_harness, after="cursor-9", limit=42)

    kwargs = list_harness.alist_kwargs()
    assert kwargs["after"] == "cursor-9"
    assert kwargs["limit"] == 42


# --------------------------------------------------------------------------- #
# Cross-cutting: router requirement, route_type, failure hook.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_list__no_router_raises_500(list_harness):
    with patch.object(proxy_server, "llm_router", None):
        with pytest.raises(ProxyException) as exc:
            await call_list(list_harness)

    assert exc.value.code == "500"
    list_harness.litellm_alist.assert_not_called()


@pytest.mark.asyncio
async def test_list__uses_alist_batches_route_type(list_harness):
    await call_list(list_harness)

    assert list_harness.pre_call.call_args.kwargs["route_type"] == "alist_batches"


@pytest.mark.asyncio
async def test_list__exception_calls_failure_hook(list_harness):
    list_harness.litellm_alist.side_effect = ValueError("provider boom")

    with pytest.raises(Exception):
        await call_list(list_harness)

    list_harness.logging.post_call_failure_hook.assert_called_once()
    assert (
        list_harness.logging.post_call_failure_hook.call_args.kwargs[
            "original_exception"
        ].args[0]
        == "provider boom"
    )


# =========================================================================== #
#                                                                             #
#   POST /v1/batches/{batch_id}/cancel  -  cancel_batch routing-contract tests #
#                                                                             #
#   Three branches, first match wins:                                         #
#     1. model-encoded batch id -> litellm.acancel_batch via model creds       #
#     2. unified batch id       -> llm_router.acancel_batch (model+batch_id    #
#                                  parsed out of the unified id)               #
#     3. fallback               -> litellm.acancel_batch via env-var provider  #
#   Every branch then writes state back via update_batch_in_database(          #
#   operation="cancel"). There is NO ManagedObjectTable read short-circuit     #
#   here (unlike retrieve).                                                    #
#                                                                             #
#   These tests pin CURRENT behavior so a refactor can't silently change it.  #
#   Two current-behavior quirks are locked deliberately and noted inline:     #
#     - SCENARIO 1 forwards the DEPLOYMENT model from creds, not the decoded   #
#       model (retrieve overrides it; cancel does not).                        #
#     - SCENARIO 3 rebuilds a CancelBatchRequest and forwards only            #
#       {custom_llm_provider, batch_id}, dropping enrichment keys.            #
# =========================================================================== #


@dataclass
class CancelHarness:
    data: Dict[str, Any]
    pre_call: AsyncMock
    add_data: AsyncMock
    get_headers: MagicMock
    provider_from_headers: MagicMock
    provider_from_query: MagicMock
    litellm_acancel: AsyncMock
    router: MagicMock
    logging: MagicMock
    creds_resolver: MagicMock
    update_batch_in_db: AsyncMock

    @property
    def router_acancel(self) -> AsyncMock:
        return self.router.acancel_batch

    def acancel_kwargs(self) -> Dict[str, Any]:
        assert self.litellm_acancel.call_count == 1
        return dict(self.litellm_acancel.call_args.kwargs)

    def router_kwargs(self) -> Dict[str, Any]:
        assert self.router_acancel.call_count == 1
        return dict(self.router_acancel.call_args.kwargs)


@pytest.fixture
def cancel_harness():
    data_holder: Dict[str, Any] = {"data": {}}
    logging = MagicMock(spec=ProxyLogging)
    logging.post_call_success_hook = AsyncMock(side_effect=lambda **kw: kw["response"])
    logging.post_call_failure_hook = AsyncMock()
    logging.update_request_status = AsyncMock()
    logging.get_proxy_hook = MagicMock(return_value=None)

    router = MagicMock(spec=Router)
    router.acancel_batch = AsyncMock(return_value=make_batch())
    router.get_deployment_credentials_with_provider = MagicMock(
        side_effect=_creds_lookup
    )

    pre_call = AsyncMock(side_effect=lambda **kw: (data_holder["data"], MagicMock()))
    # add_litellm_data_to_request is a passthrough that returns the data it got.
    add_data = AsyncMock(side_effect=lambda **kw: kw["data"])
    get_headers = MagicMock(return_value={})
    provider_from_headers = MagicMock(return_value=None)
    provider_from_query = MagicMock(return_value=None)
    litellm_acancel = AsyncMock(return_value=make_batch())
    update_batch_in_db = AsyncMock(return_value=None)

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                ProxyBaseLLMRequestProcessing,
                "common_processing_pre_call_logic",
                pre_call,
            )
        )
        stack.enter_context(
            patch.object(
                ProxyBaseLLMRequestProcessing, "get_custom_headers", get_headers
            )
        )
        stack.enter_context(
            patch.object(
                endpoints,
                "get_custom_llm_provider_from_request_headers",
                provider_from_headers,
            )
        )
        stack.enter_context(
            patch.object(
                endpoints,
                "get_custom_llm_provider_from_request_query",
                provider_from_query,
            )
        )
        stack.enter_context(
            patch.object(endpoints, "update_batch_in_database", update_batch_in_db)
        )
        stack.enter_context(patch.object(litellm, "acancel_batch", litellm_acancel))
        stack.enter_context(
            patch.object(litellm, "enable_loadbalancing_on_batch_endpoints", False)
        )
        stack.enter_context(patch.object(proxy_server, "llm_router", router))
        stack.enter_context(patch.object(proxy_server, "proxy_logging_obj", logging))
        stack.enter_context(patch.object(proxy_server, "general_settings", {}))
        stack.enter_context(patch.object(proxy_server, "proxy_config", MagicMock()))
        stack.enter_context(patch.object(proxy_server, "version", "test-version"))
        stack.enter_context(patch.object(proxy_server, "prisma_client", MagicMock()))
        stack.enter_context(
            patch.object(proxy_server, "add_litellm_data_to_request", add_data)
        )

        yield CancelHarness(
            data=data_holder,
            pre_call=pre_call,
            add_data=add_data,
            get_headers=get_headers,
            provider_from_headers=provider_from_headers,
            provider_from_query=provider_from_query,
            litellm_acancel=litellm_acancel,
            router=router,
            logging=logging,
            creds_resolver=router.get_deployment_credentials_with_provider,
            update_batch_in_db=update_batch_in_db,
        )


async def call_cancel(
    harness: CancelHarness,
    batch_id: str,
    *,
    provider: Optional[str] = None,
    user: Optional[UserAPIKeyAuth] = None,
    headers: Optional[Dict[str, str]] = None,
    query: Optional[Dict[str, str]] = None,
    data_extra: Optional[Dict[str, Any]] = None,
):
    harness.data["data"] = {"batch_id": batch_id, **(data_extra or {})}
    return await endpoints.cancel_batch(
        request=FakeRequest(headers=headers, query=query),
        batch_id=batch_id,
        fastapi_response=Response(),
        provider=provider,
        user_api_key_dict=user or UserAPIKeyAuth(api_key="sk-test"),
    )


# --------------------------------------------------------------------------- #
# SCENARIO 1 - model-encoded batch id -> litellm.acancel_batch via model creds.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cancel__model_encoded_id(cancel_harness):
    resp = await call_cancel(cancel_harness, AZURE_BATCH_ID)

    # DISPATCH - model-credential path via litellm; router untouched.
    assert cancel_harness.litellm_acancel.call_count == 1
    cancel_harness.router_acancel.assert_not_called()

    # CREDENTIALS - resolved for the model decoded from the batch id.
    cancel_harness.creds_resolver.assert_called_once_with(model_id="azure/gpt-4o")

    # SEAM PAYLOAD - exact dict. NOTE current behavior: `model` is the
    # DEPLOYMENT name from creds, NOT the decoded model (cancel, unlike
    # retrieve, does not override it). Locking this guards the difference.
    assert cancel_harness.acancel_kwargs() == {
        "custom_llm_provider": "azure",
        "batch_id": "batch_orig123",  # decoded/stripped original id
        "api_key": "sk-azure",
        "api_base": "https://azure.test",
        "model": "azure/gpt-4o-deployment",
    }

    # OUTPUT SHAPE - response id re-encoded with the DECODED model.
    assert resp.id == encode_file_id_with_model(
        "batch-provider-id", "azure/gpt-4o", id_type="batch"
    )

    # write-back tagged as a cancel.
    assert cancel_harness.update_batch_in_db.call_count == 1
    assert cancel_harness.update_batch_in_db.call_args.kwargs["operation"] == "cancel"


@pytest.mark.asyncio
async def test_cancel__model_encoded_id_forwards_deployment_model(cancel_harness):
    """Pin the current contract: cancel forwards the creds' deployment model.
    If someone adds a decoded-model override (as retrieve has), this flips and
    must be reviewed."""
    await call_cancel(cancel_harness, AZURE_BATCH_ID)

    assert cancel_harness.acancel_kwargs()["model"] == "azure/gpt-4o-deployment"


@pytest.mark.asyncio
async def test_cancel__model_encoded_beats_unified(cancel_harness):
    with patch.object(
        endpoints, "_is_base64_encoded_unified_file_id", return_value=UNIFIED_BATCH_ID
    ):
        await call_cancel(cancel_harness, AZURE_BATCH_ID)

    assert cancel_harness.litellm_acancel.call_count == 1
    cancel_harness.router_acancel.assert_not_called()
    cancel_harness.creds_resolver.assert_called_once_with(model_id="azure/gpt-4o")


# --------------------------------------------------------------------------- #
# SCENARIO 2 - unified batch id -> llm_router.acancel_batch. model and batch_id
# are parsed out of the unified id; hidden params stamped.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cancel__unified_batch_id_routes_to_router(cancel_harness):
    with patch.object(
        endpoints, "_is_base64_encoded_unified_file_id", return_value=UNIFIED_BATCH_ID
    ):
        resp = await call_cancel(cancel_harness, "batch-unified-blob")

    # DISPATCH - router fired, litellm did not, no creds lookup.
    assert cancel_harness.router_acancel.call_count == 1
    cancel_harness.litellm_acancel.assert_not_called()
    cancel_harness.creds_resolver.assert_not_called()

    # model + batch_id are extracted from the unified id and forwarded.
    assert cancel_harness.router_kwargs() == {
        "batch_id": "batch-raw-xyz",
        "model": "gpt-4o-mini",
    }

    # hidden params: unified id passed through, model_id stamped from data.
    assert resp._hidden_params["unified_batch_id"] == UNIFIED_BATCH_ID
    assert resp._hidden_params["model_id"] == "gpt-4o-mini"

    assert cancel_harness.update_batch_in_db.call_args.kwargs["operation"] == "cancel"


@pytest.mark.asyncio
async def test_cancel__unified_missing_model_id_400(cancel_harness):
    # unified id with no model_id segment -> get_model_id returns None -> 400.
    with patch.object(
        endpoints,
        "_is_base64_encoded_unified_file_id",
        return_value="litellm_proxy;llm_batch_id:batch-xyz",
    ):
        with pytest.raises(ProxyException) as exc:
            await call_cancel(cancel_harness, "batch-unified-blob")

    assert exc.value.code == "400"
    cancel_harness.router_acancel.assert_not_called()
    cancel_harness.litellm_acancel.assert_not_called()


@pytest.mark.asyncio
async def test_cancel__unified_no_router_500(cancel_harness):
    with patch.object(proxy_server, "llm_router", None), patch.object(
        endpoints, "_is_base64_encoded_unified_file_id", return_value=UNIFIED_BATCH_ID
    ):
        with pytest.raises(ProxyException) as exc:
            await call_cancel(cancel_harness, "batch-unified-blob")

    assert exc.value.code == "500"


# --------------------------------------------------------------------------- #
# SCENARIO 3 - fallback to custom_llm_provider. Rebuilds a CancelBatchRequest
# and forwards only {custom_llm_provider, batch_id}.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cancel__fallback_default_openai(cancel_harness):
    await call_cancel(cancel_harness, "batch-raw-xyz")

    assert cancel_harness.litellm_acancel.call_count == 1
    cancel_harness.router_acancel.assert_not_called()
    cancel_harness.creds_resolver.assert_not_called()  # inverse-bug guard
    # current behavior: enrichment keys dropped; only these two forwarded.
    assert cancel_harness.acancel_kwargs() == {
        "custom_llm_provider": "openai",
        "batch_id": "batch-raw-xyz",
    }
    assert cancel_harness.update_batch_in_db.call_count == 1


@pytest.mark.asyncio
async def test_cancel__fallback_provider_path_param(cancel_harness):
    await call_cancel(cancel_harness, "batch-raw-xyz", provider="anthropic")

    cancel_harness.creds_resolver.assert_not_called()
    assert cancel_harness.acancel_kwargs()["custom_llm_provider"] == "anthropic"


@pytest.mark.asyncio
async def test_cancel__fallback_provider_from_data_body(cancel_harness):
    await call_cancel(
        cancel_harness, "batch-raw-xyz", data_extra={"custom_llm_provider": "bedrock"}
    )

    assert cancel_harness.acancel_kwargs()["custom_llm_provider"] == "bedrock"


@pytest.mark.asyncio
async def test_cancel__fallback_provider_from_header(cancel_harness):
    cancel_harness.provider_from_headers.return_value = "vertex_ai"

    await call_cancel(cancel_harness, "batch-raw-xyz")

    assert cancel_harness.acancel_kwargs()["custom_llm_provider"] == "vertex_ai"


@pytest.mark.asyncio
async def test_cancel__fallback_provider_from_query(cancel_harness):
    cancel_harness.provider_from_query.return_value = "azure"

    await call_cancel(cancel_harness, "batch-raw-xyz")

    assert cancel_harness.acancel_kwargs()["custom_llm_provider"] == "azure"


@pytest.mark.xfail(
    strict=True,
    raises=ProxyException,
    reason="cancel SCENARIO 3: `provider or data.pop('custom_llm_provider')` "
    "short-circuits when provider (path param) is set, so a body "
    "custom_llm_provider is left in data and forwarded twice -> duplicate-kwarg "
    "TypeError. Intended: path param wins cleanly. Remove marker when fixed.",
)
@pytest.mark.asyncio
async def test_cancel__fallback_provider_precedence_path_over_body(cancel_harness):
    """Intended contract: provider path param beats a body custom_llm_provider.
    CURRENTLY raises because the `or` short-circuit skips the data.pop, leaving
    the body value to collide with the explicit kwarg."""
    await call_cancel(
        cancel_harness,
        "batch-raw-xyz",
        provider="anthropic",
        data_extra={"custom_llm_provider": "bedrock"},
    )

    assert cancel_harness.acancel_kwargs()["custom_llm_provider"] == "anthropic"


# --------------------------------------------------------------------------- #
# Cross-cutting: enrichment route_type and failure-hook on provider error.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cancel__uses_acancel_batch_route_type(cancel_harness):
    await call_cancel(cancel_harness, "batch-raw-xyz")

    assert cancel_harness.pre_call.call_args.kwargs["route_type"] == "acancel_batch"


@pytest.mark.asyncio
async def test_cancel__exception_calls_failure_hook(cancel_harness):
    cancel_harness.litellm_acancel.side_effect = ValueError("provider boom")

    with pytest.raises(Exception):
        await call_cancel(cancel_harness, "batch-raw-xyz")

    cancel_harness.logging.post_call_failure_hook.assert_called_once()
    assert (
        cancel_harness.logging.post_call_failure_hook.call_args.kwargs[
            "original_exception"
        ].args[0]
        == "provider boom"
    )


# =========================================================================== #
# Router-required 500 guards (one per endpoint branch that calls the router).
# These pin the defensive checks that fire when llm_router is unset.
# =========================================================================== #


@pytest.mark.asyncio
async def test_create__loadbalancing_no_router_500(harness):
    set_body(
        harness,
        {
            "input_file_id": "file-plain",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "model": "lb-model",
        },
    )
    harness.is_known_model.return_value = True
    with patch.object(
        litellm, "enable_loadbalancing_on_batch_endpoints", True
    ), patch.object(proxy_server, "llm_router", None):
        with pytest.raises(ProxyException) as exc:
            await call_create(harness)

    assert exc.value.code == "500"
    harness.router_acreate.assert_not_called()
    harness.litellm_acreate.assert_not_called()


@pytest.mark.asyncio
async def test_create__unified_no_router_500(harness):
    set_body(
        harness,
        {
            "input_file_id": "litellm_proxy_unified_id",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    with patch.object(
        endpoints, "_is_base64_encoded_unified_file_id", return_value="unified-xyz"
    ), patch.object(
        endpoints, "get_models_from_unified_file_id", return_value=["gpt-4o-mini"]
    ), patch.object(
        proxy_server, "llm_router", None
    ):
        with pytest.raises(ProxyException) as exc:
            await call_create(harness)

    assert exc.value.code == "500"


@pytest.mark.asyncio
async def test_retrieve__unified_no_router_500(retrieve_harness):
    with patch.object(
        endpoints, "_is_base64_encoded_unified_file_id", return_value=UNIFIED_BATCH_ID
    ), patch.object(proxy_server, "llm_router", None):
        with pytest.raises(ProxyException) as exc:
            await call_retrieve(retrieve_harness, "batch-unified-blob")

    assert exc.value.code == "500"
    retrieve_harness.router_aretrieve.assert_not_called()
    retrieve_harness.litellm_aretrieve.assert_not_called()
