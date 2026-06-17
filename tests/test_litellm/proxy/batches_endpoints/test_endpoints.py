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
