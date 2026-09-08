"""Live e2e: the lifecycle of a DB-stored deployment through the model management
routes, read back on every gateway replica.

Each test registers its own gpt-4o-mini mock deployment through /model/new (deleted
on teardown) with non-default pricing, context window, mode, and api_base pinned, then
walks the lifecycle up to the step it proves: the create reads back field for field,
a partial PATCH changes only the key it names, an explicit null on PATCH removes the
key from the stored row (JSON Merge Patch), a call after the price clear is billed at
the cost map's rate rather than the cleared override, and a delete removes the
deployment from /model/info and makes the model name unknown to /chat/completions.

The stored row is read back from /model/info, a control-plane route with one answer
behind it. What every gateway must agree on is which models it serves, so the create
and delete steps poll /v1/models on every URL in PROXY_REPLICA_URLS through
ProxyClient.read_model_back_everywhere, failing by name on the gateway that never converged.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from management_client import ManagementClient
from models import (
    ChatBody,
    ChatMessage,
    Clear,
    LiteLLMParamsBody,
    LiteLLMParamsPatch,
    ModelInfoBody,
    ModelInfoEntry,
    ModelInfoResponse,
    ModelNewBody,
    ModelPatchBody,
    ModelsListResponse,
    SpendLogRow,
)

pytestmark = pytest.mark.e2e

BACKEND_MODEL: Final = "gpt-4o-mini"
PINNED_API_BASE: Final = "https://pinned.example.invalid/v1"
PINNED_MAX_INPUT_TOKENS: Final = 4096
PINNED_INPUT_RATE: Final = 1e-05
UPDATED_INPUT_RATE: Final = 2e-05
PINNED_OUTPUT_RATE: Final = 3e-05

# A PATCH lands on the control plane, and each gateway picks it up on its own config
# reload, so the first call after the write can still be billed at the old rate. There
# is no price on the gateway's data-plane surface to poll, so the billing steps drive
# calls until the new rate shows up in the spend row and let the deadline be what fails.
BILLING_CONVERGENCE_TIMEOUT: Final = 90.0
BILLING_CONVERGENCE_INTERVAL: Final = 5.0


@dataclass(frozen=True, slots=True)
class Registered:
    model_name: str
    model_id: str


class _ErrorDetail(BaseModel):
    message: str


class _ErrorEnvelope(BaseModel):
    error: _ErrorDetail


def _register(client: ManagementClient, resources: ResourceManager) -> Registered:
    """Register a mock gpt-4o-mini deployment with every field under test pinned to a
    non-default value, deleted on teardown. max_input_tokens is pinned in
    litellm_params only: a value in model_info is copied into the shared cost-map
    entry for the backend model, which would leak into every other gpt-4o-mini
    deployment on the proxy."""
    model_name: Final = f"e2e-lifecycle-{unique_marker()}"
    model_id: Final = client.proxy.register_model(
        ModelNewBody(
            model_name=model_name,
            litellm_params=LiteLLMParamsBody(
                model=BACKEND_MODEL,
                mock_response="ok",
                api_base=PINNED_API_BASE,
                input_cost_per_token=PINNED_INPUT_RATE,
                output_cost_per_token=PINNED_OUTPUT_RATE,
                max_input_tokens=PINNED_MAX_INPUT_TOKENS,
            ),
            model_info=ModelInfoBody(mode="chat"),
        )
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return Registered(model_name=model_name, model_id=model_id)


def _entry(body: ModelInfoResponse, model_name: str) -> ModelInfoEntry | None:
    return next((entry for entry in body.data if entry.model_name == model_name), None)


def _stored_entry(
    client: ManagementClient,
    model_name: str,
    *,
    converged: Callable[[ModelInfoEntry], bool],
) -> ModelInfoEntry:
    """The stored /model/info row for `model_name`, once it satisfies `converged`.

    /model/info is a control-plane route: the gateways named in PROXY_REPLICA_URLS
    serve the LLM surface only, so the stored row has one answer, not one per
    gateway. What every gateway must agree on is which models it serves, and
    `_assert_served_everywhere` / `_assert_absent_everywhere` poll /v1/models for
    that."""

    def has_converged(body: ModelInfoResponse) -> bool:
        entry: Final = _entry(body, model_name)
        return entry is not None and converged(entry)

    body: Final = client.proxy.read_model_back("/model/info", ModelInfoResponse, predicate=has_converged)
    entry: Final = _entry(body, model_name)
    assert entry is not None, f"/model/info stopped listing {model_name!r} between the poll and the read"
    return entry


def _serves(body: ModelsListResponse, model_name: str) -> bool:
    return any(entry.id == model_name for entry in body.data)


def _assert_served_everywhere(client: ManagementClient, model_name: str) -> None:
    _ = client.proxy.read_model_back_everywhere(
        "/v1/models", ModelsListResponse, predicate=lambda body: _serves(body, model_name)
    )


def _assert_absent_everywhere(client: ManagementClient, model_name: str) -> None:
    _ = client.proxy.read_model_back_everywhere(
        "/v1/models", ModelsListResponse, predicate=lambda body: not _serves(body, model_name)
    )


def _assert_untouched_keys_as_created(entry: ModelInfoEntry, replica: str) -> None:
    """The keys no later step names read back byte-for-byte as /model/new wrote them."""
    params: Final = entry.litellm_params
    assert params.model == BACKEND_MODEL, f"{replica}: litellm_params.model {params.model!r} != {BACKEND_MODEL!r}"
    assert params.api_base == PINNED_API_BASE, f"{replica}: api_base {params.api_base!r} != {PINNED_API_BASE!r}"
    assert params.output_cost_per_token == PINNED_OUTPUT_RATE, (
        f"{replica}: output_cost_per_token {params.output_cost_per_token} != {PINNED_OUTPUT_RATE}"
    )
    assert entry.model_info.mode == "chat", f"{replica}: model_info.mode {entry.model_info.mode!r} != 'chat'"


def _approx_equal(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-9)


def _priced(rows: list[SpendLogRow]) -> bool:
    return any(row.metadata and row.metadata.cost_breakdown and row.metadata.cost_breakdown.input_cost for row in rows)


def _billed_input_cost(client: ManagementClient, model_name: str, key: str) -> tuple[int, float]:
    """Drive one chat completion through `model_name` and return the prompt tokens and
    input cost its spend row recorded, so a test can assert the rate the gateway actually
    billed rather than only the rate it stored."""
    chat: Final = unwrap(
        client.proxy.chat(
            key,
            ChatBody(
                model=model_name,
                messages=[ChatMessage(role="user", content=f"reply with one word {unique_marker()}")],
                max_tokens=16,
            ),
        )
    )
    assert chat.id is not None, f"chat completion carried no id to find its spend row by: {chat}"

    rows: Final = client.proxy.poll_logs_for_request_id(chat.id, predicate=_priced)
    row: Final = next((row for row in rows if row.request_id == chat.id), None)
    assert row is not None and row.metadata and row.metadata.cost_breakdown, (
        f"no priced spend row for request {chat.id} before the deadline: {rows}"
    )
    prompt_tokens: Final = row.prompt_tokens or 0
    input_cost: Final = row.metadata.cost_breakdown.input_cost
    assert prompt_tokens > 0 and input_cost is not None, f"spend row logged no prompt tokens or input cost: {row}"
    return prompt_tokens, input_cost


def _await_billed_input_cost(
    client: ManagementClient, model_name: str, key: str, *, expected_rate: float
) -> tuple[int, float]:
    """Drive calls through `model_name` until one is billed at `expected_rate`, and
    return the prompt tokens and input cost of the last spend row either way.

    Only the deadline ends the wait unsatisfied: a rate that never reaches the gateway
    comes back as the stale cost for the caller to assert on, so the rate the caller
    expects is still what decides the test."""
    deadline: Final = time.monotonic() + BILLING_CONVERGENCE_TIMEOUT
    while True:
        prompt_tokens, input_cost = _billed_input_cost(client, model_name, key)
        if _approx_equal(input_cost, prompt_tokens * expected_rate) or time.monotonic() >= deadline:
            return prompt_tokens, input_cost
        time.sleep(BILLING_CONVERGENCE_INTERVAL)


class TestModelLifecycle:
    @pytest.mark.covers("mgmt.model.add.persists")
    def test_create_reads_back_every_field_and_serves_on_every_replica(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        registered = _register(client, resources)

        entry = _stored_entry(client, registered.model_name, converged=lambda _entry: True)
        stored = "/model/info"

        _assert_untouched_keys_as_created(entry, stored)
        assert entry.litellm_params.input_cost_per_token == PINNED_INPUT_RATE, (
            f"{stored}: input_cost_per_token {entry.litellm_params.input_cost_per_token} != {PINNED_INPUT_RATE}"
        )
        assert entry.litellm_params.max_input_tokens == PINNED_MAX_INPUT_TOKENS, (
            f"{stored}: max_input_tokens {entry.litellm_params.max_input_tokens} != {PINNED_MAX_INPUT_TOKENS}"
        )
        assert entry.model_info.id == registered.model_id, (
            f"{stored}: model_info.id {entry.model_info.id!r} != {registered.model_id!r}"
        )

        _assert_served_everywhere(client, registered.model_name)

    @pytest.mark.covers("mgmt.model.update.preserves_unrelated_fields")
    def test_partial_update_changes_only_the_named_key(
        self, client: ManagementClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        registered = _register(client, resources)

        stored = client.proxy.patch_model(
            registered.model_id,
            ModelPatchBody(litellm_params=LiteLLMParamsPatch(input_cost_per_token=UPDATED_INPUT_RATE)),
        )
        assert stored.litellm_params.input_cost_per_token == UPDATED_INPUT_RATE, (
            f"PATCH response stores input_cost_per_token {stored.litellm_params.input_cost_per_token}, "
            f"sent {UPDATED_INPUT_RATE}"
        )

        entry = _stored_entry(
            client,
            registered.model_name,
            converged=lambda entry: entry.litellm_params.input_cost_per_token == UPDATED_INPUT_RATE,
        )
        stored = "/model/info"

        _assert_untouched_keys_as_created(entry, stored)
        assert entry.litellm_params.max_input_tokens == PINNED_MAX_INPUT_TOKENS, (
            f"{stored}: max_input_tokens {entry.litellm_params.max_input_tokens} != {PINNED_MAX_INPUT_TOKENS}"
        )
        assert entry.model_info.input_cost_per_token == UPDATED_INPUT_RATE, (
            f"{stored}: model_info.input_cost_per_token {entry.model_info.input_cost_per_token} "
            f"did not mirror the updated {UPDATED_INPUT_RATE}"
        )

        prompt_tokens, input_cost = _await_billed_input_cost(
            client, registered.model_name, scoped_key, expected_rate=UPDATED_INPUT_RATE
        )
        assert _approx_equal(input_cost, prompt_tokens * UPDATED_INPUT_RATE), (
            f"input_cost {input_cost} != {prompt_tokens} tokens * updated rate {UPDATED_INPUT_RATE} "
            f"= {prompt_tokens * UPDATED_INPUT_RATE}; the partial update did not reach billing"
        )

    @pytest.mark.covers("mgmt.model.update.clear_persists")
    def test_explicit_null_removes_the_key_from_the_stored_row(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        registered = _register(client, resources)

        stored = client.proxy.patch_model(
            registered.model_id,
            ModelPatchBody(litellm_params=LiteLLMParamsPatch(max_input_tokens=Clear(), input_cost_per_token=Clear())),
        )
        stored_params = stored.litellm_params.model_fields_set
        assert "max_input_tokens" not in stored_params, (
            f"stored litellm_params still carries max_input_tokens "
            f"{stored.litellm_params.max_input_tokens} after an explicit null"
        )
        assert "input_cost_per_token" not in stored_params, (
            f"stored litellm_params still carries input_cost_per_token "
            f"{stored.litellm_params.input_cost_per_token} after an explicit null"
        )
        assert "max_input_tokens" not in stored.model_info.model_fields_set, (
            f"stored model_info carries max_input_tokens {stored.model_info.max_input_tokens} after the clear"
        )
        assert "input_cost_per_token" not in stored.model_info.model_fields_set, (
            f"stored model_info still mirrors input_cost_per_token {stored.model_info.input_cost_per_token}"
        )

        cost_map_input_rate = client.proxy.model_cost_map()[BACKEND_MODEL].input_cost_per_token
        assert cost_map_input_rate is not None, f"cost map has no input rate for {BACKEND_MODEL}"
        entry = _stored_entry(
            client,
            registered.model_name,
            converged=lambda entry: "max_input_tokens" not in entry.litellm_params.model_fields_set,
        )
        stored = "/model/info"

        _assert_untouched_keys_as_created(entry, stored)
        served = entry.litellm_params.model_fields_set
        assert "max_input_tokens" not in served, (
            f"{stored}: litellm_params still serves max_input_tokens {entry.litellm_params.max_input_tokens}"
        )
        assert "input_cost_per_token" not in served, (
            f"{stored}: litellm_params still serves input_cost_per_token {entry.litellm_params.input_cost_per_token}"
        )
        assert entry.model_info.input_cost_per_token == cost_map_input_rate, (
            f"{stored}: model_info.input_cost_per_token {entry.model_info.input_cost_per_token} is not the "
            f"cost map's {cost_map_input_rate}; the cleared override {PINNED_INPUT_RATE} still resolves"
        )

    @pytest.mark.covers("mgmt.model.update.clear_persists")
    def test_cleared_price_is_billed_at_the_cost_map_rate(
        self, client: ManagementClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        registered = _register(client, resources)
        _ = client.proxy.patch_model(
            registered.model_id,
            ModelPatchBody(litellm_params=LiteLLMParamsPatch(max_input_tokens=Clear(), input_cost_per_token=Clear())),
        )
        _ = _stored_entry(
            client,
            registered.model_name,
            converged=lambda entry: "input_cost_per_token" not in entry.litellm_params.model_fields_set,
        )
        cost_map_input_rate = client.proxy.model_cost_map()[BACKEND_MODEL].input_cost_per_token
        assert cost_map_input_rate is not None, f"cost map has no input rate for {BACKEND_MODEL}"

        prompt_tokens, input_cost = _await_billed_input_cost(
            client, registered.model_name, scoped_key, expected_rate=cost_map_input_rate
        )

        assert _approx_equal(input_cost, prompt_tokens * cost_map_input_rate), (
            f"input_cost {input_cost} != {prompt_tokens} tokens * cost map rate {cost_map_input_rate} "
            f"= {prompt_tokens * cost_map_input_rate}"
        )
        assert not _approx_equal(input_cost, prompt_tokens * PINNED_INPUT_RATE), (
            f"input_cost {input_cost} is still billed at the cleared override {PINNED_INPUT_RATE}"
        )

    @pytest.mark.covers("mgmt.model.delete.persists")
    def test_delete_removes_the_deployment_everywhere(
        self, client: ManagementClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        registered = _register(client, resources)
        _ = _stored_entry(client, registered.model_name, converged=lambda _entry: True)

        client.delete_model_strict(registered.model_id)

        _assert_absent_everywhere(client, registered.model_name)
        refused = client.chat_status(scoped_key, registered.model_name, "hi this is a test")
        assert refused.status_code == 400, (
            f"chat against the deleted model must be rejected 400, got {refused.status_code}: {refused.body[:300]}"
        )
        envelope = _ErrorEnvelope.model_validate_json(refused.body)
        assert envelope.error.message, f"400 body must carry an error message: {refused.body[:300]}"
