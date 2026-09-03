"""Live e2e: what an already-issued virtual key can still do to itself.

Three post-issue behaviors of a virtual key, each driven through the admin route
that owns it and judged on live traffic:

- rotation with a grace period: `/key/{key}/regenerate` with `grace_period` keeps
  the old secret authenticating until the window closes, then stops accepting it.
  A sibling key rotated in the same test *without* a grace period is the control:
  it is refused immediately, so the graced key still being served cannot be an
  auth-cache artifact
- spend reset: `/key/{key}/reset_spend` puts the key's accumulated spend back to
  the requested value, and a key that its own max_budget had blocked serves again.
  Spend arrives on a buffered batch write, so the test settles on a value that has
  stopped moving before it resets, and afterwards asserts only relationships a
  late flush cannot violate: the reset reports a previous value no smaller than
  the settled one (spend never shrinks on its own), reports zero, and the DB is
  polled until it agrees. Exact equality against a single sample would go red on
  correct behavior the moment a flush landed between two reads
- the allow side of `allowed_routes`: a key granted the `llm_api_routes` group
  reaches the LLM endpoints, while a management route stays refused, so the grant
  is a real whitelist rather than an absent check (the deny side lives in
  tests/e2e/access_control/)

Every key is minted here and deleted on teardown; auth is judged by status code
(401 vs not-401), never by requiring a completion, so a provider hiccup cannot be
mistaken for a revocation.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from e2e_config import CHEAP_OPENAI_MODEL, unique_marker
from e2e_http import NoBody, Result, StreamingResponse, is_ok, unwrap
from lifecycle import ResourceManager
from models import (
    ChatBody,
    ChatMessage,
    EmbedBody,
    KeyGenerateBody,
    KeyInfoParams,
    LiteLLMParamsBody,
    ModelInfoBody,
    ModelNewBody,
)
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

EMBEDDING_MODEL = "openai-text-embedding-3-small"
ROUTE_NOT_ALLOWED_MARKER = "not allowed to call this route"
BUDGET_BLOCK_MARKER = "budget_exceeded"

GRACE_PERIOD = "60s"
REVOCATION_DEADLINE_SECONDS = 240.0
TINY_BUDGET = 3e-6
# Spend lands on a buffered batch write (proxy_batch_write_at), so a freshly read
# value can still be superseded. Re-read across a window wider than that flush
# before treating a value as final.
SPEND_SETTLE_SECONDS = 25.0
SPEND_SAMPLE_INTERVAL = 5.0


class RegeneratedKey(BaseModel):
    key: str


class KeyRegenerateBody(BaseModel):
    """POST /key/{key}/regenerate. `grace_period` is a duration string ("60s",
    "24h"); omitted means the old secret is revoked the moment the new one is
    issued."""

    grace_period: str | None = None


class ResetSpendBody(BaseModel):
    reset_to: float


class ResetSpendResult(BaseModel):
    """POST /key/{key}/reset_spend. Reports the spend the key carried before the
    reset alongside the value it now holds."""

    spend: float
    previous_spend: float


class ScopedKeyInfo(BaseModel):
    allowed_routes: list[str] | None = None
    models: list[str] = []
    spend: float | None = None
    max_budget: float | None = None


class ScopedKeyInfoResponse(BaseModel):
    info: ScopedKeyInfo


@dataclass(frozen=True, slots=True)
class KeyLifecycleClient:
    """The admin routes that act on an existing key, plus the LLM and management
    calls used to judge what that key may still do."""

    proxy: ProxyClient

    def key_info(self, key: str) -> ScopedKeyInfo:
        return unwrap(
            self.proxy.transport.get(
                "/key/info",
                headers=self.proxy.transport.master,
                params=KeyInfoParams(key=key),
                response_type=ScopedKeyInfoResponse,
            )
        ).info

    def regenerate(self, key: str, *, grace_period: str | None = None) -> str:
        return unwrap(
            self.proxy.transport.post(
                f"/key/{key}/regenerate",
                headers=self.proxy.transport.master,
                json=KeyRegenerateBody(grace_period=grace_period),
                response_type=RegeneratedKey,
            )
        ).key

    def reset_spend(self, key: str, *, reset_to: float) -> ResetSpendResult:
        return unwrap(
            self.proxy.transport.post(
                f"/key/{key}/reset_spend",
                headers=self.proxy.transport.master,
                json=ResetSpendBody(reset_to=reset_to),
                response_type=ResetSpendResult,
            )
        )

    def chat_status(self, key: str, model: str = CHEAP_OPENAI_MODEL) -> StreamingResponse:
        return self.proxy.transport.send(
            "/chat/completions",
            headers=self.proxy.transport.bearer(key),
            json=ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=f"reply with one word {unique_marker()}")],
                max_tokens=8,
            ),
        )

    def embed_status(self, key: str, model: str = EMBEDDING_MODEL) -> StreamingResponse:
        return self.proxy.transport.send(
            "/embeddings",
            headers=self.proxy.transport.bearer(key),
            json=EmbedBody(model=model, input=f"route group probe {unique_marker()}"),
        )

    def create_model_status(self, key: str, model_name: str) -> StreamingResponse:
        return self.proxy.transport.send(
            "/model/new",
            headers=self.proxy.transport.bearer(key),
            json=ModelNewBody(
                model_name=model_name,
                litellm_params=LiteLLMParamsBody(model="openai/gpt-4o-mini"),
                model_info=ModelInfoBody(id=model_name),
            ),
        )

    def model_catalog_status(self, key: str) -> Result[NoBody]:
        return self.proxy.transport.get(
            "/v1/models",
            headers=self.proxy.transport.bearer(key),
            params=NoBody(),
            response_type=NoBody,
        )


@pytest.fixture
def keys(proxy: ProxyClient) -> KeyLifecycleClient:
    return KeyLifecycleClient(proxy=proxy)


def _poll[T](attempt: Callable[[], T | None], *, deadline_seconds: float, interval: float, failure: str) -> T:
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        found = attempt()
        if found is not None:
            return found
        time.sleep(interval)
    pytest.fail(failure)


def _mint(keys: KeyLifecycleClient, resources: ResourceManager, body: KeyGenerateBody) -> str:
    key = keys.proxy.generate_key(body)
    resources.defer(lambda: keys.proxy.delete_key(key))
    return key


def _assert_authenticates(keys: KeyLifecycleClient, key: str, context: str) -> None:
    outcome = keys.chat_status(key)
    assert outcome.status_code != 401, f"{context}: the key was refused at auth ({outcome.body[:300]})"


def _settled_spend(keys: KeyLifecycleClient, key: str) -> float:
    """The key's recorded spend once it has stopped moving.

    Spend reaches the DB on a buffered batch write, so a single sample can be
    taken mid-flight and a value read now can be superseded a moment later. The
    key this runs against has made exactly one successful call (every later one
    is refused over budget and costs nothing), so once a non-zero value appears
    there is no second increment left to land. That is asserted rather than
    assumed: the value is re-read across a settle window wider than the batch
    write, and a change means either a flush was still outstanding or the one
    call got counted twice.
    """
    first = _poll(
        lambda: (lambda spend: spend if spend is not None and spend > 0 else None)(keys.key_info(key).spend),
        deadline_seconds=keys.proxy.poll_timeout,
        interval=5.0,
        failure="/key/info never recorded any spend for the key, so the reset would have nothing to clear",
    )
    deadline = time.monotonic() + SPEND_SETTLE_SECONDS
    while time.monotonic() < deadline:
        time.sleep(SPEND_SAMPLE_INTERVAL)
        again = keys.key_info(key).spend
        assert again == first, (
            f"the key's recorded spend moved from {first} to {again} while settling, after a single "
            "successful call; a second increment means a write was still outstanding or the call was "
            "billed twice"
        )
    return first


def _await_rejected(keys: KeyLifecycleClient, key: str, *, deadline_seconds: float, failure: str) -> None:
    _poll(
        lambda: True if keys.chat_status(key).status_code == 401 else None,
        deadline_seconds=deadline_seconds,
        interval=5.0,
        failure=failure,
    )


class TestKeyRegenerationGracePeriod:
    @pytest.mark.covers("other.key_mgmt.regenerate.grace_period_honored")
    def test_old_key_serves_through_the_grace_period_then_is_revoked(
        self, keys: KeyLifecycleClient, resources: ResourceManager
    ) -> None:
        graced = _mint(keys, resources, KeyGenerateBody(models=[CHEAP_OPENAI_MODEL]))
        control = _mint(keys, resources, KeyGenerateBody(models=[CHEAP_OPENAI_MODEL]))
        _assert_authenticates(keys, graced, "before rotation the key to be graced")
        _assert_authenticates(keys, control, "before rotation the control key")

        rotated_graced = keys.regenerate(graced, grace_period=GRACE_PERIOD)
        resources.defer(lambda: keys.proxy.delete_key(rotated_graced))
        rotated_control = keys.regenerate(control)
        resources.defer(lambda: keys.proxy.delete_key(rotated_control))
        assert rotated_graced != graced, "regenerate handed back the same secret, so nothing rotated"

        _assert_authenticates(keys, graced, f"inside the {GRACE_PERIOD} grace period the rotated-out key")
        _assert_authenticates(keys, rotated_graced, "the replacement key")

        _await_rejected(
            keys,
            control,
            deadline_seconds=keys.proxy.poll_timeout,
            failure=(
                "a key rotated with no grace period was still accepted at auth; the graced key's "
                "survival cannot be attributed to the grace period"
            ),
        )

        _await_rejected(
            keys,
            graced,
            deadline_seconds=REVOCATION_DEADLINE_SECONDS,
            failure=(
                f"the rotated-out key was still accepted at auth well past its {GRACE_PERIOD} grace "
                "period, so the window never closes"
            ),
        )
        _assert_authenticates(keys, rotated_graced, "after the grace period closed the replacement key")


class TestKeySpendReset:
    @pytest.mark.covers("other.key_mgmt.spend_reset.resets_to_value")
    def test_reset_spend_clears_recorded_spend_and_unblocks_the_key(
        self, keys: KeyLifecycleClient, resources: ResourceManager
    ) -> None:
        key = _mint(keys, resources, KeyGenerateBody(models=[CHEAP_OPENAI_MODEL], max_budget=TINY_BUDGET))

        first = keys.chat_status(key)
        assert first.status_code == 200, (
            f"the key must serve its first call before its budget is spent, got {first.status_code}: "
            f"{first.body[:300]}"
        )

        blocked = _poll(
            lambda: (lambda outcome: outcome if BUDGET_BLOCK_MARKER in outcome.body else None)(keys.chat_status(key)),
            deadline_seconds=keys.proxy.poll_timeout,
            interval=3.0,
            failure=f"the key never got blocked over its {TINY_BUDGET} budget, so there is no block to reset away",
        )
        assert blocked.status_code == 429, (
            f"an over-budget call must be refused 429, got {blocked.status_code}: {blocked.body[:300]}"
        )

        spent = _settled_spend(keys, key)
        assert spent > TINY_BUDGET, (
            f"/key/info recorded {spent}, at or under the {TINY_BUDGET} cap the key was refused for; the "
            "reset would then be clearing something other than the over-budget spend"
        )

        reset = keys.reset_spend(key, reset_to=0.0)
        assert reset.previous_spend >= spent, (
            f"reset_spend reports previous_spend {reset.previous_spend}, less than the {spent} the key had "
            "settled on. Spend accounting only ever grows as buffered writes land, so a late flush can raise "
            "this value; nothing may lower it except the reset itself"
        )
        assert reset.spend == 0.0, f"reset_spend to 0.0 reports the key still holding {reset.spend}"

        cleared = _poll(
            lambda: (lambda spend: spend if spend == 0.0 else None)(keys.key_info(key).spend),
            deadline_seconds=keys.proxy.poll_timeout,
            interval=5.0,
            failure=(
                f"/key/info never reported the key's spend back at 0.0 after the reset; it had settled on "
                f"{spent} beforehand"
            ),
        )
        assert cleared == 0.0
        budget = keys.key_info(key).max_budget
        assert budget == TINY_BUDGET, (
            f"the reset must not disturb the key's budget; /key/info reports max_budget {budget}"
        )

        _poll(
            lambda: (lambda outcome: True if BUDGET_BLOCK_MARKER not in outcome.body else None)(
                keys.chat_status(key)
            ),
            deadline_seconds=keys.proxy.poll_timeout,
            interval=5.0,
            failure="the key was still refused over budget after its spend was reset to 0.0",
        )


class TestVirtualKeyRouteGroupGrant:
    @pytest.mark.covers("other.auth.virtual_key.route_group_allowed")
    def test_llm_route_group_grants_the_llm_endpoints_and_nothing_else(
        self, keys: KeyLifecycleClient, resources: ResourceManager
    ) -> None:
        key = _mint(keys, resources, KeyGenerateBody(models=[], allowed_routes=["llm_api_routes"]))

        granted = keys.key_info(key).allowed_routes
        assert granted == ["llm_api_routes"], (
            f"/key/info reports allowed_routes {granted}, configured ['llm_api_routes']"
        )

        chat = keys.chat_status(key)
        assert chat.status_code == 200, (
            f"a key granted the llm_api_routes group must reach /chat/completions, got {chat.status_code}: "
            f"{chat.body[:300]}"
        )

        embeddings = keys.embed_status(key)
        assert embeddings.status_code == 200, (
            f"the same grant must cover /embeddings, got {embeddings.status_code}: {embeddings.body[:300]}"
        )

        catalog = keys.model_catalog_status(key)
        assert is_ok(catalog), f"the grant must cover the /v1/models catalog LLM clients read, got {catalog}"

        denied = keys.create_model_status(key, f"e2e-route-group-{unique_marker()}")
        assert denied.status_code == 403, (
            f"the grant is a whitelist: a management route must stay refused 403, got {denied.status_code}: "
            f"{denied.body[:300]}"
        )
        assert ROUTE_NOT_ALLOWED_MARKER in denied.body, (
            f"the 403 must be a route-permission denial, got: {denied.body[:300]}"
        )
