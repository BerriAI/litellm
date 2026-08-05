"""Live e2e: the v3 dynamic rate limiter's saturation-aware priority reservation.

Covers quota_management.ratelimit.priority_generous / priority_strict: with
`dynamic_rate_limiter_v3` enabled, a model's TPM capacity is split into priority
reservations, but a reservation is only enforced once the model is saturated.

- Generous mode (recorded usage below the saturation threshold): a key whose
  priority reserves 25% of capacity keeps serving past its reservation,
  borrowing the idle capacity (priority_generous.picks_under_tpm)
- Strict mode (recorded usage at/over the threshold): the over-reservation key
  is blocked with the priority-flavored 429 while a key of a different priority,
  still inside its own reservation, is served (priority_strict.picks_under_tpm)

The proxy under test must run with this config (and LITELLM_LICENSE set, since
priority reservation is a premium feature):

    litellm_settings:
      callbacks: ["dynamic_rate_limiter_v3"]
      priority_reservation:
        prod: 0.5
        dev: 0.25
      priority_reservation_settings:
        saturation_threshold: 0.5
        saturation_check_cache_ttl: 1

The constants below mirror those values; if the proxy runs different ones the
tests fail with a message naming the required config rather than skipping.

The limiter counts a request against the model-wide window pre-call, but tokens
only land on the counters after each response completes (there is no pre-call
token reservation at the model level), so recorded saturation always trails the
traffic that produced it. The tests therefore drive spend by summing each
body's usage.total_tokens (the counter can never be ahead of that sum) and poll
for the strict-mode block instead of expecting it on an exact call. Each test
creates its own /model/new deployment so its 60s rate-limit window and counters
are isolated from concurrent runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from e2e_config import unique_marker
from e2e_http import StreamingResponse, require_successful_call
from lifecycle import ResourceManager
from models import KeyGenerateBody, KeyMetadata, LiteLLMParamsBody
from quota_client import QuotaClient

pytestmark = pytest.mark.e2e

BACKEND = "anthropic/claude-haiku-4-5-20251001"
MODEL_TPM = 400
DEV_PRIORITY = "dev"
PROD_PRIORITY = "prod"
DEV_RESERVED_TOKENS = int(MODEL_TPM * 0.25)
SATURATION_TOKENS = int(MODEL_TPM * 0.5)
CHAT_MAX_TOKENS = 16
WINDOW_SECONDS = 60
WINDOW_MARGIN_SECONDS = 10
STRICT_POLL_SPEND_CEILING = int(MODEL_TPM * 0.7)

REQUIRED_CONFIG_HINT = (
    "the proxy must run litellm_settings.callbacks=['dynamic_rate_limiter_v3'] with "
    "priority_reservation {prod: 0.5, dev: 0.25} and priority_reservation_settings "
    "{saturation_threshold: 0.5, saturation_check_cache_ttl: 1}; see this module's docstring"
)


class _ChatUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    total_tokens: int


class _ChatBodyWithUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    usage: _ChatUsage


def _total_tokens(outcome: StreamingResponse) -> int:
    try:
        return _ChatBodyWithUsage.model_validate_json(outcome.body).usage.total_tokens
    except ValidationError:
        pytest.fail(f"successful chat body must report usage.total_tokens, got: {outcome.body[:300]}")


@dataclass(frozen=True, slots=True)
class _Fixture:
    model: str
    dev_key: str
    prod_key: str


def _dynamic_limited_model(client: QuotaClient, resources: ResourceManager, label: str) -> _Fixture:
    model = f"e2e-dynpri-{label}-{unique_marker()}"
    model_id = client.proxy.create_model(
        model,
        LiteLLMParamsBody(model=BACKEND, api_key="os.environ/ANTHROPIC_API_KEY", tpm=MODEL_TPM),
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))

    def _priority_key(priority: str) -> str:
        key = client.proxy.generate_key(
            KeyGenerateBody(
                models=[model],
                metadata=KeyMetadata(priority=priority),
                key_alias=f"e2e-dynpri-{label}-{priority}-{unique_marker()}",
            )
        )
        resources.defer(lambda: client.proxy.delete_key(key))
        return key

    return _Fixture(model=model, dev_key=_priority_key(DEV_PRIORITY), prod_key=_priority_key(PROD_PRIORITY))


def _chat(client: QuotaClient, key: str, model: str) -> StreamingResponse:
    return client.chat(key, model, f"reply with one word {unique_marker()}", max_tokens=CHAT_MAX_TOKENS)


@dataclass(frozen=True, slots=True)
class _FirstOk:
    sent_at: float
    response: StreamingResponse


def _first_ok(client: QuotaClient, key: str, model: str) -> _FirstOk:
    """First successful call on a fresh key opens the model's rate-limit window;
    `sent_at` (captured before the winning send) is a lower bound on the window
    start. A fresh key may briefly 401 until the auth cache picks it up, so
    retry 401s to a deadline; a 401 never reaches the limiter, so only the
    successful call consumes budget."""
    deadline = time.monotonic() + client.proxy.poll_timeout
    while True:
        sent_at = time.monotonic()
        outcome = _chat(client, key, model)
        if outcome.ok:
            return _FirstOk(sent_at=sent_at, response=outcome)
        if outcome.status_code != 401 or time.monotonic() >= deadline:
            require_successful_call(outcome)
        time.sleep(client.proxy.poll_interval)


def _window_guard(first: _FirstOk, spent: int) -> None:
    assert time.monotonic() < first.sent_at + WINDOW_SECONDS - WINDOW_MARGIN_SECONDS, (
        f"only {spent} tokens of spend landed before the {WINDOW_SECONDS}s rate-limit window could "
        "roll; this test needs every call inside one window"
    )


class TestDynamicRateLimitPriority:
    @pytest.mark.covers(
        "quota_management.ratelimit.priority_generous.picks_under_tpm",
        exercised_on=["chat_completions"],
    )
    def test_generous_mode_lets_priority_borrow_past_reservation(
        self, client: QuotaClient, resources: ResourceManager
    ) -> None:
        fixture = _dynamic_limited_model(client, resources, "generous")

        info = client.proxy.key_info(fixture.dev_key)
        assert info.metadata is not None and info.metadata.priority == DEV_PRIORITY, (
            f"/key/info must echo the key's priority metadata, got {info.metadata}"
        )

        first = _first_ok(client, fixture.dev_key, fixture.model)
        spent = _total_tokens(first.response)
        while spent <= DEV_RESERVED_TOKENS:
            _window_guard(first, spent)
            assert spent < SATURATION_TOKENS, (
                f"spend reached the saturation threshold ({spent} of {SATURATION_TOKENS}) before "
                f"crossing the dev reservation ({DEV_RESERVED_TOKENS}); shrink per-call spend to "
                "keep the borrowing claim observable"
            )
            outcome = _chat(client, fixture.dev_key, fixture.model)
            assert outcome.status_code != 429, (
                f"dev key was blocked at {spent} recorded tokens, under the saturation threshold "
                f"({SATURATION_TOKENS} of {MODEL_TPM}); generous mode must let it borrow past its "
                f"{DEV_RESERVED_TOKENS}-token reservation. If the limiter is missing entirely, "
                f"{REQUIRED_CONFIG_HINT}. 429 body: {outcome.body[:300]}"
            )
            require_successful_call(outcome)
            spent += _total_tokens(outcome)

        assert spent > DEV_RESERVED_TOKENS

    @pytest.mark.skip(
        reason=(
            "LIT-5118: the stage proxy does not run the dynamic_rate_limiter_v3 callbacks + "
            "priority_reservation config this module's docstring requires (zero limiter log lines "
            "on any pod during the 2026-08-02 run), so strict enforcement can never engage there"
        )
    )
    @pytest.mark.covers(
        "quota_management.ratelimit.priority_strict.picks_under_tpm",
        exercised_on=["chat_completions"],
    )
    def test_strict_mode_blocks_saturated_priority_but_serves_the_other(
        self, client: QuotaClient, resources: ResourceManager
    ) -> None:
        fixture = _dynamic_limited_model(client, resources, "strict")

        prod_warmup = _first_ok(client, fixture.prod_key, fixture.model)
        first = _first_ok(client, fixture.dev_key, fixture.model)
        prod_spent = _total_tokens(prod_warmup.response)
        dev_spent = _total_tokens(first.response)

        while prod_spent + dev_spent < SATURATION_TOKENS:
            _window_guard(prod_warmup, prod_spent + dev_spent)
            outcome = _chat(client, fixture.dev_key, fixture.model)
            assert outcome.status_code != 429, (
                f"dev key was blocked at {prod_spent + dev_spent} recorded tokens, before the "
                f"saturation threshold ({SATURATION_TOKENS} of {MODEL_TPM}); strict enforcement "
                f"must not engage early. 429 body: {outcome.body[:300]}"
            )
            require_successful_call(outcome)
            dev_spent += _total_tokens(outcome)

        while True:
            _window_guard(prod_warmup, prod_spent + dev_spent)
            assert prod_spent + dev_spent < STRICT_POLL_SPEND_CEILING, (
                f"dev key was still served at {prod_spent + dev_spent} tokens, past the saturation "
                f"threshold ({SATURATION_TOKENS}) and {DEV_RESERVED_TOKENS}-token dev reservation; "
                f"strict priority enforcement never engaged. Check that {REQUIRED_CONFIG_HINT}"
            )
            outcome = _chat(client, fixture.dev_key, fixture.model)
            if outcome.status_code == 429:
                assert "Priority-based rate limit exceeded" in outcome.body, (
                    f"the saturated dev key must get the priority-flavored 429, got: {outcome.body[:300]}"
                )
                assert outcome.headers.get("x-litellm-priority") == DEV_PRIORITY, (
                    f"the 429 must attribute the blocked priority, headers: "
                    f"{ {k: v for k, v in outcome.headers.items() if 'litellm' in k} }"
                )
                break
            require_successful_call(outcome)
            dev_spent += _total_tokens(outcome)

        prod_outcome = _chat(client, fixture.prod_key, fixture.model)
        require_successful_call(prod_outcome)
        prod_spent += _total_tokens(prod_outcome)
        assert prod_spent < int(MODEL_TPM * 0.5), (
            f"the prod fairness claim needs prod spend ({prod_spent}) inside its reservation "
            f"({int(MODEL_TPM * 0.5)}); shrink per-call spend"
        )
