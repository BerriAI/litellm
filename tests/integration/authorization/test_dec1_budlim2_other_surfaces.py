"""Contract: budgets and rate limits on embeddings, text completions, audio transcriptions, image edits and
batches.

Every one of these routes runs the inference checks at auth (``RouteChecks.runs_inference_checks``), so an
exhausted key is refused on all of them before the provider. What differs per surface is the budget hold
(``_max_cost_for_cost_info`` in litellm/proxy/spend_tracking/budget_reservation.py):

- embeddings, rerank and moderations are input only routes (``_is_input_only_route``): no output is ever held,
  so with free input an embedding takes no hold at all;
- text completions multiply the output estimate by ``max(n, best_of, 1)`` and clamp it to the cost map ceiling
  of the model (4096 for gpt-3.5-turbo-instruct);
- transcriptions are priced per second (whisper-1 has no token prices), so no estimate can be made and no hold is
  taken; the budget is enforced only on the stored spend at auth;
- image edits hold ``n * per image price`` when the model's ``mode`` is ``image_generation`` or ``image_edit``;
- batch creation names no model in its body, so no hold is taken; spend arrives when the batch completes.

The rate limiter's pre-body check refuses a cached, exhausted key on every metered route before the body is
read, including multipart uploads and ``POST /v1/batches`` (#51675), but skips bodyless GETs such as a batch
retrieve.

Price: input free, output 0.004 a token. The scripted text completion provider reports 2 output tokens (0.008).
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.client import Gateway
from tests.integration.authorization.dec1_budlim2_support import (
    OUTPUT_RATE,
    TEXT_REAL_COST,
    TRANSCRIPT,
    chat,
    embedding,
    embedding_model,
    image_edit,
    image_edit_model,
    message_of,
    openai_error,
    surface_model,
    surface_provider,
    text_completion,
    text_model,
    transcription,
    transcription_model,
)
from tests.integration.authorization.dec1_limits_support import hashed, limit_message
from tests.integration.authorization.dec1_ratelim_support import NO_ANSWER, status_before_body_completes
from tests.integration.spend.dec1_budgets_support import (
    background,
    counter,
    counter_settles,
    key_counter,
    redis_client,
)

pytestmark: Final = pytest.mark.shares_worker_state


@pytest.mark.timeout(180)
def test_an_embedding_with_free_input_takes_no_hold_even_when_output_is_priced(gateway: Gateway) -> None:
    """An embedding deployment has free input and an output price of 1.0 a token. A key with ``max_budget`` 0.01
    sends an embedding that the provider holds. Because embeddings are input only, the estimate is 0 and no hold is
    taken: the key counter stays absent or 0 while it is in flight, and a second embedding on the same key is
    served meanwhile. A port that applies the 16384 token output default to embeddings would pin the budget."""
    with surface_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = embedding_model(scenario, provider, output_cost_per_token=1.0)
        key: Final = scenario.key(models=[model], max_budget=0.01)
        in_flight: Final = run(lambda: embedding(gateway, model, key, hold=True))
        provider.wait_for(1)
        try:
            assert counter(cache, key_counter(key)) in (None, 0.0)
            assert embedding(gateway, model, key).status_code == 200
        finally:
            provider.release()
        assert in_flight.result(timeout=60).status_code == 200


def test_an_exhausted_key_is_refused_on_embeddings_with_the_openai_budget_envelope(gateway: Gateway) -> None:
    """A key with ``max_budget: 0`` calls ``/v1/embeddings``. It is refused with HTTP 422 and
    ``{"error": {"type": "budget_exceeded", "code": "422"}}`` before the provider is called."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = embedding_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=0)
        error: Final = openai_error(embedding(gateway, model, key), 422)
        assert (error["type"], error["code"]) == ("budget_exceeded", "422"), error
        assert provider.posts() == ()


@pytest.mark.timeout(180)
@pytest.mark.parametrize(("field", "count"), (("n", 2), ("best_of", 3)))
def test_a_text_completion_hold_is_multiplied_by_n_or_best_of(gateway: Gateway, field: str, count: int) -> None:
    """A key with ``max_budget`` 10 sends a held ``/v1/completions`` call with ``max_tokens: 10`` and ``n: 2`` or
    ``best_of: 3``. The hold is ``10 * 0.004 * count``: 0.08 for ``n: 2`` and 0.12 for ``best_of: 3``
    (``_get_output_multiplier`` takes the larger of both). After the answer it settles to the reported cost 0.008.
    ``best_of`` exists only on text completions, so a port that copies the chat estimator would hold 0.04."""
    with surface_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = text_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=10.0)
        in_flight: Final = run(lambda: text_completion(gateway, model, key, hold=True, max_tokens=10, **{field: count}))
        provider.wait_for(1)
        assert counter(cache, key_counter(key)) == pytest.approx(10 * OUTPUT_RATE * count)
        provider.release()
        assert in_flight.result(timeout=60).status_code == 200
        counter_settles(cache, key_counter(key), TEXT_REAL_COST)


@pytest.mark.timeout(180)
def test_an_uncapped_text_completion_holds_the_instruct_model_ceiling_not_16384(gateway: Gateway) -> None:
    """A key with ``max_budget`` 100 sends a held ``/v1/completions`` call without ``max_tokens`` to
    gpt-3.5-turbo-instruct, whose cost map ``max_output_tokens`` is 4096. The hold is 4096 * 0.004 = 16.384, not
    the 16384 token default (65.536). UNVERIFIED: assumes the router's model group info of a custom priced
    deployment still carries the cost map ceiling."""
    with surface_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = text_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=100.0)
        in_flight: Final = run(lambda: text_completion(gateway, model, key, hold=True))
        provider.wait_for(1)
        assert counter(cache, key_counter(key)) == pytest.approx(4096 * OUTPUT_RATE)
        provider.release()
        in_flight.result(timeout=60)


def test_a_text_completion_the_provider_fails_releases_its_hold(gateway: Gateway) -> None:
    """The provider answers 500 to a ``/v1/completions`` call with ``max_tokens: 10`` on a key with
    ``max_budget`` 1.0. The caller gets a 500 and the key counter returns to 0."""
    with surface_provider(failing_status=500) as provider, gateway.scenario() as scenario, redis_client() as cache:
        model: Final = text_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        assert text_completion(gateway, model, key, max_tokens=10).status_code == 500
        counter_settles(cache, key_counter(key), 0.0)


@pytest.mark.timeout(180)
def test_a_transcription_takes_no_hold_because_whisper_is_priced_per_second(gateway: Gateway) -> None:
    """A key with ``max_budget`` 0.05 has a held ``/v1/audio/transcriptions`` call (whisper-1, priced per second
    in the cost map) in flight. No estimate can be made, so no hold is taken: the key counter stays absent or 0 and
    a chat call on the same key with ``max_tokens: 10`` (hold 0.04) is served meanwhile. UNVERIFIED: assumes the
    multipart ``prompt`` field reaches the provider so the hold marker works."""
    with surface_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        whisper: Final = transcription_model(scenario, provider)
        chat_model: Final = surface_model(scenario, provider)
        key: Final = scenario.key(models=[whisper, chat_model], max_budget=0.05)
        in_flight: Final = run(lambda: transcription(gateway, whisper, key, hold=True))
        provider.wait_for(1)
        try:
            assert counter(cache, key_counter(key)) in (None, 0.0)
            assert chat(gateway, chat_model, key, max_tokens=10).status_code == 200
        finally:
            provider.release()
        answered: Final = in_flight.result(timeout=60)
        assert answered.status_code == 200, answered.text
        assert answered.json()["text"] == TRANSCRIPT


def test_an_exhausted_key_is_refused_on_transcriptions_before_the_provider(gateway: Gateway) -> None:
    """A key with ``max_budget: 0`` uploads audio to ``/v1/audio/transcriptions``. It is refused with HTTP 422
    ``budget_exceeded`` and the provider receives nothing: without a hold the stored spend check at auth is the
    only budget guard on this route, so it must stay."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = transcription_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=0)
        assert openai_error(transcription(gateway, model, key), 422)["type"] == "budget_exceeded"
        assert provider.posts() == ()


def test_transcriptions_are_refused_over_the_key_rpm_limit(gateway: Gateway) -> None:
    """A key with ``rpm_limit: 1`` sends two transcriptions. The second is HTTP 429 with ``error.type``
    ``throttling_error`` and the ``api_key`` requests text, and the provider receives one upload."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = transcription_model(scenario, provider)
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert transcription(gateway, model, key).status_code == 200
        error: Final = openai_error(transcription(gateway, model, key), 429)
        assert error["type"] == "throttling_error", error
        assert message_of(error).startswith(limit_message("api_key", hashed(key), "requests", 1))
        assert len(provider.posts()) == 1


def test_a_transcription_uses_a_request_but_no_tokens_of_the_key_windows(gateway: Gateway) -> None:
    """A key with ``rpm_limit: 5`` and ``tpm_limit: 1000`` sends one transcription. Its response reports
    ``x-ratelimit-api_key-remaining-requests: 4``. UNVERIFIED: the token reservation for an audio upload is not
    pinned (the limiter estimates text from ``prompt``); only the request count is asserted."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = transcription_model(scenario, provider)
        key: Final = scenario.key(models=[model], rpm_limit=5, tpm_limit=1000)
        served: Final = transcription(gateway, model, key)
        assert served.status_code == 200, served.text
        assert served.headers["x-ratelimit-api_key-remaining-requests"] == "4"


def test_image_edits_are_refused_over_the_key_rpm_limit(gateway: Gateway) -> None:
    """A key with ``rpm_limit: 1`` sends two ``/v1/images/edits`` uploads. The second is HTTP 429
    ``throttling_error`` with the ``api_key`` requests text, before the provider."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = image_edit_model(scenario, provider)
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert image_edit(gateway, model, key).status_code == 200
        error: Final = openai_error(image_edit(gateway, model, key), 429)
        assert message_of(error).startswith(limit_message("api_key", hashed(key), "requests", 1))
        assert len(provider.posts()) == 1


def test_an_exhausted_key_is_refused_on_image_edits_before_the_provider(gateway: Gateway) -> None:
    """A key with ``max_budget: 0`` sends ``/v1/images/edits``. HTTP 422 ``budget_exceeded``, provider not
    called."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = image_edit_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=0)
        assert openai_error(image_edit(gateway, model, key), 422)["type"] == "budget_exceeded"
        assert provider.posts() == ()


@pytest.mark.timeout(180)
def test_an_image_edit_holds_n_times_the_per_image_price(gateway: Gateway) -> None:
    """An image edit deployment declares ``mode: image_edit`` and ``output_cost_per_image: 0.5`` in its
    ``model_info``. A key with ``max_budget`` 10 sends an edit with ``n: 3``; while the provider answers the
    hold on the key counter is 3 * 0.5 = 1.5
    (``_estimate_image_generation_cost``). UNVERIFIED: custom image prices are known to be
    lost for image generation (#52103); if the same happens to edits, no hold is taken and this fails for that
    reason."""
    with surface_provider() as provider, gateway.scenario() as scenario, redis_client() as cache, background() as run:
        model: Final = image_edit_model(scenario, provider, mode="image_edit", output_cost_per_image=0.5)
        key: Final = scenario.key(models=[model], max_budget=10.0)
        in_flight: Final = run(lambda: image_edit(gateway, model, key, n=3, hold=True))
        provider.wait_for(1)
        try:
            assert counter(cache, key_counter(key)) == pytest.approx(3 * 0.5)
        finally:
            provider.release()
        assert in_flight.result(timeout=60).status_code == 200


def test_an_exhausted_key_cannot_create_a_batch(gateway: Gateway) -> None:
    """A key with ``max_budget: 0`` posts ``/v1/batches`` with a made up ``input_file_id``. The budget check at
    auth runs before the handler looks at the file, so the answer is HTTP 422 ``budget_exceeded`` (not a 4xx about
    the file), and nothing reaches the provider. Batch spend only lands when a batch completes, so this auth check
    is what stops an exhausted key from queueing more work."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = surface_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=0)
        refused: Final = gateway.request(
            "POST",
            "/v1/batches",
            {"input_file_id": "file-dec1-budlim2", "endpoint": "/v1/chat/completions", "completion_window": "24h"},
            key=key,
        )
        assert openai_error(refused, 422)["type"] == "budget_exceeded"
        assert provider.posts() == ()


def test_a_cached_key_over_its_rpm_is_refused_on_batch_creation_before_the_body(gateway: Gateway) -> None:
    """A key with ``rpm_limit: 1`` uses its request on chat (which caches the key). A raw ``POST /v1/batches`` that
    declares an 8 MB body but sends 1 KB is answered 429 at once, without waiting for the rest of the body:
    batch creation is metered, so #51675 keeps it in the pre-body check."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = surface_model(scenario, provider)
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert chat(gateway, model, key, max_tokens=5).status_code == 200
        assert status_before_body_completes(gateway, key, path="/v1/batches", model=model) == 429


def test_a_cached_key_over_its_rpm_can_still_retrieve_a_batch(gateway: Gateway) -> None:
    """A key with ``rpm_limit: 1`` uses its request on chat. A bodyless ``GET /v1/batches/batch_dec1_missing`` is
    not refused by the pre-body rate limit check (``_is_bodyless_get``, #51675): whatever the handler answers for
    the unknown id, it is not 429, so a client can keep polling its batches while it is at its limit."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = surface_model(scenario, provider)
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert chat(gateway, model, key, max_tokens=5).status_code == 200
        polled: Final = gateway.request("GET", "/v1/batches/batch_dec1_missing", key=key)
        assert polled.status_code != 429, polled.text


def test_a_cached_key_over_its_rpm_is_refused_on_embeddings_before_the_body(gateway: Gateway) -> None:
    """A key with ``rpm_limit: 1`` uses its request on chat. A raw ``POST /v1/embeddings`` declaring an 8 MB body is
    answered 429 after only 1 KB was sent: the key's request window needs no body, so every metered route is
    refused before buffering. A port that moves the check after body parsing reads every retried payload."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = surface_model(scenario, provider)
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert chat(gateway, model, key, max_tokens=5).status_code == 200
        status: Final = status_before_body_completes(gateway, key, path="/v1/embeddings", model=model)
        assert status not in (NO_ANSWER, 200), status
        assert status == 429
