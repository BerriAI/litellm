"""Key limits on the non-chat endpoints, and the legacy limiter's wire format.

The contract: a key's ``rpm_limit`` and ``tpm_limit`` cover every generation endpoint with ONE window per
key. Embeddings, the Responses API, Anthropic Messages and text completions are admitted, refused, counted and
reported like chat completions, with these endpoint-specific rules for the token reservation:

- ``/v1/embeddings``: input only (40 characters is 10 tokens), no output is reserved and no output cap is
  ever added to the request.
- ``/v1/responses``: input from ``input``, output from ``max_output_tokens`` with a minimum of 16. Under a small
  token limit an uncapped request is forwarded with ``max_output_tokens`` set to a quarter of the limit.
- ``/v1/messages``: input from ``messages``, output from ``max_tokens``. The 429 is Anthropic-shaped:
  ``{"type": "error", "error": {"type": "rate_limit_error", "message": ...}}`` with the same message text.
- ``/v1/completions``: input from ``prompt``, output from ``max_tokens``.

The last tests pin the legacy limiter that ``LEGACY_MULTI_INSTANCE_RATE_LIMITING=true`` switches back on,
because its headers and message are a different public format: unscoped ``x-ratelimit-limit-requests`` /
``x-ratelimit-remaining-requests``, ``x-litellm-key-remaining-requests-<model>`` for per-model limits, and a
429 that starts ``LiteLLM Rate Limit Handler for rate limit type = key. Crossed TPM / RPM / Max Parallel
Request Limit.``

Read from ``litellm/proxy/hooks/parallel_request_limiter_v3.py`` (``_estimate_input_and_output_tokens``,
``_get_explicit_output_cap``, ``_apply_implicit_output_cap``), ``litellm/proxy/hooks/parallel_request_limiter.py``
and ``litellm/proxy/common_utils/callback_utils.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.wire import Wire, wire_server
from tests.integration.authorization.dec1_limits_support import (
    ANTHROPIC_MODEL,
    DEFAULT_USAGE,
    WINDOW_SECONDS,
    anthropic_messages,
    assert_limited,
    chat,
    complete,
    counter_key,
    embed,
    error_message,
    hashed,
    limit_message,
    limits_proxy,
    posts,
    provider,
    redis_integer,
    respond,
    scope_headers,
    shared_redis,
    statuses,
)

LEGACY_PREFIX: Final = (
    "LiteLLM Rate Limit Handler for rate limit type = key. Crossed TPM / RPM / Max Parallel Request Limit. "
    "current rpm: 1, rpm limit: 1, "
)


def _embedding_model(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(model="openai/text-embedding-3-small", api_base=f"{wire.url}/v1")


def _anthropic_model(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(model=f"anthropic/{ANTHROPIC_MODEL}", api_base=wire.url, api_key="synthetic-anthropic-key")


def _completion_model(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(model="text-completion-openai/gpt-3.5-turbo-instruct", api_base=f"{wire.url}/v1")


def _tokens_settled(key: str, expected: int) -> None:
    with shared_redis() as redis:
        name: Final = counter_key("api_key", hashed(key), "tokens")
        eventually(lambda: redis_integer(redis, name), lambda value: value == expected, seconds=20)


@contextmanager
def _legacy_proxy(gateway: Gateway, directory: Path) -> Iterator[Gateway]:
    with limits_proxy(gateway, directory, environment={"LEGACY_MULTI_INSTANCE_RATE_LIMITING": "true"}) as candidate:
        yield candidate


def test_embeddings_are_refused_over_the_key_rpm_limit(gateway: Gateway) -> None:
    """A key with rpm_limit=1 sends two ``/v1/embeddings`` requests. The first must answer 200 with
    ``x-ratelimit-api_key-limit-requests: 1`` and ``-remaining-requests: 0``; the second must be the standard
    OpenAI-shaped 429 for ``api_key`` requests, and the provider must have been called once."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = _embedding_model(scenario, wire)
        key: Final = scenario.key(models=[model], rpm_limit=1)
        served: Final = embed(gateway, model, key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-requests": "1", "remaining-requests": "0"}
        assert_limited(embed(gateway, model, key), "api_key", hashed(key), "requests", 1)
        assert len(posts(wire)) == 1


def test_an_embedding_reserves_its_input_tokens_and_no_output(gateway: Gateway) -> None:
    """A key has tpm_limit=10 and embeds a 40 character input (10 tokens). The request must be served with
    ``remaining-tokens: 0``: exactly the input estimate is reserved and nothing is added for output."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = _embedding_model(scenario, wire)
        key: Final = scenario.key(models=[model], tpm_limit=10)
        served: Final = embed(gateway, model, key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-tokens": "10", "remaining-tokens": "0"}


def test_an_embedding_larger_than_the_tpm_limit_is_refused(gateway: Gateway) -> None:
    """A key has tpm_limit=9 and embeds a 40 character input (10 tokens). It must be refused with the 429 for
    ``api_key`` tokens reading ``Current limit: 9, Remaining: 9`` and never reach the provider."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = _embedding_model(scenario, wire)
        key: Final = scenario.key(models=[model], tpm_limit=9)
        assert_limited(embed(gateway, model, key), "api_key", hashed(key), "tokens", 9, remaining=9)
        assert posts(wire) == ()


def test_an_embedding_is_forwarded_without_an_output_cap_under_a_small_tpm_limit(gateway: Gateway) -> None:
    """A key has tpm_limit=100, small enough that a chat request would get ``max_tokens: 25`` added. An embedding
    request must reach the provider with only the fields the caller sent plus the provider model: no
    ``max_tokens`` and no ``max_output_tokens``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = _embedding_model(scenario, wire)
        key: Final = scenario.key(models=[model], tpm_limit=100)
        assert embed(gateway, model, key).status_code == 200
        (forwarded,) = posts(wire)
        assert "max_tokens" not in forwarded
        assert "max_output_tokens" not in forwarded


def test_an_embedding_is_charged_the_prompt_tokens_the_provider_reported(gateway: Gateway) -> None:
    """A key has tpm_limit=1000 and the embedding provider bills 11 prompt tokens. The key's token window must
    settle at 11."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = _embedding_model(scenario, wire)
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        assert embed(gateway, model, key).status_code == 200
        _tokens_settled(key, DEFAULT_USAGE.prompt)
        assert len(posts(wire)) == 1


def test_responses_requests_are_refused_over_the_key_rpm_limit(gateway: Gateway) -> None:
    """A key with rpm_limit=1 sends two ``/v1/responses`` requests. The first must answer 200 with the api_key
    request headers (1, 0); the second must be the standard OpenAI-shaped 429 with ``retry-after: 60``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        served: Final = respond(gateway, model, key, max_output_tokens=16)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-requests": "1", "remaining-requests": "0"}
        refused: Final = respond(gateway, model, key, max_output_tokens=16)
        assert_limited(refused, "api_key", hashed(key), "requests", 1)
        assert refused.headers["retry-after"] == str(WINDOW_SECONDS)
        assert len(posts(wire)) == 1


def test_a_responses_request_reserves_max_output_tokens_with_a_floor_of_sixteen(gateway: Gateway) -> None:
    """A key has tpm_limit=1000. A ``/v1/responses`` request with ``max_output_tokens: 100`` must report
    ``remaining-tokens: 890`` (10 input + 100). A second with ``max_output_tokens: 4`` must reserve 10 + 16, not
    10 + 4, because the Responses API has a minimum output of 16 tokens."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        first: Final = respond(gateway, model, key, max_output_tokens=100)
        assert first.status_code == 200, first.text
        assert scope_headers(first, "api_key") == {"limit-tokens": "1000", "remaining-tokens": "890"}
        _tokens_settled(key, DEFAULT_USAGE.total)
        second: Final = respond(gateway, model, key, max_output_tokens=4)
        assert second.status_code == 200, second.text
        assert scope_headers(second, "api_key") == {
            "limit-tokens": "1000",
            "remaining-tokens": str(1000 - DEFAULT_USAGE.total - 26),
        }


def test_an_uncapped_responses_request_gets_max_output_tokens_added_under_a_small_tpm_limit(
    gateway: Gateway,
) -> None:
    """A key has tpm_limit=100 and sends a ``/v1/responses`` request with no ``max_output_tokens``. The provider
    must receive ``max_output_tokens: 25`` and no ``max_tokens``, and the reservation is 10 + 25
    (``remaining-tokens: 65``). The Responses API ignores ``max_tokens``, so the cap must use its own field."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=100)
        served: Final = respond(gateway, model, key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-tokens": "100", "remaining-tokens": "65"}
        (forwarded,) = posts(wire)
        assert forwarded["max_output_tokens"] == 25
        assert "max_tokens" not in forwarded


def test_a_responses_request_is_charged_the_usage_the_provider_reported(gateway: Gateway) -> None:
    """A key has tpm_limit=1000 and the Responses provider bills 11 input and 7 output tokens. The key's token
    window must settle at 18."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        assert respond(gateway, model, key, max_output_tokens=100).status_code == 200
        _tokens_settled(key, DEFAULT_USAGE.total)
        assert len(posts(wire)) == 1


def test_messages_requests_over_the_key_rpm_limit_get_an_anthropic_shaped_429(gateway: Gateway) -> None:
    """A key with rpm_limit=1 sends two ``/v1/messages`` requests to an Anthropic model. The second must answer
    429 with the body ``{"type": "error", "error": {"type": "rate_limit_error", "message": "Rate limit exceeded
    for api_key: <hash>. Limit type: requests. Current limit: 1, Remaining: 0. Limit resets at: ..."}}`` and
    ``retry-after: 60``. Anthropic SDKs only recognise the error in this shape.

    Product bug: the limiter raises ``ProxyRateLimitError`` with ``headers={"retry-after": ...}`` and chat
    completions forward them (``handle_llm_api_exception`` merges ``failure.headers``), but the ``/v1/messages``
    handler builds its error from ``get_custom_headers`` only (litellm/proxy/anthropic_endpoints/endpoints.py
    ``proxy_exception_from_http_exception(answered, headers)``), so ``retry-after`` is dropped. The
    ``RateLimitError`` docstring says these headers are preserved across the proxy boundary."""
    pytest.skip("BUG: /v1/messages 429 from the key rate limiter drops the limiter's retry-after header")
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_model(scenario, wire)
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert anthropic_messages(gateway, model, key).status_code == 200
        refused: Final = anthropic_messages(gateway, model, key)
        assert refused.status_code == 429, refused.text
        body: Final = JSON_OBJECT.validate_json(refused.content)
        assert body["type"] == "error", refused.text
        error: Final = object_value(body["error"])
        assert error["type"] == "rate_limit_error", refused.text
        assert string_value(error["message"]).startswith(limit_message("api_key", hashed(key), "requests", 1))
        assert refused.headers["retry-after"] == str(WINDOW_SECONDS)
        assert len(posts(wire)) == 1


def test_a_messages_request_reserves_its_input_plus_max_tokens(gateway: Gateway) -> None:
    """A key has tpm_limit=1000 and sends a ``/v1/messages`` request with a 40 character prompt and
    ``max_tokens: 16``. The 200 must carry ``x-ratelimit-api_key-limit-tokens: 1000`` and
    ``-remaining-tokens: 974`` (10 + 16 reserved)."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_model(scenario, wire)
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        served: Final = anthropic_messages(gateway, model, key, max_tokens=16)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-tokens": "1000", "remaining-tokens": "974"}


def test_a_messages_request_too_large_for_the_tpm_limit_is_refused_before_the_provider(gateway: Gateway) -> None:
    """A key has tpm_limit=20 and sends a ``/v1/messages`` request with ``max_tokens: 500``. It must answer 429 in
    the Anthropic error shape with the ``api_key`` tokens message (``Current limit: 20, Remaining: 20``) and
    the provider must receive nothing."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_model(scenario, wire)
        key: Final = scenario.key(models=[model], tpm_limit=20)
        refused: Final = anthropic_messages(gateway, model, key, max_tokens=500)
        assert refused.status_code == 429, refused.text
        assert error_message(refused).startswith(limit_message("api_key", hashed(key), "tokens", 20, 20))
        assert posts(wire) == ()


def test_a_messages_request_is_charged_input_plus_output_tokens(gateway: Gateway) -> None:
    """A key has tpm_limit=1000 and the Anthropic provider bills 11 input and 7 output tokens for a
    ``/v1/messages`` call. The key's token window must settle at 18."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_model(scenario, wire)
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        assert anthropic_messages(gateway, model, key, max_tokens=100).status_code == 200
        _tokens_settled(key, DEFAULT_USAGE.total)
        assert len(posts(wire)) == 1


def test_text_completions_are_limited_and_reserved_from_the_prompt(gateway: Gateway) -> None:
    """A key has rpm_limit=1 and tpm_limit=1000 and sends two ``/v1/completions`` requests with a 40 character
    prompt and ``max_tokens: 20``. The first must answer 200 with ``remaining-requests: 0`` and
    ``remaining-tokens: 970``; the second must be the 429 for ``api_key`` requests.

    Pinned current behaviour: the ``/v1/completions`` route does not go through ``handle_llm_api_exception``. Its
    catch all (litellm/proxy/proxy_server.py ``completion``, ``route_failure_proxy_exception(e, str(e.message))``)
    uses the exception's ``message``, so the limiter's text arrives prefixed with ``litellm.RateLimitError: ``
    where chat completions return the bare limiter text."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = _completion_model(scenario, wire)
        key: Final = scenario.key(models=[model], rpm_limit=1, tpm_limit=1000)
        served: Final = complete(gateway, model, key, max_tokens=20)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {
            "limit-requests": "1",
            "remaining-requests": "0",
            "limit-tokens": "1000",
            "remaining-tokens": "970",
        }
        assert_limited(
            complete(gateway, model, key, max_tokens=20),
            "api_key",
            hashed(key),
            "requests",
            1,
            message_prefix="litellm.RateLimitError: ",
        )
        assert len(posts(wire)) == 1


def test_one_request_window_covers_every_endpoint_the_key_calls(gateway: Gateway) -> None:
    """A key has rpm_limit=3 and calls chat completions, embeddings and the Responses API once each. All three
    are served with ``remaining-requests`` counting down 2, 1, 0, and a fourth request on any endpoint must be
    refused: there is one window per key, not one per endpoint."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        chat_model: Final = scenario.model(api_base=f"{wire.url}/v1")
        embedding_model: Final = _embedding_model(scenario, wire)
        key: Final = scenario.key(models=[chat_model, embedding_model], rpm_limit=3)
        served: Final = [
            chat(gateway, chat_model, key),
            embed(gateway, embedding_model, key),
            respond(gateway, chat_model, key, max_output_tokens=16),
        ]
        assert statuses(served) == [200, 200, 200], [response.text for response in served]
        assert [scope_headers(response, "api_key")["remaining-requests"] for response in served] == ["2", "1", "0"]
        assert_limited(embed(gateway, embedding_model, key), "api_key", hashed(key), "requests", 3)
        assert len(posts(wire)) == 3


def test_the_legacy_limiter_refuses_with_its_own_message_and_a_seconds_to_next_minute_retry_after(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A proxy runs with ``LEGACY_MULTI_INSTANCE_RATE_LIMITING=true`` and a key has rpm_limit=1. The second
    request must answer 429 with a message starting ``LiteLLM Rate Limit Handler for rate limit type = key.
    Crossed TPM / RPM / Max Parallel Request Limit. current rpm: 1, rpm limit: 1,`` and a ``retry-after`` that
    is the number of seconds to the next wall-clock minute (between 0 and 60), not the fixed 60 of the default
    limiter."""
    with (
        wire_server(provider()) as wire,
        _legacy_proxy(gateway, tmp_path) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert chat(candidate, model, key).status_code == 200
        refused: Final = eventually(
            lambda: chat(candidate, model, key), lambda response: response.status_code == 429, seconds=5
        )
        assert error_message(refused).startswith(LEGACY_PREFIX), refused.text
        assert 0 < float(refused.headers["retry-after"]) <= 60
        assert "rate_limit_type" not in refused.headers


def test_the_legacy_limiter_reports_unscoped_request_headers_for_the_key(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy runs with the legacy limiter and a key has rpm_limit=5. A served response must carry
    ``x-ratelimit-limit-requests: 5`` and ``x-ratelimit-remaining-requests: 4`` and no
    ``x-ratelimit-api_key-*`` header, which only the default limiter writes."""
    with (
        wire_server(provider()) as wire,
        _legacy_proxy(gateway, tmp_path) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=5)
        served: Final = chat(candidate, model, key)
        assert served.status_code == 200, served.text
        assert served.headers["x-ratelimit-limit-requests"] == "5"
        assert served.headers["x-ratelimit-remaining-requests"] == "4"
        assert scope_headers(served, "api_key") == {}


def test_the_legacy_limiter_reports_per_model_remaining_as_x_litellm_key_headers(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A proxy runs with the legacy limiter and a key has ``model_rpm_limit`` 5 and ``model_tpm_limit`` 1000 for a
    model. Its first served response must carry ``x-litellm-key-remaining-requests-<model>: 4`` and
    ``x-litellm-key-remaining-tokens-<model>: 1000`` (no tokens have been billed yet). These two header families
    exist only under the legacy limiter."""
    with (
        wire_server(provider()) as wire,
        _legacy_proxy(gateway, tmp_path) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], model_rpm_limit={model: 5}, model_tpm_limit={model: 1000})
        served: Final = chat(candidate, model, key)
        assert served.status_code == 200, served.text
        assert served.headers[f"x-litellm-key-remaining-requests-{model}"] == "4"
        assert served.headers[f"x-litellm-key-remaining-tokens-{model}"] == "1000"


def test_the_default_limiter_never_sends_x_litellm_key_remaining_headers(gateway: Gateway) -> None:
    """On the default proxy a key has ``model_rpm_limit`` 5 for a model. A served response must carry no header
    whose name starts with ``x-litellm-key-remaining-``: the default limiter reports per-model remaining counts
    only as ``x-ratelimit-model_per_key-remaining-requests`` (here 4)."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], model_rpm_limit={model: 5})
        served: Final = chat(gateway, model, key)
        assert served.status_code == 200, served.text
        assert [name for name in served.headers if name.startswith("x-litellm-key-remaining-")] == []
        assert scope_headers(served, "model_per_key") == {"limit-requests": "5", "remaining-requests": "4"}
