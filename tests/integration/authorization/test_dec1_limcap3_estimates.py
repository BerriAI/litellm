"""Edge cases of the token reservation the v3 limiter takes before the provider is called.

The contract, read from ``_estimate_input_and_output_tokens``, ``_get_explicit_output_cap``,
``get_output_candidate_count``, ``_parse_output_cap_value`` and ``_apply_implicit_output_cap`` in
``litellm/proxy/hooks/parallel_request_limiter_v3.py`` (lines 970 to 1460):

- Input is ``max(1, characters // 4)`` when the prompt has any characters and 0 when it has none.
- The explicit output cap is the LARGEST of ``max_tokens`` and ``max_completion_tokens`` (``max(..., default=None)``
  at line 1141). A value that parses as a number (``int(float(value))``) counts, booleans never do, and an explicit
  0 is a real cap (``is not None``, line 1145).
- Output with no cap is ``max(input, floor)`` and the floor is ``min(1024, max(1, tpm // 4))``; but when input is 0
  and there is no cap, the output estimate is 0 too (line 1449).
- The output part is multiplied by the largest of ``n``, ``best_of`` and Google ``candidateCount`` (line 1153),
  never by the number of prompts in a text completion prompt list.
- ``_apply_implicit_output_cap`` writes the floor into the request whenever it is below 1024 and the caller set no
  cap, independent of what was reserved, so an empty prompt is still capped although it reserved nothing.
- Google GenAI native routes read ``generationConfig.maxOutputTokens`` and write the implicit cap there.

Each test reads ``x-ratelimit-api_key-remaining-tokens`` (limit minus everything reserved in the window, this request
included) or the 429 text. The Rust limiter (#51537) re-implements this estimator; every number here is one a port
can silently change.
"""

from __future__ import annotations

from typing import Final

from tests.integration._support.client import Gateway
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limcap3_support import completion_model, generate_content
from tests.integration.authorization.dec1_limits_support import (
    assert_limited,
    chat,
    hashed,
    posts,
    probe,
    provider,
    scope_headers,
)


def test_the_larger_of_max_tokens_and_max_completion_tokens_is_reserved(gateway: Gateway) -> None:
    """A key with tpm_limit=1000 sends ``max_tokens: 20`` and ``max_completion_tokens: 50`` together. The reservation
    is 10 input + 50 output, so the 200 carries ``remaining-tokens: 940`` (parallel_request_limiter_v3.py:1135-1141
    take the maximum of both fields). A port that reads only ``max_tokens`` would report 970."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        served: Final = chat(gateway, model, key, max_tokens=20, max_completion_tokens=50)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-tokens": "1000", "remaining-tokens": "940"}


def test_an_explicit_max_tokens_of_zero_reserves_only_the_prompt(gateway: Gateway) -> None:
    """A key with tpm_limit=1000 sends ``max_tokens: 0``. Zero is an explicit cap (``_has_explicit_output_cap`` checks
    ``is not None``, parallel_request_limiter_v3.py:1145), so the reservation is 10 + 0 and the 200 carries
    ``remaining-tokens: 990``, not 990 - 1024 worth of floor."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        served: Final = chat(gateway, model, key, max_tokens=0)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-tokens": "1000", "remaining-tokens": "990"}


def test_an_explicit_max_tokens_of_zero_is_not_replaced_by_the_implicit_cap(gateway: Gateway) -> None:
    """A key with tpm_limit=100 (implicit cap 25) sends ``max_tokens: 0``. The provider must receive
    ``max_tokens: 0`` unchanged: ``_apply_implicit_output_cap`` returns early on an explicit cap
    (parallel_request_limiter_v3.py:1206-1212). UNVERIFIED: that the OpenAI transformation forwards a zero
    ``max_tokens`` instead of dropping it."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=100)
        assert chat(gateway, model, key, max_tokens=0).status_code == 200
        (forwarded,) = posts(wire)
        assert forwarded["max_tokens"] == 0


def test_a_numeric_string_max_tokens_is_reserved_as_its_number(gateway: Gateway) -> None:
    """A key with tpm_limit=1000 sends ``max_tokens: "30"`` (a string). ``_parse_output_cap_value`` parses strings
    with ``int(float(value))`` (parallel_request_limiter_v3.py:970), so the reservation is 10 + 30 and the 200 carries
    ``remaining-tokens: 960``. UNVERIFIED: that the proxy forwards a string ``max_tokens`` without a 400."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        served: Final = chat(gateway, model, key, max_tokens="30")
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-tokens": "1000", "remaining-tokens": "960"}


def test_a_fractional_max_tokens_is_reserved_rounded_down(gateway: Gateway) -> None:
    """A key with tpm_limit=1000 sends ``max_tokens: 30.9``. ``int(float(30.9))`` is 30
    (parallel_request_limiter_v3.py:975), so the 200 carries ``remaining-tokens: 960``. UNVERIFIED: that the request
    is not refused for a non-integer ``max_tokens`` before the limiter runs."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        served: Final = chat(gateway, model, key, max_tokens=30.9)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-tokens": "1000", "remaining-tokens": "960"}


def test_a_prompt_shorter_than_four_characters_still_reserves_one_input_token(gateway: Gateway) -> None:
    """A key with tpm_limit=1000 sends the three character prompt ``abc`` with ``max_tokens: 20``. Input is
    ``max(1, 3 // 4)`` = 1 (parallel_request_limiter_v3.py:1440), so the reservation is 21 and the 200 carries
    ``remaining-tokens: 979``. The model name is fresh, so no response cache entry can answer it."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        served: Final = chat(gateway, model, key, text="abc", max_tokens=20)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-tokens": "1000", "remaining-tokens": "979"}


def test_an_empty_prompt_without_an_output_cap_reserves_nothing(gateway: Gateway) -> None:
    """A key with tpm_limit=1000 sends one user message with empty content and no output cap. Input is 0, and with
    no cap and no input the output estimate is 0 too (parallel_request_limiter_v3.py:1449), so the 200 carries
    ``remaining-tokens: 1000``. UNVERIFIED: that an empty content string reaches the limiter (no earlier 400)."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        served: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": ""}]}, key=key
        )
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-tokens": "1000", "remaining-tokens": "1000"}


def test_an_empty_prompt_is_still_given_the_implicit_output_cap(gateway: Gateway) -> None:
    """Same empty prompt under tpm_limit=1000: although nothing was reserved, ``_apply_implicit_output_cap`` only
    looks at the floor (``min(1024, 1000 // 4)`` = 250) and the missing cap, so the provider must receive
    ``max_tokens: 250`` (parallel_request_limiter_v3.py:1176-1227). The reservation and the cap disagree here; a port
    that derives one from the other changes either the header or the forwarded body."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        served: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": ""}]}, key=key
        )
        assert served.status_code == 200, served.text
        (forwarded,) = posts(wire)
        assert forwarded["max_tokens"] == 250


def test_a_text_completion_prompt_list_reserves_every_prompt_but_one_output(gateway: Gateway) -> None:
    """A key with tpm_limit=1000 sends ``/v1/completions`` with a list of two 40 character prompts and
    ``max_tokens: 5``. Input is ``(40 + 40) // 4`` = 20 and the output is not multiplied by the number of prompts
    (only ``n`` / ``best_of`` multiply it, parallel_request_limiter_v3.py:1153), so the 200 carries
    ``remaining-tokens: 975``. A provider answers one completion per prompt, so this under-reserves on purpose or not;
    pinned as current behaviour."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = completion_model(scenario, wire)
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        served: Final = gateway.request(
            "POST", "/v1/completions", {"model": model, "prompt": [probe(), probe()], "max_tokens": 5}, key=key
        )
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-tokens": "1000", "remaining-tokens": "975"}


def test_best_of_multiplies_the_text_completion_output_reservation(gateway: Gateway) -> None:
    """A key with tpm_limit=1000 sends ``/v1/completions`` with a 40 character prompt, ``max_tokens: 10`` and
    ``best_of: 3``. The reservation is 10 + 3 x 10 = 40, so the 200 carries ``remaining-tokens: 960``
    (``get_output_candidate_count``, parallel_request_limiter_v3.py:1153). ``best_of`` exists only on text
    completions, so a port that copies the chat estimator would report 980."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = completion_model(scenario, wire)
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        served: Final = gateway.request(
            "POST", "/v1/completions", {"model": model, "prompt": probe(), "max_tokens": 10, "best_of": 3}, key=key
        )
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-tokens": "1000", "remaining-tokens": "960"}


def test_the_larger_of_n_and_best_of_multiplies_the_output_not_their_product(gateway: Gateway) -> None:
    """A key with tpm_limit=1000 sends ``/v1/completions`` with ``max_tokens: 10``, ``n: 2`` and ``best_of: 3``. The
    multiplier is ``max(n, best_of)`` = 3, not 6, so the reservation is 40 and the 200 carries
    ``remaining-tokens: 960`` (parallel_request_limiter_v3.py:1167-1173)."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = completion_model(scenario, wire)
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        served: Final = gateway.request(
            "POST",
            "/v1/completions",
            {"model": model, "prompt": probe(), "max_tokens": 10, "n": 2, "best_of": 3},
            key=key,
        )
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-tokens": "1000", "remaining-tokens": "960"}


def test_a_generate_content_request_is_refused_when_max_output_tokens_times_candidates_exceeds_the_limit(
    gateway: Gateway,
) -> None:
    """A key with tpm_limit=45 calls ``/v1beta/models/<model>:generateContent`` with a 40 character part and
    ``generationConfig: {maxOutputTokens: 20, candidateCount: 2}``. The estimate is 10 + 2 x 20 = 50 > 45, so the
    request is refused with the ``api_key`` tokens 429 reading ``Current limit: 45, Remaining: 45`` and nothing
    reaches the provider (parallel_request_limiter_v3.py:1116-1124 and 1153-1166). UNVERIFIED: that this route uses
    a Google GenAI native call type and that the OpenAI-shaped 429 envelope is kept on it."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=45)
        refused: Final = generate_content(gateway, model, key, {"maxOutputTokens": 20, "candidateCount": 2})
        assert_limited(refused, "api_key", hashed(key), "tokens", 45, remaining=45)
        assert posts(wire) == ()


def test_a_generate_content_request_with_one_candidate_fits_the_same_limit(gateway: Gateway) -> None:
    """The same key (tpm_limit=45) sends ``maxOutputTokens: 20`` with ``candidateCount: 1``. The estimate is
    10 + 20 = 30 <= 45, so it is served and the provider receives exactly one request. Paired with the test above it
    shows ``candidateCount`` is what pushed the first request over. UNVERIFIED as above."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=45)
        served: Final = generate_content(gateway, model, key, {"maxOutputTokens": 20, "candidateCount": 1})
        assert served.status_code == 200, served.text
        assert len(posts(wire)) == 1


def test_an_uncapped_generate_content_request_gets_the_implicit_cap_in_generation_config(gateway: Gateway) -> None:
    """A key with tpm_limit=100 calls ``generateContent`` with no ``generationConfig``. The limiter writes
    ``generationConfig.maxOutputTokens: 25`` (a quarter of the limit) into the request
    (parallel_request_limiter_v3.py:1215-1222), which the Google-to-OpenAI adapter forwards as an output cap of 25.
    UNVERIFIED: that the adapter maps ``maxOutputTokens`` to ``max_tokens`` (not ``max_completion_tokens``) on the
    OpenAI wire, so the assertion accepts either field name but insists on the value 25."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=100)
        assert generate_content(gateway, model, key).status_code == 200
        (forwarded,) = posts(wire)
        assert forwarded.get("max_tokens", forwarded.get("max_completion_tokens")) == 25, forwarded
