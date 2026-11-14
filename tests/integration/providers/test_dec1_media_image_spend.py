"""Image generation money and access contracts: what a call costs, what is written to `LiteLLM_SpendLogs`, and
which key, team, budget and rate-limit rules stop a call before the provider is reached.

A DALL-E style deployment is billed per image (`output_cost_per_image` times the image count) or per pixel
(`input_cost_per_pixel` times width times height times the image count). The image count is the `n` the caller
asked for, or the number of images returned when `n` was omitted. The same number must appear in the
`x-litellm-response-cost` header, the spend row, and the key and team totals. Rates are set on the deployment in
each test, so no vendor price is pinned.
"""

from __future__ import annotations

import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.wire import Reply, Request, wire_server
from integration.providers.dec1_media_support import (
    B64_IMAGE,
    OPENAI_KEY,
    PROMPT,
    error_message,
    error_of,
    image_reply,
    json_body,
    logged_rows,
    openai_error_reply,
    response_cost,
    settled_key_spend,
    targets,
)

_GENERATIONS: Final = "/v1/images/generations"
_PRICE: Final = 0.25


def _images(count: int) -> Reply:
    return image_reply(*({"b64_json": B64_IMAGE} for _ in range(count)))


def _one_image(_request: Request) -> Reply:
    return _images(1)


def _prompt() -> str:
    return f"{PROMPT} {uuid.uuid4().hex}"


def test_image_generation_bills_the_per_image_price_times_the_requested_n(gateway: Gateway) -> None:
    """The deployment costs 0.25 per image and the caller asks for `n: 3`. The response cost header must be
    0.75, and one spend row must land for the key with `call_type` `aimage_generation`, `status` `success`,
    `spend` 0.75, `model_group` equal to the public model name and `custom_llm_provider` `openai`."""
    with wire_server(lambda _request: _images(3)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/dall-e-2", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY, output_cost_per_image=_PRICE
        )
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST", _GENERATIONS, {"model": model, "prompt": _prompt(), "n": 3}, key=key
        )
        assert response.status_code == 200, response.text
        assert response_cost(response) == pytest.approx(3 * _PRICE)
        row: Final = logged_rows(key)[0]
        assert (row["call_type"], row["status"], row["model_group"], row["custom_llm_provider"]) == (
            "aimage_generation",
            "success",
            model,
            "openai",
        ), row
        assert float(str(row["spend"])) == pytest.approx(3 * _PRICE)
        assert len(wire.drain()) == 1


def test_image_generation_without_n_bills_the_number_of_images_returned(gateway: Gateway) -> None:
    """The caller omits `n` and the upstream returns two images. The cost must be two times the per-image price:
    with no requested count the returned `data` length is the billing basis
    (litellm/litellm_core_utils/llm_cost_calc/utils.py:2327)."""
    with wire_server(lambda _request: _images(2)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/dall-e-2", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY, output_cost_per_image=_PRICE
        )
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request("POST", _GENERATIONS, {"model": model, "prompt": _prompt()}, key=key)
        assert response.status_code == 200, response.text
        assert response_cost(response) == pytest.approx(2 * _PRICE)
        assert float(str(logged_rows(key)[0]["spend"])) == pytest.approx(2 * _PRICE)
        assert len(wire.drain()) == 1


def test_image_generation_bills_a_per_pixel_price_by_the_requested_size(gateway: Gateway) -> None:
    """The deployment sets only `input_cost_per_pixel` (1e-8) and the caller asks for one `1792x1024` image. The
    cost must be 1792 * 1024 * 1e-8, so the requested size is the billing basis for pixel-priced models
    (litellm/cost_calculator.py:3085)."""
    with wire_server(_one_image) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/dall-e-3", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY, input_cost_per_pixel=1e-8
        )
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST", _GENERATIONS, {"model": model, "prompt": _prompt(), "size": "1792x1024"}, key=key
        )
        assert response.status_code == 200, response.text
        assert response_cost(response) == pytest.approx(1792 * 1024 * 1e-8)
        assert float(str(logged_rows(key)[0]["spend"])) == pytest.approx(1792 * 1024 * 1e-8)
        assert len(wire.drain()) == 1


def test_image_generation_cost_is_added_to_the_team_spend(gateway: Gateway) -> None:
    """A key that belongs to a team generates one image priced at 0.25. The team's `spend` column in
    `LiteLLM_TeamTable` must reach 0.25 and the spend row must carry the team id, so team budgets and reports
    see image traffic."""
    with wire_server(_one_image) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/dall-e-3", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY, output_cost_per_image=_PRICE
        )
        team: Final = scenario.team()
        key: Final = scenario.key(models=[model], team_id=team)
        response: Final = gateway.request("POST", _GENERATIONS, {"model": model, "prompt": _prompt()}, key=key)
        assert response.status_code == 200, response.text
        eventually(
            lambda: gateway.rows('SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,)),
            lambda rows: float(str(rows[0]["spend"])) == pytest.approx(_PRICE),
            seconds=70,
        )
        logged: Final = eventually(
            lambda: gateway.rows('SELECT team_id, spend FROM "LiteLLM_SpendLogs" WHERE team_id = %s', (team,)),
            lambda rows: len(rows) == 1,
            seconds=70,
        )
        assert float(str(logged[0]["spend"])) == pytest.approx(_PRICE)
        assert len(wire.drain()) == 1


def test_image_generation_spend_row_records_the_end_user_and_request_tags(gateway: Gateway) -> None:
    """The caller sends `user: <end user>` and `metadata.tags: [<tag>]`. The spend row must store the end user in
    `end_user` and the tag in `request_tags`, which is how per-customer and per-tag spend reports are built."""
    end_user: Final = f"customer-{uuid.uuid4().hex[:10]}"
    tag: Final = f"media-{uuid.uuid4().hex[:10]}"
    with wire_server(_one_image) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/dall-e-3", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY, output_cost_per_image=_PRICE
        )
        response: Final = gateway.request(
            "POST",
            _GENERATIONS,
            {"model": model, "prompt": _prompt(), "user": end_user, "metadata": {"tags": [tag]}},
        )
        assert response.status_code == 200, response.text
        rows: Final = eventually(
            lambda: gateway.rows(
                'SELECT end_user, request_tags::text AS tags, call_type FROM "LiteLLM_SpendLogs" WHERE end_user = %s',
                (end_user,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows[0]["call_type"] == "aimage_generation", rows
        assert tag in str(rows[0]["tags"]), rows
        assert len(wire.drain()) == 1


def test_failed_image_generation_logs_a_failure_row_with_zero_spend(gateway: Gateway) -> None:
    """The upstream refuses the prompt with a 400. The caller gets 400, exactly one spend row lands for the key
    with `status` `failure` and `spend` 0, and the key's own spend stays 0: a refused image is never billed."""

    def respond(_request: Request) -> Reply:
        return openai_error_reply(400, "scripted prompt refusal", "invalid_request_error")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/dall-e-3", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY, output_cost_per_image=_PRICE
        )
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST", _GENERATIONS, {"model": model, "prompt": _prompt(), "num_retries": 0}, key=key
        )
        assert response.status_code == 400, response.text
        row: Final = logged_rows(key)[0]
        assert row["status"] == "failure", row
        assert float(str(row["spend"])) == 0.0
        assert settled_key_spend(key, 0.0) == 0.0
        assert len(wire.drain()) == 1


def test_image_generation_is_refused_once_the_key_budget_is_spent(gateway: Gateway) -> None:
    """A key has `max_budget` 0.2 and an image costs 0.25. The first call is served. Once the key's spend has
    settled at 0.25 the second call must answer 422 with error type `budget_exceeded` and must not reach the
    upstream."""
    with wire_server(_one_image) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/dall-e-3", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY, output_cost_per_image=_PRICE
        )
        key: Final = scenario.key(models=[model], max_budget=0.2)
        first: Final = gateway.request("POST", _GENERATIONS, {"model": model, "prompt": _prompt()}, key=key)
        assert first.status_code == 200, first.text
        settled_key_spend(key, _PRICE)
        refused: Final = eventually(
            lambda: gateway.request("POST", _GENERATIONS, {"model": model, "prompt": _prompt()}, key=key),
            lambda response: response.status_code != 200,
            seconds=30,
        )
        assert refused.status_code == 422, refused.text
        assert error_of(refused)["type"] == "budget_exceeded", refused.text
        assert targets(wire) == [("POST", _GENERATIONS)]


@pytest.mark.parametrize("path", ("/v1/images/generations", "/images/generations"))
def test_image_generation_enforces_key_model_access(gateway: Gateway, path: str) -> None:
    """A key is limited to one image model. Calling another image model with it must answer 403 with error type
    `key_model_access_denied` and `param` `model` without reaching any upstream; the allowed model is served."""
    with wire_server(_one_image) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(model="openai/dall-e-3", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        other: Final = scenario.model(model="openai/dall-e-3", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        key: Final = scenario.key(models=[allowed])
        denied: Final = gateway.request("POST", path, {"model": other, "prompt": PROMPT}, key=key)
        assert denied.status_code == 403, denied.text
        error: Final = error_of(denied)
        assert (error["type"], error["param"]) == ("key_model_access_denied", "model"), denied.text
        assert wire.drain() == ()
        granted: Final = gateway.request("POST", path, {"model": allowed, "prompt": PROMPT}, key=key)
        assert granted.status_code == 200, granted.text
        assert len(wire.drain()) == 1


def test_image_generation_counts_against_the_key_rpm_limit(gateway: Gateway) -> None:
    """A key has `rpm_limit` 1. Its first image generation is served and the second, sent right after, must
    answer 429 without reaching the upstream: image calls consume the same per-key request window as chat."""
    with wire_server(_one_image) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/dall-e-3", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        key: Final = scenario.key(models=[model], rpm_limit=1)
        first: Final = gateway.request("POST", _GENERATIONS, {"model": model, "prompt": _prompt()}, key=key)
        second: Final = gateway.request("POST", _GENERATIONS, {"model": model, "prompt": _prompt()}, key=key)
        assert (first.status_code, second.status_code) == (200, 429), (first.text, second.text)
        assert len(wire.drain()) == 1


def test_image_generation_request_fallback_serves_from_the_second_deployment(gateway: Gateway) -> None:
    """The primary image deployment's upstream always answers 500. The caller lists a second image model in
    `fallbacks`. The call must answer 200 with the second upstream's image, the second upstream must be called
    once with its own provider model name, and the primary must have been tried at least once."""

    def failing(_request: Request) -> Reply:
        return openai_error_reply(500, "scripted outage", "server_error")

    def fallback(request: Request) -> Reply:
        assert json_body(request)["model"] == "dall-e-2"
        return image_reply({"b64_json": "ZmFsbGJhY2s="})

    with wire_server(failing) as primary, wire_server(fallback) as secondary, gateway.scenario() as scenario:
        first: Final = scenario.model(model="openai/dall-e-3", api_base=f"{primary.url}/v1", api_key=OPENAI_KEY)
        second: Final = scenario.model(model="openai/dall-e-2", api_base=f"{secondary.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request(
            "POST", _GENERATIONS, {"model": first, "prompt": _prompt(), "fallbacks": [second], "num_retries": 0}
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"][0]["b64_json"] == "ZmFsbGJhY2s="
        assert len(primary.drain()) >= 1
        assert len(secondary.drain()) == 1


def test_image_generation_retries_a_failed_attempt_when_num_retries_is_set(gateway: Gateway) -> None:
    """The upstream answers 500 once and then succeeds. With `num_retries: 1` in the request the caller must get
    200 and the upstream must have seen exactly two attempts with identical bodies. UNVERIFIED: that the
    request-level `num_retries` is honoured on the image route the same way as on chat; see
    litellm/router.py async_function_with_fallbacks."""
    prompt: Final = _prompt()
    seen: Final[list[dict[str, object]]] = []  # mutable-ok: the peer thread records each attempt for the assertion

    def respond(request: Request) -> Reply:
        seen.append(dict(json_body(request)))
        if len(seen) == 1:
            return openai_error_reply(500, "scripted transient failure", "server_error")
        return image_reply({"b64_json": B64_IMAGE})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/dall-e-3", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request(
            "POST", _GENERATIONS, {"model": model, "prompt": prompt, "num_retries": 1}
        )
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 2
        assert seen[0] == seen[1] == {"model": "dall-e-3", "prompt": prompt}


def test_image_generation_content_policy_refusal_reaches_the_caller_as_400(gateway: Gateway) -> None:
    """The upstream refuses a prompt with 400 and code `content_policy_violation`. The caller must get 400 with
    the upstream's message inside `error.message`, and the upstream must be called once: a policy refusal is not
    retried."""
    refusal: Final = "Your request was rejected as a result of our safety system."

    def respond(_request: Request) -> Reply:
        return openai_error_reply(400, refusal, "invalid_request_error", "content_policy_violation")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/dall-e-3", api_base=f"{wire.url}/v1", api_key=OPENAI_KEY)
        response: Final = gateway.request("POST", _GENERATIONS, {"model": model, "prompt": _prompt()})
        assert response.status_code == 400, response.text
        assert refusal in error_message(response), response.text
        assert len(wire.drain()) == 1
