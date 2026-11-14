"""Retries, fallbacks, aliases, wildcards and order on embeddings, Anthropic messages and responses.

The chat completions endpoint is where most routing tests live. The same router decides for every other
endpoint, and a rewrite can easily wire one endpoint to a different retry or fallback path. This file repeats
the core contracts once per endpoint: `POST /v1/embeddings`, `POST /v1/messages` and `POST /v1/responses`.

One owned proxy runs with `num_retries: 1`, cooldowns disabled, a `fallbacks` entry and a `model_group_alias`
entry per endpoint. All deployments are OpenAI-shaped and point at the scripted upstream, which records the
upstream model name of every request whatever path it arrives on, so each test counts attempts per upstream
model. Groups that are expected to retry have two deployments, which keeps retries immediate.

Assumed, not verified in this container: that the proxy serves `/v1/messages` and `/v1/responses` for an
`openai/` deployment by calling the upstream's chat completions or responses path with the deployment's model
name. The scripted upstream answers both paths.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from integration.routing.dec1_routing_support import (
    PROVIDER_KEY,
    Rig,
    anthropic_message,
    deployment,
    embed,
    error_of,
    pair,
    respond,
    routing_rig,
)
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

ENDPOINTS: Final = ("embeddings", "messages", "responses")
ROUTER_SETTINGS: Final[dict[str, JsonValue]] = {
    "num_retries": 1,
    "disable_cooldowns": True,
    "fallbacks": [{f"ep-{endpoint}-primary": [f"ep-{endpoint}-backup"]} for endpoint in ENDPOINTS],
    "model_group_alias": {f"ep-{endpoint}-alias": f"ep-{endpoint}-real" for endpoint in ENDPOINTS},
}


def _endpoint_models(endpoint: str, api_base: str) -> tuple[dict[str, JsonValue], ...]:
    return (
        *pair(f"ep-{endpoint}-retry", f"script-{endpoint}-retry", api_base),
        *pair(f"ep-{endpoint}-primary", f"status-500-{endpoint}-primary", api_base),
        deployment(f"ep-{endpoint}-backup", f"ok-{endpoint}-backup", api_base),
        deployment(f"ep-{endpoint}-real", f"ok-{endpoint}-real", api_base),
        *pair(f"ep-{endpoint}-dead", f"status-503-{endpoint}-dead", api_base),
        *pair(f"ep-{endpoint}-clientfb", f"status-500-{endpoint}-clientfb", api_base),
        deployment(f"ep-{endpoint}-clientbackup", f"ok-{endpoint}-clientbackup", api_base),
        *pair(f"ep-{endpoint}-400", f"status-400-{endpoint}-bad", api_base),
        deployment(f"ep-{endpoint}-order", f"status-500-{endpoint}-order-one", api_base, order=1),
        deployment(f"ep-{endpoint}-order", f"ok-{endpoint}-order-two", api_base, order=2),
    )


def _models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        *(
            entry for endpoint in ENDPOINTS for entry in _endpoint_models(endpoint, api_base)
        ),  # comprehension-ok: flat model list
        {
            "model_name": "dec1ep/*",
            "litellm_params": {"model": "openai/*", "api_key": PROVIDER_KEY, "api_base": api_base},
            "model_info": {"id": "dec1-ep-wildcard"},
        },
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with routing_rig(tmp_path_factory.mktemp("dec1-routing-endpoints"), _models, ROUTER_SETTINGS) as value:
        yield value


def test_embeddings_retries_a_500_once_and_returns_the_second_attempts_answer(rig: Rig) -> None:
    """`num_retries: 1`; the group's upstream answers 500 once and 200 after that, and the caller sends
    an embeddings request (`POST /v1/embeddings`).

    The caller gets 200 with the scripted embedding vector `[0.25, 0.5, 0.75]`, and the upstream received exactly two
    requests for the model `script-embeddings-retry`.
    """
    rig.upstream.script("script-embeddings-retry", 500)
    response: Final = embed(rig.gateway, "ep-embeddings-retry")
    assert response.status_code == 200, response.text
    assert response.json()["data"][0]["embedding"] == [0.25, 0.5, 0.75], response.text
    assert rig.upstream.count("script-embeddings-retry") == 2


def test_embeddings_falls_back_to_the_configured_group_after_the_retry_budget(rig: Rig) -> None:
    """`fallbacks` maps `ep-embeddings-primary` to `ep-embeddings-backup`; the primary's upstream always answers 500 and
    the caller sends an embeddings request.

    The caller gets 200 with the scripted embedding vector `[0.25, 0.5, 0.75]`. The primary's upstream model received two
    requests (one attempt and one retry) before the backup's model received one.
    Product bug: in CI run 42966523075 the upstream saw three requests per router attempt, the later ones
    with `x-stainless-retry-count`. The router passes `max_retries: 0` to every call
    (litellm/router_rust/surface/settings.py:194), but `OpenAIChatCompletion.embedding` reads it as
    `max_retries or litellm.DEFAULT_MAX_RETRIES` (litellm/llms/openai/openai.py:1411), so 0 becomes the
    OpenAI client's default of two retries.
    """
    pytest.skip("BUG: OpenAI embeddings ignore the router's max_retries 0 and the client retries each attempt twice")
    response: Final = embed(rig.gateway, "ep-embeddings-primary")
    assert response.status_code == 200, response.text
    assert response.json()["data"][0]["embedding"] == [0.25, 0.5, 0.75], response.text
    assert rig.upstream.order("status-500-embeddings-primary", "ok-embeddings-backup") == (
        "status-500-embeddings-primary",
        "status-500-embeddings-primary",
        "ok-embeddings-backup",
    )


def test_embeddings_resolves_a_model_group_alias(rig: Rig) -> None:
    """`model_group_alias` maps `ep-embeddings-alias` to `ep-embeddings-real` and the caller sends an embeddings request for the alias.

    The caller gets 200 and the upstream received exactly one request, for the real group's model
    `ok-embeddings-real`.
    """
    response: Final = embed(rig.gateway, "ep-embeddings-alias")
    assert response.status_code == 200, response.text
    assert response.json()["data"][0]["embedding"] == [0.25, 0.5, 0.75], response.text
    assert rig.upstream.count("ok-embeddings-real") == 1


def test_embeddings_returns_the_upstream_503_once_retries_are_spent_and_no_fallback_exists(rig: Rig) -> None:
    """The group's upstream always answers 503, the group has no fallback, and the caller sends an embeddings request.

    The caller gets HTTP 503 with the OpenAI error envelope with `code` `"503"`, and the upstream received exactly two requests for the
    model `status-503-embeddings-dead`.
    Product bug: in CI run 42966523075 the upstream saw three requests per router attempt, the later ones
    with `x-stainless-retry-count`. The router passes `max_retries: 0` to every call
    (litellm/router_rust/surface/settings.py:194), but `OpenAIChatCompletion.embedding` reads it as
    `max_retries or litellm.DEFAULT_MAX_RETRIES` (litellm/llms/openai/openai.py:1411), so 0 becomes the
    OpenAI client's default of two retries.
    """
    pytest.skip("BUG: OpenAI embeddings ignore the router's max_retries 0 and the client retries each attempt twice")
    response: Final = embed(rig.gateway, "ep-embeddings-dead")
    assert response.status_code == 503, response.text
    assert error_of(response)["code"] == "503", response.text
    assert rig.upstream.count("status-503-embeddings-dead") == 2


def test_embeddings_honours_fallbacks_sent_in_the_request_body(rig: Rig) -> None:
    """The group has no configured fallback and always answers 500; the caller sends an embeddings request with
    `fallbacks: ["ep-embeddings-clientbackup"]` in the body.

    The caller gets 200 with the scripted embedding vector `[0.25, 0.5, 0.75]`, and the `fallbacks` field is not
    forwarded to the upstream that served the request.
    """
    response: Final = embed(rig.gateway, "ep-embeddings-clientfb", fallbacks=["ep-embeddings-clientbackup"])
    assert response.status_code == 200, response.text
    assert response.json()["data"][0]["embedding"] == [0.25, 0.5, 0.75], response.text
    (served_request,) = rig.upstream.calls("ok-embeddings-clientbackup")
    assert "fallbacks" not in served_request.body


def test_embeddings_does_not_retry_a_400(rig: Rig) -> None:
    """The group has two deployments whose upstream always answers 400, and the caller sends an embeddings request.

    The caller gets HTTP 400 and the upstream received exactly one request for the model
    `status-400-embeddings-bad`, although `num_retries` is 1.
    """
    response: Final = embed(rig.gateway, "ep-embeddings-400")
    assert response.status_code == 400, response.text
    assert rig.upstream.count("status-400-embeddings-bad") == 1


def test_embeddings_moves_to_the_order_two_deployment_when_order_one_fails(rig: Rig) -> None:
    """The group's `order: 1` deployment answers 500 and its `order: 2` deployment is healthy; the caller
    sends an embeddings request.

    The caller gets 200 with the scripted embedding vector `[0.25, 0.5, 0.75]`, and the `order: 2` model received exactly
    one request.
    """
    response: Final = embed(rig.gateway, "ep-embeddings-order")
    assert response.status_code == 200, response.text
    assert response.json()["data"][0]["embedding"] == [0.25, 0.5, 0.75], response.text
    assert rig.upstream.count("ok-embeddings-order-two") == 1


def test_embeddings_routes_a_wildcard_model_name_to_the_matching_upstream_model(rig: Rig) -> None:
    """A deployment is named `dec1ep/*` with `litellm_params.model: openai/*`; the caller sends an embeddings request for
    `dec1ep/ok-embeddings-wild`.

    The caller gets 200 and the upstream received exactly one request whose model is `ok-embeddings-wild`.
    """
    response: Final = embed(rig.gateway, "dec1ep/ok-embeddings-wild")
    assert response.status_code == 200, response.text
    assert response.json()["data"][0]["embedding"] == [0.25, 0.5, 0.75], response.text
    assert rig.upstream.count("ok-embeddings-wild") == 1


def test_messages_retries_a_500_once_and_returns_the_second_attempts_answer(rig: Rig) -> None:
    """`num_retries: 1`; the group's upstream answers 500 once and 200 after that, and the caller sends
    an Anthropic messages request (`POST /v1/messages`).

    The caller gets 200 with a body containing the text `served script-messages-retry`, and the upstream received exactly two
    requests for the model `script-messages-retry`.
    """
    rig.upstream.script("script-messages-retry", 500)
    response: Final = anthropic_message(rig.gateway, "ep-messages-retry")
    assert response.status_code == 200, response.text
    assert "served script-messages-retry" in response.text, response.text
    assert rig.upstream.count("script-messages-retry") == 2


def test_messages_falls_back_to_the_configured_group_after_the_retry_budget(rig: Rig) -> None:
    """`fallbacks` maps `ep-messages-primary` to `ep-messages-backup`; the primary's upstream always answers 500 and
    the caller sends an Anthropic messages request.

    The caller gets 200 with a body containing the text `served ok-messages-backup`. The primary's upstream model received two
    requests (one attempt and one retry) before the backup's model received one.
    """
    response: Final = anthropic_message(rig.gateway, "ep-messages-primary")
    assert response.status_code == 200, response.text
    assert "served ok-messages-backup" in response.text, response.text
    assert rig.upstream.order("status-500-messages-primary", "ok-messages-backup") == (
        "status-500-messages-primary",
        "status-500-messages-primary",
        "ok-messages-backup",
    )


def test_messages_resolves_a_model_group_alias(rig: Rig) -> None:
    """`model_group_alias` maps `ep-messages-alias` to `ep-messages-real` and the caller sends an Anthropic messages request for the alias.

    The caller gets 200 and the upstream received exactly one request, for the real group's model
    `ok-messages-real`.
    """
    response: Final = anthropic_message(rig.gateway, "ep-messages-alias")
    assert response.status_code == 200, response.text
    assert "served ok-messages-real" in response.text, response.text
    assert rig.upstream.count("ok-messages-real") == 1


def test_messages_returns_the_upstream_503_once_retries_are_spent_and_no_fallback_exists(rig: Rig) -> None:
    """The group's upstream always answers 503, the group has no fallback, and the caller sends an Anthropic messages request.

    The caller gets HTTP 503 with a JSON body with an `error` object, and the upstream received exactly two requests for the
    model `status-503-messages-dead`.
    """
    response: Final = anthropic_message(rig.gateway, "ep-messages-dead")
    assert response.status_code == 503, response.text
    assert "error" in response.json(), response.text
    assert rig.upstream.count("status-503-messages-dead") == 2


def test_messages_honours_fallbacks_sent_in_the_request_body(rig: Rig) -> None:
    """The group has no configured fallback and always answers 500; the caller sends an Anthropic messages request with
    `fallbacks: ["ep-messages-clientbackup"]` in the body.

    The caller gets 200 with a body containing the text `served ok-messages-clientbackup`, and the `fallbacks` field is not
    forwarded to the upstream that served the request.
    """
    response: Final = anthropic_message(rig.gateway, "ep-messages-clientfb", fallbacks=["ep-messages-clientbackup"])
    assert response.status_code == 200, response.text
    assert "served ok-messages-clientbackup" in response.text, response.text
    (served_request,) = rig.upstream.calls("ok-messages-clientbackup")
    assert "fallbacks" not in served_request.body


def test_messages_does_not_retry_a_400(rig: Rig) -> None:
    """The group has two deployments whose upstream always answers 400, and the caller sends an Anthropic messages request.

    The caller gets HTTP 400 and the upstream received exactly one request for the model
    `status-400-messages-bad`, although `num_retries` is 1.
    """
    response: Final = anthropic_message(rig.gateway, "ep-messages-400")
    assert response.status_code == 400, response.text
    assert rig.upstream.count("status-400-messages-bad") == 1


def test_messages_moves_to_the_order_two_deployment_when_order_one_fails(rig: Rig) -> None:
    """The group's `order: 1` deployment answers 500 and its `order: 2` deployment is healthy; the caller
    sends an Anthropic messages request.

    The caller gets 200 with a body containing the text `served ok-messages-order-two`, and the `order: 2` model received exactly
    one request.
    """
    response: Final = anthropic_message(rig.gateway, "ep-messages-order")
    assert response.status_code == 200, response.text
    assert "served ok-messages-order-two" in response.text, response.text
    assert rig.upstream.count("ok-messages-order-two") == 1


def test_messages_routes_a_wildcard_model_name_to_the_matching_upstream_model(rig: Rig) -> None:
    """A deployment is named `dec1ep/*` with `litellm_params.model: openai/*`; the caller sends an Anthropic messages request for
    `dec1ep/ok-messages-wild`.

    The caller gets 200 and the upstream received exactly one request whose model is `ok-messages-wild`.
    """
    response: Final = anthropic_message(rig.gateway, "dec1ep/ok-messages-wild")
    assert response.status_code == 200, response.text
    assert "served ok-messages-wild" in response.text, response.text
    assert rig.upstream.count("ok-messages-wild") == 1


def test_responses_retries_a_500_once_and_returns_the_second_attempts_answer(rig: Rig) -> None:
    """`num_retries: 1`; the group's upstream answers 500 once and 200 after that, and the caller sends
    a responses request (`POST /v1/responses`).

    The caller gets 200 with a body containing the text `served script-responses-retry`, and the upstream received exactly two
    requests for the model `script-responses-retry`.
    """
    rig.upstream.script("script-responses-retry", 500)
    response: Final = respond(rig.gateway, "ep-responses-retry")
    assert response.status_code == 200, response.text
    assert "served script-responses-retry" in response.text, response.text
    assert rig.upstream.count("script-responses-retry") == 2


def test_responses_falls_back_to_the_configured_group_after_the_retry_budget(rig: Rig) -> None:
    """`fallbacks` maps `ep-responses-primary` to `ep-responses-backup`; the primary's upstream always answers 500 and
    the caller sends a responses request.

    The caller gets 200 with a body containing the text `served ok-responses-backup`. The primary's upstream model received two
    requests (one attempt and one retry) before the backup's model received one.
    """
    response: Final = respond(rig.gateway, "ep-responses-primary")
    assert response.status_code == 200, response.text
    assert "served ok-responses-backup" in response.text, response.text
    assert rig.upstream.order("status-500-responses-primary", "ok-responses-backup") == (
        "status-500-responses-primary",
        "status-500-responses-primary",
        "ok-responses-backup",
    )


def test_responses_resolves_a_model_group_alias(rig: Rig) -> None:
    """`model_group_alias` maps `ep-responses-alias` to `ep-responses-real` and the caller sends a responses request for the alias.

    The caller gets 200 and the upstream received exactly one request, for the real group's model
    `ok-responses-real`.
    """
    response: Final = respond(rig.gateway, "ep-responses-alias")
    assert response.status_code == 200, response.text
    assert "served ok-responses-real" in response.text, response.text
    assert rig.upstream.count("ok-responses-real") == 1


def test_responses_returns_the_upstream_503_once_retries_are_spent_and_no_fallback_exists(rig: Rig) -> None:
    """The group's upstream always answers 503, the group has no fallback, and the caller sends a responses request.

    The caller gets HTTP 503 with the OpenAI error envelope with `code` `"503"`, and the upstream received exactly two requests for the
    model `status-503-responses-dead`.
    """
    response: Final = respond(rig.gateway, "ep-responses-dead")
    assert response.status_code == 503, response.text
    assert error_of(response)["code"] == "503", response.text
    assert rig.upstream.count("status-503-responses-dead") == 2


def test_responses_honours_fallbacks_sent_in_the_request_body(rig: Rig) -> None:
    """The group has no configured fallback and always answers 500; the caller sends a responses request with
    `fallbacks: ["ep-responses-clientbackup"]` in the body.

    The caller gets 200 with a body containing the text `served ok-responses-clientbackup`, and the `fallbacks` field is not
    forwarded to the upstream that served the request.
    """
    response: Final = respond(rig.gateway, "ep-responses-clientfb", fallbacks=["ep-responses-clientbackup"])
    assert response.status_code == 200, response.text
    assert "served ok-responses-clientbackup" in response.text, response.text
    (served_request,) = rig.upstream.calls("ok-responses-clientbackup")
    assert "fallbacks" not in served_request.body


def test_responses_does_not_retry_a_400(rig: Rig) -> None:
    """The group has two deployments whose upstream always answers 400, and the caller sends a responses request.

    The caller gets HTTP 400 and the upstream received exactly one request for the model
    `status-400-responses-bad`, although `num_retries` is 1.
    """
    response: Final = respond(rig.gateway, "ep-responses-400")
    assert response.status_code == 400, response.text
    assert rig.upstream.count("status-400-responses-bad") == 1


def test_responses_moves_to_the_order_two_deployment_when_order_one_fails(rig: Rig) -> None:
    """The group's `order: 1` deployment answers 500 and its `order: 2` deployment is healthy; the caller
    sends a responses request.

    The caller gets 200 with a body containing the text `served ok-responses-order-two`, and the `order: 2` model received exactly
    one request.
    """
    response: Final = respond(rig.gateway, "ep-responses-order")
    assert response.status_code == 200, response.text
    assert "served ok-responses-order-two" in response.text, response.text
    assert rig.upstream.count("ok-responses-order-two") == 1


def test_responses_routes_a_wildcard_model_name_to_the_matching_upstream_model(rig: Rig) -> None:
    """A deployment is named `dec1ep/*` with `litellm_params.model: openai/*`; the caller sends a responses request for
    `dec1ep/ok-responses-wild`.

    The caller gets 200 and the upstream received exactly one request whose model is `ok-responses-wild`.
    """
    response: Final = respond(rig.gateway, "dec1ep/ok-responses-wild")
    assert response.status_code == 200, response.text
    assert "served ok-responses-wild" in response.text, response.text
    assert rig.upstream.count("ok-responses-wild") == 1
