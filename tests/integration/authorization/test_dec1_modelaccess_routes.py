"""Model access on every inference surface, and the exact refusal each surface answers.

Contract:

* Every OpenAI shaped route answers a model denial with HTTP 403 and exactly
  `{"error": {"message": <fixed text naming the model>, "type": "<layer>_model_access_denied", "param":
  "model", "code": "403"}}` (litellm/proxy/auth/model_access_denied.py, `ProxyErrorTypes`).
* The Anthropic routes (`/v1/messages`, `/v1/messages/count_tokens`) answer the same denial in the Anthropic
  envelope: `{"type": "error", "error": {"type": "permission_error", "message": <same text>,
  "litellm_error_type": "<layer>_model_access_denied"}}`
  (litellm/proxy/common_utils/anthropic_error_payload.py `anthropic_error_content`, `_with_litellm_error_type`).
* The model is read from the body `model`, from the path on Google and Azure style routes, and on file and
  batch routes also from the `x-litellm-model` header, the `model` query parameter and `target_model_names`
  (litellm/proxy/auth/auth_utils.py `get_model_from_request`, `_extract_model_candidates_from_request`).
* An end user (customer) list answers its own message and `customer_model_access_denied`.

`/v1/messages` is moving to Rust on more providers this month, and the auth front door after it, so the
envelope and the model source per route are pinned here. Existing files already pin chat, completions,
embeddings, responses and moderations for the key layer only (`test_dec1_keys_model_access.py`); this file
adds the team and user layers, the Anthropic envelope and the other surfaces.
"""

from __future__ import annotations

import uuid
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway
from tests.integration.authorization.dec1_modelaccess_support import (
    CUSTOMER_DENIED,
    CUSTOMER_MODEL_DENIED_TEMPLATE,
    KEY_DENIED,
    MODEL_DENIED_TEMPLATE,
    TEAM_DENIED,
    USER_DENIED,
    assert_model_denied,
    assert_served,
    body_of,
    chat,
    marker,
    post_as,
    upstream_requests,
)


def _anthropic_denial(model: str, error_type: str) -> dict[str, JsonValue]:
    return {
        "type": "error",
        "error": {
            "type": "permission_error",
            "message": MODEL_DENIED_TEMPLATE.format(model=model),
            "litellm_error_type": error_type,
        },
    }


def _messages_body(model: str, text: str | None = None) -> dict[str, JsonValue]:
    return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": text or marker()}]}


def test_a_key_denial_on_v1_messages_is_the_exact_anthropic_envelope(gateway: Gateway) -> None:
    """A key limited to model A posts `/v1/messages` for model B.

    The answer must be 403 with exactly the Anthropic error body: outer `type: "error"`, inner
    `type: "permission_error"`, the fixed denial message naming B, and `litellm_error_type:
    "key_model_access_denied"`. The upstream must not be called. UNVERIFIED: assumes no `request_id` or
    `litellm_call_id` field is added when the caller sends no `x-request-id` and the call id setting is off.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        text: Final = marker()
        refused: Final = post_as(gateway, key, "/v1/messages", _messages_body(other, text))
        assert refused.status_code == 403, refused.text
        assert body_of(refused) == _anthropic_denial(other, KEY_DENIED), refused.text
        assert upstream_requests(gateway, text) == []


def test_a_team_denial_on_v1_messages_carries_the_team_error_type(gateway: Gateway) -> None:
    """A team allows A; its key (no list) posts `/v1/messages` for B.

    The answer must be 403 with the Anthropic body whose `litellm_error_type` is `team_model_access_denied`.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        team: Final = scenario.team(models=[allowed])
        key: Final = scenario.key(team_id=team)
        refused: Final = post_as(gateway, key, "/v1/messages", _messages_body(other))
        assert refused.status_code == 403, refused.text
        assert body_of(refused) == _anthropic_denial(other, TEAM_DENIED), refused.text


def test_a_user_denial_on_v1_messages_carries_the_user_error_type(gateway: Gateway) -> None:
    """A user allows A; their personal key posts `/v1/messages` for B.

    The answer must be 403 with the Anthropic body whose `litellm_error_type` is `user_model_access_denied`.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        user: Final = scenario.user(models=[allowed])
        key: Final = scenario.key(user_id=user)
        refused: Final = post_as(gateway, key, "/v1/messages", _messages_body(other))
        assert refused.status_code == 403, refused.text
        assert body_of(refused) == _anthropic_denial(other, USER_DENIED), refused.text


def test_an_allowed_model_on_v1_messages_is_served(gateway: Gateway) -> None:
    """A key limited to model A posts `/v1/messages` for A (an OpenAI shaped deployment on the scripted upstream).

    The answer must be 200 with an Anthropic `message` body (`type: "message"`), proving the denial above is
    about the model and not the route. The deployment is `hosted_vllm`, not `openai`: `/v1/messages` sends an
    `openai` deployment to the Responses API (`_should_route_to_responses_api`,
    litellm/llms/anthropic/messages/native/handler.py:881-885), and the scripted upstream has no `/v1/responses`
    scenario, while `hosted_vllm` is bridged to `/v1/chat/completions`, which it answers.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model(model="hosted_vllm/gpt-4o-mini")
        key: Final = scenario.key(models=[allowed])
        served: Final = post_as(gateway, key, "/v1/messages", _messages_body(allowed))
        assert served.status_code == 200, served.text
        assert body_of(served)["type"] == "message", served.text


def test_count_tokens_checks_the_body_model_like_messages(gateway: Gateway) -> None:
    """A key limited to model A posts `/v1/messages/count_tokens` for model B.

    The answer must be 403 with the Anthropic denial body for `key_model_access_denied`: counting tokens
    names a model and is subject to the same list. UNVERIFIED: assumed from `ANTHROPIC_MESSAGES_ROUTES`
    containing the route and the body `model` being read for every LLM API route.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        refused: Final = post_as(
            gateway,
            key,
            "/v1/messages/count_tokens",
            {"model": other, "messages": [{"role": "user", "content": marker()}]},
        )
        assert refused.status_code == 403, refused.text
        assert body_of(refused) == _anthropic_denial(other, KEY_DENIED), refused.text


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/responses", {"input": "dec1 model access"}),
        ("/v1/embeddings", {"input": "dec1 model access"}),
        ("/embeddings", {"input": "dec1 model access"}),
        ("/v1/completions", {"prompt": "dec1 model access"}),
    ],
)
def test_a_team_denial_is_the_exact_openai_body_on_every_openai_route(
    gateway: Gateway, path: str, payload: dict[str, JsonValue]
) -> None:
    """A team allows A; its key (no list) posts each OpenAI shaped route for B.

    Each answer must be 403 with exactly the OpenAI denial body of type `team_model_access_denied`.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        team: Final = scenario.team(models=[allowed])
        key: Final = scenario.key(team_id=team)
        assert_model_denied(post_as(gateway, key, path, {"model": other, **payload}), other, TEAM_DENIED)


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/rerank", {"query": "dec1", "documents": ["a", "b"]}),
        ("/v1/images/generations", {"prompt": "dec1 model access"}),
    ],
)
def test_a_key_denial_is_the_exact_openai_body_on_other_inference_routes(
    gateway: Gateway, path: str, payload: dict[str, JsonValue]
) -> None:
    """A key limited to A posts rerank and image generation for B.

    Each answer must be 403 with exactly the OpenAI denial body of type `key_model_access_denied`: the
    check runs at auth before any route-specific validation.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        assert_model_denied(post_as(gateway, key, path, {"model": other, **payload}), other, KEY_DENIED)


def test_the_google_generate_content_route_checks_the_model_in_the_path(gateway: Gateway) -> None:
    """A key limited to A posts `/v1beta/models/<B>:generateContent` with a body that names no model.

    The answer must be 403 with exactly the OpenAI denial body naming B: on this route the model comes from
    the path (`_HANDLER_MODEL_ROUTES`). UNVERIFIED: the envelope is assumed to be the OpenAI one, since the
    route is not in the Anthropic set.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        refused: Final = post_as(
            gateway,
            key,
            f"/v1beta/models/{other}:generateContent",
            {"contents": [{"role": "user", "parts": [{"text": marker()}]}]},
        )
        assert_model_denied(refused, other, KEY_DENIED)


def test_a_body_model_cannot_override_the_google_path_model_for_the_check(gateway: Gateway) -> None:
    """A key limited to A posts `/v1beta/models/<B>:generateContent` with body `model: <A>`.

    The answer must still be 403 naming B: the path model is what the handler serves, so the body cannot
    be used to pass the check for another model. UNVERIFIED: `resolve_inference_model` decides between path
    and body; pinned from `_handler_selected_model` returning the path model when the body names another.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        refused: Final = post_as(
            gateway,
            key,
            f"/v1beta/models/{other}:generateContent",
            {"model": allowed, "contents": [{"role": "user", "parts": [{"text": marker()}]}]},
        )
        assert refused.status_code == 403, refused.text


def test_a_batch_listing_routed_by_the_model_header_checks_that_model(gateway: Gateway) -> None:
    """A key limited to A calls `GET /v1/batches` with header `x-litellm-model: <B>`.

    The answer must be 403 with exactly the OpenAI denial body naming B: on file and batch routes the
    routing header is a model source (`MODEL_ROUTING_HEADER_NAME`).
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        refused: Final = gateway.request("GET", "/v1/batches", key=key, headers={"x-litellm-model": other})
        assert_model_denied(refused, other, KEY_DENIED)


def test_a_batch_listing_routed_by_the_model_query_parameter_checks_that_model(gateway: Gateway) -> None:
    """A key limited to A calls `GET /v1/batches?model=<B>`.

    The answer must be 403 with exactly the OpenAI denial body naming B.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        refused: Final = gateway.request("GET", "/v1/batches", key=key, params={"model": other})
        assert_model_denied(refused, other, KEY_DENIED)


def test_a_file_listing_with_a_query_target_model_checks_that_model(gateway: Gateway) -> None:
    """A key limited to A calls `GET /v1/files?target_model_names=<B>`.

    The answer must be 403 naming B with type `key_model_access_denied`: `/files` reads target models from
    the query too. UNVERIFIED: a single name is assumed to be passed through as a string candidate.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        refused: Final = gateway.request("GET", "/v1/files", key=key, params={"target_model_names": other})
        assert_model_denied(refused, other, KEY_DENIED)


def test_the_model_header_is_not_a_model_source_on_chat_completions(gateway: Gateway) -> None:
    """A key limited to A posts a chat completion for A with header `x-litellm-model: <B>`.

    The request must be served with 200: the routing header only counts on file, batch, skill, eval and
    interaction routes, so it neither denies nor reroutes a chat call.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        served: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": allowed, "messages": [{"role": "user", "content": marker()}]},
            key=key,
            headers={"x-litellm-model": other},
        )
        assert_served(served)


def test_a_customer_list_denial_has_its_own_message_and_type(gateway: Gateway) -> None:
    """A customer is created with `models: [A]`; an unrestricted key calls B with `user: <customer>`.

    The answer must be 403 with exactly `{"error": {"message": "The requested model '<B>' is not in the
    allowed models for this customer. Check the models this customer can use and try again.", "type":
    "customer_model_access_denied", "param": "model", "code": "403"}}`.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        customer: Final = f"dec1-ma-customer-{uuid.uuid4().hex}"
        gateway.post("/customer/new", {"user_id": customer, "models": [allowed]})
        scenario.cleanups.callback(gateway.post, "/customer/delete", {"user_ids": [customer]})
        key: Final = scenario.key()
        refused: Final = chat(gateway, key, other, user=customer)
        assert refused.status_code == 403, refused.text
        assert body_of(refused) == {
            "error": {
                "message": CUSTOMER_MODEL_DENIED_TEMPLATE.format(model=other),
                "type": CUSTOMER_DENIED,
                "param": "model",
                "code": "403",
            }
        }, refused.text


def test_a_model_list_in_the_body_is_checked_name_by_name(gateway: Gateway) -> None:
    """A key limited to A posts a chat completion whose `model` is the list `[A, B]`.

    The answer must be 403 `key_model_access_denied` naming B: a list model is checked element by element
    (`_can_object_call_model` recurses over lists), so one disallowed entry refuses the whole request.
    UNVERIFIED: the chat route may reject a list `model` with 400 before auth; pin whichever it is.
    """
    with gateway.scenario() as scenario:
        allowed: Final = scenario.model()
        other: Final = scenario.model()
        key: Final = scenario.key(models=[allowed])
        refused: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": [allowed, other], "messages": [{"role": "user", "content": marker()}]},
            key=key,
        )
        assert refused.status_code in (400, 403), refused.text
        if refused.status_code == 403:
            assert_model_denied(refused, other, KEY_DENIED)


def test_a_route_without_a_model_is_not_subject_to_model_lists(gateway: Gateway) -> None:
    """A key limited to one model calls `GET /v1/batches` with no model header, query or body.

    The request names no model, so it must not be refused with a model access denial (any other status is
    acceptable here). A port that treats "no model" as "denied" would break batch and file polling.
    """
    with gateway.scenario() as scenario:
        key: Final = scenario.key(models=[scenario.model()])
        response: Final = gateway.request("GET", "/v1/batches", key=key)
        assert response.status_code != 403 or "model_access_denied" not in response.text, response.text
