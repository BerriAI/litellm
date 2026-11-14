"""Serving the proxy under a path prefix with `SERVER_ROOT_PATH`.

Setting the environment variable `SERVER_ROOT_PATH=/dec1-root` makes the application's root path `/dec1-root`
(`litellm/proxy/proxy_server.py:2168`). Every route is then reachable as `/dec1-root/<route>`, auth and routing
decide on the route with the prefix removed (`get_request_route`, `litellm/proxy/auth/auth_utils.py:948`), and the
OpenAPI document advertises the prefix. A gateway in front of the proxy must pass the path through byte for byte:
it must not strip the prefix, add it a second time, or normalise it. One proxy is booted with the prefix for this
module. `tests/integration/configuration/test_openapi_docs_route.py` already pins the OpenAPI `servers` entry.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import list_value, object_value
from integration._support.process import graceful_stop_seconds
from integration.compatibility.dec1_http_support import (
    CHAT,
    OWNED_CHAT_MODEL,
    OWNED_EMBEDDING_MODEL,
    OwnedDoor,
    chat_body,
    owned_front_door,
    path_of,
)

ROOT: Final = "/dec1-root"

pytestmark: Final = [
    pytest.mark.timeout(2 * graceful_stop_seconds() + 120),
    pytest.mark.xdist_group("dec1-http-root-path"),
]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedDoor]:
    directory: Final = tmp_path_factory.mktemp("dec1-http-root-path")
    environment: Final = {"SERVER_ROOT_PATH": ROOT, "LITELLM_UI_PATH": str(directory / "ui")}
    with owned_front_door(directory, environment=environment) as door:
        yield door


def test_the_liveliness_probe_answers_under_the_prefix(rig: OwnedDoor) -> None:
    """The proxy runs with `SERVER_ROOT_PATH=/dec1-root`. `GET /dec1-root/health/liveliness` without a key must
    answer 200 with the JSON string `"I'm alive!"`, the same as the unprefixed probe on a default proxy."""
    response: Final = rig.gateway.client.get(f"{ROOT}/health/liveliness")
    assert response.status_code == 200
    assert response.json() == "I'm alive!"


def test_a_chat_call_under_the_prefix_reaches_the_provider_without_the_prefix(rig: OwnedDoor) -> None:
    """The master key posts a chat call to `/dec1-root/v1/chat/completions`. The call must succeed with 200 and
    the provider must receive it on `/v1/chat/completions`: the prefix belongs to the proxy's public address and
    must never leak into the provider URL."""
    rig.upstream.drain()
    response: Final = rig.gateway.request("POST", f"{ROOT}{CHAT}", chat_body(OWNED_CHAT_MODEL))
    assert response.status_code == 200, response.text
    (call,) = rig.calls()
    assert path_of(call) == "/v1/chat/completions"


def test_an_embeddings_call_under_the_prefix_without_v1_is_served(rig: OwnedDoor) -> None:
    """The master key posts to `/dec1-root/embeddings`, the prefix combined with the route alias that has no
    `/v1`. The call must succeed with 200 and return one embedding: aliases keep working under a prefix."""
    response: Final = rig.gateway.request(
        "POST", f"{ROOT}/embeddings", {"model": OWNED_EMBEDDING_MODEL, "input": f"dec1 {uuid.uuid4().hex}"}
    )
    assert response.status_code == 200, response.text
    assert len(list_value(object_value(response.json())["data"])) == 1


def test_a_missing_key_under_the_prefix_is_the_standard_401(rig: OwnedDoor) -> None:
    """A chat call is posted to `/dec1-root/v1/chat/completions` with no key. The proxy must answer 401 with the
    OpenAI error envelope and `type` equal to `auth_error`: auth recognises the route after removing the prefix,
    so the prefixed LLM route is neither public nor unknown."""
    response: Final = rig.gateway.client.post(f"{ROOT}{CHAT}", json=chat_body(OWNED_CHAT_MODEL))
    assert response.status_code == 401, response.text
    assert object_value(object_value(response.json())["error"])["type"] == "auth_error"


def test_a_virtual_key_limited_to_a_model_is_enforced_under_the_prefix(rig: OwnedDoor) -> None:
    """A virtual key that may only call the embedding model posts a chat call for the chat model to
    `/dec1-root/v1/chat/completions`. The proxy must refuse it with 401 or 403 and the provider must receive
    nothing: model access checks depend on recognising the prefixed path as the chat route. UNVERIFIED: the
    refusal status for a model outside the key's list is assumed to be 401 (`key_model_access_denied`)."""
    rig.upstream.drain()
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(models=[OWNED_EMBEDDING_MODEL])
        response: Final = rig.gateway.request("POST", f"{ROOT}{CHAT}", chat_body(OWNED_CHAT_MODEL), key=key)
    assert response.status_code in (401, 403), response.text
    assert rig.calls() == ()


def test_an_unknown_path_under_the_prefix_is_a_404_detail(rig: OwnedDoor) -> None:
    """The master key GETs `/dec1-root/dec1-missing-<id>`. The proxy must answer 404 with exactly
    `{"detail": "Not Found"}`, the same shape as on a proxy without a prefix."""
    response: Final = rig.gateway.request("GET", f"{ROOT}/dec1-missing-{uuid.uuid4().hex}")
    assert (response.status_code, response.json()) == (404, {"detail": "Not Found"})


def test_the_models_list_under_the_prefix_names_the_configured_models(rig: OwnedDoor) -> None:
    """The master key GETs `/dec1-root/v1/models`. The answer must be 200 and its `data` must include the ids
    `dec1-http-chat` and `dec1-http-embedding` from this proxy's config."""
    response: Final = rig.gateway.request("GET", f"{ROOT}/v1/models")
    assert response.status_code == 200, response.text
    ids: Final = {object_value(entry)["id"] for entry in list_value(object_value(response.json())["data"])}
    assert {OWNED_CHAT_MODEL, OWNED_EMBEDDING_MODEL} <= ids


def test_the_unprefixed_chat_route_is_still_served(rig: OwnedDoor) -> None:
    """The master key posts a chat call to `/v1/chat/completions` with no prefix on the proxy that has
    `SERVER_ROOT_PATH=/dec1-root`. The call is served with 200. UNVERIFIED: the web framework matches a route
    with or without the root path, and the harness itself relies on unprefixed health probes to boot this proxy.
    Pinned because a gateway that routes by prefix must know both spellings are live."""
    response: Final = rig.gateway.request("POST", CHAT, chat_body(OWNED_CHAT_MODEL))
    assert response.status_code == 200, response.text


def test_a_trailing_slash_redirect_under_the_prefix_keeps_the_prefix(rig: OwnedDoor) -> None:
    """The master key posts to `/dec1-root/v1/chat/completions/` with a trailing slash and does not follow
    redirects. The proxy must answer 307 with a `Location` whose path is `/dec1-root/v1/chat/completions`.
    UNVERIFIED: assumes the slash redirect is computed from the full request path, prefix included. A redirect
    that dropped the prefix would send clients to a path their load balancer does not route."""
    response: Final = rig.gateway.client.post(
        f"{ROOT}{CHAT}/",
        json=chat_body(OWNED_CHAT_MODEL),
        headers={"Authorization": f"Bearer {rig.gateway.key}"},
        follow_redirects=False,
    )
    assert response.status_code == 307, response.text
    assert response.headers["location"].split("?", 1)[0].endswith(f"{ROOT}{CHAT}")
