"""Azure OpenAI Assistants and threads, still routed through `assistant_settings` with `custom_llm_provider: azure`.

The proxy's `/v1/assistants` and `/v1/threads` routes call `llm_router.a*` helpers that read
`assistant_settings` from the config (`proxy_server.py`, `litellm/assistants/main.py`). With
`custom_llm_provider: azure` they use the AzureOpenAI SDK (`litellm/llms/azure/assistants.py`), so Azure sees the
OpenAI Assistants paths under `/openai/` with `?api-version=<v>` and the key in `api-key`. The existing
`test_assistants_route_errors_wire.py` covers error mapping on the OpenAI provider and a credential-less Azure
proxy; this file pins the successful Azure wire shape.

One owned proxy per module, whose `assistant_settings.litellm_params.api_base` is a fixed-port owned peer.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit

import pytest
from integration._support.client import gateway_from_environment
from integration._support.ports import allocate_port
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._assistants_route_errors_support import owned_config
from integration.providers.dec1_azure_support import API_KEY, json_reply, query_of

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 120)]

ASSISTANTS_API_VERSION: Final = "2024-05-01-preview"


def _respond(request: Request) -> Reply:
    path: Final = urlsplit(request.target).path
    if path.endswith("/assistants") and request.method == "GET":
        return json_reply({"object": "list", "data": [], "first_id": None, "last_id": None, "has_more": False})
    if path.endswith("/assistants"):
        return json_reply(
            {"id": "asst_dec1", "object": "assistant", "created_at": 1, "model": "gpt-4o-dec1", "tools": []}
        )
    if path.endswith("/threads"):
        return json_reply({"id": "thread_dec1", "object": "thread", "created_at": 1, "metadata": {}})
    if path.endswith("/messages") and request.method == "POST":
        return json_reply(
            {
                "id": "msg_dec1",
                "object": "thread.message",
                "created_at": 1,
                "thread_id": "thread_dec1",
                "role": "user",
                "content": [{"type": "text", "text": {"value": "hi", "annotations": []}}],
                "attachments": [],
                "metadata": {},
            }
        )
    if path.endswith("/runs"):
        return json_reply(
            {
                "id": "run_dec1",
                "object": "thread.run",
                "created_at": 1,
                "thread_id": "thread_dec1",
                "assistant_id": "asst_dec1",
                "status": "queued",
                "model": "gpt-4o-dec1",
                "instructions": "",
                "tools": [],
            }
        )
    return json_reply({"object": "list", "data": []})


@dataclass(frozen=True, slots=True)
class _Rig:
    owned: OwnedProxy
    wire: Wire


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("dec1-azure3-assistants")
    port: Final = allocate_port()
    settings: Final = {
        "custom_llm_provider": "azure",
        "litellm_params": {
            "api_base": f"http://127.0.0.1:{port}",
            "api_key": API_KEY,
            "api_version": ASSISTANTS_API_VERSION,
            "max_retries": 0,
        },
    }
    with (
        wire_server(_respond, port=port) as wire,
        gateway_from_environment() as shared,
        owned_proxy_process(shared, directory, {}, config=owned_config(directory, (), settings), workers=1) as owned,
    ):
        yield _Rig(owned, wire)


def _calls(rig: _Rig) -> list[Request]:
    return [r for r in rig.wire.drain() if "/openai/" in urlsplit(r.target).path]


def test_listing_assistants_gets_openai_assistants_with_the_api_version_and_key(rig: _Rig) -> None:
    """The caller lists `GET /v1/assistants`.

    Azure must get one `GET /openai/assistants?api-version=2024-05-01-preview` with `api-key:
    synthetic-azure-key`, and the caller gets 200 with `object: list`.
    """
    rig.wire.drain()
    response: Final = rig.owned.gateway.request("GET", "/v1/assistants")
    assert response.status_code == 200, response.text
    assert response.json()["object"] == "list"
    (call,) = _calls(rig)
    assert (call.method, urlsplit(call.target).path, query_of(call)) == (
        "GET",
        "/openai/assistants",
        {"api-version": ASSISTANTS_API_VERSION},
    )
    assert call.headers.get("api-key") == API_KEY


def test_creating_an_assistant_posts_the_callers_fields(rig: _Rig) -> None:
    """The caller creates an assistant with `model: gpt-4o-dec1`, `name: <marker>` and `instructions: be brief`.

    Azure must get `POST /openai/assistants` whose JSON body carries those three values unchanged; the model is
    the Azure deployment name the caller chose, not rewritten by the proxy.
    """
    rig.wire.drain()
    name: Final = f"dec1-{uuid.uuid4().hex}"
    response: Final = rig.owned.gateway.request(
        "POST", "/v1/assistants", {"model": "gpt-4o-dec1", "name": name, "instructions": "be brief"}
    )
    assert response.status_code == 200, response.text
    (call,) = _calls(rig)
    body: Final = json.loads(call.body)
    assert (urlsplit(call.target).path, body["model"], body["name"], body["instructions"]) == (
        "/openai/assistants",
        "gpt-4o-dec1",
        name,
        "be brief",
    )


def test_creating_a_thread_posts_openai_threads(rig: _Rig) -> None:
    """The caller creates `POST /v1/threads` with one user message.

    Azure must get `POST /openai/threads?api-version=2024-05-01-preview` whose body `messages[0].content` is the
    caller's text, and the caller gets the thread id `thread_dec1`.
    """
    rig.wire.drain()
    response: Final = rig.owned.gateway.request(
        "POST", "/v1/threads", {"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 200, response.text
    assert response.json()["id"] == "thread_dec1"
    (call,) = _calls(rig)
    assert (urlsplit(call.target).path, query_of(call)) == ("/openai/threads", {"api-version": ASSISTANTS_API_VERSION})
    assert json.loads(call.body)["messages"][0]["content"] == "hi"


def test_adding_a_message_posts_to_the_threads_messages_route(rig: _Rig) -> None:
    """The caller adds `POST /v1/threads/thread_dec1/messages` with `role: user`, `content: hi`.

    Azure must get `POST /openai/threads/thread_dec1/messages?api-version=2024-05-01-preview` with that body.
    """
    rig.wire.drain()
    response: Final = rig.owned.gateway.request(
        "POST", "/v1/threads/thread_dec1/messages", {"role": "user", "content": "hi"}
    )
    assert response.status_code == 200, response.text
    (call,) = _calls(rig)
    assert urlsplit(call.target).path == "/openai/threads/thread_dec1/messages"
    assert {k: json.loads(call.body)[k] for k in ("role", "content")} == {"role": "user", "content": "hi"}


def test_running_a_thread_posts_the_assistant_id(rig: _Rig) -> None:
    """The caller runs `POST /v1/threads/thread_dec1/runs` with `assistant_id: asst_dec1`.

    Azure must get `POST /openai/threads/thread_dec1/runs?api-version=2024-05-01-preview` whose body has
    `assistant_id: asst_dec1`, and the caller gets `status: queued`.
    """
    rig.wire.drain()
    response: Final = rig.owned.gateway.request("POST", "/v1/threads/thread_dec1/runs", {"assistant_id": "asst_dec1"})
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "queued"
    (call,) = _calls(rig)
    assert (urlsplit(call.target).path, json.loads(call.body)["assistant_id"]) == (
        "/openai/threads/thread_dec1/runs",
        "asst_dec1",
    )


def test_assistants_calls_carry_the_assistants_v2_beta_header(rig: _Rig) -> None:
    """Any Assistants call through the SDK.

    The OpenAI SDK's beta resources send `OpenAI-Beta: assistants=v2`, which Azure needs to pick the v2 Assistants
    API, so the upstream request must carry it. UNVERIFIED: read from the pinned OpenAI SDK's beta resources,
    not from litellm.
    """
    rig.wire.drain()
    rig.owned.gateway.request("GET", "/v1/assistants")
    (call,) = _calls(rig)
    assert call.headers.get("openai-beta") == "assistants=v2"


def test_the_callers_virtual_key_never_reaches_azure_assistants(rig: _Rig) -> None:
    """The caller authenticates with the proxy master key.

    No upstream header may contain that key: Azure sees only the configured `api-key`, and no `Authorization`
    header carrying the caller's credential.
    """
    rig.wire.drain()
    rig.owned.gateway.request("GET", "/v1/assistants")
    (call,) = _calls(rig)
    assert all(rig.owned.gateway.key not in value for value in call.headers.values())
