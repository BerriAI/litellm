"""Contract: OpenAI resource APIs that need process-level settings: Conversations and stored chat completions through
the OpenAI passthrough, and fine-tuning jobs through `finetune_settings`.

One owned proxy per module runs with `OPENAI_API_BASE` and `OPENAI_API_KEY` pointing at a local peer and a config
whose `finetune_settings` and `files_settings` also point at the peer (with `organization` set on fine-tuning).

Conversations (`/v1/conversations`) and stored chat completions (`GET/POST/DELETE /v1/chat/completions/{id}`,
`/messages`) have no native proxy route, so the only way through the proxy is `/openai/{endpoint:path}`
(`llm_passthrough_endpoints.py:3683`). That route sends the method, path, query and body to
`OPENAI_API_BASE` with the environment key as `authorization: Bearer` and `api-key`
(`BaseOpenAIPassThroughHandler._assemble_headers`, `:4201`), never the caller's virtual key.

Fine-tuning (`fine_tuning_endpoints/endpoints.py`) routes by `custom_llm_provider` to the matching
`finetune_settings` entry: create strips an `openai/` model prefix and sends only `model`, `training_file`,
`hyperparameters`, `suffix`, `validation_file`, `integrations` and `seed` (`litellm/fine_tuning/main.py:128`); list
and retrieve without a provider answer 400, create without one answers 500 (pinned as a BUG skip). The existing
`test_openai_fine_tuning.py` pins the basic create, list and cancel round trip and is not repeated.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from typing import Final

import httpx
import pytest
from integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment, list_value, object_value
from integration._support.process import owned_proxy
from integration.providers.dec1_openai2_support import (
    ENV_OPENAI_KEY,
    FINETUNE_KEY,
    FINETUNE_ORG,
    calls_to,
    owned_config,
    resource_respond,
)
from integration.providers.dec1_openai_support import Call, Peer, peer
from pydantic import JsonValue


@pytest.fixture(scope="module")
def provider() -> Iterator[Peer]:
    with peer(resource_respond) as owned:
        yield owned


@pytest.fixture(scope="module")
def isolated(provider: Peer, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("dec1-openai2-resources")
    with (
        gateway_from_environment() as shared,
        owned_proxy(
            shared,
            directory,
            {"OPENAI_API_BASE": provider.url, "OPENAI_API_KEY": ENV_OPENAI_KEY},
            config=owned_config(directory, f"{provider.url}/v1", organization=FINETUNE_ORG),
            remove_environment=tuple(
                name for name in os.environ if name in ("OPENAI_BASE_URL", "OPENAI_ORGANIZATION", "OPENAI_PROJECT")
            ),
        ) as candidate,
    ):
        yield candidate


def _new_calls(provider: Peer, action: Callable[[], object]) -> tuple[Call, ...]:
    before: Final = len(provider.all())
    action()
    return provider.all()[before:]


def _ok(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    return JSON_OBJECT.validate_json(response.content)


def test_conversation_create_reaches_openai_through_the_passthrough(isolated: Gateway, provider: Peer) -> None:
    """The caller posts `{"metadata": {"topic": "dec1"}, "items": [<one user message>]}` to
    `/openai/v1/conversations`.

    The provider receives one `POST /v1/conversations` with exactly that body and
    `authorization: Bearer <OPENAI_API_KEY>`, and the caller gets `object: conversation` with the provider id.
    """
    sent: Final[dict[str, JsonValue]] = {
        "metadata": {"topic": "dec1"},
        "items": [{"type": "message", "role": "user", "content": "dec1 hello"}],
    }
    response: list[dict[str, JsonValue]] = []
    calls: Final = _new_calls(
        provider, lambda: response.append(_ok(isolated.request("POST", "/openai/v1/conversations", sent)))
    )
    (call,) = calls_to(calls, "/conversations")
    assert (call.method, call.path, call.body) == ("POST", "/v1/conversations", sent)
    assert call.headers["authorization"] == f"Bearer {ENV_OPENAI_KEY}"
    assert (response[0]["object"], response[0]["id"]) == ("conversation", "conv_dec1openai2"), response


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/v1/conversations/conv_dec1openai2", None),
        ("POST", "/v1/conversations/conv_dec1openai2", {"metadata": {"topic": "renamed"}}),
        ("DELETE", "/v1/conversations/conv_dec1openai2", None),
        (
            "POST",
            "/v1/conversations/conv_dec1openai2/items",
            {"items": [{"type": "message", "role": "user", "content": "more"}]},
        ),
        ("GET", "/v1/conversations/conv_dec1openai2/items/msg_dec1openai2", None),
        ("DELETE", "/v1/conversations/conv_dec1openai2/items/msg_dec1openai2", None),
    ],
    ids=["retrieve", "update", "delete", "items_create", "item_retrieve", "item_delete"],
)
def test_each_conversation_route_keeps_its_method_path_and_body(
    isolated: Gateway, provider: Peer, method: str, path: str, body: dict[str, JsonValue] | None
) -> None:
    """The caller calls `<method> /openai<path>` with the given body.

    The provider receives exactly one request with the same method, the same path and the same JSON body, and
    the caller gets 200. A rewrite that only knows POST passthrough or re-roots paths would break the item routes.
    """
    calls: Final = _new_calls(provider, lambda: _ok(isolated.request(method, f"/openai{path}", body)))
    (call,) = calls_to(calls, "/conversations")
    assert (call.method, call.path) == (method, path)
    if body is not None:
        assert call.body == body


def test_conversation_item_list_forwards_its_query(isolated: Gateway, provider: Peer) -> None:
    """The caller lists `/openai/v1/conversations/conv_dec1openai2/items?limit=5&order=asc&include=message.output_text.logprobs`.

    The provider receives `GET /v1/conversations/conv_dec1openai2/items` with the same three query values and
    the caller gets a `list` whose first item is the provider's message.
    """
    page: list[dict[str, JsonValue]] = []
    calls: Final = _new_calls(
        provider,
        lambda: page.append(
            _ok(
                isolated.request(
                    "GET",
                    "/openai/v1/conversations/conv_dec1openai2/items",
                    params={"limit": "5", "order": "asc", "include": "message.output_text.logprobs"},
                )
            )
        ),
    )
    (call,) = calls_to(calls, "/conversations")
    assert {key: list(call.query[key]) for key in ("limit", "order", "include")} == {
        "limit": ["5"],
        "order": ["asc"],
        "include": ["message.output_text.logprobs"],
    }, call.query
    assert object_value(list_value(page[0]["data"])[0])["id"] == "msg_dec1openai2", page


def test_the_caller_virtual_key_never_reaches_openai_on_passthrough(isolated: Gateway, provider: Peer) -> None:
    """A conversation retrieve is sent through `/openai/...` with the owned proxy's master key.

    No header the provider receives contains the caller's key; `authorization` and `api-key` carry only the
    `OPENAI_API_KEY` of the proxy environment.
    """
    calls: Final = _new_calls(
        provider, lambda: _ok(isolated.request("GET", "/openai/v1/conversations/conv_dec1openai2"))
    )
    (call,) = calls_to(calls, "/conversations")
    assert all(isolated.key not in value for value in call.headers.values()), dict(call.headers)
    assert (call.headers["authorization"], call.headers["api-key"]) == (f"Bearer {ENV_OPENAI_KEY}", ENV_OPENAI_KEY)


def test_the_native_v1_conversations_path_is_not_served(isolated: Gateway, provider: Peer) -> None:
    """The caller posts to `/v1/conversations` without the `/openai` prefix.

    The proxy answers 404 (no native route) and the provider sees nothing, so a port that adds a native route
    must do it on purpose. UNVERIFIED: no catch-all route answers this path with a different status.
    """
    statuses: list[int] = []
    calls: Final = _new_calls(
        provider,
        lambda: statuses.append(isolated.request("POST", "/v1/conversations", {"metadata": {}}).status_code),
    )
    assert statuses == [404], statuses
    assert not calls_to(calls, "/conversations"), calls


def test_a_responses_call_naming_a_conversation_reaches_openai_with_it(isolated: Gateway, provider: Peer) -> None:
    """A Responses passthrough call `/openai/v1/responses` carries `conversation: "conv_dec1openai2"`.

    The provider body keeps `conversation` and `model` unchanged; the passthrough does not rewrite Responses
    bodies.
    """
    sent: Final[dict[str, JsonValue]] = {"model": "gpt-4o-mini", "input": "dec1", "conversation": "conv_dec1openai2"}
    calls: Final = _new_calls(provider, lambda: _ok(isolated.request("POST", "/openai/v1/responses", sent)))
    (call,) = calls_to(calls, "/responses")
    assert call.body == sent


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/v1/chat/completions/chatcmpl-dec1stored", None),
        ("POST", "/v1/chat/completions/chatcmpl-dec1stored", {"metadata": {"reviewed": "yes"}}),
        ("DELETE", "/v1/chat/completions/chatcmpl-dec1stored", None),
        ("GET", "/v1/chat/completions/chatcmpl-dec1stored/messages", None),
    ],
    ids=["retrieve", "update_metadata", "delete", "messages"],
)
def test_each_stored_chat_completion_route_passes_through(
    isolated: Gateway, provider: Peer, method: str, path: str, body: dict[str, JsonValue] | None
) -> None:
    """The caller calls `<method> /openai<path>` for a stored chat completion.

    The provider receives the same method and path (and the same `metadata` body on update) with the
    environment key, and the caller gets 200. These routes only exist through the passthrough today.
    """
    calls: Final = _new_calls(provider, lambda: _ok(isolated.request(method, f"/openai{path}", body)))
    (call,) = calls_to(calls, "/chat/completions")
    assert (call.method, call.path) == (method, path)
    assert call.headers["authorization"] == f"Bearer {ENV_OPENAI_KEY}"
    if body is not None:
        assert call.body == body


def test_stored_chat_completion_list_forwards_model_and_metadata_filters(isolated: Gateway, provider: Peer) -> None:
    """The caller lists `/openai/v1/chat/completions?model=gpt-4o-mini&metadata[team]=blue&limit=3&order=asc`.

    The provider receives `GET /v1/chat/completions` with the same four query parameters, including the bracketed
    metadata key unchanged.
    """
    calls: Final = _new_calls(
        provider,
        lambda: _ok(
            isolated.request(
                "GET",
                "/openai/v1/chat/completions",
                params={"model": "gpt-4o-mini", "metadata[team]": "blue", "limit": "3", "order": "asc"},
            )
        ),
    )
    (call,) = [c for c in calls_to(calls, "/chat/completions") if c.method == "GET"]
    assert {key: list(call.query.get(key, [])) for key in ("model", "metadata[team]", "limit", "order")} == {
        "model": ["gpt-4o-mini"],
        "metadata[team]": ["blue"],
        "limit": ["3"],
        "order": ["asc"],
    }, call.query


def test_fine_tuning_create_forwards_the_openai_fields_with_the_settings_key(isolated: Gateway, provider: Peer) -> None:
    """The caller creates a job with `custom_llm_provider: openai`, model `openai/gpt-4.1-mini-2025-04-14`,
    hyperparameters, `suffix`, `seed` and `validation_file`.

    The provider receives `POST /v1/fine_tuning/jobs` with `authorization: Bearer <finetune_settings key>` and a
    body of `model: gpt-4.1-mini-2025-04-14` (prefix stripped), `training_file`, the same hyperparameters,
    `suffix`, `seed` and `validation_file`; `custom_llm_provider` is not sent.
    """
    sent: Final[dict[str, JsonValue]] = {
        "custom_llm_provider": "openai",
        "model": "openai/gpt-4.1-mini-2025-04-14",
        "training_file": "file-dec1-openai2-train",
        "validation_file": "file-dec1-openai2-valid",
        "hyperparameters": {"n_epochs": 3, "batch_size": 4, "learning_rate_multiplier": 0.1},
        "suffix": "dec1",
        "seed": 7,
    }
    calls: Final = _new_calls(provider, lambda: _ok(isolated.request("POST", "/v1/fine_tuning/jobs", sent)))
    (call,) = calls_to(calls, "/fine_tuning/jobs")
    assert call.headers["authorization"] == f"Bearer {FINETUNE_KEY}"
    assert call.body == {
        "model": "gpt-4.1-mini-2025-04-14",
        "training_file": "file-dec1-openai2-train",
        "validation_file": "file-dec1-openai2-valid",
        "hyperparameters": {"n_epochs": 3, "batch_size": 4, "learning_rate_multiplier": 0.1},
        "suffix": "dec1",
        "seed": 7,
    }


def test_fine_tuning_settings_organization_reaches_openai(isolated: Gateway, provider: Peer) -> None:
    """`finetune_settings` for openai sets `organization: org-dec1-openai2-finetune`; the caller creates a job.

    The provider request carries `openai-organization: org-dec1-openai2-finetune`, read from
    `optional_params.organization` in `create_fine_tuning_job` (`litellm/fine_tuning/main.py:203`).
    """
    sent: Final[dict[str, JsonValue]] = {
        "custom_llm_provider": "openai",
        "model": "gpt-4.1-mini-2025-04-14",
        "training_file": "file-dec1-openai2-train",
    }
    calls: Final = _new_calls(provider, lambda: _ok(isolated.request("POST", "/v1/fine_tuning/jobs", sent)))
    (call,) = calls_to(calls, "/fine_tuning/jobs")
    assert call.headers["openai-organization"] == FINETUNE_ORG


def test_fine_tuning_create_forwards_the_training_method(isolated: Gateway, provider: Peer) -> None:
    """The caller creates a DPO job with `method: {"type": "dpo", "dpo": {"hyperparameters": {"beta": 0.1}}}`.

    The provider body must carry the same `method` object; without it OpenAI runs supervised fine-tuning on a
    preference dataset and fails or trains the wrong objective.
    """
    pytest.skip(
        "BUG: _build_fine_tuning_job_data only sends model, training_file, hyperparameters, suffix, "
        "validation_file, integrations and seed, so method (and metadata) never reach OpenAI"
    )
    method: Final[dict[str, JsonValue]] = {"type": "dpo", "dpo": {"hyperparameters": {"beta": 0.1}}}
    sent: Final[dict[str, JsonValue]] = {
        "custom_llm_provider": "openai",
        "model": "gpt-4.1-mini-2025-04-14",
        "training_file": "file-dec1-openai2-train",
        "method": method,
    }
    calls: Final = _new_calls(provider, lambda: _ok(isolated.request("POST", "/v1/fine_tuning/jobs", sent)))
    (call,) = calls_to(calls, "/fine_tuning/jobs")
    assert call.body["method"] == method


def test_fine_tuning_retrieve_reaches_the_job_path(isolated: Gateway, provider: Peer) -> None:
    """The caller calls `GET /v1/fine_tuning/jobs/ftjob-dec1openai2?custom_llm_provider=openai`.

    The provider receives `GET /v1/fine_tuning/jobs/ftjob-dec1openai2` with the settings key and the caller gets
    `object: fine_tuning.job` with that id.
    """
    body: list[dict[str, JsonValue]] = []
    calls: Final = _new_calls(
        provider,
        lambda: body.append(
            _ok(
                isolated.request(
                    "GET", "/v1/fine_tuning/jobs/ftjob-dec1openai2", params={"custom_llm_provider": "openai"}
                )
            )
        ),
    )
    (call,) = calls_to(calls, "/fine_tuning/jobs")
    assert (call.method, call.path) == ("GET", "/v1/fine_tuning/jobs/ftjob-dec1openai2")
    assert call.headers["authorization"] == f"Bearer {FINETUNE_KEY}"
    assert (body[0]["object"], body[0]["id"]) == ("fine_tuning.job", "ftjob-dec1openai2"), body


def test_fine_tuning_list_forwards_after_and_limit(isolated: Gateway, provider: Peer) -> None:
    """The caller lists `/v1/fine_tuning/jobs?custom_llm_provider=openai&after=ftjob-dec1prev&limit=2`.

    The provider receives `GET /v1/fine_tuning/jobs` with `after=ftjob-dec1prev` and `limit=2`, and
    `custom_llm_provider` is not part of the provider query.
    """
    calls: Final = _new_calls(
        provider,
        lambda: _ok(
            isolated.request(
                "GET",
                "/v1/fine_tuning/jobs",
                params={"custom_llm_provider": "openai", "after": "ftjob-dec1prev", "limit": "2"},
            )
        ),
    )
    (call,) = calls_to(calls, "/fine_tuning/jobs")
    assert (list(call.query["after"]), list(call.query["limit"])) == (["ftjob-dec1prev"], ["2"]), call.query
    assert "custom_llm_provider" not in call.query, call.query


@pytest.mark.parametrize(
    "path", ["/v1/fine_tuning/jobs", "/v1/fine_tuning/jobs/ftjob-dec1openai2"], ids=["list", "retrieve"]
)
def test_fine_tuning_reads_without_a_provider_are_400(isolated: Gateway, provider: Peer, path: str) -> None:
    """The caller lists or retrieves fine-tuning jobs without `custom_llm_provider` and with a raw job id.

    The proxy answers 400 with `Invalid request, No litellm managed file id or custom_llm_provider provided.` and
    the provider sees nothing.
    """
    responses: list[int] = []
    texts: list[str] = []

    def call() -> None:
        response = isolated.request("GET", path)
        responses.append(response.status_code)
        texts.append(response.text)

    calls: Final = _new_calls(provider, call)
    assert responses == [400], texts
    assert "No litellm managed file id or custom_llm_provider provided" in texts[0]
    assert not calls_to(calls, "/fine_tuning"), calls


def test_fine_tuning_create_without_a_provider_is_a_client_error(isolated: Gateway, provider: Peer) -> None:
    """The caller creates a job with a raw training file id and no `custom_llm_provider`.

    The proxy should answer 400 like list and retrieve do, naming the missing provider; the provider sees
    nothing.
    """
    pytest.skip(
        "BUG: create raises a bare ValueError('Invalid request, No litellm managed file id or custom_llm_provider "
        "provided.') that handle_exception_on_proxy answers as 500, while list and retrieve raise HTTPException 400"
    )
    response: Final = isolated.request(
        "POST", "/v1/fine_tuning/jobs", {"model": "gpt-4.1-mini-2025-04-14", "training_file": "file-dec1-raw"}
    )
    assert response.status_code == 400, response.text
    assert "custom_llm_provider" in response.text


def test_fine_tuning_routes_answer_for_the_unprefixed_alias(isolated: Gateway, provider: Peer) -> None:
    """The caller lists jobs on `/fine_tuning/jobs?custom_llm_provider=openai` (no `/v1`).

    The provider receives `GET /v1/fine_tuning/jobs` and the caller gets 200: both route spellings are served
    (`endpoints.py:237`).
    """
    calls: Final = _new_calls(
        provider,
        lambda: _ok(isolated.request("GET", "/fine_tuning/jobs", params={"custom_llm_provider": "openai"})),
    )
    (call,) = calls_to(calls, "/fine_tuning/jobs")
    assert call.path == "/v1/fine_tuning/jobs"
