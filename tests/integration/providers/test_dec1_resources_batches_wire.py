"""Contract: `/v1/batches` create, retrieve, cancel and list, on the provider path and the managed path.

Provider path (raw provider ids, no `model`): the provider comes from the URL prefix, the body field
`custom_llm_provider`, or the `custom-llm-provider` header, and defaults to `openai`. The proxy forwards the
OpenAI Batch API call (`POST /v1/batches`, `GET /v1/batches/{id}`, `POST /v1/batches/{id}/cancel`; Azure uses
`/openai/batches...?api-version=`) with a deployment of that provider the key may use, and returns the
provider's object and raw id. Only string-valued `metadata` entries are forwarded, and a team can force
`output_expires_after` through team metadata `enforced_batch_output_expires_after`.

Managed path (file uploaded with `target_model_names`): the caller only ever sees unified ids. A managed
file id is unpadded base64url of
`litellm_proxy:<mime>;unified_id,<uuid>;target_model_names,<models>;llm_output_file_id,<provider id>;llm_output_file_model_id,<deployment id>`
and a managed batch id is unpadded base64url of `litellm_proxy;model_id:<deployment id>;llm_batch_id:<provider id>`.
The proxy swaps them for provider ids on the way out and back on the way in, and records ownership in
`LiteLLM_ManagedFileTable` and `LiteLLM_ManagedObjectTable`.

`GET /v1/batches` never calls a provider while the managed-files hook is loaded: it lists the caller's own
rows from `LiteLLM_ManagedObjectTable`, newest first, with `limit` (0 to 100, default 20) and an `after`
cursor that must be one of the caller's batch ids.

Assumed: `dec1_resources_support.booted_rig`; the scripted upstream names a batch `batch_<input file stem>`.
The batch pre-call hook may download the input file to count tokens, so tests look for the batch request
among the upstream requests instead of requiring it to be the only one.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import eventually
from integration._support.wire import Request
from integration.providers.dec1_resources_support import (
    AZURE_API_VERSION,
    AZURE_MODEL_KEY,
    JSON_OBJECT,
    OPENAI_MODEL,
    OPENAI_MODEL_KEY,
    OTHER_MODEL_KEY,
    Rig,
    batch_line,
    bearer,
    booted_rig,
    decoded,
    error_of,
    json_body,
    jsonl,
    path_of,
    query_of,
    unique,
)
from pydantic import JsonValue

_OPENAI_DEPLOYMENT_KEYS: Final = frozenset({OPENAI_MODEL_KEY, OTHER_MODEL_KEY})
_MANAGED_FILE: Final = re.compile(
    r"^litellm_proxy:[^;]+;unified_id,[0-9a-f-]{36};target_model_names,(?P<models>[^;]*);"
    r"llm_output_file_id,(?P<provider_id>[^;]+);llm_output_file_model_id,(?P<model_id>[^;]+)$"
)
_MANAGED_BATCH: Final = re.compile(r"^litellm_proxy;model_id:(?P<model_id>[^;]+);llm_batch_id:(?P<provider_id>.+)$")
_EMPTY_PAGE: Final[dict[str, JsonValue]] = {
    "object": "list",
    "data": [],
    "first_id": None,
    "last_id": None,
    "has_more": False,
}


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with booted_rig(tmp_path_factory.mktemp("dec1-resources-batches-wire")) as booted:
        yield booted


def _create_body(input_file_id: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {"input_file_id": input_file_id, "endpoint": "/v1/chat/completions", "completion_window": "24h", **fields}


def _sent(rig: Rig, method: str, path: str) -> Request:
    """The one upstream request with this method and path among everything sent since the last drain."""
    matches: Final = [request for request in rig.drained() if (request.method, path_of(request)) == (method, path)]
    assert len(matches) == 1, matches
    return matches[0]


def _managed_file(rig: Rig, key: str, stem: str) -> str:
    """A managed batch file whose provider id is `file-<stem>`. The proxy re-uploads the rewritten file as
    `modified_file.jsonl`, so the scripted provider names it after the first line's `custom_id`."""
    content: Final = jsonl(batch_line(OPENAI_MODEL, custom_id=stem))
    return rig.uploaded_id(f"{stem}.jsonl", content, key=key, target_model_names=OPENAI_MODEL)


def _managed_batch(rig: Rig, key: str, stem: str) -> str:
    created: Final = rig.proxy.request("POST", "/v1/batches", _create_body(_managed_file(rig, key, stem)), key=key)
    assert created.status_code == 200, created.text
    identity: Final = JSON_OBJECT.validate_json(created.content)["id"]
    assert isinstance(identity, str)
    return identity


def _listed_ids(rig: Rig, key: str | None, **params: str) -> list[JsonValue]:
    response: Final = rig.proxy.request("GET", "/v1/batches", key=key, params=params)
    assert response.status_code == 200, response.text
    data: Final = JSON_OBJECT.validate_json(response.content)["data"]
    assert isinstance(data, list)
    return [JSON_OBJECT.validate_python(entry)["id"] for entry in data]


def test_create_forwards_input_file_endpoint_window_and_string_metadata(rig: Rig) -> None:
    """The master key creates a batch from a raw provider file id with `completion_window: "24h"` and
    `metadata: {"team": "blue", "job": "nightly"}`.

    The upstream must receive one `POST /v1/batches` whose JSON has exactly those four fields with the same
    values, sent with an OpenAI deployment key. The caller gets 200 and the provider's batch object with the
    raw id `batch_<stem>` and status `validating`.
    """
    stem: Final = unique("plain")
    rig.wire.drain()
    response: Final = rig.proxy.request(
        "POST", "/v1/batches", _create_body(f"file-{stem}", metadata={"team": "blue", "job": "nightly"})
    )
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert (body["id"], body["object"], body["status"]) == (f"batch_{stem}", "batch", "validating")
    request: Final = _sent(rig, "POST", "/v1/batches")
    assert bearer(request) in _OPENAI_DEPLOYMENT_KEYS
    assert json_body(request) == {
        "input_file_id": f"file-{stem}",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {"team": "blue", "job": "nightly"},
    }


def test_create_drops_metadata_entries_that_are_not_strings(rig: Rig) -> None:
    """A batch is created with `metadata: {"team": "blue", "attempt": 3, "labels": {"a": "b"}}`.

    OpenAI accepts only string values in batch metadata, so the upstream body must carry
    `metadata: {"team": "blue"}` and the request must still succeed.
    """
    stem: Final = unique("mixed-metadata")
    rig.wire.drain()
    response: Final = rig.proxy.request(
        "POST",
        "/v1/batches",
        _create_body(f"file-{stem}", metadata={"team": "blue", "attempt": 3, "labels": {"a": "b"}}),
    )
    assert response.status_code == 200, response.text
    assert json_body(_sent(rig, "POST", "/v1/batches"))["metadata"] == {"team": "blue"}


def test_create_never_forwards_proxy_internal_metadata(rig: Rig) -> None:
    """A virtual key in a team creates a batch and sends no `metadata`.

    The proxy tracks the caller in its own request metadata, but none of it may reach the provider: the
    upstream body has no `metadata` (or a null one) and no key that starts with `user_api_key`, and no
    `litellm_metadata` field.
    """
    stem: Final = unique("no-metadata")
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(team_id=scenario.team())
        rig.wire.drain()
        response: Final = rig.proxy.request("POST", "/v1/batches", _create_body(f"file-{stem}"), key=key)
        assert response.status_code == 200, response.text
        sent: Final = json_body(_sent(rig, "POST", "/v1/batches"))
        assert sent.get("metadata") is None
        assert "litellm_metadata" not in sent
        assert "user_api_key" not in rig.wire.url and b"user_api_key" not in str(sent).encode()


@pytest.mark.parametrize("surface", ["body", "header", "path"])
def test_create_naming_azure_reaches_the_azure_batches_route(rig: Rig, surface: str) -> None:
    """Azure is named in the body field `custom_llm_provider`, in the `custom-llm-provider` header, or with the
    `/azure/v1/batches` URL prefix.

    Each must produce one `POST /openai/batches?api-version=2024-10-21` with the Azure deployment's `api-key`,
    and the body sent upstream must not contain the proxy-only field `custom_llm_provider`.
    """
    stem: Final = unique(f"azure-{surface}")
    rig.wire.drain()
    match surface:
        case "body":
            response = rig.proxy.request(
                "POST", "/v1/batches", _create_body(f"file-{stem}", custom_llm_provider="azure")
            )
        case "header":
            response = rig.proxy.request(
                "POST", "/v1/batches", _create_body(f"file-{stem}"), headers={"custom-llm-provider": "azure"}
            )
        case _:
            response = rig.proxy.request("POST", "/azure/v1/batches", _create_body(f"file-{stem}"))
    assert response.status_code == 200, response.text
    request: Final = _sent(rig, "POST", "/openai/batches")
    assert query_of(request) == {"api-version": [AZURE_API_VERSION]}
    assert request.headers.get("api-key") == AZURE_MODEL_KEY
    assert "custom_llm_provider" not in json_body(request)


def test_create_with_two_different_model_headers_is_400(rig: Rig) -> None:
    """A batch create carries two `x-litellm-model` headers naming different models and no `model` in the body
    or query.

    The proxy cannot tell which deployment was meant, so it answers 400 with the message `Send one
    x-litellm-model header. This request has several naming different models` and creates nothing upstream.
    """
    rig.wire.drain()
    response: Final = rig.proxy.client.post(
        "/v1/batches",
        json=_create_body("file-two-headers"),
        headers=[
            ("Authorization", f"Bearer {rig.proxy.key}"),
            ("x-litellm-model", OPENAI_MODEL),
            ("x-litellm-model", "another-model"),
        ],
    )
    assert response.status_code == 400, response.text
    error: Final = error_of(response)
    assert error["message"] == "Send one x-litellm-model header. This request has several naming different models"
    assert error["type"] == "invalid_request_error"
    assert rig.drained() == ()


def test_team_enforced_batch_output_expiry_is_added_to_the_upstream_body(rig: Rig) -> None:
    """A team has metadata `enforced_batch_output_expires_after = {"anchor": "created_at", "seconds": 86400}` and
    one of its keys creates a batch without `output_expires_after`.

    The upstream body must carry `output_expires_after: {"anchor": "created_at", "seconds": 86400}`.
    """
    stem: Final = unique("enforced-output")
    with rig.proxy.scenario() as scenario:
        team: Final = scenario.team(
            metadata={"enforced_batch_output_expires_after": {"anchor": "created_at", "seconds": 86400}}
        )
        key: Final = scenario.key(team_id=team)
        rig.wire.drain()
        response: Final = rig.proxy.request("POST", "/v1/batches", _create_body(f"file-{stem}"), key=key)
        assert response.status_code == 200, response.text
        assert json_body(_sent(rig, "POST", "/v1/batches"))["output_expires_after"] == {
            "anchor": "created_at",
            "seconds": 86400,
        }


def test_malformed_team_enforced_batch_output_expiry_is_500(rig: Rig) -> None:
    """A team's `enforced_batch_output_expires_after` has no `seconds` and its key creates a batch.

    The create fails with 500 and a message naming the metadata field as malformed, and no batch is created
    upstream.
    """
    with rig.proxy.scenario() as scenario:
        team: Final = scenario.team(metadata={"enforced_batch_output_expires_after": {"anchor": "created_at"}})
        key: Final = scenario.key(team_id=team)
        rig.wire.drain()
        response: Final = rig.proxy.request("POST", "/v1/batches", _create_body("file-malformed-team"), key=key)
        assert response.status_code == 500, response.text
        assert "team metadata field 'enforced_batch_output_expires_after' is malformed" in response.text
        assert [request for request in rig.drained() if request.method == "POST"] == []


@pytest.mark.parametrize("prefix", ["/v1", ""])
def test_retrieve_forwards_the_raw_batch_id_and_returns_the_provider_object(rig: Rig, prefix: str) -> None:
    """The master key retrieves a raw provider batch id on `/v1/batches/{id}` and on the unversioned alias.

    The upstream must receive one `GET /v1/batches/<id>` with an OpenAI deployment key; the caller gets the
    provider's object with the same raw id and status `in_progress`.
    """
    batch_id: Final = unique("batch_retrieve")
    rig.wire.drain()
    response: Final = rig.proxy.request("GET", f"{prefix}/batches/{batch_id}")
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert (body["id"], body["status"]) == (batch_id, "in_progress")
    request: Final = _sent(rig, "GET", f"/v1/batches/{batch_id}")
    assert bearer(request) in _OPENAI_DEPLOYMENT_KEYS


def test_cancel_posts_to_the_provider_cancel_route_and_returns_its_status(rig: Rig) -> None:
    """The master key cancels a raw provider batch id.

    The upstream must receive one `POST /v1/batches/<id>/cancel`; the caller gets 200 and the provider's
    object, whose status is `cancelling` (cancel is asynchronous at the provider, the proxy does not rewrite
    the status to `cancelled`).
    """
    batch_id: Final = unique("batch_cancel")
    rig.wire.drain()
    response: Final = rig.proxy.request("POST", f"/v1/batches/{batch_id}/cancel")
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert (body["id"], body["status"]) == (batch_id, "cancelling")
    _sent(rig, "POST", f"/v1/batches/{batch_id}/cancel")


@pytest.mark.parametrize("surface", ["query", "header", "path"])
def test_retrieve_and_cancel_naming_azure_reach_the_azure_routes(rig: Rig, surface: str) -> None:
    """A raw batch id is retrieved and then cancelled with Azure named in the query, header or URL prefix.

    The upstream must receive `GET /openai/batches/<id>` and `POST /openai/batches/<id>/cancel`, both with
    `api-version=2024-10-21` and the Azure deployment's `api-key`.
    """
    batch_id: Final = unique("batch_azure")
    prefix: Final = "/azure/v1" if surface == "path" else "/v1"
    params: Final = {"custom_llm_provider": "azure"} if surface == "query" else None
    headers: Final = {"custom-llm-provider": "azure"} if surface == "header" else None
    rig.wire.drain()
    retrieved: Final = rig.proxy.request("GET", f"{prefix}/batches/{batch_id}", params=params, headers=headers)
    assert retrieved.status_code == 200, retrieved.text
    request: Final = _sent(rig, "GET", f"/openai/batches/{batch_id}")
    assert query_of(request) == {"api-version": [AZURE_API_VERSION]}
    assert request.headers.get("api-key") == AZURE_MODEL_KEY
    cancelled: Final = rig.proxy.request("POST", f"{prefix}/batches/{batch_id}/cancel", params=params, headers=headers)
    assert cancelled.status_code == 200, cancelled.text
    _sent(rig, "POST", f"/openai/batches/{batch_id}/cancel")


def test_managed_upload_returns_a_unified_file_id_and_records_the_owner(rig: Rig) -> None:
    """A virtual key uploads a batch file with `target_model_names` naming one deployment.

    The upstream receives one `POST /v1/files` with that deployment's key. The id the caller gets is not the
    provider id: decoded, it matches the unified file format and embeds the target model name and the provider
    file id `file-<stem>`. A row with that `unified_file_id` exists in `LiteLLM_ManagedFileTable` and its
    `created_by` is set, which is what every later ownership check reads.
    """
    stem: Final = unique("managed-file")
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        rig.wire.drain()
        file_id: Final = _managed_file(rig, key, stem)
        assert bearer(_sent(rig, "POST", "/v1/files")) == OPENAI_MODEL_KEY
        matched: Final = _MANAGED_FILE.match(decoded(file_id))
        assert matched is not None, decoded(file_id)
        assert (matched.group("models"), matched.group("provider_id")) == (OPENAI_MODEL, f"file-{stem}")
        rows: Final = rig.proxy.rows(
            'SELECT created_by FROM "LiteLLM_ManagedFileTable" WHERE unified_file_id = %s', (file_id,)
        )
        assert len(rows) == 1 and rows[0]["created_by"] is not None, rows


def test_managed_batch_create_swaps_ids_in_both_directions(rig: Rig) -> None:
    """A virtual key creates a batch from its managed file id, naming no model.

    The upstream `POST /v1/batches` must carry the provider file id `file-<stem>` as `input_file_id`. The
    caller must get a batch whose `id` decodes to `litellm_proxy;model_id:<deployment>;llm_batch_id:batch_<stem>`
    and whose `input_file_id` is the managed file id it sent, never the provider id.
    """
    stem: Final = unique("managed-batch")
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        file_id: Final = _managed_file(rig, key, stem)
        rig.wire.drain()
        created: Final = rig.proxy.request("POST", "/v1/batches", _create_body(file_id), key=key)
        assert created.status_code == 200, created.text
        request: Final = _sent(rig, "POST", "/v1/batches")
        assert json_body(request)["input_file_id"] == f"file-{stem}"
        assert bearer(request) == OPENAI_MODEL_KEY
        body: Final = JSON_OBJECT.validate_json(created.content)
        matched: Final = _MANAGED_BATCH.match(decoded(str(body["id"])))
        assert matched is not None, decoded(str(body["id"]))
        assert matched.group("provider_id") == f"batch_{stem}"
        assert body["input_file_id"] == file_id


def test_managed_batch_create_records_one_owned_row(rig: Rig) -> None:
    """A virtual key creates a managed batch.

    `LiteLLM_ManagedObjectTable` must gain exactly one row for the returned id with `file_purpose = 'batch'`,
    `model_object_id` equal to the provider batch id, and `created_by` set. The cost poller and the listing
    both read this row, so a batch without it is never billed and never listed.
    """
    stem: Final = unique("managed-row")
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        batch_id: Final = _managed_batch(rig, key, stem)
        rows: Final = eventually(
            lambda: rig.proxy.rows(
                'SELECT file_purpose, model_object_id, created_by FROM "LiteLLM_ManagedObjectTable" '
                "WHERE unified_object_id = %s",
                (batch_id,),
            ),
            lambda found: len(found) == 1,
        )
        assert (rows[0]["file_purpose"], rows[0]["model_object_id"]) == ("batch", f"batch_{stem}")
        assert rows[0]["created_by"] is not None


def test_managed_batch_retrieve_and_cancel_reach_the_provider_with_the_provider_id(rig: Rig) -> None:
    """The owner retrieves and then cancels its managed batch by unified id.

    The upstream must receive `GET /v1/batches/batch_<stem>` and `POST /v1/batches/batch_<stem>/cancel` with
    the deployment key the batch was created with; both responses must carry the unified id the caller used,
    with statuses `in_progress` and `cancelling`.
    """
    stem: Final = unique("managed-lifecycle")
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        batch_id: Final = _managed_batch(rig, key, stem)
        rig.wire.drain()
        retrieved: Final = rig.proxy.request("GET", f"/v1/batches/{batch_id}", key=key)
        assert retrieved.status_code == 200, retrieved.text
        assert bearer(_sent(rig, "GET", f"/v1/batches/batch_{stem}")) == OPENAI_MODEL_KEY
        retrieved_body: Final = JSON_OBJECT.validate_json(retrieved.content)
        assert (retrieved_body["id"], retrieved_body["status"]) == (batch_id, "in_progress")
        cancelled: Final = rig.proxy.request("POST", f"/v1/batches/{batch_id}/cancel", key=key)
        assert cancelled.status_code == 200, cancelled.text
        assert bearer(_sent(rig, "POST", f"/v1/batches/batch_{stem}/cancel")) == OPENAI_MODEL_KEY
        cancelled_body: Final = JSON_OBJECT.validate_json(cancelled.content)
        assert (cancelled_body["id"], cancelled_body["status"]) == (batch_id, "cancelling")


def test_another_key_cannot_retrieve_or_cancel_a_managed_batch(rig: Rig) -> None:
    """Two virtual keys with no user and no team exist; the first creates a managed batch.

    The second key gets 403 on both `GET /v1/batches/{id}` and `POST /v1/batches/{id}/cancel`, and neither call
    reaches the provider. For such keys the owner is the key itself (its hashed token).
    """
    stem: Final = unique("managed-private")
    with rig.proxy.scenario() as scenario:
        owner: Final = scenario.key()
        stranger: Final = scenario.key()
        batch_id: Final = _managed_batch(rig, owner, stem)
        rig.wire.drain()
        assert rig.proxy.request("GET", f"/v1/batches/{batch_id}", key=stranger).status_code == 403
        assert rig.proxy.request("POST", f"/v1/batches/{batch_id}/cancel", key=stranger).status_code == 403
        assert rig.drained() == ()


def test_list_shows_a_key_only_its_own_batches_without_calling_a_provider(rig: Rig) -> None:
    """Two virtual keys each create one managed batch, then each lists `/v1/batches`.

    Each key sees exactly its own unified batch id, the master key sees both, and no list request reaches the
    provider: the listing is read from `LiteLLM_ManagedObjectTable`.
    """
    with rig.proxy.scenario() as scenario:
        first: Final = scenario.key()
        second: Final = scenario.key()
        first_batch: Final = _managed_batch(rig, first, unique("list-first"))
        second_batch: Final = _managed_batch(rig, second, unique("list-second"))
        rig.wire.drain()
        assert _listed_ids(rig, first) == [first_batch]
        assert _listed_ids(rig, second) == [second_batch]
        everything: Final = _listed_ids(rig, None, limit="100")
        assert first_batch in everything and second_batch in everything
        assert [
            request for request in rig.drained() if request.method == "GET" and path_of(request) == "/v1/batches"
        ] == []


def test_list_pages_newest_first_with_limit_and_after(rig: Rig) -> None:
    """A virtual key creates two managed batches, then lists with `limit=1` and follows the cursor.

    The first page holds the newer batch with `has_more: true` and `first_id == last_id == <that id>`. The page
    after that id holds the older batch with `has_more: false`. The page after the older id is empty.
    """
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        older: Final = _managed_batch(rig, key, unique("page-older"))
        newer: Final = _managed_batch(rig, key, unique("page-newer"))
        first: Final = JSON_OBJECT.validate_json(
            rig.proxy.request("GET", "/v1/batches", key=key, params={"limit": "1"}).content
        )
        assert (first["first_id"], first["last_id"], first["has_more"]) == (newer, newer, True)
        second: Final = JSON_OBJECT.validate_json(
            rig.proxy.request("GET", "/v1/batches", key=key, params={"limit": "1", "after": newer}).content
        )
        assert (second["first_id"], second["has_more"]) == (older, False)
        assert _listed_ids(rig, key, after=older) == []


def test_list_with_limit_zero_is_an_empty_page(rig: Rig) -> None:
    """A key that owns a managed batch lists with `limit=0`.

    Zero is a valid limit for batches (unlike files): the answer is 200 with exactly `{"object": "list",
    "data": [], "first_id": null, "last_id": null, "has_more": false}`.
    """
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        _managed_batch(rig, key, unique("limit-zero"))
        response: Final = rig.proxy.request("GET", "/v1/batches", key=key, params={"limit": "0"})
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content) == _EMPTY_PAGE


@pytest.mark.parametrize(
    ("limit", "message"),
    [
        ("-1", "Invalid 'limit': integer below minimum value. Expected a value >= 0, but got -1 instead."),
        ("101", "Invalid 'limit': integer above maximum value. Expected a value <= 100, but got 101 instead."),
    ],
)
def test_list_limit_outside_zero_to_one_hundred_is_400(rig: Rig, limit: str, message: str) -> None:
    """`GET /v1/batches?limit=` is called with -1 or 101.

    The proxy answers 400 with `param: limit`, `type: invalid_request_error` and OpenAI's wording for the
    bound. The batch range is 0 to 100, which differs from the file range of 1 to 10000.
    """
    response: Final = rig.proxy.request("GET", "/v1/batches", params={"limit": limit})
    assert response.status_code == 400, response.text
    error: Final = error_of(response)
    assert (error["message"], error["param"], error["type"]) == (message, "limit", "invalid_request_error")


def test_list_after_a_cursor_the_caller_does_not_own_is_400(rig: Rig) -> None:
    """A fresh key lists batches with `after` set to an id that is not one of its batches.

    The proxy answers 400 with the message `Invalid 'after' cursor: no batch found with id '<id>'.` rather than
    an empty page, and the same answer is given for another tenant's real batch id, so the cursor cannot be
    used to probe for ids.
    """
    with rig.proxy.scenario() as scenario:
        owner: Final = scenario.key()
        stranger: Final = scenario.key()
        real: Final = _managed_batch(rig, owner, unique("cursor-owner"))
        for cursor in ("batch_does-not-exist", real):
            response = rig.proxy.request("GET", "/v1/batches", key=stranger, params={"after": cursor})
            assert response.status_code == 400, response.text
            assert f"Invalid 'after' cursor: no batch found with id '{cursor}'." in response.text


@pytest.mark.parametrize(
    ("path", "params", "param", "message"),
    [
        (
            "/openai/v1/batches",
            {},
            "provider",
            "Filtering by 'provider' is not supported when using managed batches.",
        ),
        (
            "/v1/batches",
            {"target_model_names": OPENAI_MODEL},
            "target_model_names",
            "Filtering by 'target_model_names' is not supported when using managed batches.",
        ),
    ],
    ids=["provider-prefix", "target-model-names"],
)
def test_list_filters_are_refused_while_batches_are_managed(
    rig: Rig, path: str, params: dict[str, str], param: str, message: str
) -> None:
    """Batches are listed through the provider URL prefix, or with `target_model_names`.

    The managed listing cannot filter by either (unified ids do not carry that information), so the proxy
    answers 400 with `type: invalid_request_error`, the matching `param` and the exact message, instead of
    returning an unfiltered page.
    """
    response: Final = rig.proxy.request("GET", path, params=params)
    assert response.status_code == 400, response.text
    error: Final = error_of(response)
    assert (error["message"], error["param"], error["type"]) == (message, param, "invalid_request_error")
