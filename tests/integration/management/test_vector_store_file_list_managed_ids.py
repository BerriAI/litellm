import base64
import hashlib
import json
import re
import signal
import threading
import uuid
from collections.abc import Callable, Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final
from urllib.parse import parse_qs, urlsplit

import httpx
import psutil
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import AsyncOpenAI, OpenAI
from pydantic import JsonValue, TypeAdapter

MANAGED_PREFIX: Final = "litellm_proxy:"
CARRIED_PROVIDER_FILE_ID: Final = re.compile(r"(?:^|;)llm_output_file_id,([^;]+)")
UPLOAD_FILENAME: Final = re.compile(rb'filename="([^"]+)"')
FILE_PATH: Final = re.compile(r"^/v1/files/([^/]+)$")
STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
MANAGED_FILE_ROW: Final = (
    'SELECT flat_model_file_ids, created_by, team_id FROM "LiteLLM_ManagedFileTable" WHERE unified_file_id = %s'
)

Listing = Callable[[Request], Reply]


def _provider_file_id(bearer: str, filename: str) -> str:
    return "file-" + hashlib.sha256(f"{bearer}:{filename}".encode()).hexdigest()[:16]


def _bearer(request: Request) -> str:
    return request.headers.get("authorization", "").removeprefix("Bearer ")


def _query(request: Request) -> dict[str, list[str]]:
    return parse_qs(urlsplit(request.target).query, keep_blank_values=True)


def _json(response: httpx.Response) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(response.content)


def _json_reply(body: Mapping[str, JsonValue], status: int = 200) -> Reply:
    return Reply(status=status, body=json.dumps(body).encode())


def _file_object(file_id: str) -> dict[str, JsonValue]:
    return {
        "id": file_id,
        "object": "file",
        "bytes": 12,
        "created_at": 1700000000,
        "filename": "notes.txt",
        "purpose": "user_data",
        "status": "processed",
    }


def _store_file(store: str, file_id: JsonValue) -> dict[str, JsonValue]:
    return {
        "id": file_id,
        "object": "vector_store.file",
        "usage_bytes": 123,
        "created_at": 1700000001,
        "vector_store_id": store,
        "status": "completed",
        "last_error": None,
        "chunking_strategy": {"type": "static", "static": {"max_chunk_size_tokens": 800, "chunk_overlap_tokens": 400}},
        "attributes": {},
    }


def _page(store: str, file_ids: tuple[JsonValue, ...], *, has_more: bool = False) -> dict[str, JsonValue]:
    return {
        "object": "list",
        "data": [_store_file(store, file_id) for file_id in file_ids],
        "first_id": file_ids[0] if file_ids else None,
        "last_id": file_ids[-1] if file_ids else None,
        "has_more": has_more,
    }


def _constant_listing(store: str, *file_ids: JsonValue) -> Listing:
    return lambda _: _json_reply(_page(store, file_ids))


def _paged_listing(store: str, first: str, second: str) -> Listing:
    def listing(request: Request) -> Reply:
        if _query(request).get("after") == [first]:
            return _json_reply(_page(store, (second,)))
        return _json_reply(_page(store, (first,), has_more=True))

    return listing


def _provider_error(status: int, message: str) -> dict[str, JsonValue]:
    return {"error": {"message": message, "type": "provider_error", "code": str(status)}}


def _error_listing(status: int, message: str) -> Callable[[str, str], Listing]:
    return lambda _store, _bearer: lambda _: _json_reply(_provider_error(status, message), status)


def _html_listing() -> Callable[[str, str], Listing]:
    return lambda _store, _bearer: lambda _: Reply(body=b"<html>upstream maintenance</html>", content_type="text/html")


def _two_pages(store: str, bearer: str) -> Listing:
    return _paged_listing(store, _provider_file_id(bearer, "a.txt"), _provider_file_id(bearer, "b.txt"))


def _raw_then_uploaded(raw_id: str) -> Callable[[str, str], Listing]:
    return lambda store, bearer: _constant_listing(store, raw_id, _provider_file_id(bearer, "a.txt"))


def _uploaded_then_integer(store: str, bearer: str) -> Listing:
    return _constant_listing(store, _provider_file_id(bearer, "a.txt"), 7)


def _provider(store: str, listing: Listing) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        if request.method == "POST" and path == "/v1/files":
            filename: Final = UPLOAD_FILENAME.search(request.body)
            assert filename is not None, request.body[:200]
            return _json_reply(_file_object(_provider_file_id(_bearer(request), filename.group(1).decode())))
        if request.method == "POST" and path == f"/v1/vector_stores/{store}/files":
            return _json_reply(_store_file(store, JSON_OBJECT.validate_json(request.body)["file_id"]))
        if request.method == "GET" and path == f"/v1/vector_stores/{store}/files":
            return listing(request)
        file: Final = FILE_PATH.match(path)
        if request.method == "GET" and file:
            return _json_reply(_file_object(file.group(1)))
        if request.method == "DELETE" and file:
            return _json_reply({"id": file.group(1), "object": "file", "deleted": True})
        return _json_reply({"error": {"message": f"unscripted {request.method} {request.target}"}}, 404)

    return respond


def _decoded(managed_file_id: str) -> str:
    decoded: Final = base64.urlsafe_b64decode(managed_file_id + "=" * (-len(managed_file_id) % 4)).decode()
    assert decoded.startswith(MANAGED_PREFIX), decoded
    return decoded


def _carried_provider_file_id(managed_file_id: str) -> str:
    carried: Final = CARRIED_PROVIDER_FILE_ID.search(_decoded(managed_file_id))
    assert carried is not None, managed_file_id
    return carried.group(1)


def _upload(gateway: Gateway, key: str, target_model_names: str, filename: str) -> str:
    uploaded: Final = gateway.request_multipart(
        "/v1/files",
        {"purpose": "user_data", "target_model_names": target_model_names},
        {"file": (filename, f"notes in {filename}\n".encode(), "text/plain")},
        key=key,
    )
    assert uploaded.status_code == 200, uploaded.text
    return string_value(_json(uploaded)["id"])


def _listed(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, response.text
    return _json(response)


def _ids(page: Mapping[str, JsonValue]) -> tuple[JsonValue, ...]:
    data: Final = page["data"]
    assert isinstance(data, list), page
    return tuple(object_value(entry)["id"] for entry in data)


def _sdk_base_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/") + "/v1"


def _models_over_a_fresh_connection(gateway: Gateway, _: int) -> frozenset[str]:
    with httpx.Client(base_url=gateway.client.base_url, timeout=15, trust_env=False) as client:
        listed: Final = client.get("/v1/models", headers={"Authorization": f"Bearer {gateway.key}"})
    assert listed.status_code == 200, listed.text
    data: Final = _json(listed)["data"]
    assert isinstance(data, list), listed.text
    return frozenset(string_value(object_value(entry)["id"]) for entry in data)


def _every_worker_serves(gateway: Gateway, model: str) -> bool:
    with ThreadPoolExecutor(max_workers=16) as pool:
        rounds: Final = tuple(
            tuple(pool.map(partial(_models_over_a_fresh_connection, gateway), range(16))) for _ in range(2)
        )
    return all(model in seen for round_ in rounds for seen in round_)


def _wait_until_every_worker_serves(gateway: Gateway, model: str) -> None:
    eventually(lambda: _every_worker_serves(gateway, model), lambda served: served, seconds=90)


@dataclass(frozen=True, slots=True)
class _Member:
    team: str
    user: str
    key: str


def _member(scenario: Scenario, *models: str) -> _Member:
    team: Final = scenario.team(models=list(models))
    user: Final = scenario.member(team)
    return _Member(team, user, scenario.key(team_id=team, user_id=user))


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    scenario: Scenario
    wire: Wire
    store: str
    bearer: str
    model: str

    def file_id(self, filename: str) -> str:
        return _provider_file_id(self.bearer, filename)

    def upload(self, key: str, filename: str) -> str:
        managed: Final = _upload(self.gateway, key, self.model, filename)
        assert _carried_provider_file_id(managed) == self.file_id(filename), _decoded(managed)
        return managed

    def list(
        self,
        key: str,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        *,
        query: str | None = None,
    ) -> httpx.Response:
        suffix: Final = "" if query is None else f"?{query}"
        return self.gateway.request(
            "GET", f"/v1/vector_stores/{self.store}/files{suffix}", key=key, params=params, headers=headers
        )

    def listed(self, key: str, params: Mapping[str, str] | None = None) -> dict[str, JsonValue]:
        return _listed(self.list(key, params if params is not None else {"model": self.model}))

    def list_requests(self) -> tuple[Request, ...]:
        return tuple(
            request
            for request in self.wire.drain()
            if (request.method, urlsplit(request.target).path) == ("GET", f"/v1/vector_stores/{self.store}/files")
        )

    def single_list_request(self) -> Request:
        (request,) = self.list_requests()
        return request


@contextmanager
def _rig(gateway: Gateway, *filenames: str, listing: Callable[[str, str], Listing] | None = None) -> Generator[_Rig]:
    store: Final = "vs_" + uuid.uuid4().hex
    bearer: Final = "provider-key-" + uuid.uuid4().hex[:8]
    served: Final = (
        listing(store, bearer)
        if listing is not None
        else _constant_listing(store, *(_provider_file_id(bearer, filename) for filename in filenames))
    )
    with gateway.scenario() as scenario, wire_server(_provider(store, served)) as wire:
        model: Final = scenario.model(api_base=wire.url + "/v1", api_key=bearer)
        _wait_until_every_worker_serves(gateway, model)
        yield _Rig(gateway, scenario, wire, store, bearer, model)


def test_raw_httpx_list_returns_the_uploaders_managed_ids(gateway: Gateway) -> None:
    with _rig(gateway, "a.txt", "b.txt") as rig:
        member: Final = _member(rig.scenario, rig.model)
        managed_a: Final = rig.upload(member.key, "a.txt")
        managed_b: Final = rig.upload(member.key, "b.txt")
        assert read_rows(MANAGED_FILE_ROW, (managed_a,)) == [
            {"flat_model_file_ids": [rig.file_id("a.txt")], "created_by": member.user, "team_id": member.team}
        ]
        page: Final = rig.listed(member.key)
        assert _ids(page) == (managed_a, managed_b), page
        assert (page["first_id"], page["last_id"]) == (managed_a, managed_b), page
        assert page["has_more"] is False, page
        listed: Final = rig.single_list_request()
        assert _query(listed) == {}, listed.target
        assert listed.headers["authorization"] == f"Bearer {rig.bearer}", listed.headers


def test_attach_by_managed_id_sends_the_provider_file_id_and_lists_it_back_managed(gateway: Gateway) -> None:
    with _rig(gateway, "a.txt") as rig:
        member: Final = _member(rig.scenario, rig.model)
        managed_a: Final = rig.upload(member.key, "a.txt")
        attached: Final = rig.gateway.request(
            "POST", f"/v1/vector_stores/{rig.store}/files", {"file_id": managed_a}, key=member.key
        )
        assert attached.status_code == 200, attached.text
        assert _json(attached)["id"] == managed_a, attached.text
        attach_path: Final = f"/v1/vector_stores/{rig.store}/files"
        attach_bodies: Final = [
            JSON_OBJECT.validate_json(request.body)
            for request in rig.wire.drain()
            if (request.method, request.target) == ("POST", attach_path)
        ]
        assert attach_bodies == [{"file_id": rig.file_id("a.txt")}], attach_bodies
        assert _ids(rig.listed(member.key)) == (managed_a,)


def test_openai_sdk_sync_auto_pager_walks_pages_with_managed_cursors(gateway: Gateway) -> None:
    with _rig(gateway, listing=_two_pages) as rig:
        member: Final = _member(rig.scenario, rig.model)
        managed_a: Final = rig.upload(member.key, "a.txt")
        managed_b: Final = rig.upload(member.key, "b.txt")
        with OpenAI(base_url=_sdk_base_url(gateway), api_key=member.key, max_retries=0) as client:
            first: Final = client.vector_stores.files.list(rig.store, limit=1, extra_query={"model": rig.model})
            assert [file.id for file in first.data] == [managed_a], first.model_dump_json()
            assert first.has_more is True, first.model_dump_json()
            second: Final = first.get_next_page()
            assert [file.id for file in second.data] == [managed_b], second.model_dump_json()
            assert second.has_more is False, second.model_dump_json()
        queries: Final = [_query(request) for request in rig.list_requests()]
        assert queries == [{"limit": ["1"]}, {"after": [rig.file_id("a.txt")], "limit": ["1"]}], queries


async def test_openai_sdk_async_auto_pager_walks_pages_with_managed_cursors(gateway: Gateway) -> None:
    with _rig(gateway, listing=_two_pages) as rig:
        member: Final = _member(rig.scenario, rig.model)
        managed_a: Final = rig.upload(member.key, "a.txt")
        managed_b: Final = rig.upload(member.key, "b.txt")
        async with AsyncOpenAI(base_url=_sdk_base_url(gateway), api_key=member.key, max_retries=0) as client:
            first: Final = await client.vector_stores.files.list(rig.store, limit=1, extra_query={"model": rig.model})
            assert [file.id for file in first.data] == [managed_a], first.model_dump_json()
            second: Final = await first.get_next_page()
            assert [file.id for file in second.data] == [managed_b], second.model_dump_json()
        queries: Final = [_query(request) for request in rig.list_requests()]
        assert queries == [{"limit": ["1"]}, {"after": [rig.file_id("a.txt")], "limit": ["1"]}], queries


def test_after_cursor_with_a_managed_id_reaches_the_provider_decoded(gateway: Gateway) -> None:
    with _rig(gateway, "b.txt") as rig:
        member: Final = _member(rig.scenario, rig.model)
        managed_a: Final = rig.upload(member.key, "a.txt")
        managed_b: Final = rig.upload(member.key, "b.txt")
        assert _ids(rig.listed(member.key, {"model": rig.model, "after": managed_a})) == (managed_b,)
        assert _query(rig.single_list_request()) == {"after": [rig.file_id("a.txt")]}


def test_before_cursor_with_a_managed_id_reaches_the_provider_decoded(gateway: Gateway) -> None:
    with _rig(gateway, "a.txt") as rig:
        member: Final = _member(rig.scenario, rig.model)
        managed_a: Final = rig.upload(member.key, "a.txt")
        managed_b: Final = rig.upload(member.key, "b.txt")
        assert _ids(rig.listed(member.key, {"model": rig.model, "before": managed_b})) == (managed_a,)
        assert _query(rig.single_list_request()) == {"before": [rig.file_id("b.txt")]}


def test_after_cursor_with_a_raw_provider_id_is_forwarded_verbatim(gateway: Gateway) -> None:
    with _rig(gateway, "b.txt") as rig:
        member: Final = _member(rig.scenario, rig.model)
        managed_b: Final = rig.upload(member.key, "b.txt")
        raw_cursor: Final = "file-" + uuid.uuid4().hex[:16]
        page: Final = rig.listed(member.key, {"model": rig.model, "after": raw_cursor})
        assert _query(rig.single_list_request()) == {"after": [raw_cursor]}
        assert _ids(page) == (managed_b,), page


def test_model_header_routing_returns_managed_ids(gateway: Gateway) -> None:
    with _rig(gateway, "a.txt") as rig:
        member: Final = _member(rig.scenario, rig.model)
        managed_a: Final = rig.upload(member.key, "a.txt")
        page: Final = _listed(rig.list(member.key, {}, {"x-litellm-model": rig.model}))
        assert _ids(page) == (managed_a,), page
        assert _query(rig.single_list_request()) == {}


def test_managed_vector_store_registry_routing_returns_managed_ids(gateway: Gateway) -> None:
    with _rig(gateway, "a.txt") as rig:
        member: Final = _member(rig.scenario, rig.model)
        managed_a: Final = rig.upload(member.key, "a.txt")
        registry_bearer: Final = "registry-key-" + uuid.uuid4().hex[:8]
        gateway.post(
            "/vector_store/new",
            {
                "vector_store_id": rig.store,
                "custom_llm_provider": "openai",
                "vector_store_name": "managed-ids-registry",
                "litellm_params": {"api_base": rig.wire.url + "/v1", "api_key": registry_bearer},
            },
        )
        rig.scenario.cleanups.callback(gateway.post, "/vector_store/delete", {"vector_store_id": rig.store})
        page: Final = _listed(rig.list(member.key, {}))
        assert _ids(page) == (managed_a,), page
        listed: Final = rig.single_list_request()
        assert listed.headers["authorization"] == f"Bearer {registry_bearer}", listed.headers
        assert _query(listed) == {}, listed.target


def test_team_model_fallback_routing_returns_managed_ids(gateway: Gateway) -> None:
    with _rig(gateway, "a.txt") as rig:
        member: Final = _member(rig.scenario, rig.model)
        managed_a: Final = rig.upload(member.key, "a.txt")
        page: Final = _listed(rig.list(member.key, {}))
        assert _ids(page) == (managed_a,), page
        listed: Final = rig.single_list_request()
        assert listed.headers["authorization"] == f"Bearer {rig.bearer}", listed.headers


def test_teammate_sees_the_uploaders_managed_ids(gateway: Gateway) -> None:
    with _rig(gateway, "a.txt") as rig:
        uploader: Final = _member(rig.scenario, rig.model)
        teammate_user: Final = rig.scenario.member(uploader.team)
        teammate_key: Final = rig.scenario.key(team_id=uploader.team, user_id=teammate_user)
        managed_a: Final = rig.upload(uploader.key, "a.txt")
        assert _ids(rig.listed(teammate_key)) == (managed_a,)


def test_proxy_admin_sees_every_managed_id(gateway: Gateway) -> None:
    with _rig(gateway, "a.txt") as rig:
        uploader: Final = _member(rig.scenario, rig.model)
        managed_a: Final = rig.upload(uploader.key, "a.txt")
        assert _ids(rig.listed(gateway.key)) == (managed_a,)


def test_stranger_in_another_team_sees_raw_provider_ids(gateway: Gateway) -> None:
    with _rig(gateway, "a.txt") as rig:
        uploader: Final = _member(rig.scenario, rig.model)
        stranger: Final = _member(rig.scenario, rig.model)
        managed_a: Final = rig.upload(uploader.key, "a.txt")
        assert len(read_rows(MANAGED_FILE_ROW, (managed_a,))) == 1
        page: Final = rig.listed(stranger.key)
        assert _ids(page) == (rig.file_id("a.txt"),), page
        assert (page["first_id"], page["last_id"]) == (rig.file_id("a.txt"), rig.file_id("a.txt")), page


def test_service_account_upload_is_shared_with_its_team_only(gateway: Gateway) -> None:
    with _rig(gateway, "a.txt") as rig:
        teammate: Final = _member(rig.scenario, rig.model)
        service_account: Final = rig.scenario.key(team_id=teammate.team)
        stranger: Final = _member(rig.scenario, rig.model)
        managed_a: Final = rig.upload(service_account, "a.txt")
        assert read_rows(MANAGED_FILE_ROW, (managed_a,)) == [
            {"flat_model_file_ids": [rig.file_id("a.txt")], "created_by": None, "team_id": teammate.team}
        ]
        assert _ids(rig.listed(service_account)) == (managed_a,)
        assert _ids(rig.listed(teammate.key)) == (managed_a,)
        assert _ids(rig.listed(stranger.key)) == (rig.file_id("a.txt"),)


def test_key_without_user_or_team_owns_its_upload_alone(gateway: Gateway) -> None:
    with _rig(gateway, "a.txt") as rig:
        owner: Final = rig.scenario.key()
        sibling: Final = rig.scenario.key()
        managed_a: Final = rig.upload(owner, "a.txt")
        assert read_rows(MANAGED_FILE_ROW, (managed_a,)) == [
            {
                "flat_model_file_ids": [rig.file_id("a.txt")],
                "created_by": f"key:{hashlib.sha256(owner.encode()).hexdigest()}",
                "team_id": None,
            }
        ]
        assert _ids(rig.listed(owner)) == (managed_a,)
        assert _ids(rig.listed(sibling)) == (rig.file_id("a.txt"),)


def test_file_attached_by_raw_provider_id_stays_raw_beside_a_managed_one(gateway: Gateway) -> None:
    raw_id: Final = "file-raw-" + uuid.uuid4().hex[:12]
    with _rig(gateway, listing=_raw_then_uploaded(raw_id)) as rig:
        member: Final = _member(rig.scenario, rig.model)
        managed_a: Final = rig.upload(member.key, "a.txt")
        attached: Final = rig.gateway.request(
            "POST", f"/v1/vector_stores/{rig.store}/files", {"file_id": raw_id, "model": rig.model}, key=member.key
        )
        assert attached.status_code == 200, attached.text
        assert _json(attached)["id"] == raw_id, attached.text
        assert _ids(rig.listed(member.key)) == (raw_id, managed_a)


def test_multi_model_upload_maps_only_the_provider_id_the_managed_id_carries(gateway: Gateway) -> None:
    with _rig(gateway, "a.txt") as first, _rig(gateway, "a.txt") as second:
        member: Final = _member(first.scenario, first.model, second.model)
        managed_a: Final = _upload(gateway, member.key, f"{first.model},{second.model}", "a.txt")
        (row,) = read_rows(MANAGED_FILE_ROW, (managed_a,))
        flat_ids: Final = row["flat_model_file_ids"]
        assert isinstance(flat_ids, list), row
        assert sorted(string_value(value) for value in flat_ids) == sorted(
            (first.file_id("a.txt"), second.file_id("a.txt"))
        ), row
        carried: Final = _carried_provider_file_id(managed_a)
        assert carried in {first.file_id("a.txt"), second.file_id("a.txt")}, carried
        first_ids: Final = _ids(first.listed(member.key))
        second_ids: Final = _ids(second.listed(member.key))
        assert first_ids == ((managed_a,) if carried == first.file_id("a.txt") else (first.file_id("a.txt"),))
        assert second_ids == ((managed_a,) if carried == second.file_id("a.txt") else (second.file_id("a.txt"),))


def test_deleting_the_managed_file_makes_its_listing_raw_again(gateway: Gateway) -> None:
    with _rig(gateway, "a.txt") as rig:
        member: Final = _member(rig.scenario, rig.model)
        managed_a: Final = rig.upload(member.key, "a.txt")
        assert _ids(rig.listed(member.key)) == (managed_a,)
        deleted: Final = gateway.request("DELETE", f"/v1/files/{managed_a}", key=member.key)
        assert deleted.status_code == 200, deleted.text
        assert _json(deleted)["deleted"] is True, deleted.text
        assert read_rows(MANAGED_FILE_ROW, (managed_a,)) == []
        assert _ids(rig.listed(member.key)) == (rig.file_id("a.txt"),)
        deletes: Final = [request.target for request in rig.wire.drain() if request.method == "DELETE"]
        assert deletes == [f"/v1/files/{rig.file_id('a.txt')}"], deletes


def test_empty_page_is_returned_unchanged(gateway: Gateway) -> None:
    with _rig(gateway) as rig:
        member: Final = _member(rig.scenario, rig.model)
        rig.upload(member.key, "a.txt")
        assert rig.listed(member.key) == _page(rig.store, ())


def test_duplicate_provider_ids_in_one_page_are_both_mapped(gateway: Gateway) -> None:
    with _rig(gateway, "a.txt", "a.txt") as rig:
        member: Final = _member(rig.scenario, rig.model)
        managed_a: Final = rig.upload(member.key, "a.txt")
        page: Final = rig.listed(member.key)
        assert _ids(page) == (managed_a, managed_a), page
        assert (page["first_id"], page["last_id"]) == (managed_a, managed_a), page


def test_mixed_page_maps_only_the_managed_entries_and_the_matching_edge_ids(gateway: Gateway) -> None:
    raw_id: Final = "file-raw-" + uuid.uuid4().hex[:12]
    with _rig(gateway, listing=_raw_then_uploaded(raw_id)) as rig:
        member: Final = _member(rig.scenario, rig.model)
        managed_a: Final = rig.upload(member.key, "a.txt")
        page: Final = rig.listed(member.key)
        assert _ids(page) == (raw_id, managed_a), page
        assert (page["first_id"], page["last_id"]) == (raw_id, managed_a), page


def test_repeated_identical_lists_each_reach_the_provider_and_each_map(gateway: Gateway) -> None:
    with _rig(gateway, "a.txt") as rig:
        member: Final = _member(rig.scenario, rig.model)
        managed_a: Final = rig.upload(member.key, "a.txt")
        assert _ids(rig.listed(member.key)) == (managed_a,)
        assert _ids(rig.listed(member.key)) == (managed_a,)
        targets: Final = [request.target for request in rig.list_requests()]
        assert targets == [f"/v1/vector_stores/{rig.store}/files"] * 2, targets


def test_duplicated_managed_after_cursor_reaches_the_provider_once_decoded(gateway: Gateway) -> None:
    with _rig(gateway, "b.txt") as rig:
        member: Final = _member(rig.scenario, rig.model)
        managed_a: Final = rig.upload(member.key, "a.txt")
        managed_b: Final = rig.upload(member.key, "b.txt")
        page: Final = _listed(rig.list(member.key, query=f"model={rig.model}&after={managed_a}&after={managed_a}"))
        assert _ids(page) == (managed_b,), page
        assert _query(rig.single_list_request()) == {"after": [rig.file_id("a.txt")]}


def _unpadded(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


@pytest.mark.parametrize(
    "cursor",
    (
        pytest.param("12345", id="integer-like"),
        pytest.param("", id="empty"),
        pytest.param("x" * 5000, id="five-kilobyte"),
        pytest.param(_unpadded(b"litellm_proxy:text/plain;unified_id,abc"), id="managed-without-provider-id"),
        pytest.param(_unpadded(b"\xff\xfe\xfd\xfc"), id="non-utf8-base64"),
    ),
)
def test_unmappable_after_cursors_are_forwarded_verbatim(gateway: Gateway, cursor: str) -> None:
    with _rig(gateway, "b.txt") as rig:
        member: Final = _member(rig.scenario, rig.model)
        managed_b: Final = rig.upload(member.key, "b.txt")
        page: Final = _listed(rig.list(member.key, {"model": rig.model, "after": cursor}))
        assert _query(rig.single_list_request()) == {"after": [cursor]}
        liveliness: Final = gateway.request("GET", "/health/liveliness")
        assert liveliness.status_code == 200, liveliness.text
        assert _ids(page) == (managed_b,), page


def test_two_different_after_values_forward_the_last_one(gateway: Gateway) -> None:
    with _rig(gateway, "b.txt") as rig:
        member: Final = _member(rig.scenario, rig.model)
        managed_b: Final = rig.upload(member.key, "b.txt")
        page: Final = _listed(rig.list(member.key, query=f"model={rig.model}&after=first-value&after=second-value"))
        assert _query(rig.single_list_request()) == {"after": ["second-value"]}
        assert _ids(page) == (managed_b,), page


@pytest.mark.parametrize("status", (401, 404, 500))
def test_provider_errors_reach_the_caller_and_other_models_keep_mapping(gateway: Gateway, status: int) -> None:
    message: Final = f"provider refused listing {uuid.uuid4().hex[:8]}"
    with _rig(gateway, listing=_error_listing(status, message)) as failing, _rig(gateway, "a.txt") as healthy:
        member: Final = _member(failing.scenario, failing.model, healthy.model)
        managed_a: Final = healthy.upload(member.key, "a.txt")
        failed: Final = failing.list(member.key, {"model": failing.model})
        assert _json(failed) == _provider_error(status, message), failed.text
        assert len(failing.list_requests()) == 1
        assert _ids(healthy.listed(member.key)) == (managed_a,)
        liveliness: Final = gateway.request("GET", "/health/liveliness")
        assert liveliness.status_code == 200, liveliness.text


def test_non_json_provider_body_is_an_error_response_and_other_models_keep_mapping(gateway: Gateway) -> None:
    with _rig(gateway, listing=_html_listing()) as failing, _rig(gateway, "a.txt") as healthy:
        member: Final = _member(failing.scenario, failing.model, healthy.model)
        managed_a: Final = healthy.upload(member.key, "a.txt")
        failed: Final = failing.list(member.key, {"model": failing.model})
        assert failed.status_code == 500, failed.text
        assert string_value(object_value(_json(failed)["error"])["message"]), failed.text
        assert _ids(healthy.listed(member.key)) == (managed_a,)
        liveliness: Final = gateway.request("GET", "/health/liveliness")
        assert liveliness.status_code == 200, liveliness.text


def test_non_string_ids_in_a_page_are_left_alone_while_strings_map(gateway: Gateway) -> None:
    with _rig(gateway, listing=_uploaded_then_integer) as rig:
        member: Final = _member(rig.scenario, rig.model)
        managed_a: Final = rig.upload(member.key, "a.txt")
        page: Final = rig.listed(member.key)
        assert _ids(page) == (managed_a, 7), page
        assert (page["first_id"], page["last_id"]) == (managed_a, 7), page


def test_retrieving_the_managed_file_still_resolves_to_the_provider_file(gateway: Gateway) -> None:
    with _rig(gateway, "a.txt") as rig:
        member: Final = _member(rig.scenario, rig.model)
        managed_a: Final = rig.upload(member.key, "a.txt")
        retrieved: Final = gateway.request("GET", f"/v1/files/{managed_a}", key=member.key)
        assert retrieved.status_code == 200, retrieved.text
        file: Final = _json(retrieved)
        assert (file["id"], file["object"], file["purpose"]) == (managed_a, "file", "user_data"), retrieved.text


def _burst(gateway: Gateway, store: str, key: str, model: str, size: int) -> tuple[httpx.Response, ...]:
    def one(_: int) -> httpx.Response:
        return gateway.request("GET", f"/v1/vector_stores/{store}/files", key=key, params={"model": model})

    with ThreadPoolExecutor(max_workers=size) as pool:
        return tuple(pool.map(one, range(size)))


@pytest.mark.timeout(180)
def test_provider_outage_mid_burst_fails_loudly_and_mapping_resumes_after_recovery(gateway: Gateway) -> None:
    store: Final = "vs_" + uuid.uuid4().hex
    bearer: Final = "provider-key-" + uuid.uuid4().hex[:8]
    provider_a: Final = _provider_file_id(bearer, "a.txt")
    respond: Final = _provider(store, _constant_listing(store, provider_a))
    with gateway.scenario() as scenario:
        with wire_server(respond) as wire:
            model: Final = scenario.model(api_base=wire.url + "/v1", api_key=bearer)
            _wait_until_every_worker_serves(gateway, model)
            member: Final = _member(scenario, model)
            managed_a: Final = _upload(gateway, member.key, model, "a.txt")
            assert _carried_provider_file_id(managed_a) == provider_a
            served: Final = _burst(gateway, store, member.key, model, 40)
            assert [_ids(_listed(response)) for response in served] == [(managed_a,)] * 40
            assert sum(1 for request in wire.drain() if request.method == "GET") == 40
            port: Final = urlsplit(wire.url).port
        assert port is not None
        failed: Final = _burst(gateway, store, member.key, model, 20)
        assert [response.status_code for response in failed] == [500] * 20, [r.text for r in failed[:3]]
        for response in failed:
            assert string_value(object_value(_json(response)["error"])["message"]), response.text
        liveliness: Final = gateway.request("GET", "/health/liveliness")
        assert liveliness.status_code == 200, liveliness.text
        with wire_server(respond, port=port) as revived:
            recovered: Final = _burst(gateway, store, member.key, model, 40)
            assert [_ids(_listed(response)) for response in recovered] == [(managed_a,)] * 40
            assert sum(1 for request in revived.drain() if request.method == "GET") == 40


def _open_connections_to(pid: int, url: str) -> int:
    port: Final = urlsplit(url).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def _tolerant_list(gateway: Gateway, store: str, key: str, model: str) -> httpx.Response | None:
    try:
        return gateway.request("GET", f"/v1/vector_stores/{store}/files", key=key, params={"model": model})
    except httpx.HTTPError:
        return None


@pytest.mark.timeout(300)
def test_worker_sigkill_mid_burst_leaves_the_sibling_mapping_ids(gateway: Gateway, tmp_path: Path) -> None:
    store: Final = "vs_" + uuid.uuid4().hex
    bearer: Final = "provider-key-" + uuid.uuid4().hex[:8]
    provider_a: Final = _provider_file_id(bearer, "a.txt")
    release: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()

    def held_listing(request: Request) -> Reply:
        held.put(request.target)
        assert release.wait(timeout=120), "The burst was never released"
        return _json_reply(_page(store, (provider_a,)))

    with gateway.scenario() as scenario, wire_server(_provider(store, held_listing)) as wire:
        model: Final = scenario.model(api_base=wire.url + "/v1", api_key=bearer)
        _wait_until_every_worker_serves(gateway, model)
        member: Final = _member(scenario, model)
        managed_a: Final = _upload(gateway, member.key, model, "a.txt")
        with owned_proxy_process(gateway, tmp_path, {}, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = eventually(
                lambda: tuple(int(match.group(1)) for match in STARTED_WORKER.finditer(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=60,
            )
            with ThreadPoolExecutor(max_workers=20) as pool:
                burst: Final = tuple(
                    pool.submit(_tolerant_list, candidate, store, member.key, model) for _ in range(20)
                )
                eventually(held.qsize, lambda size: size == 20, seconds=60)
                held_by: Final = MappingProxyType({pid: _open_connections_to(pid, wire.url) for pid in workers})
                assert sum(held_by.values()) == 20, held_by
                victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
                victim: Final = psutil.Process(victim_pid)
                victim.suspend()
                victim.send_signal(signal.SIGKILL)
                release.set()
                served: Final = tuple(result for future in burst if (result := future.result()) is not None)
            assert held_by[survivor_pid] >= 1, held_by
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            for response in served:
                assert _ids(_listed(response)) == (managed_a,)
            assert psutil.Process(survivor_pid).is_running()
            follow_up: Final = eventually(
                lambda: _tolerant_list(candidate, store, member.key, model),
                lambda response: response is not None and response.status_code == 200,
                seconds=60,
            )
            assert follow_up is not None
            assert _ids(_listed(follow_up)) == (managed_a,)
