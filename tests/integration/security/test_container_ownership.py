import json
import uuid
from collections.abc import Callable
from email.parser import BytesParser
from email.policy import HTTP
from typing import Final
from urllib.parse import parse_qsl, urlsplit

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import OpenAI, PermissionDeniedError
from openai.types import ContainerListResponse, ContainerRetrieveResponse
from openai.types.containers import FileListResponse, FileRetrieveResponse
from pydantic import BaseModel, TypeAdapter

from litellm.responses.utils import ResponsesAPIRequestUtils

_FILE_ID: Final = "cfile_ownership"
_FILE_BYTES: Final = b"tenant secret contents"
_FORBIDDEN: Final = {"detail": "Forbidden"}
# Non-admins overfetch upstream pages of 100, then trim to client limit; order/after and admin limits pass unchanged.
_OWNED_LIST_UPSTREAM_PAGE_SIZE: Final = 100
_Part = tuple[str, str | None, str, bytes]


class _Deleted(BaseModel):
    id: str
    object: str
    deleted: bool


class _ContainerPage(BaseModel):
    object: str
    data: tuple[ContainerListResponse, ...]
    first_id: str | None
    last_id: str | None
    has_more: bool


_ERROR: Final = TypeAdapter(dict[str, str])


def _container(container_id: str, name: str) -> dict[str, object]:
    return {"id": container_id, "object": "container", "created_at": 1, "status": "running", "name": name}


def _file(container_id: str) -> dict[str, object]:
    return {
        "id": _FILE_ID,
        "object": "container.file",
        "container_id": container_id,
        "created_at": 1,
        "bytes": len(_FILE_BYTES),
        "path": "/mnt/data/notes.txt",
        "source": "user",
    }


def _multipart_parts(request: Request) -> tuple[_Part, ...]:
    content_type: Final = request.headers["content-type"]
    assert content_type.startswith("multipart/form-data; boundary="), content_type
    envelope: Final = f"content-type: {content_type}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.body
    return tuple(
        (
            str(part.get_param("name", header="content-disposition")),
            part.get_filename(),
            part.get_content_type(),
            bytes(part.get_payload(decode=True)),
        )
        for part in parsed.iter_parts()
    )


def _container_upstream(container_ids: tuple[str, ...]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        segments: Final = path.removeprefix("/v1/containers").strip("/").split("/")
        if request.method == "POST" and path == "/v1/containers":
            name: Final = str(json.loads(request.body)["name"])
            return Reply(body=json.dumps(_container(container_ids[int(name.rsplit("-", 1)[1])], name)).encode())
        container_id: Final = segments[0]
        match request.method, segments[1:]:
            case "GET", []:
                return Reply(body=json.dumps(_container(container_id, "c-0")).encode())
            case "DELETE", []:
                return Reply(
                    body=json.dumps({"id": container_id, "object": "container.deleted", "deleted": True}).encode()
                )
            case "GET", ["files"]:
                return Reply(
                    body=json.dumps(
                        {
                            "object": "list",
                            "data": [_file(container_id)],
                            "first_id": _FILE_ID,
                            "last_id": _FILE_ID,
                            "has_more": False,
                        }
                    ).encode()
                )
            case "POST", ["files"]:
                assert _multipart_parts(request) == (("file", "notes.txt", "text/plain", _FILE_BYTES),), request.body
                return Reply(body=json.dumps(_file(container_id)).encode())
            case "GET", ["files", _]:
                return Reply(body=json.dumps(_file(container_id)).encode())
            case "GET", ["files", _, "content"]:
                return Reply(body=_FILE_BYTES, content_type="application/octet-stream")
            case "DELETE", ["files", _]:
                return Reply(
                    body=json.dumps({"id": _FILE_ID, "object": "container.file.deleted", "deleted": True}).encode()
                )
        return Reply(status=404, body=b'{"error": {"message": "unscripted container route"}}')

    return respond


def _user_key(scenario: Scenario) -> tuple[str, str]:
    user_id: Final = scenario.user(user_role="internal_user")
    return user_id, scenario.key(user_id=user_id)


def _client(gateway: Gateway, key: str, model: str) -> OpenAI:
    client: Final = OpenAI(base_url=str(gateway.client.base_url.join("/v1")), api_key=key, max_retries=0)
    eventually(lambda: tuple(entry.id for entry in client.models.list()), lambda ids: model in ids)
    return client


def _routable_over_alias(gateway: Gateway, model: str) -> None:
    def listed() -> tuple[str, ...]:
        response: Final = gateway.client.get("/models", headers={"Authorization": f"Bearer {gateway.key}"})
        assert response.status_code == 200, response.text
        return tuple(str(entry["id"]) for entry in response.json()["data"])

    eventually(listed, lambda ids: model in ids)


def _create(client: OpenAI, model: str, index: int) -> str:
    created: Final = client.containers.create(name=f"c-{index}", extra_body={"model": model})
    assert created.object == "container", created.model_dump_json()
    return created.id


def _owner_rows(original_container_id: str) -> list[dict[str, object]]:
    return read_rows(
        'SELECT unified_object_id, file_purpose, created_by FROM "LiteLLM_ManagedObjectTable" WHERE model_object_id=%s',
        (f"container:openai:{original_container_id}",),
    )


def _requests(wire: Wire) -> list[tuple[str, str, str]]:
    return [(request.method, request.target, request.headers["authorization"]) for request in wire.drain()]


def _refused_with_sdk(call: Callable[[], object]) -> None:
    with pytest.raises(PermissionDeniedError) as refused:
        call()
    assert refused.value.status_code == 403, refused.value.response.text
    assert _ERROR.validate_json(refused.value.response.text) == _FORBIDDEN, refused.value.response.text


def _refused_over_alias(response: httpx.Response) -> None:
    assert response.status_code == 403, response.text
    assert _ERROR.validate_json(response.text) == _FORBIDDEN, response.text


def test_container_routes_refuse_another_users_key_and_serve_the_owner(gateway: Gateway) -> None:
    original: Final = f"cntr_{uuid.uuid4().hex}"
    provider_key: Final = f"container-key-{uuid.uuid4().hex}"
    with wire_server(_container_upstream((original,))) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=f"{wire.url}/v1", api_key=provider_key)
        owner_id, owner_key = _user_key(scenario)
        _other_id, other_key = _user_key(scenario)
        owner: Final = _client(gateway, owner_key, model)
        other: Final = _client(gateway, other_key, model)
        _routable_over_alias(gateway, model)
        created: Final = owner.containers.create(
            name="c-0",
            expires_after={"anchor": "last_active_at", "minutes": 20},
            file_ids=["file-seed"],
            extra_body={"model": model},
        )
        container_id: Final = created.id
        assert ResponsesAPIRequestUtils.decode_container_id_to_original(container_id) == original, container_id
        assert _owner_rows(original) == [
            {"unified_object_id": container_id, "file_purpose": "container", "created_by": owner_id}
        ]
        create_requests: Final = wire.drain()
        assert [
            (request.method, request.target, request.headers["authorization"]) for request in create_requests
        ] == [("POST", "/v1/containers", f"Bearer {provider_key}")]
        assert json.loads(create_requests[0].body) == {
            "name": "c-0",
            "expires_after": {"anchor": "last_active_at", "minutes": 20},
            "file_ids": ["file-seed"],
        }, create_requests[0].body

        _refused_with_sdk(lambda: other.containers.retrieve(container_id))
        _refused_with_sdk(lambda: other.containers.files.list(container_id))
        _refused_with_sdk(lambda: other.containers.files.create(container_id, file=("x.txt", b"x", "text/plain")))
        _refused_with_sdk(lambda: other.containers.files.retrieve(_FILE_ID, container_id=container_id))
        _refused_with_sdk(lambda: other.containers.files.content.retrieve(_FILE_ID, container_id=container_id))
        _refused_with_sdk(lambda: other.containers.files.delete(_FILE_ID, container_id=container_id))
        _refused_with_sdk(lambda: other.containers.delete(container_id))
        alias: Final = f"/containers/{container_id}"
        other_auth: Final = {"Authorization": f"Bearer {other_key}"}
        _refused_over_alias(gateway.client.get(alias, headers=other_auth))
        _refused_over_alias(gateway.client.get(f"{alias}/files", headers=other_auth))
        _refused_over_alias(
            gateway.client.post(f"{alias}/files", files={"file": ("x.txt", b"x", "text/plain")}, headers=other_auth)
        )
        _refused_over_alias(gateway.client.get(f"{alias}/files/{_FILE_ID}", headers=other_auth))
        _refused_over_alias(gateway.client.get(f"{alias}/files/{_FILE_ID}/content", headers=other_auth))
        _refused_over_alias(gateway.client.delete(f"{alias}/files/{_FILE_ID}", headers=other_auth))
        _refused_over_alias(gateway.client.delete(alias, headers=other_auth))
        assert _requests(wire) == []

        retrieved: Final = owner.containers.retrieve(container_id)
        assert retrieved == ContainerRetrieveResponse.model_validate(_container(original, "c-0")), retrieved
        over_alias: Final = gateway.client.get(alias, headers={"Authorization": f"Bearer {owner_key}"})
        assert over_alias.status_code == 200, over_alias.text
        assert ContainerRetrieveResponse.model_validate_json(over_alias.text) == retrieved, over_alias.text
        listed: Final = owner.containers.files.list(container_id)
        assert listed.data == [FileListResponse.model_validate(_file(original))], listed
        uploaded: Final = owner.containers.files.create(container_id, file=("notes.txt", _FILE_BYTES, "text/plain"))
        assert uploaded.id == _FILE_ID, uploaded
        fetched: Final = owner.containers.files.retrieve(_FILE_ID, container_id=container_id)
        assert fetched == FileRetrieveResponse.model_validate(_file(original)), fetched
        content: Final = owner.containers.files.content.retrieve(_FILE_ID, container_id=container_id)
        assert content.read() == _FILE_BYTES
        file_deleted: Final = gateway.client.delete(
            f"/v1/containers/{container_id}/files/{_FILE_ID}", headers={"Authorization": f"Bearer {owner_key}"}
        )
        assert file_deleted.status_code == 200, file_deleted.text
        assert _Deleted.model_validate_json(file_deleted.text) == _Deleted(
            id=_FILE_ID, object="container.file.deleted", deleted=True
        ), file_deleted.text
        deleted: Final = gateway.client.delete(
            f"/v1/containers/{container_id}", headers={"Authorization": f"Bearer {owner_key}"}
        )
        assert deleted.status_code == 200, deleted.text
        assert _Deleted.model_validate_json(deleted.text) == _Deleted(
            id=original, object="container.deleted", deleted=True
        ), deleted.text
        prefix: Final = f"/v1/containers/{original}"
        auth: Final = f"Bearer {provider_key}"
        assert _requests(wire) == [
            ("GET", prefix, auth),
            ("GET", prefix, auth),
            ("GET", f"{prefix}/files", auth),
            ("POST", f"{prefix}/files", auth),
            ("GET", f"{prefix}/files/{_FILE_ID}", auth),
            ("GET", f"{prefix}/files/{_FILE_ID}/content", auth),
            ("DELETE", f"{prefix}/files/{_FILE_ID}", auth),
            ("DELETE", prefix, auth),
        ]


def _listing_upstream(
    container_ids: tuple[str, ...], pages: dict[str | None, tuple[tuple[str, ...], bool]]
) -> Callable[[Request], Reply]:
    created: Final = _container_upstream(container_ids)

    def respond(request: Request) -> Reply:
        if request.method != "GET" or urlsplit(request.target).path != "/v1/containers":
            return created(request)
        after: Final = dict(parse_qsl(urlsplit(request.target).query)).get("after")
        ids, has_more = pages[after]
        return Reply(
            body=json.dumps(
                {
                    "object": "list",
                    "data": [_container(container_id, container_id) for container_id in ids],
                    "first_id": ids[0],
                    "last_id": ids[-1],
                    "has_more": has_more,
                }
            ).encode()
        )

    return respond


def _page(ids: tuple[str, ...], has_more: bool) -> _ContainerPage:
    return _ContainerPage(
        object="list",
        data=tuple(
            ContainerListResponse.model_validate(_container(container_id, container_id)) for container_id in ids
        ),
        first_id=ids[0] if ids else None,
        last_id=ids[-1] if ids else None,
        has_more=has_more,
    )


def test_container_list_returns_only_the_callers_containers_across_upstream_pages(gateway: Gateway) -> None:
    run: Final = uuid.uuid4().hex
    own_first, own_second, own_third, foreign_first, foreign_second, ownerless = (
        f"cntr_{name}_{run}" for name in ("a1", "a2", "a3", "b1", "b2", "orphan")
    )
    created_ids: Final = (own_first, own_second, own_third, foreign_first, foreign_second)
    first_page: Final = (own_first, foreign_first, own_second, ownerless)
    second_page: Final = (own_third, foreign_second)
    pages: Final = {
        None: (first_page, True),
        ownerless: (second_page, False),
        own_second: ((ownerless, *second_page), False),
    }
    provider_key: Final = f"container-key-{uuid.uuid4().hex}"
    with wire_server(_listing_upstream(created_ids, pages)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=f"{wire.url}/v1", api_key=provider_key)
        _owner_id, owner_key = _user_key(scenario)
        _other_id, other_key = _user_key(scenario)
        owner: Final = _client(gateway, owner_key, model)
        other: Final = _client(gateway, other_key, model)
        _routable_over_alias(gateway, model)
        for index in range(2):
            _create(owner, model, index)
        for index in range(3, 5):
            _create(other, model, index)
        assert _requests(wire) == [
            ("POST", "/v1/containers", f"Bearer {provider_key}"),
            ("POST", "/v1/containers", f"Bearer {provider_key}"),
            ("POST", "/v1/containers", f"Bearer {provider_key}"),
            ("POST", "/v1/containers", f"Bearer {provider_key}"),
        ]

        first: Final = owner.containers.with_raw_response.list(limit=2, order="desc", extra_query={"model": model})
        assert _ContainerPage.model_validate_json(first.http_response.text) == _page((own_first, own_second), False), (
            first.http_response.text
        )
        assert [container.id for container in first.parse().data] == [own_first, own_second]
        auth: Final = f"Bearer {provider_key}"
        assert _requests(wire) == [
            ("GET", f"/v1/containers?limit={_OWNED_LIST_UPSTREAM_PAGE_SIZE}&order=desc", auth),
            (
                "GET",
                f"/v1/containers?after={ownerless}&limit={_OWNED_LIST_UPSTREAM_PAGE_SIZE}&order=desc",
                auth,
            ),
        ]

        _create(owner, model, 2)
        assert _requests(wire) == [("POST", "/v1/containers", f"Bearer {provider_key}")]

        listed_after_create: Final = owner.containers.with_raw_response.list(
            limit=2, order="desc", extra_query={"model": model}
        )
        assert _ContainerPage.model_validate_json(listed_after_create.http_response.text) == _page(
            (own_first, own_second), True
        ), listed_after_create.http_response.text
        assert [container.id for container in listed_after_create.parse().data] == [own_first, own_second]
        assert _requests(wire) == [
            ("GET", f"/v1/containers?limit={_OWNED_LIST_UPSTREAM_PAGE_SIZE}&order=desc", auth),
            (
                "GET",
                f"/v1/containers?after={ownerless}&limit={_OWNED_LIST_UPSTREAM_PAGE_SIZE}&order=desc",
                auth,
            ),
        ]

        alias_page: Final = gateway.client.get(
            "/containers",
            params={"limit": "2", "after": own_second, "order": "desc", "model": model},
            headers={"Authorization": f"Bearer {owner_key}"},
        )
        assert alias_page.status_code == 200, alias_page.text
        assert _ContainerPage.model_validate_json(alias_page.text) == _page((own_third,), False), alias_page.text
        assert _requests(wire) == [
            ("GET", f"/v1/containers?after={own_second}&limit={_OWNED_LIST_UPSTREAM_PAGE_SIZE}&order=desc", auth)
        ]

        admin: Final = _client(gateway, gateway.key, model).containers.with_raw_response.list(
            limit=2, order="desc", extra_query={"model": model}
        )
        assert _ContainerPage.model_validate_json(admin.http_response.text) == _page(first_page, True), (
            admin.http_response.text
        )
        assert _requests(wire) == [("GET", "/v1/containers?limit=2&order=desc", auth)]
