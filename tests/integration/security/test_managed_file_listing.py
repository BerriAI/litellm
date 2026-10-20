import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, JsonValue

pytestmark: Final = pytest.mark.shares_worker_state

UNMANAGED_FILE: Final = "file-uploaded-straight-to-the-provider"
UPLOAD: Final = b"isolation check\n"


@dataclass(frozen=True, slots=True)
class Deployment:
    provider: str
    model: str
    litellm_model: str
    api_key: str
    api_path: str


OPENAI: Final = Deployment("openai", "gpt-4.1-managed-listing", "openai/gpt-4.1", "sk-managed-listing", "/v1")
ANTHROPIC: Final = Deployment(
    "anthropic", "claude-managed-listing", "anthropic/claude-haiku-5-5", "sk-ant-managed-listing", ""
)
DEPLOYMENTS: Final = (pytest.param(OPENAI, id="openai"), pytest.param(ANTHROPIC, id="anthropic"))


@dataclass(frozen=True, slots=True)
class ListCall:
    path: str
    params: Mapping[str, str] = MappingProxyType({})
    headers: Mapping[str, str] = MappingProxyType({})


LIST_FORMS: Final = (
    pytest.param(lambda d: ListCall(f"/{d.provider}/v1/files"), id="provider-path"),
    pytest.param(lambda d: ListCall("/v1/files", params={"provider": d.provider}), id="provider-query"),
    pytest.param(lambda d: ListCall("/v1/files", headers={"custom-llm-provider": d.provider}), id="provider-header"),
    pytest.param(lambda d: ListCall("/v1/files", params={"target_model_names": d.model}), id="target-model-names"),
    pytest.param(lambda d: ListCall("/v1/files", headers={"x-litellm-model": d.model}), id="model-header"),
)


class FilePage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    data: list[dict[str, JsonValue]]
    has_more: bool

    @property
    def ids(self) -> list[str]:
        return [str(entry["id"]) for entry in self.data]


def _openai_file(file_id: str) -> dict[str, JsonValue]:
    return {
        "id": file_id,
        "object": "file",
        "bytes": len(UPLOAD),
        "created_at": 1700000000,
        "filename": "notes.txt",
        "purpose": "user_data",
        "status": "processed",
    }


def _anthropic_file(file_id: str) -> dict[str, JsonValue]:
    return {
        "id": file_id,
        "type": "file",
        "filename": "notes.txt",
        "mime_type": "text/plain",
        "size_bytes": len(UPLOAD),
        "created_at": "2026-01-01T00:00:00Z",
        "downloadable": False,
    }


def _reply(body: Mapping[str, JsonValue]) -> Reply:
    return Reply(body=json.dumps(body).encode())


def _list_reply(provider: str, file_ids: tuple[str, ...]) -> Reply:
    if provider == ANTHROPIC.provider:
        return _reply(
            {
                "data": [_anthropic_file(file_id) for file_id in file_ids],
                "has_more": False,
                "first_id": file_ids[0],
                "last_id": file_ids[-1],
            }
        )
    return _reply({"object": "list", "data": [_openai_file(file_id) for file_id in file_ids]})


def _responder(provider_files: Mapping[str, tuple[str, ...]]) -> Callable[[Request], Reply]:
    uploads: Final = {provider: count() for provider in provider_files}

    def respond(request: Request) -> Reply:
        target: Final = urlsplit(request.target)
        path: Final = target.path
        provider: Final = ANTHROPIC.provider if "x-api-key" in request.headers else OPENAI.provider
        files: Final = provider_files[provider]
        if (request.method, path) == ("GET", "/v1/models"):
            return _reply({"object": "list", "data": []})
        if (request.method, path) == ("POST", "/v1/files"):
            uploaded: Final = files[next(uploads[provider])]
            return _reply(_anthropic_file(uploaded) if provider == ANTHROPIC.provider else _openai_file(uploaded))
        if (request.method, path) == ("GET", "/v1/files"):
            limit: Final = int(parse_qs(target.query).get("limit", ["10000"])[0])
            return _list_reply(provider, (UNMANAGED_FILE, *reversed(files))[:limit])
        return Reply(status=404, body=json.dumps({"error": {"message": f"unscripted {request.target}"}}).encode())

    return respond


def _config(wire: Wire) -> dict[str, JsonValue]:
    return {
        "model_list": [
            {
                "model_name": deployment.model,
                "litellm_params": {
                    "model": deployment.litellm_model,
                    "api_key": deployment.api_key,
                    "api_base": f"{wire.url}{deployment.api_path}",
                },
            }
            for deployment in (OPENAI, ANTHROPIC)
        ],
        "litellm_settings": {"require_managed_files": True},
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY", "database_url": "os.environ/DATABASE_URL"},
    }


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    owner: str
    intruder: str
    managed_files: tuple[str, ...]

    def list(self, call: ListCall, key: str | None = None) -> FilePage:
        response: Final = self.proxy.request("GET", call.path, key=key, params=call.params, headers=call.headers)
        assert response.status_code == 200, response.text
        return FilePage.model_validate_json(response.text)


def _upload(proxy: Gateway, key: str) -> str:
    with OpenAI(base_url=f"{str(proxy.client.base_url).rstrip('/')}/openai/v1", api_key=key, max_retries=0) as client:
        uploaded: Final = client.files.with_raw_response.create(
            file=("notes.txt", UPLOAD, "text/plain"),
            purpose="user_data",
            extra_body={"target_model_names": f"{OPENAI.model},{ANTHROPIC.model}"},
        )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.parse().id


@contextmanager
def _rig(directory: Path) -> Iterator[Rig]:
    build: Final = uuid.uuid4().hex
    provider_files: Final = {
        OPENAI.provider: (f"file-older-{build}", f"file-newer-{build}"),
        ANTHROPIC.provider: (f"file_older_{build}", f"file_newer_{build}"),
    }
    with gateway_from_environment() as shared, wire_server(_responder(provider_files)) as wire:
        config: Final = directory / "config.yaml"
        config.write_text(yaml.safe_dump(_config(wire)))
        with owned_proxy(shared, directory, {}, config=config) as proxy, proxy.scenario() as scenario:
            owner: Final = scenario.key(user_id=scenario.user())
            intruder: Final = scenario.key(user_id=scenario.user())
            older: Final = _upload(proxy, owner)
            newer: Final = _upload(proxy, owner)
            yield Rig(proxy, owner, intruder, (newer, older))


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _rig(tmp_path_factory.mktemp("managed-file-listing")) as built:
        yield built


@pytest.mark.parametrize("deployment", DEPLOYMENTS)
@pytest.mark.parametrize("form", LIST_FORMS)
def test_another_key_lists_none_of_the_files_on_the_shared_key(
    rig: Rig, deployment: Deployment, form: Callable[[Deployment], ListCall]
) -> None:
    page: Final = rig.list(form(deployment), key=rig.intruder)
    assert (page.ids, page.has_more) == ([], False)


@pytest.mark.parametrize("deployment", DEPLOYMENTS)
@pytest.mark.parametrize("form", LIST_FORMS)
def test_owner_lists_their_own_managed_files_newest_first(
    rig: Rig, deployment: Deployment, form: Callable[[Deployment], ListCall]
) -> None:
    page: Final = rig.list(form(deployment), key=rig.owner)
    assert (page.ids, page.has_more) == (list(rig.managed_files), False)


def test_owner_pages_through_their_managed_files_with_limit_and_after(rig: Rig) -> None:
    first: Final = rig.list(ListCall("/openai/v1/files", params={"limit": "1"}), key=rig.owner)
    second: Final = rig.list(ListCall("/openai/v1/files", params={"limit": "1", "after": first.ids[0]}), key=rig.owner)
    assert [(page.ids, page.has_more) for page in (first, second)] == [
        ([rig.managed_files[0]], True),
        ([rig.managed_files[1]], False),
    ]


@pytest.mark.parametrize("deployment", DEPLOYMENTS)
def test_master_key_lists_managed_files_instead_of_the_provider_key(rig: Rig, deployment: Deployment) -> None:
    listed: Final = rig.list(ListCall(f"/{deployment.provider}/v1/files")).ids
    assert UNMANAGED_FILE not in listed
    assert set(rig.managed_files) <= set(listed)
