from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from hashlib import sha256
from typing import Final, TypeVar

import httpx
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.database import read_rows

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
T = TypeVar("T")


def object_value(value: JsonValue) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_python(value)


def string_value(value: JsonValue) -> str:
    assert isinstance(value, str), f"Expected a string, received {type(value).__name__}"
    return value


def delete_key_if_present(candidate: Gateway, key: str) -> None:
    digest: Final = sha256(key.encode()).hexdigest()
    if read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,)):
        candidate.post("/key/delete", {"keys": [key]})
    assert read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,)) == []


def eventually(
    read: Callable[[], T],
    satisfied: Callable[[T], bool],
    seconds: float = 10,
    return_last_on_timeout: bool = False,
) -> T:
    deadline: Final = time.monotonic() + seconds
    while True:
        observed: Final = read()
        if satisfied(observed):
            return observed
        if return_last_on_timeout and time.monotonic() >= deadline:
            return observed
        assert time.monotonic() < deadline, f"State did not converge: {observed!r}"
        time.sleep(0.1)


@dataclass(frozen=True, slots=True)
class Gateway:
    client: httpx.Client
    key: str
    upstream_url: str

    def request(
        self,
        method: str,
        path: str,
        body: Mapping[str, JsonValue] | None = None,
        *,
        key: str | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        request_headers: Final = {
            "Authorization": f"Bearer {self.key if key is None else key}",
            **(headers or {}),
        }
        return self.client.request(
            method,
            path,
            json=body,
            params=params,
            headers=request_headers,
        )

    def request_multipart(
        self,
        path: str,
        fields: Mapping[str, str],
        files: Mapping[str, tuple[str, bytes, str]],
        *,
        key: str | None = None,
    ) -> httpx.Response:
        return self.client.post(
            path,
            data=fields,
            files=files,
            headers={"Authorization": f"Bearer {self.key if key is None else key}"},
        )

    def post(self, path: str, body: Mapping[str, JsonValue], *, key: str | None = None) -> dict[str, JsonValue]:
        response: Final = self.request("POST", path, body, key=key)
        assert response.status_code == 200, f"POST {path}: {response.status_code} {response.text}"
        return JSON_OBJECT.validate_json(response.content)

    def get(self, path: str, params: Mapping[str, str] | None = None) -> dict[str, JsonValue]:
        response: Final = self.request("GET", path, params=params)
        assert response.status_code == 200, f"GET {path}: {response.status_code} {response.text}"
        return JSON_OBJECT.validate_json(response.content)

    def chat(self, model: str, *, key: str | None = None, text: str = "integration control") -> dict[str, JsonValue]:
        return self.post(
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": text}]},
            key=key,
        )

    @contextmanager
    def scenario(self) -> Iterator[Scenario]:
        with ExitStack() as cleanups:
            yield Scenario(self, cleanups)


@dataclass(frozen=True, slots=True)
class Scenario:
    gateway: Gateway
    cleanups: ExitStack

    def key(self, **fields: JsonValue) -> str:
        created: Final = self.gateway.post("/key/generate", fields)
        token: Final = string_value(created["key"])
        self.cleanups.callback(self.delete_key, token)
        return token

    def team(self, **fields: JsonValue) -> str:
        created: Final = self.gateway.post("/team/new", {"team_alias": f"integration-{uuid.uuid4().hex}", **fields})
        identity: Final = string_value(created["team_id"])
        self.cleanups.callback(self.delete_team, identity)
        return identity

    def delete_team(self, identity: str) -> None:
        self.gateway.post("/team/delete", {"team_ids": [identity]})
        assert read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (identity,)) == []

    def project(self, team_id: str, **fields: JsonValue) -> str:
        created: Final = self.gateway.post(
            "/project/new", {"team_id": team_id, "project_alias": f"integration-{uuid.uuid4().hex}", **fields}
        )
        identity: Final = string_value(created["project_id"])
        self.cleanups.callback(self.delete_project, identity)
        return identity

    def delete_project(self, identity: str) -> None:
        response: Final = self.gateway.request("DELETE", "/project/delete", {"project_ids": [identity]})
        assert response.status_code == 200, response.text
        assert read_rows('SELECT project_id FROM "LiteLLM_ProjectTable" WHERE project_id = %s', (identity,)) == []

    def budget(self, **fields: JsonValue) -> str:
        created: Final = self.gateway.post("/budget/new", fields)
        identity: Final = string_value(created["budget_id"])
        self.cleanups.callback(self.delete_budget, identity)
        return identity

    def delete_budget(self, identity: str) -> None:
        self.gateway.post("/budget/delete", {"id": identity})
        assert read_rows('SELECT budget_id FROM "LiteLLM_BudgetTable" WHERE budget_id = %s', (identity,)) == []

    def user(self, **fields: JsonValue) -> str:
        created: Final = self.gateway.post(
            "/user/new", {"user_id": f"integration-{uuid.uuid4().hex}", "auto_create_key": False, **fields}
        )
        identity: Final = string_value(created["user_id"])
        self.cleanups.callback(self.delete_user, identity)
        return identity

    def delete_user(self, identity: str) -> None:
        response: Final = self.gateway.request("POST", "/user/delete", {"user_ids": [identity]})
        assert response.status_code == 200 and response.json() == 1, response.text
        assert read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s', (identity,)) == []

    def delete_key(self, token: str) -> None:
        self.gateway.post("/key/delete", {"keys": [token]})
        hashed: Final = sha256(token.encode()).hexdigest()
        assert read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE token = %s', (hashed,)) == []
        info: Final = object_value(self.gateway.get("/key/info", {"key": hashed})["info"])
        assert info["status"] == "deleted", f"Deleted key still served as live: {info['status']}"

    def delete_model(self, identity: str) -> None:
        self.gateway.post("/model/delete", {"id": identity})
        entries: Final = self.gateway.get("/model/info")["data"]
        assert isinstance(entries, list)
        assert all(object_value(object_value(entry)["model_info"])["id"] != identity for entry in entries)
        assert read_rows('SELECT model_id FROM "LiteLLM_ProxyModelTable" WHERE model_id = %s', (identity,)) == []

    def model(self, *, model_info: Mapping[str, JsonValue] | None = None, **parameters: JsonValue) -> str:
        name: Final = f"integration-{uuid.uuid4().hex}"
        created: Final = self.gateway.post(
            "/model/new",
            {
                "model_name": name,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "integration-provider-key",
                    "api_base": f"{self.gateway.upstream_url}/v1",
                    **parameters,
                },
                "model_info": dict(model_info) if model_info is not None else {},
            },
        )
        identity: Final = string_value(object_value(created["model_info"])["id"])
        self.cleanups.callback(self.delete_model, identity)
        return name


@contextmanager
def gateway_from_environment() -> Iterator[Gateway]:
    url: Final = os.environ["INTEGRATION_PROXY_URL"]
    upstream: Final = os.environ["INTEGRATION_UPSTREAM_URL"]
    with httpx.Client(base_url=url, timeout=15, trust_env=False) as client:
        yield Gateway(client, os.environ["INTEGRATION_MASTER_KEY"], upstream)


def _set_team_admin_permissions(gateway: Gateway, fields: Sequence[str]) -> None:
    response: Final = gateway.request("PATCH", "/update/ui_settings", {"team_admin_editable_team_fields": list(fields)})
    assert response.status_code == 200, response.text


@contextmanager
def team_admin_permissions(gateway: Gateway, fields: Sequence[str]) -> Iterator[None]:
    """Grant team admins ``fields`` proxy-wide for the block, then restore the prior grant."""
    original: Final = object_value(gateway.get("/get/ui_settings")["values"]).get("team_admin_editable_team_fields")
    _set_team_admin_permissions(gateway, fields)
    try:
        yield
    finally:
        _set_team_admin_permissions(gateway, [str(field) for field in original] if isinstance(original, list) else ())
