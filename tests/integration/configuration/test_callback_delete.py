import json
import os
import uuid
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, JsonValue, eventually, list_value, object_value
from integration._support.database import read_rows, scratch_database
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server

CALLBACK: Final = "generic_api"
SIBLING: Final = "langsmith"
LIST_TYPES: Final = MappingProxyType(
    {
        "success_callback": "success",
        "failure_callback": "failure",
        "callbacks": "success_and_failure",
    }
)
SINK_ENVIRONMENT: Final = MappingProxyType(
    {
        "GENERIC_LOGGER_ENDPOINT": "http://127.0.0.1:9/unused",
        "LANGSMITH_API_KEY": "synthetic-langsmith-key",
        "LANGSMITH_BASE_URL": "http://127.0.0.1:9",
    }
)
NOT_FOUND: Final = f"Callback '{CALLBACK}' not found in active configuration"
WORKER_SYNC_SECONDS: Final = 40


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    database_url: str


@contextmanager
def _rig(gateway: Gateway, tmp_path: Path, file_settings: Mapping[str, JsonValue]) -> Generator[Rig]:
    config: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    path: Final = tmp_path / f"callback_delete_{uuid.uuid4().hex}.yaml"
    path.write_text(
        yaml.safe_dump({**config, "litellm_settings": {**object_value(config["litellm_settings"]), **file_settings}})
    )
    with (
        scratch_database() as database_url,
        owned_proxy(
            gateway,
            tmp_path,
            {"DATABASE_URL": database_url, **SINK_ENVIRONMENT},
            config=path,
            remove_environment=("DATABASE_URL_READ_REPLICA",),
            workers=2,
        ) as proxy,
    ):
        yield Rig(proxy, database_url)


@pytest.fixture
def rig(gateway: Gateway, tmp_path: Path) -> Iterator[Rig]:
    with _rig(gateway, tmp_path, {}) as created:
        yield created


def _configured(proxy: Gateway) -> tuple[tuple[str, str], ...]:
    rows: Final = proxy.get("/get/config/callbacks")["callbacks"]
    assert isinstance(rows, list)
    return tuple(
        sorted(
            (str(row["name"]), str(row["type"]))
            for row in (object_value(entry) for entry in rows)
            if not row.get("read_only")
        )
    )


def _listed(proxy: Gateway, expected: tuple[tuple[str, str], ...]) -> None:
    eventually(lambda: _configured(proxy), lambda rows: rows == expected, seconds=WORKER_SYNC_SECONDS)


def _stored_settings(database_url: str) -> Mapping[str, JsonValue]:
    rows: Final = read_rows(
        'SELECT param_value FROM "LiteLLM_Config" WHERE param_name = %s',
        ("litellm_settings",),
        database_url=database_url,
    )
    assert len(rows) == 1, rows
    value: Final = rows[0]["param_value"]
    return object_value(json.loads(value) if isinstance(value, str) else value)


def _configure(
    proxy: Gateway, settings: Mapping[str, JsonValue], environment: Mapping[str, JsonValue] | None = None
) -> None:
    body: Final[Mapping[str, JsonValue]] = {
        "litellm_settings": dict(settings),
        "environment_variables": None if environment is None else dict(environment),
    }
    assert proxy.post("/config/update", body) == {"message": "Config updated successfully"}


def _delete(proxy: Gateway, name: str) -> tuple[int, Mapping[str, JsonValue]]:
    response: Final = proxy.request("POST", "/config/callback/delete", {"callback_name": name})
    return response.status_code, JSON_OBJECT.validate_json(response.content)


@pytest.mark.parametrize("list_key", tuple(LIST_TYPES))
def test_callback_listed_under_any_settings_list_is_deleted_from_the_listing_and_the_store(
    rig: Rig, list_key: str
) -> None:
    _configure(rig.proxy, {list_key: [CALLBACK]})
    _listed(rig.proxy, ((CALLBACK, LIST_TYPES[list_key]),))

    status, body = _delete(rig.proxy, CALLBACK)

    assert status == 200, body
    assert (body["removed_callback"], body["remaining_callbacks"]) == (CALLBACK, [])
    assert _stored_settings(rig.database_url)[list_key] == []
    _listed(rig.proxy, ())


def test_callback_in_both_success_and_failure_lists_is_removed_from_both_in_one_delete(rig: Rig) -> None:
    _configure(rig.proxy, {"success_callback": [CALLBACK], "failure_callback": [CALLBACK]})
    _listed(rig.proxy, ((CALLBACK, "failure"), (CALLBACK, "success")))

    status, body = _delete(rig.proxy, CALLBACK)

    assert status == 200, body
    stored: Final = _stored_settings(rig.database_url)
    assert (stored["success_callback"], stored["failure_callback"]) == ([], [])
    _listed(rig.proxy, ())


def test_delete_keeps_other_callbacks_and_reports_what_is_left_in_the_list_it_edited(rig: Rig) -> None:
    _configure(rig.proxy, {"callbacks": [SIBLING, CALLBACK], "failure_callback": [SIBLING]})
    _listed(rig.proxy, ((CALLBACK, "success_and_failure"), (SIBLING, "failure"), (SIBLING, "success_and_failure")))

    status, body = _delete(rig.proxy, CALLBACK)

    assert status == 200, body
    assert body["remaining_callbacks"] == [SIBLING]
    stored: Final = _stored_settings(rig.database_url)
    assert (stored["callbacks"], stored["failure_callback"]) == ([SIBLING], [SIBLING])
    _listed(rig.proxy, ((SIBLING, "failure"), (SIBLING, "success_and_failure")))


def test_delete_matches_the_callback_name_case_insensitively(rig: Rig) -> None:
    _configure(rig.proxy, {"failure_callback": [CALLBACK.upper()], "callbacks": [CALLBACK]})
    _listed(rig.proxy, ((CALLBACK.upper(), "failure"), (CALLBACK, "success_and_failure")))

    status, body = _delete(rig.proxy, CALLBACK.upper())

    assert status == 200, body
    assert body["removed_callback"] == CALLBACK
    _listed(rig.proxy, ())


def test_delete_of_an_unconfigured_callback_is_a_404_and_changes_nothing(rig: Rig) -> None:
    _configure(rig.proxy, {"callbacks": [SIBLING]})
    _listed(rig.proxy, ((SIBLING, "success_and_failure"),))
    before: Final = _stored_settings(rig.database_url)

    status, body = _delete(rig.proxy, CALLBACK)

    assert (status, body) == (404, {"detail": {"error": NOT_FOUND}})
    assert _stored_settings(rig.database_url) == before
    _listed(rig.proxy, ((SIBLING, "success_and_failure"),))


@pytest.mark.parametrize(
    ("file_key", "database_key"), (("failure_callback", "success_callback"), ("success_callback", "failure_callback"))
)
def test_delete_removes_the_database_copy_and_keeps_the_copy_the_config_file_sets(
    gateway: Gateway, tmp_path: Path, file_key: str, database_key: str
) -> None:
    with _rig(gateway, tmp_path, {file_key: [CALLBACK]}) as file_owned:
        _configure(file_owned.proxy, {database_key: [CALLBACK]})
        _listed(file_owned.proxy, ((CALLBACK, "failure"), (CALLBACK, "success")))

        status, body = _delete(file_owned.proxy, CALLBACK)

        assert status == 200, body
        assert body["remaining_callbacks"] == []
        assert _stored_settings(file_owned.database_url)[database_key] == []
        _listed(file_owned.proxy, ((CALLBACK, LIST_TYPES[file_key]),))


def test_delete_of_a_callback_only_the_config_file_lists_is_refused_and_names_the_file(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _rig(gateway, tmp_path, {"callbacks": [CALLBACK]}) as file_owned:
        _listed(file_owned.proxy, ((CALLBACK, "success_and_failure"),))

        status, body = _delete(file_owned.proxy, CALLBACK)

        assert status == 400, body
        detail: Final = object_value(body["detail"])
        assert (detail["keys"], detail["section"]) == (["callbacks"], "litellm_settings")
        assert "set in the config file" in str(detail["error"])
        _listed(file_owned.proxy, ((CALLBACK, "success_and_failure"),))


def _completion(request: Request) -> Reply:
    message: Final = object_value(list_value(JSON_OBJECT.validate_json(request.body)["messages"])[0])
    if str(message["content"]).endswith("-failure"):
        return Reply(
            status=400,
            body=json.dumps(
                {"error": {"type": "invalid_request_error", "code": "synthetic", "message": "synthetic failure"}}
            ).encode(),
        )
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
        ).encode()
    )


@dataclass(slots=True)
class Sink:
    wire: Wire
    bodies: list[bytes]  # mutable-ok: drain() consumes the queue, later reads keep earlier batches

    def received(self, markers: tuple[str, ...]) -> bool:
        self.bodies.extend(request.body for request in self.wire.drain())
        return all(any(marker.encode() in body for body in self.bodies) for marker in markers)


@dataclass(frozen=True, slots=True)
class Pair:
    first: Gateway
    peer: Gateway
    database_url: str
    provider: Wire
    generic: Sink
    langsmith: Sink

    def running(self, name: str) -> tuple[bool, bool]:
        return (_runs(self.first, name), _runs(self.peer, name))


def _runs(proxy: Gateway, name: str) -> bool:
    active: Final = list_value(proxy.get("/active/callbacks")["all_litellm_callbacks"])
    return any(name in str(entry).lower() for entry in active)


@contextmanager
def _pair(gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Pair]:
    with (
        scratch_database() as database_url,
        wire_server(_completion) as provider,
        wire_server(lambda _: Reply()) as generic,
        wire_server(lambda _: Reply()) as langsmith,
    ):
        monkeypatch.setenv("DATABASE_URL", database_url)
        environment: Final = {
            "DATABASE_URL": database_url,
            "DEFAULT_FLUSH_INTERVAL_SECONDS": "1",
            "LITELLM_STORE_AUDIT_LOGS": "True",
        }
        removed: Final = (
            "DATABASE_URL_READ_REPLICA",
            "GENERIC_LOGGER_ENDPOINT",
            "GENERIC_LOGGER_HEADERS",
            "LANGSMITH_API_KEY",
            "LANGSMITH_BASE_URL",
        )
        with (
            owned_proxy(gateway, tmp_path, environment, remove_environment=removed) as first,
            owned_proxy(gateway, tmp_path, environment, remove_environment=removed) as peer,
        ):
            yield Pair(first, peer, database_url, provider, Sink(generic, []), Sink(langsmith, []))


def _send_on_both(pair: Pair, model: str, key: str, marker: str, outcome: str) -> tuple[str, str]:
    markers: Final = (f"{marker}-first-{outcome}", f"{marker}-peer-{outcome}")
    for proxy, text in zip((pair.first, pair.peer), markers, strict=True):
        response = proxy.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": text}]}, key=key
        )
        assert response.status_code == (400 if outcome == "failure" else 200), response.text
    return markers


@pytest.mark.parametrize(("list_key", "outcome"), (("failure_callback", "failure"), ("callbacks", "success")))
def test_deleted_callback_stops_receiving_events_on_both_processes(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, list_key: str, outcome: str
) -> None:
    assert os.environ.get("LITELLM_LICENSE"), "LITELLM_LICENSE must be set: audit logs are an enterprise feature"
    marker: Final = "callback-delete-" + uuid.uuid4().hex
    with _pair(gateway, tmp_path, monkeypatch) as pair, pair.first.scenario() as scenario:
        _configure(
            pair.first,
            {list_key: [SIBLING, CALLBACK]},
            {
                "GENERIC_LOGGER_ENDPOINT": pair.generic.wire.url,
                "LANGSMITH_API_KEY": "langsmith-key-" + marker,
                "LANGSMITH_BASE_URL": pair.langsmith.wire.url,
            },
        )
        eventually(lambda: pair.running(SIBLING), lambda running: running == (True, True), seconds=WORKER_SYNC_SECONDS)
        model: Final = scenario.model(api_base=pair.provider.url + "/v1")
        key: Final = scenario.key(models=[model])
        before: Final = _send_on_both(pair, model, key, marker + "-before", outcome)
        eventually(lambda: pair.generic.received(before), bool, seconds=25)
        eventually(lambda: pair.langsmith.received(before), bool, seconds=25)

        status, body = _delete(pair.first, SIBLING)

        assert status == 200, body
        assert (body["removed_callback"], body["remaining_callbacks"]) == (SIBLING, [CALLBACK])
        assert _stored_settings(pair.database_url)[list_key] == [CALLBACK]
        audit: Final = eventually(
            lambda: read_rows(
                'SELECT before_value, updated_values FROM "LiteLLM_AuditLog" WHERE object_id = %s AND action = %s',
                ("litellm_settings", "deleted"),
                database_url=pair.database_url,
            ),
            bool,
            seconds=10,
        )
        assert len(audit) == 1, audit
        before_value: Final = list_value(object_value(audit[0]["before_value"])[list_key])
        assert sorted(str(name) for name in before_value) == sorted([SIBLING, CALLBACK]), audit
        assert audit[0]["updated_values"] == {list_key: [CALLBACK]}, audit
        eventually(
            lambda: pair.running(SIBLING), lambda running: running == (False, False), seconds=WORKER_SYNC_SECONDS
        )

        after: Final = _send_on_both(pair, model, key, marker + "-after", outcome)
        eventually(lambda: pair.generic.received(after), bool, seconds=25)
        _configure(pair.first, {list_key: [CALLBACK, SIBLING]})
        eventually(lambda: pair.running(SIBLING), lambda running: running == (True, True), seconds=WORKER_SYNC_SECONDS)
        barrier: Final = _send_on_both(pair, model, key, marker + "-barrier", outcome)
        eventually(lambda: pair.langsmith.received(barrier), bool, seconds=25)

        leaked: Final = tuple(text for text in after if any(text.encode() in body for body in pair.langsmith.bodies))
        assert leaked == (), leaked
