import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import httpx
import psycopg
import pytest
from integration._support.client import Gateway, delete_key_if_present, eventually, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from psycopg import sql
from pydantic import BaseModel, JsonValue, TypeAdapter


@pytest.mark.covers("other.database.regeneration.writer_updates_dependent_grants")
def test_key_regeneration_uses_writer_with_a_real_readonly_reader(gateway: Gateway, tmp_path: Path) -> None:
    role: Final = f"integration_reader_{uuid.uuid4().hex}"
    url: Final = os.environ["DATABASE_URL"]
    parsed: Final = urlsplit(url)
    reader_url: Final = urlunsplit(
        parsed._replace(netloc=f"{role}:integration-reader-password@{parsed.hostname}:{parsed.port}")
    )
    with psycopg.connect(url, autocommit=True) as admin:
        admin.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD 'integration-reader-password' NOSUPERUSER NOINHERIT").format(
                sql.Identifier(role)
            )
        )
        try:
            admin.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(role)))
            admin.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(sql.Identifier(role)))
            admin.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(sql.Identifier(role)))
            with psycopg.connect(reader_url, autocommit=True) as reader:
                assert reader.execute("SHOW transaction_read_only").fetchone() == ("on",)
                with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
                    reader.execute('UPDATE "LiteLLM_VerificationToken" SET blocked = true WHERE false')
            with owned_proxy(gateway, tmp_path, {"DATABASE_URL_READ_REPLICA": reader_url}) as candidate:
                assert read_rows("SELECT pid FROM pg_stat_activity WHERE usename=%s", (role,)), (
                    "Candidate reader was never connected"
                )
                with gateway.scenario() as scenario:
                    model: Final = scenario.model()
                    outside: Final = scenario.model()
                    old: Final = string_value(candidate.post("/key/generate", {"models": [outside]})["key"])
                    new: Final = f"sk-integration-{uuid.uuid4().hex}"
                    scenario.cleanups.callback(delete_key_if_present, gateway, old)
                    scenario.cleanups.callback(delete_key_if_present, gateway, new)
                    old_hash: Final = sha256(old.encode()).hexdigest()
                    before: Final = candidate.request(
                        "POST",
                        "/v1/chat/completions",
                        {"model": model, "messages": [{"role": "user", "content": "no grant yet"}]},
                        key=old,
                    )
                    assert before.status_code == 403 and before.json()["error"]["type"] == "key_model_access_denied", (
                        before.text
                    )
                    response: Final = candidate.request(
                        "POST",
                        "/v1/access_group",
                        {
                            "access_group_name": f"integration-{uuid.uuid4().hex}",
                            "access_model_names": [model],
                            "assigned_key_ids": [old_hash],
                        },
                    )
                    assert response.status_code == 201, response.text
                    group: Final = string_value(response.json()["access_group_id"])
                    try:
                        with psycopg.connect(url) as blocker, ThreadPoolExecutor(max_workers=1) as executor:
                            blocker.execute('LOCK TABLE "LiteLLM_AccessGroupTable" IN ACCESS EXCLUSIVE MODE')
                            pending: Final = executor.submit(candidate.request, "GET", f"/v1/access_group/{group}")
                            try:
                                reached: Final = eventually(
                                    lambda: read_rows(
                                        "SELECT usename FROM pg_stat_activity WHERE %s=ANY(pg_blocking_pids(pid)) "
                                        "AND usename=%s AND query LIKE 'SELECT%%'",
                                        (blocker.info.backend_pid, role),
                                    ),
                                    bool,
                                    seconds=3,
                                )
                                assert reached == [{"usename": role}]
                            finally:
                                blocker.rollback()
                            selected: Final = pending.result(timeout=5)
                            assert selected.status_code == 200 and selected.json()["access_group_id"] == group, (
                                selected.text
                            )
                        assert candidate.chat(model, key=old)["usage"]["total_tokens"] == 40
                        regenerated: Final = candidate.post(
                            "/key/regenerate", {"key": old, "new_key": new, "grace_period": "0s"}
                        )
                        assert regenerated["key"] == new
                        new_hash: Final = sha256(new.encode()).hexdigest()
                        assert new != old
                        assert read_rows(
                            'SELECT assigned_key_ids FROM "LiteLLM_AccessGroupTable" WHERE access_group_id=%s', (group,)
                        ) == [{"assigned_key_ids": [new_hash]}]
                        assert read_rows(
                            'SELECT token, access_group_ids FROM "LiteLLM_VerificationToken" WHERE token=ANY(%s)',
                            ([old_hash, new_hash],),
                        ) == [{"token": new_hash, "access_group_ids": [group]}]
                        assert candidate.chat(model, key=new)["usage"]["total_tokens"] == 40
                        assert candidate.chat(outside, key=new)["usage"]["total_tokens"] == 40
                        denied: Final = candidate.request(
                            "POST",
                            "/v1/chat/completions",
                            {"model": model, "messages": [{"role": "user", "content": "rotated key"}]},
                            key=old,
                        )
                        assert (
                            denied.status_code == 401 and denied.json()["error"]["type"] == "token_not_found_in_db"
                        ), denied.text
                    finally:
                        deleted: Final = gateway.request("DELETE", f"/v1/access_group/{group}")
                        assert deleted.status_code == 204, deleted.text
                        assert (
                            read_rows(
                                'SELECT access_group_id FROM "LiteLLM_AccessGroupTable" WHERE access_group_id=%s',
                                (group,),
                            )
                            == []
                        )
        finally:
            admin.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
            admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
    assert read_rows("SELECT rolname FROM pg_roles WHERE rolname=%s", (role,)) == []


class _Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class _AssistantMessage(BaseModel):
    content: str


class _Choice(BaseModel):
    message: _AssistantMessage


class _ChatResponse(BaseModel):
    id: str
    choices: tuple[_Choice, ...]
    usage: _Usage


class _GenerateKeyResponse(BaseModel):
    key: str


class _Error(BaseModel):
    type: str


class _ErrorResponse(BaseModel):
    error: _Error


class _SpendLog(BaseModel):
    api_key: str
    spend: float


class _KeySpend(BaseModel):
    spend: float


class _DeprecatedKey(BaseModel):
    active_token_id: str
    grace_seconds: float


_CHAT_BODY: Final = TypeAdapter(dict[str, JsonValue])
_EXPECTED_CHAT: Final = ("POST", "/v1/chat/completions")


def _chat_body(model: str, prompt: str) -> dict[str, JsonValue]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }


def _wire_reply(identity: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "chat.completion",
                "created": 1,
                "model": "integration-keys-auth-grace",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "wire response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40},
            }
        ).encode()
    )


def _chat(gateway: Gateway, model: str, key: str, prompt: str) -> httpx.Response:
    return gateway.request("POST", "/v1/chat/completions", _chat_body(model, prompt), key=key)


def _assert_success(response: httpx.Response) -> _ChatResponse:
    assert response.status_code == 200, response.text
    payload: Final = _ChatResponse.model_validate_json(response.content)
    assert payload.choices[0].message.content == "wire response", response.text
    assert payload.usage == _Usage(prompt_tokens=20, completion_tokens=20, total_tokens=40), response.text
    return payload


@pytest.mark.parametrize("spelling", ("body", "path"))
def test_regenerate_with_a_grace_period_keeps_the_old_key_serving_billed_to_the_new_row(
    gateway: Gateway, spelling: str
) -> None:
    assert os.environ.get("LITELLM_KEY_ROTATION_GRACE_PERIOD") is None
    backend: Final = "integration-keys-auth-grace"
    prompts: Final = (
        "grace old key " + uuid.uuid4().hex,
        "grace new key " + uuid.uuid4().hex,
        "revoked second key " + uuid.uuid4().hex,
        "final generated key " + uuid.uuid4().hex,
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        body: Final = _CHAT_BODY.validate_json(request.body)
        assert body in tuple(_chat_body(backend, prompt) for prompt in prompts), body
        return _wire_reply("chatcmpl-" + uuid.uuid4().hex)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{backend}",
            api_base=f"{wire.url}/v1",
            api_key="synthetic-openai-key",
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        created: Final = gateway.request("POST", "/key/generate", {"models": [model]})
        assert created.status_code == 200, created.text
        key_one: Final = _GenerateKeyResponse.model_validate_json(created.content).key
        scenario.cleanups.callback(delete_key_if_present, gateway, key_one)
        hash_one: Final = sha256(key_one.encode()).hexdigest()

        generated_two: Final = (
            gateway.request("POST", "/key/regenerate", {"key": key_one, "grace_period": "1h"})
            if spelling == "body"
            else gateway.request("POST", f"/key/{hash_one}/regenerate", {"grace_period": "1h"})
        )
        assert generated_two.status_code == 200, generated_two.text
        key_two: Final = _GenerateKeyResponse.model_validate_json(generated_two.content).key
        scenario.cleanups.callback(delete_key_if_present, gateway, key_two)
        hash_two: Final = sha256(key_two.encode()).hexdigest()
        old_response: Final = _assert_success(_chat(gateway, model, key_one, prompts[0]))
        new_response: Final = _assert_success(_chat(gateway, model, key_two, prompts[1]))
        assert tuple((request.method, request.target) for request in wire.drain()) == (_EXPECTED_CHAT,) * 2

        old_spend: Final = TypeAdapter(tuple[_SpendLog, ...]).validate_python(
            eventually(
                lambda: read_rows(
                    'SELECT api_key, spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                    (old_response.id,),
                ),
                lambda rows: len(rows) == 1,
                seconds=70,
            )
        )
        new_spend: Final = TypeAdapter(tuple[_SpendLog, ...]).validate_python(
            eventually(
                lambda: read_rows(
                    'SELECT api_key, spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                    (new_response.id,),
                ),
                lambda rows: len(rows) == 1,
                seconds=70,
            )
        )
        assert old_spend[0].api_key == hash_two
        assert old_spend[0].spend > 0
        assert new_spend[0].api_key == hash_two
        assert new_spend[0].spend > 0
        key_spend: Final = TypeAdapter(tuple[_KeySpend, ...]).validate_python(
            eventually(
                lambda: read_rows(
                    'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s',
                    (hash_two,),
                ),
                lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) > 0,
                seconds=70,
            )
        )
        assert key_spend[0].spend > 0
        deprecated: Final = TypeAdapter(tuple[_DeprecatedKey, ...]).validate_python(
            read_rows(
                "SELECT active_token_id, EXTRACT(EPOCH FROM revoke_at - now())::float8 AS grace_seconds "
                'FROM "LiteLLM_DeprecatedVerificationToken" WHERE token=%s',
                (hash_one,),
            )
        )
        assert len(deprecated) == 1
        assert deprecated[0].active_token_id == hash_two
        assert deprecated[0].grace_seconds == pytest.approx(3600, abs=120)

        generated_three: Final = (
            gateway.request("POST", "/key/regenerate", {"key": key_two})
            if spelling == "body"
            else gateway.request("POST", f"/key/{hash_two}/regenerate", {})
        )
        assert generated_three.status_code == 200, generated_three.text
        key_three: Final = _GenerateKeyResponse.model_validate_json(generated_three.content).key
        scenario.cleanups.callback(delete_key_if_present, gateway, key_three)
        refused: Final = _chat(gateway, model, key_two, prompts[2])
        assert refused.status_code == 401, refused.text
        assert _ErrorResponse.model_validate_json(refused.content).error.type == "token_not_found_in_db", refused.text
        assert wire.drain() == (), "the revoked second key reached the upstream"
        assert (
            _assert_success(_chat(gateway, model, key_three, prompts[3])).choices[0].message.content == "wire response"
        )
        assert tuple((request.method, request.target) for request in wire.drain()) == (_EXPECTED_CHAT,)
