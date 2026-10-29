import json
import math
import uuid
from collections.abc import Iterator, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import pytest
from integration._support.client import (
    Gateway,
    eventually,
    gateway_from_environment,
    held,
    object_value,
    string_value,
)
from integration._support.forward_proxy import tunnelling_forward_proxy
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 180)

_MODEL: Final = "gpt-5.6"
_INPUT_TOKENS: Final = 1200
_OUTPUT_TOKENS: Final = 300
_READS: Final = 3
_SPEND_LANDING_SECONDS: Final = 70
_LATE_CHARGE_WINDOW_SECONDS: Final = 10
_COST_MAP: Final = Path("litellm/model_prices_and_context_window_backup.json")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _response(response_id: str, status: str) -> Mapping[str, JsonValue]:
    finished: Final = status == "completed"
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1,
        "status": status,
        "background": True,
        "error": None,
        "incomplete_details": None,
        "model": _MODEL,
        "output": [
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "pong", "annotations": []}],
            }
        ]
        if finished
        else [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": _INPUT_TOKENS,
            "output_tokens": _OUTPUT_TOKENS,
            "total_tokens": _INPUT_TOKENS + _OUTPUT_TOKENS,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
        if finished
        else None,
        "metadata": {},
    }


def _openai(request: Request) -> Reply:
    path: Final = urlsplit(request.target).path
    if request.method == "POST" and path == "/v1/responses":
        return Reply(body=json.dumps(_response(f"resp_{uuid.uuid4().hex}", "queued")).encode())
    return Reply(body=json.dumps(_response(path.rsplit("/", 1)[-1], "completed")).encode())


def _port(url: str) -> int:
    port: Final = urlsplit(url).port
    assert port is not None, url
    return port


@pytest.fixture(scope="module")
def proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("passthrough-background-response")
    certificate: Final = write_self_signed_cert(directory, ("api.openai.com",))
    with (
        wire_server(_openai, tls=server_context(*certificate)) as upstream,
        tunnelling_forward_proxy(_port(upstream.url)) as tunnel,
        gateway_from_environment() as base,
        owned_proxy(
            base,
            directory,
            {
                "HTTPS_PROXY": tunnel.url,
                "NO_PROXY": "127.0.0.1,localhost",
                "AIOHTTP_TRUST_ENV": "True",
                "SSL_VERIFY": "False",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                "OPENAI_API_KEY": "sk-scripted-openai",
            },
            workers=2,
        ) as owned,
    ):
        yield owned


def _price() -> float:
    row: Final = _JSON_OBJECT.validate_python(_JSON_OBJECT.validate_json(_COST_MAP.read_bytes())[_MODEL])
    input_rate: Final = row["input_cost_per_token"]
    output_rate: Final = row["output_cost_per_token"]
    assert isinstance(input_rate, float) and isinstance(output_rate, float), row
    return _INPUT_TOKENS * input_rate + _OUTPUT_TOKENS * output_rate


def _passthrough(proxy: Gateway, method: str, path: str, key: str, body: Mapping[str, JsonValue] | None = None) -> str:
    response: Final = proxy.request(method, f"/openai_passthrough/v1/responses{path}", body, key=key)
    assert response.status_code == 200, response.text
    payload: Final = object_value(response.json())
    return string_value(payload["id"]) if method == "POST" else string_value(payload["status"])


def _number(value: JsonValue) -> float:
    assert isinstance(value, int | float), value
    return float(value)


def _spend_of(proxy: Gateway, table: str, column: str, identity: str) -> float:
    return _number(proxy.rows(f'SELECT spend FROM "{table}" WHERE {column} = %s', (identity,))[0]["spend"])


def _key_spend(proxy: Gateway, key: str) -> float:
    return _number(object_value(proxy.get("/key/info", {"key": key})["info"])["spend"])


def _logged_spend(proxy: Gateway, key: str) -> float:
    rows: Final = proxy.rows(
        'SELECT spend FROM "LiteLLM_SpendLogs" WHERE api_key = %s', (sha256(key.encode()).hexdigest(),)
    )
    return math.fsum(_number(row["spend"]) for row in rows)


def test_a_background_response_read_through_the_openai_passthrough_bills_its_key_team_user_and_org_once(
    proxy: Gateway,
) -> None:
    price: Final = _price()
    with proxy.scenario() as scenario:
        organization_id: Final = scenario.organization()
        team_id: Final = scenario.team(organization_id=organization_id)
        user_id: Final = scenario.org_member(organization_id, "internal_user")
        proxy.post("/team/member_add", {"team_id": team_id, "member": {"role": "user", "user_id": user_id}})
        key: Final = scenario.key(team_id=team_id, user_id=user_id)
        response_id: Final = _passthrough(
            proxy, "POST", "", key, {"model": _MODEL, "input": "bill me once", "background": True}
        )
        reads: Final = tuple(_passthrough(proxy, "GET", f"/{response_id}", key) for _ in range(_READS))
        assert reads == ("completed",) * _READS

        def spent() -> tuple[float, ...]:
            return (
                _key_spend(proxy, key),
                _spend_of(proxy, "LiteLLM_TeamTable", "team_id", team_id),
                _spend_of(proxy, "LiteLLM_UserTable", "user_id", user_id),
                _spend_of(proxy, "LiteLLM_OrganizationTable", "organization_id", organization_id),
            )

        held(
            spent,
            lambda observed: all(math.isclose(value, price, rel_tol=1e-9) for value in observed),
            holding=_LATE_CHARGE_WINDOW_SECONDS,
            seconds=_SPEND_LANDING_SECONDS,
        )
        eventually(
            lambda: (_logged_spend(proxy, key), _key_spend(proxy, key)),
            lambda observed: math.isclose(observed[0], observed[1], rel_tol=1e-9),
            seconds=_SPEND_LANDING_SECONDS,
        )
