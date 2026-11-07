import json
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from litellm.constants import KEY_REVOKE_MAX_REQUESTS_PER_SOURCE, KEY_REVOKE_WINDOW_SECONDS
from litellm.proxy.utils import hash_token
from tests.integration._support.client import eventually, gateway_from_environment, string_value
from tests.integration._support.process import OwnedProxy, OwnedState, WorkerHealthcheck, owned_proxy_process

MODEL: Final = "integration-key-revoke"
_FRESH: Final = httpx.Limits(max_keepalive_connections=0)


def _config(directory: Path, upstream_url: str) -> Path:
    config: Final = directory / "key_revoke.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": MODEL,
                        "litellm_params": {
                            "model": f"openai/{MODEL}",
                            "api_base": f"{upstream_url}/v1",
                            "api_key": "integration-provider-key",
                        },
                    }
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "trusted_proxy_ranges": ["127.0.0.1/32", "::1/128"],
                },
            }
        )
    )
    return config


@pytest.fixture(scope="module")
def revoking(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("key-revoke").resolve()
    with (
        gateway_from_environment() as shared,
        owned_proxy_process(
            shared,
            directory,
            {"STORE_MODEL_IN_DB": "False"},
            config=_config(directory, shared.upstream_url),
            workers=2,
            healthcheck=WorkerHealthcheck.LENIENT,
            state=OwnedState.PRIVATE,
        ) as owned,
    ):
        yield owned


def _source() -> str:
    return f"198.51.{uuid.uuid4().int % 250}.{uuid.uuid4().int % 250 + 1}"


def _post(owned: OwnedProxy, path: str, body: Mapping[str, JsonValue], headers: Mapping[str, str]) -> httpx.Response:
    with httpx.Client(base_url=owned.gateway.client.base_url, timeout=30, trust_env=False, limits=_FRESH) as client:
        return client.post(path, json=body, headers=dict(headers))


def _revoke(owned: OwnedProxy, keys: JsonValue, source: str) -> httpx.Response:
    return _post(owned, "/key/revoke", {"keys": keys}, {"X-Forwarded-For": source})


def _chat(owned: OwnedProxy, key: str) -> httpx.Response:
    return _post(
        owned,
        "/v1/chat/completions",
        {"model": MODEL, "messages": [{"role": "user", "content": "key revoke"}]},
        {"Authorization": f"Bearer {key}"},
    )


def _workers(owned: OwnedProxy) -> tuple[int, int]:
    first, second = sorted(eventually(owned.worker_pids, lambda pids: len(pids) == 2))
    return first, second


def _chat_on_each_worker(owned: OwnedProxy, key: str) -> tuple[httpx.Response, ...]:
    def on(idle: int) -> httpx.Response:
        with owned.suspended(idle):
            return _chat(owned, key)

    return tuple(on(idle) for idle in reversed(_workers(owned)))


def _key(owned: OwnedProxy) -> str:
    return string_value(owned.gateway.post("/key/generate", {"models": [MODEL]})["key"])


def _blocked(owned: OwnedProxy, key: str) -> bool:
    rows: Final = owned.gateway.rows(
        'SELECT blocked FROM "LiteLLM_VerificationToken" WHERE token = %s', (hash_token(key),)
    )
    assert len(rows) == 1, rows
    return rows[0]["blocked"] is True


def test_a_revoked_key_is_refused_by_every_worker_that_had_it_cached(revoking: OwnedProxy) -> None:
    key: Final = _key(revoking)
    assert [response.status_code for response in _chat_on_each_worker(revoking, key)] == [200, 200]

    response: Final = _revoke(revoking, key, _source())

    assert (response.status_code, response.content) == (202, b"{}")
    refused: Final = eventually(
        lambda: _chat_on_each_worker(revoking, key),
        lambda answers: [answer.status_code for answer in answers] == [401, 401],
        return_last_on_timeout=True,
    )
    assert [answer.status_code for answer in refused] == [401, 401], [answer.text for answer in refused]
    assert all("Key is blocked" in answer.text for answer in refused), [answer.text for answer in refused]
    assert _blocked(revoking, key)


def test_every_key_state_gets_the_same_answer_and_only_raw_keys_block(revoking: OwnedProxy) -> None:
    live: Final = _key(revoking)
    already_blocked: Final = _key(revoking)
    revoking.gateway.post("/key/block", {"key": already_blocked})
    hashed_only: Final = _key(revoking)
    mixed_live: Final = _key(revoking)
    submissions: Final[tuple[JsonValue, ...]] = (
        live,
        already_blocked,
        f"sk-never-existed-{uuid.uuid4().hex}",
        hash_token(hashed_only),
        "not a key",
        [hash_token(hashed_only), "junk", f"sk-never-{uuid.uuid4().hex}", mixed_live],
    )

    answers: Final = tuple(_revoke(revoking, keys, _source()) for keys in submissions)

    assert [(answer.status_code, answer.content) for answer in answers] == [(202, b"{}")] * len(submissions)
    assert (_blocked(revoking, live), _blocked(revoking, mixed_live)) == (True, True)
    assert not _blocked(revoking, hashed_only)
    assert _chat(revoking, hashed_only).status_code == 200


def test_the_hourly_allowance_is_counted_once_across_workers(revoking: OwnedProxy) -> None:
    source: Final = _source()
    never_existed: Final = f"sk-never-existed-{uuid.uuid4().hex}"
    half: Final = KEY_REVOKE_MAX_REQUESTS_PER_SOURCE // 2

    def on(idle: int) -> tuple[int, ...]:
        with revoking.suspended(idle):
            return tuple(_revoke(revoking, never_existed, source).status_code for _ in range(half))

    admitted: Final = tuple(status for idle in reversed(_workers(revoking)) for status in on(idle))
    refused: Final = _revoke(revoking, never_existed, source)

    assert admitted == (202,) * (half * 2)
    assert refused.status_code == 429, refused.text
    assert 0 < int(refused.headers["retry-after"]) <= KEY_REVOKE_WINDOW_SECONDS
    assert _revoke(revoking, never_existed, _source()).status_code == 202
