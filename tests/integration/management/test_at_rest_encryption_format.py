"""At-rest encryption writes the versioned AES-256-GCM format and keeps reading every older format.

The proxy, Postgres and the router run for real. Older rows are built by hand from the wire formats litellm
owns (nacl SecretBox and AES-GCM under the raw SHA-256 derivation) and inserted straight into the model table,
the way a deployment upgraded in place would find them. Every read is proven at the scripted upstream: the
Authorization header it receives is the plaintext key the proxy decrypted.
"""

import base64
import hashlib
import json
import os
import signal
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import httpx
import nacl.secret
import psycopg
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy_process

V3_PREFIX: Final = "v3:gcm:"
V2_PREFIX: Final = "v2:gcm:"


def _salt_key() -> str:
    return os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt")


def _sha256_key() -> bytes:
    return hashlib.sha256(_salt_key().encode()).digest()


def _legacy_nacl_ciphertext(plaintext: str) -> str:
    sealed: Final = nacl.secret.SecretBox(_sha256_key()).encrypt(plaintext.encode())
    return base64.urlsafe_b64encode(bytes(sealed)).decode()


def _legacy_v2_ciphertext(plaintext: str) -> str:
    nonce: Final = os.urandom(12)
    blob: Final = AESGCM(_sha256_key()).encrypt(nonce, plaintext.encode(), None)
    return V2_PREFIX + base64.urlsafe_b64encode(nonce + blob).decode()


def _stored_api_key(model_id: str) -> str:
    rows: Final = read_rows(
        "SELECT litellm_params->>'api_key' AS api_key FROM \"LiteLLM_ProxyModelTable\" WHERE model_id = %s",
        (model_id,),
    )
    assert len(rows) == 1, rows
    return string_value(rows[0]["api_key"])


def _stored_credential_api_key(name: str) -> str:
    rows: Final = read_rows(
        "SELECT credential_values->>'api_key' AS api_key FROM \"LiteLLM_CredentialsTable\" WHERE credential_name = %s",
        (name,),
    )
    assert len(rows) == 1, rows
    return string_value(rows[0]["api_key"])


def _insert_model_row(gateway: Gateway, scenario: Scenario, ciphertext: str) -> tuple[str, str]:
    model_id: Final = f"enc-{uuid.uuid4().hex}"
    model_name: Final = f"integration-{uuid.uuid4().hex}"
    params: Final = {
        "model": "openai/gpt-4o-mini",
        "api_key": ciphertext,
        "api_base": f"{gateway.upstream_url}/v1",
    }
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        connection.execute(
            'INSERT INTO "LiteLLM_ProxyModelTable" '
            "(model_id, model_name, litellm_params, model_info, created_by, updated_by) "
            "VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s)",
            (model_id, model_name, json.dumps(params), json.dumps({"id": model_id}), "integration", "integration"),
        )
    scenario.cleanups.callback(_delete_model_row_if_present, gateway, model_id)
    return model_id, model_name


def _delete_model_row_if_present(gateway: Gateway, model_id: str) -> None:
    response: Final = gateway.request("POST", "/model/delete", {"id": model_id})
    assert response.status_code in (200, 400, 404), response.text
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        connection.execute('DELETE FROM "LiteLLM_ProxyModelTable" WHERE model_id = %s', (model_id,))


def _new_model(gateway: Gateway, scenario: Scenario, api_key: str) -> tuple[str, str]:
    model_name: Final = f"integration-{uuid.uuid4().hex}"
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": model_name,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": api_key,
                "api_base": f"{gateway.upstream_url}/v1",
            },
            "model_info": {},
        },
    )
    model_id: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, model_id)
    return model_id, model_name


def _drain_upstream(gateway: Gateway) -> tuple[tuple[str, str], ...]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=15, trust_env=False) as upstream:
        observed: Final = upstream.get("/__observations")
        observed.raise_for_status()
        requests: Final = JSON_OBJECT.validate_json(observed.content)["requests"]
    assert isinstance(requests, list), requests
    entries: Final = tuple(object_value(value) for value in requests)
    return tuple((json.dumps(entry["body"]), string_value(entry["authorization"])) for entry in entries)


def _authorizations_for(observed: tuple[tuple[str, str], ...], marker: str) -> tuple[str, ...]:
    return tuple(authorization for body, authorization in observed if marker in body)


def _chat_reaches_upstream_with(gateway: Gateway, model_name: str, plaintext_key: str) -> None:
    marker: Final = f"marker-{uuid.uuid4().hex}"
    response: Final = eventually(
        lambda: gateway.request(
            "POST", "/v1/chat/completions", {"model": model_name, "messages": [{"role": "user", "content": marker}]}
        ),
        lambda observed: observed.status_code == 200,
        seconds=70,
    )
    assert response.status_code == 200, response.text
    assert _authorizations_for(_drain_upstream(gateway), marker) == (f"Bearer {plaintext_key}",)


def _model_table_count(gateway: Gateway, counter: str) -> int:
    report: Final = object_value(gateway.get("/credentials/migrate-encryption/check")["report"])
    value: Final = object_value(object_value(report["locations"])["model_table"])[counter]
    assert isinstance(value, int), report
    return value


def test_new_model_api_key_is_stored_as_v3_gcm_and_decrypts_for_the_upstream_call(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        plaintext_key: Final = f"sk-upstream-{uuid.uuid4().hex}"
        model_id, model_name = _new_model(gateway, scenario, plaintext_key)
        stored: Final = _stored_api_key(model_id)
        assert stored.startswith(V3_PREFIX), f"expected a {V3_PREFIX} ciphertext, stored {stored!r}"
        assert plaintext_key not in stored
        _chat_reaches_upstream_with(gateway, model_name, plaintext_key)


def test_new_credential_values_are_stored_as_v3_gcm(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        name: Final = f"credential-{uuid.uuid4().hex}"
        plaintext_key: Final = f"sk-credential-{uuid.uuid4().hex}"
        gateway.post(
            "/credentials",
            {"credential_name": name, "credential_values": {"api_key": plaintext_key}, "credential_info": {}},
        )
        scenario.cleanups.callback(gateway.request, "DELETE", f"/credentials/{name}")
        stored: Final = _stored_credential_api_key(name)
        assert stored.startswith(V3_PREFIX), f"expected a {V3_PREFIX} ciphertext, stored {stored!r}"
        assert plaintext_key not in stored


def test_legacy_nacl_model_row_still_decrypts_for_the_upstream_call(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        plaintext_key: Final = f"sk-nacl-{uuid.uuid4().hex}"
        _, model_name = _insert_model_row(gateway, scenario, _legacy_nacl_ciphertext(plaintext_key))
        _chat_reaches_upstream_with(gateway, model_name, plaintext_key)


def test_v2_gcm_model_row_still_decrypts_for_the_upstream_call(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        plaintext_key: Final = f"sk-v2-{uuid.uuid4().hex}"
        _, model_name = _insert_model_row(gateway, scenario, _legacy_v2_ciphertext(plaintext_key))
        _chat_reaches_upstream_with(gateway, model_name, plaintext_key)


def test_encryption_check_counts_v3_as_migrated_and_nacl_as_legacy(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        migrated_before: Final = _model_table_count(gateway, "already_v2")
        legacy_before: Final = _model_table_count(gateway, "legacy")
        _new_model(gateway, scenario, f"sk-upstream-{uuid.uuid4().hex}")
        _insert_model_row(gateway, scenario, _legacy_nacl_ciphertext(f"sk-nacl-{uuid.uuid4().hex}"))
        _insert_model_row(gateway, scenario, _legacy_v2_ciphertext(f"sk-v2-{uuid.uuid4().hex}"))
        migrated_after: Final = _model_table_count(gateway, "already_v2")
        assert migrated_after == migrated_before + 3 + 1, (
            "expected every litellm_params value of the new model (model, api_key, api_base) plus the v2 row "
            f"counted as migrated, before {migrated_before} after {migrated_after}"
        )
        assert _model_table_count(gateway, "legacy") == legacy_before + 1


def test_migrate_encryption_runs_under_default_settings_and_rewrites_legacy_rows(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        plaintext_key: Final = f"sk-nacl-{uuid.uuid4().hex}"
        model_id, model_name = _insert_model_row(gateway, scenario, _legacy_nacl_ciphertext(plaintext_key))
        migrated: Final = gateway.request("POST", "/credentials/migrate-encryption", {})
        assert migrated.status_code == 200, migrated.text
        rewritten: Final = _stored_api_key(model_id)
        assert rewritten.startswith(V3_PREFIX), f"expected a {V3_PREFIX} ciphertext, stored {rewritten!r}"
        _chat_reaches_upstream_with(gateway, model_name, plaintext_key)


def test_undecryptable_model_row_does_not_stop_other_models_from_serving(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        _insert_model_row(gateway, scenario, base64.b64encode(os.urandom(48)).decode())
        healthy_key: Final = f"sk-upstream-{uuid.uuid4().hex}"
        _, healthy_model = _new_model(gateway, scenario, healthy_key)
        _chat_reaches_upstream_with(gateway, healthy_model, healthy_key)
        assert gateway.request("GET", "/health/liveliness").status_code == 200


def _burst_request(gateway: Gateway, model: str, marker: str) -> tuple[str, int]:
    try:
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": marker}]}
        )
        return marker, response.status_code
    except httpx.TransportError:
        return marker, 0


def test_proxy_restart_mid_burst_reloads_every_encryption_format_from_the_database(
    gateway: Gateway, tmp_path: Path
) -> None:
    with gateway.scenario() as scenario:
        keys: Final = (f"sk-nacl-{uuid.uuid4().hex}", f"sk-v2-{uuid.uuid4().hex}", f"sk-upstream-{uuid.uuid4().hex}")
        _, nacl_model = _insert_model_row(gateway, scenario, _legacy_nacl_ciphertext(keys[0]))
        _, v2_model = _insert_model_row(gateway, scenario, _legacy_v2_ciphertext(keys[1]))
        _, v3_model = _new_model(gateway, scenario, keys[2])
        models: Final = (nacl_model, v2_model, v3_model)
        expected: Final = dict(zip(models, (f"Bearer {key}" for key in keys), strict=True))
        markers: Final = tuple(f"burst-{uuid.uuid4().hex}" for _ in range(24))
        with owned_proxy_process(gateway, tmp_path, {}, workers=2) as first:
            for model, key in zip(models, keys, strict=True):
                _chat_reaches_upstream_with(first.gateway, model, key)
            with ThreadPoolExecutor(max_workers=24) as pool:
                futures: Final = tuple(
                    pool.submit(_burst_request, first.gateway, models[index % 3], marker)
                    for index, marker in enumerate(markers)
                )
                eventually(lambda: sum(future.done() for future in futures), lambda done: done >= 6, seconds=60)
                os.killpg(first.process.pid, signal.SIGTERM)
                outcomes: Final = tuple(future.result(timeout=120) for future in futures)
        served: Final = frozenset(marker for marker, status in outcomes if status == 200)
        assert served, outcomes
        observed: Final = _drain_upstream(gateway)
        for index, marker in enumerate(markers):
            seen = _authorizations_for(observed, marker)
            assert len(seen) <= 1, (marker, seen)
            if marker in served:
                assert seen == (expected[models[index % 3]],), (marker, seen)
        with owned_proxy_process(gateway, tmp_path, {}, workers=2) as restarted:
            for model, key in zip(models, keys, strict=True):
                _chat_reaches_upstream_with(restarted.gateway, model, key)
