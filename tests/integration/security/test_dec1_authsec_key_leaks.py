"""The caller's key never lands in spend rows, callback payloads, audit rows or the proxy log, whatever carried it.

Contract (litellm/proxy/litellm_pre_call_utils.py builds `proxy_server_request` from `clean_headers` output and
logs only `redact_credential_headers`; litellm/proxy/spend_tracking/spend_tracking_utils.py stores the key as its
SHA-256; litellm/_logging.py redacts `sk-` and `Bearer` shaped values in log records; litellm/proxy/_types.py
`LiteLLM_AuditLogs.mask_api_keys` masks key fields in audit rows):

* `security/test_stored_config_slots.py` (slot A1) already sweeps every table, sink and route for a virtual key
  sent as `Authorization: Bearer`. This file covers the other carriers the Python proxy accepts and the Rust
  extractor does not: the five provider style headers and the `generateContent` `?key=` query parameter, plus
  the custom key header, the proxy log at DEBUG, and audit rows for key creation and rotation.
* Each test plants the key as a canary (`_canary.canary("A1")`, searched raw, base64, URL encoded and gzipped by
  `find_canary`) and checks a positive control so a sweep that sees nothing proves something.

The rig is `_sinks.canary_rig`: an owned proxy with `store_prompts_in_spend_logs: true`, a `generic_api` sink
callback, and a provider double, with the proxy log captured in `rig.owned.log`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import eventually, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request
from integration.security._canary import MARKER, Canary, canary, find_canary
from integration.security._sinks import CONFIG_MODEL, GENERIC_SINK, Rig, canary_rig, chat_upstream
from integration.security.dec1_authsec_support import GEMINI_REPLY, digest, raw_post

CARRIERS: Final = ("x-api-key", "API-Key", "x-goog-api-key", "Ocp-Apim-Subscription-Key", "x-litellm-api-key")
CUSTOM_KEY_HEADER: Final = "x-dec1-proxy-key"
GEMINI_CANARY_MODEL: Final = "dec1-authsec-canary-gemini"

pytestmark: Final = pytest.mark.timeout(300)


def _upstream(request: Request) -> Reply:
    if "generateContent" in request.target:
        return Reply(body=GEMINI_REPLY)
    return chat_upstream(request)


def _add_gemini_and_custom_header(config: dict[str, object], provider_url: str) -> None:
    models: Final = config["model_list"]
    assert isinstance(models, list)
    models.append(
        {
            "model_name": GEMINI_CANARY_MODEL,
            "litellm_params": {"model": "gemini/gemini-2.5-flash", "api_base": provider_url, "api_key": "dec1-gemini"},
        }
    )
    general: Final = config["general_settings"]
    assert isinstance(general, dict)
    general["litellm_key_header_name"] = CUSTOM_KEY_HEADER


def _spend_row(request_id: str) -> str:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT to_jsonb(s)::text AS row FROM "LiteLLM_SpendLogs" s WHERE request_id=%s', (request_id,)
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return string_value(rows[0]["row"])


def _sink_blobs(rig: Rig, marker: Canary) -> tuple[bytes, ...]:
    sink: Final = rig.sinks[GENERIC_SINK]
    eventually(lambda: sink.carrying(marker.core), bool, seconds=30)
    return tuple(request.body for request in sink.requests())


@pytest.mark.parametrize("header", CARRIERS)
def test_a_key_sent_in_a_provider_style_header_never_lands_in_the_spend_row_or_the_sink(
    tmp_path: Path, header: str
) -> None:
    """A virtual key (canary A1) calls the config model through one provider style header, with a marker canary
    in the prompt. Positive control: the spend row and the generic sink payload must contain the marker (prompts
    are stored). The spend row's `api_key` must be the key's SHA-256, and neither the spend row nor any sink
    payload may contain the raw key in any encoding."""
    a1: Final = canary("A1")
    marker: Final = canary(MARKER)
    with (
        canary_rig(tmp_path, configure=_add_gemini_and_custom_header, upstream=_upstream) as rig,
        rig.proxy.scenario() as scenario,
    ):
        scenario.key(key=a1.value, models=[CONFIG_MODEL])
        served = raw_post(
            rig.proxy,
            "/v1/chat/completions",
            {"model": CONFIG_MODEL, "messages": [{"role": "user", "content": f"carrier {marker.value}"}]},
            {header: a1.value},
        )
        assert served.status_code == 200, served.text
        row: Final = _spend_row(string_value(served.json()["id"]))
        sink: Final = _sink_blobs(rig, marker)
    assert find_canary(row, (marker,)) != (), "Positive control: the prompt was not stored in the spend row"
    assert json.loads(row)["api_key"] == digest(a1.value)
    assert find_canary(row, (a1,)) == (), "spend row holds the caller's key"
    for blob in sink:
        assert find_canary(blob, (a1,)) == (), "generic sink payload holds the caller's key"


def test_a_key_sent_in_the_custom_key_header_never_lands_in_the_spend_row_or_the_sink(tmp_path: Path) -> None:
    """With `litellm_key_header_name: x-dec1-proxy-key`, a virtual key (canary A1) calls the config model through
    that header as `Bearer <key>`. The spend row's `api_key` must be the SHA-256, and neither the spend row nor the
    sink payloads may contain the raw key; the custom header is dropped by `clean_headers` before
    `proxy_server_request` is built."""
    a1: Final = canary("A1")
    marker: Final = canary(MARKER)
    with (
        canary_rig(tmp_path, configure=_add_gemini_and_custom_header, upstream=_upstream) as rig,
        rig.proxy.scenario() as scenario,
    ):
        scenario.key(key=a1.value, models=[CONFIG_MODEL])
        served = raw_post(
            rig.proxy,
            "/v1/chat/completions",
            {"model": CONFIG_MODEL, "messages": [{"role": "user", "content": f"custom {marker.value}"}]},
            {CUSTOM_KEY_HEADER: f"Bearer {a1.value}"},
        )
        assert served.status_code == 200, served.text
        row: Final = _spend_row(string_value(served.json()["id"]))
        sink: Final = _sink_blobs(rig, marker)
    assert json.loads(row)["api_key"] == digest(a1.value)
    assert find_canary(row, (a1,)) == ()
    assert all(find_canary(blob, (a1,)) == () for blob in sink)


def test_a_query_key_on_generate_content_never_lands_in_the_spend_row_the_sink_or_the_provider_target(
    tmp_path: Path,
) -> None:
    """A virtual key (canary A1) calls `POST /v1beta/models/<gemini model>:generateContent?key=<key>`. The request
    URL with its query string is part of `proxy_server_request`, so this is the carrier most likely to leak. The
    call must be served; the spend row, every sink payload and the provider request target must not contain the
    key. UNVERIFIED: whether `proxy_server_request.url` reaches the stored row or the sink; read from
    `litellm_pre_call_utils.py` (`"url": str(request.url)`)."""
    a1: Final = canary("A1")
    marker: Final = canary(MARKER)
    with (
        canary_rig(tmp_path, configure=_add_gemini_and_custom_header, upstream=_upstream) as rig,
        rig.proxy.scenario() as scenario,
    ):
        scenario.key(key=a1.value, models=[GEMINI_CANARY_MODEL])
        served = rig.proxy.client.post(
            f"/v1beta/models/{GEMINI_CANARY_MODEL}:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": f"query {marker.value}"}]}]},
            params={"key": a1.value},
        )
        assert served.status_code == 200, served.text
        provider_calls: Final = rig.provider.carrying(marker.value)
        assert len(provider_calls) == 1, "Positive control: the Gemini call never reached the provider double"
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT to_jsonb(s)::text AS row FROM "LiteLLM_SpendLogs" s WHERE api_key=%s', (digest(a1.value),)
            ),
            lambda found: len(found) >= 1,
            seconds=70,
        )
        sink: Final = _sink_blobs(rig, marker)
    assert a1.value not in provider_calls[0].target
    assert all(find_canary(string_value(row["row"]), (a1,)) == () for row in rows)
    assert all(find_canary(blob, (a1,)) == () for blob in sink)


@pytest.mark.parametrize("header", ("Authorization", "x-api-key", CUSTOM_KEY_HEADER))
def test_the_debug_proxy_log_never_holds_the_virtual_key(tmp_path: Path, header: str) -> None:
    """With `LITELLM_LOG=DEBUG` (which logs request headers, the custom key header lookup with every header, and
    auth decisions), a virtual key (canary A1) makes one served call through the given header and the same key
    with one extra character makes one refused call. After the proxy stops, the captured log must contain the
    marker (positive control) and must not contain either key value in any encoding."""
    a1: Final = canary("A1")
    marker: Final = canary(MARKER)
    with canary_rig(
        tmp_path, configure=_add_gemini_and_custom_header, environment={"LITELLM_LOG": "DEBUG"}, upstream=_upstream
    ) as rig:
        with rig.proxy.scenario() as scenario:
            scenario.key(key=a1.value, models=[CONFIG_MODEL])
            for value in (a1.value, a1.value + "x"):
                carried = f"Bearer {value}" if header != "x-api-key" else value
                raw_post(
                    rig.proxy,
                    "/v1/chat/completions",
                    {"model": CONFIG_MODEL, "messages": [{"role": "user", "content": f"log {marker.value}"}]},
                    {header: carried},
                )
        log: Final = rig.owned.log
    data: Final = log.read_bytes()
    assert find_canary(data, (marker,)) != (), f"Sensitivity control: the marker never reached {log}"
    assert find_canary(data, (a1,)) == (), f"the DEBUG proxy log holds the caller's key ({header})"


def test_the_debug_proxy_log_never_holds_the_master_key(tmp_path: Path) -> None:
    """With `LITELLM_LOG=DEBUG`, the master key (canary A2, set as `LITELLM_MASTER_KEY`) makes an admin call
    (`GET /user/list`) and a chat call, and the master key plus one character makes a refused call. The captured
    log must hold the marker and must not hold the master key in any encoding: the master key comparison and the
    alias substitution (`LITELLM_PROXY_MASTER_KEY_ALIAS`) must never print it."""
    a2: Final = canary("A2")
    marker: Final = canary(MARKER)
    with canary_rig(
        tmp_path,
        configure=_add_gemini_and_custom_header,
        environment={"LITELLM_LOG": "DEBUG", "LITELLM_MASTER_KEY": a2.value},
        upstream=_upstream,
    ) as rig:
        assert rig.proxy.request("GET", "/user/list").status_code == 200
        rig.proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": CONFIG_MODEL, "messages": [{"role": "user", "content": f"admin {marker.value}"}]},
        )
        assert rig.proxy.request("GET", "/user/list", key=a2.value + "x").status_code == 401
        log: Final = rig.owned.log
    data: Final = log.read_bytes()
    assert find_canary(data, (marker,)) != ()
    assert find_canary(data, (a2,)) == (), "the DEBUG proxy log holds the master key"


def test_the_key_created_audit_row_holds_no_plaintext_key(tmp_path: Path) -> None:
    """With audit logging on (`LITELLM_STORE_AUDIT_LOGS=true`), the admin creates a key (canary A1) through
    `/key/generate`. An audit row with `action` `created` and `object_id` equal to the key's SHA-256 must appear,
    and neither `updated_values` nor `before_value` may contain the raw key: the key generate hook dumps the whole
    `GenerateKeyResponse`, whose `key` field is the plaintext key, and relies on `mask_api_keys` to mask it."""
    a1: Final = canary("A1")
    with (
        canary_rig(tmp_path, environment={"LITELLM_STORE_AUDIT_LOGS": "true"}) as rig,
        rig.proxy.scenario() as scenario,
    ):
        scenario.key(key=a1.value)
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT to_jsonb(a)::text AS row FROM "LiteLLM_AuditLog" a WHERE object_id=%s AND action=%s',
                (digest(a1.value), "created"),
            ),
            lambda found: len(found) == 1,
            seconds=30,
        )
    assert find_canary(string_value(rows[0]["row"]), (a1,)) == (), "the created audit row holds the plaintext key"


def test_the_key_rotated_audit_row_holds_neither_the_old_nor_the_new_key(tmp_path: Path) -> None:
    """With audit logging on, the admin creates a key (canary A1) and rotates it with `POST /key/regenerate`
    giving a new custom value (canary A1, second instance). The `rotated` audit row must exist and contain neither
    raw value in any encoding."""
    old: Final = canary("A1")
    new: Final = canary("A1")
    with canary_rig(tmp_path, environment={"LITELLM_STORE_AUDIT_LOGS": "true"}) as rig:
        created: Final = rig.proxy.post("/key/generate", {"key": old.value})
        assert string_value(created["key"]) == old.value
        rotated: Final = rig.proxy.post("/key/regenerate", {"key": old.value, "new_key": new.value})
        assert string_value(rotated["key"]) == new.value
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT to_jsonb(a)::text AS row FROM "LiteLLM_AuditLog" a WHERE action=%s AND updated_at >= now() - '
                "interval '5 minutes'",
                ("rotated",),
            ),
            lambda found: len(found) >= 1,
            seconds=30,
        )
        rig.proxy.post("/key/delete", {"keys": [new.value]})
    for row in rows:
        assert find_canary(string_value(row["row"]), (old, new)) == (), "the rotated audit row holds a raw key"
