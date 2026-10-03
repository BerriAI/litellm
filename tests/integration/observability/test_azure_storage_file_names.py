import json
import re
import uuid
from pathlib import Path
from typing import Final

from _azure_storage_support import (
    SINK_HOSTS,
    RecordingDataLakeSink,
    azure_storage_config,
    azure_storage_environment,
)
from _s3_v2_support import surface_reply
from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import Reply, Request, wire_server

from litellm.constants import MAX_LITELLM_CALL_ID_LENGTH

ADLS_SAFE_FILE_NAME: Final = re.compile(r"^[A-Za-z0-9._+-]+\.json$")
WORKERS: Final = 2


def _responses_id(candidate: Gateway, model: str, key: str, marker: str) -> str:
    response: Final = candidate.request("POST", "/v1/responses", {"model": model, "input": marker}, key=key)
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def test_responses_ids_with_base64_padding_land_under_adls_safe_names(gateway: Gateway, tmp_path: Path) -> None:
    """A /v1/responses id is `resp_` plus base64 with `=` padding decided by the encoded length, so upstream ids
    of several lengths yield both `=` and `==` padded ids; each must land as a file the service accepts."""
    marker: Final = f"azure-{uuid.uuid4().hex[:8]}"
    sink: Final = RecordingDataLakeSink()
    cert, key = write_self_signed_cert(tmp_path, SINK_HOSTS)
    with (
        wire_server(surface_reply) as provider,
        wire_server(sink.respond, tls=server_context(cert, key), keep_alive=True) as store,
    ):
        environment: Final = {**azure_storage_environment(store.url, cert), "DEFAULT_FLUSH_INTERVAL_SECONDS": "1"}
        config: Final = azure_storage_config(tmp_path)
        with (
            owned_proxy(gateway, tmp_path, environment, config=config, workers=WORKERS) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            api_key: Final = scenario.key(models=[model])
            answered: Final = tuple(
                _responses_id(candidate, model, api_key, f"{marker}-{'x' * extra}") for extra in range(6)
            )
            assert {response_id.count("=") for response_id in answered} >= {1, 2}, answered
            eventually(
                lambda: len(sink.stored()) + len(sink.unauthenticated_targets()),
                lambda settled: settled >= len(answered),
                seconds=60,
            )
            assert sink.unauthenticated_targets() == (), sink.unauthenticated_targets()
            assert frozenset(str(payload["id"]) for payload in sink.payloads().values()) == frozenset(answered), tuple(
                sink.stored()
            )
            names: Final = tuple(path.rsplit("/", 1)[1] for path in sink.stored())
            assert all(ADLS_SAFE_FILE_NAME.match(name) for name in names), names
            assert len(frozenset(names)) == len(answered), names
        assert provider.drain()


def _embedding_reply(request: Request) -> Reply:
    assert request.method == "POST" and request.target.endswith("/embeddings"), request.target
    return Reply(
        body=json.dumps(
            {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            }
        ).encode()
    )


def _failing_chat_reply(request: Request) -> Reply:
    assert request.method == "POST" and request.target.endswith("/chat/completions"), request.target
    return Reply(status=500, body=json.dumps({"error": {"message": "upstream rejected the request"}}).encode())


def _log_names_by_call_id(
    gateway: Gateway,
    tmp_path: Path,
    call_ids: tuple[str, ...],
    *,
    inputs: tuple[str, ...] | None = None,
    failing: bool = False,
) -> dict[str, str]:
    sink: Final = RecordingDataLakeSink()
    cert, key = write_self_signed_cert(tmp_path, SINK_HOSTS)
    with (
        wire_server(_failing_chat_reply if failing else _embedding_reply) as provider,
        wire_server(sink.respond, tls=server_context(cert, key), keep_alive=True) as store,
    ):
        environment: Final = {**azure_storage_environment(store.url, cert), "DEFAULT_FLUSH_INTERVAL_SECONDS": "1"}
        config: Final = azure_storage_config(tmp_path)
        with (
            owned_proxy(gateway, tmp_path, environment, config=config, workers=WORKERS) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model="openai/gpt-4.1-nano" if failing else "openai/text-embedding-3-small",
                api_base=provider.url + "/v1",
                api_key="synthetic-provider-key",
            )
            api_key: Final = scenario.key(models=[model])
            responses: Final = tuple(
                candidate.request(
                    "POST",
                    "/v1/chat/completions" if failing else "/v1/embeddings",
                    {"model": model, "messages": [{"role": "user", "content": text}]}
                    if failing
                    else {"model": model, "input": text},
                    key=api_key,
                    headers={"x-litellm-call-id": call_id},
                )
                for call_id, text in zip(call_ids, inputs or call_ids, strict=True)
            )
            assert all(response.status_code == (500 if failing else 200) for response in responses), tuple(
                response.text for response in responses
            )
            eventually(
                lambda: len(sink.stored()) + len(sink.duplicated()) + len(sink.unauthenticated_targets()),
                lambda settled: settled >= len(call_ids),
                seconds=60,
            )
            assert sink.unauthenticated_targets() == (), sink.unauthenticated_targets()
            assert sink.duplicated() == (), f"a later log overwrote an earlier one at {sink.duplicated()}"
        assert provider.drain()
    return {path.split("/", 3)[3]: str(payload["id"]) for path, payload in sink.payloads().items()}


def test_client_call_ids_differing_only_by_slash_or_underscore_land_in_separate_files(
    gateway: Gateway, tmp_path: Path
) -> None:
    """An embedding response carries no id, so its log is named after the caller's `x-litellm-call-id`. Two
    caller ids that differ only by `/` and `_` are two requests and must leave two logs, neither overwriting
    the other"""
    marker: Final = f"svc-{uuid.uuid4().hex[:8]}"
    call_ids: Final = (f"{marker}/req-1", f"{marker}_req-1")
    assert _log_names_by_call_id(gateway, tmp_path, call_ids) == {f"{call_id}.json": call_id for call_id in call_ids}


def test_client_call_ids_with_parent_segments_stay_inside_the_log_directory(gateway: Gateway, tmp_path: Path) -> None:
    """A caller's `x-litellm-call-id` names its log file, so a `..` segment in it must not climb out of the
    dated log directory into another day's folder or another filesystem"""
    marker: Final = uuid.uuid4().hex[:8]
    call_ids: Final = (f"../other-filesystem/{marker}", f"../2026-09-30/{marker}", f"%2e%2e/other-filesystem/{marker}")
    assert _log_names_by_call_id(gateway, tmp_path, call_ids) == {
        f".._other-filesystem_{marker}.json": call_ids[0],
        f".._2026-09-30_{marker}.json": call_ids[1],
        f"%2e%2e_other-filesystem_{marker}.json": call_ids[2],
    }


def test_client_call_ids_with_dot_or_empty_segments_keep_their_own_files(gateway: Gateway, tmp_path: Path) -> None:
    """A `.` or empty segment in a caller's `x-litellm-call-id` collapses on the Data Lake path, so `svc/./x` would
    overwrite the log of `svc/x` and `svc//x` would fail to upload. Each id must still leave its own log"""
    marker: Final = uuid.uuid4().hex[:8]
    call_ids: Final = (f"{marker}/x", f"{marker}/./x", f"{marker}//x")
    assert _log_names_by_call_id(gateway, tmp_path, call_ids) == {
        f"{marker}/x.json": call_ids[0],
        f"{marker}_._x.json": call_ids[1],
        f"{marker}__x.json": call_ids[2],
    }


def test_failed_requests_with_look_alike_call_ids_keep_separate_failure_logs(gateway: Gateway, tmp_path: Path) -> None:
    """A failed request has no response id, so its failure log is named after the caller's `x-litellm-call-id`. Two
    failures whose ids differ only by `/` and `_` must leave two failure logs"""
    marker: Final = f"fail-{uuid.uuid4().hex[:8]}"
    call_ids: Final = (f"{marker}/req-1", f"{marker}_req-1")
    assert _log_names_by_call_id(gateway, tmp_path, call_ids, failing=True) == {
        f"{call_id}.json": call_id for call_id in call_ids
    }


def test_cache_hits_with_look_alike_call_ids_keep_separate_logs(gateway: Gateway, tmp_path: Path) -> None:
    """A cached embedding is served without reaching the provider, and its log is named after the caller's call id
    plus a cache-hit suffix. Two cache hits whose ids differ only by `/` and `_` must still leave two logs"""
    marker: Final = f"hit-{uuid.uuid4().hex[:8]}"
    call_ids: Final = (f"{marker}/warm", f"{marker}/req-1", f"{marker}_req-1")
    logs: Final = _log_names_by_call_id(gateway, tmp_path, call_ids, inputs=(marker, marker, marker))
    assert {name: payload_id for name, payload_id in logs.items() if "_cache_hit" not in payload_id} == {
        f"{call_ids[0]}.json": call_ids[0]
    }, logs
    cache_hits: Final = {name: payload_id for name, payload_id in logs.items() if "_cache_hit" in payload_id}
    assert sorted(payload_id.split("_cache_hit")[0] for payload_id in cache_hits.values()) == sorted(call_ids[1:]), logs
    assert all(name == f"{payload_id}.json" for name, payload_id in cache_hits.items()), logs


def test_longest_oversized_and_blank_call_ids_each_leave_one_log(gateway: Gateway, tmp_path: Path) -> None:
    """The longest accepted caller id keeps its own name, while a 5 KB or blank `x-litellm-call-id` falls back to a
    generated id, so none of the three requests loses its log"""
    marker: Final = f"edge-{uuid.uuid4().hex[:8]}/"
    longest: Final = marker + "x" * (MAX_LITELLM_CALL_ID_LENGTH - len(marker))
    call_ids: Final = (longest, marker + "x" * 5000, "")
    logs: Final = _log_names_by_call_id(gateway, tmp_path, call_ids, inputs=(f"{marker}0", f"{marker}1", f"{marker}2"))
    assert logs.get(f"{longest}.json") == longest, tuple(logs)
    generated: Final = frozenset(payload_id for payload_id in logs.values() if payload_id != longest)
    assert len(logs) == 3 and len(generated) == 2 and not generated & frozenset(call_ids), tuple(logs)
    assert all(logs[f"{payload_id}.json"] == payload_id for payload_id in generated), tuple(logs)
