import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import httpx
import pytest

_SCRIPT: Final = Path(__file__).resolve().parents[3] / "db_scripts" / "strip_vector_store_internal_metadata.py"
_spec: Final = importlib.util.spec_from_file_location("strip_vector_store_internal_metadata", _SCRIPT)
strip_script: Final = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = strip_script
_spec.loader.exec_module(strip_script)

POLLUTED: Final = {
    "id": "vs_1",
    "metadata": {
        "user_api_key": "hash-value-1",
        "user_api_key_hash": "hash-value-1",
        "user_api_key_user_email": "member@example.com",
        "endpoint": "http://proxy/v1/vector_stores",
        "deployment": "openai/gpt-5",
        "team": "llmproxy",
        "trace_id": "trace-123",
    },
}
CLEAN: Final = {"id": "vs_2", "metadata": {"team": "llmproxy"}}
NO_METADATA: Final = {"id": "vs_3", "metadata": None}


def _fake_openai(
    stores: tuple[Mapping[str, object], ...], page_size: int = 100, post_status: int = 200
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: Final[list[httpx.Request]] = []
    by_id: Final = {store["id"]: store for store in stores}

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path: Final = request.url.path.removeprefix("/v1")
        if request.method == "POST":
            if post_status != 200:
                return httpx.Response(post_status, json={"error": {"message": "upstream rejected the update"}})
            return httpx.Response(200, json=by_id[path.removeprefix("/vector_stores/")])
        if path == "/vector_stores":
            after: Final = request.url.params.get("after")
            start: Final = 0 if after is None else [store["id"] for store in stores].index(after) + 1
            page: Final = stores[start : start + page_size]
            return httpx.Response(200, json={"data": list(page), "has_more": start + page_size < len(stores)})
        return httpx.Response(200, json=by_id[path.removeprefix("/vector_stores/")])

    return httpx.MockTransport(handle), seen


def _run(args: list[str], transport: httpx.MockTransport) -> int:
    return strip_script.main(["--api-key", "sk-test", *args], transport=transport)


def _posts(seen: list[httpx.Request]) -> list[tuple[str, dict[str, object]]]:
    return [(request.url.path, json.loads(request.content)) for request in seen if request.method == "POST"]


def test_dry_run_lists_matching_key_names_without_values_and_sends_no_update(capsys):
    transport, seen = _fake_openai((POLLUTED, CLEAN))

    exit_code: Final = _run([], transport)

    out: Final = capsys.readouterr().out
    assert exit_code == 0
    assert (
        "vs_1: would delete 5 keys: deployment, endpoint, user_api_key, user_api_key_hash, user_api_key_user_email"
        in out
    )
    assert "scanned=2 matched=1 updated=0 failed=0" in out
    assert "hash-value-1" not in out
    assert "member@example.com" not in out
    assert _posts(seen) == []
    assert {request.headers["authorization"] for request in seen} == {"Bearer sk-test"}


def test_apply_nulls_exactly_the_proxy_written_keys_and_leaves_caller_keys_alone(capsys):
    transport, seen = _fake_openai((POLLUTED, CLEAN, NO_METADATA))

    exit_code: Final = _run(["--apply"], transport)

    assert exit_code == 0
    assert _posts(seen) == [
        (
            "/v1/vector_stores/vs_1",
            {
                "metadata": {
                    "deployment": None,
                    "endpoint": None,
                    "user_api_key": None,
                    "user_api_key_hash": None,
                    "user_api_key_user_email": None,
                }
            },
        )
    ]
    assert "scanned=3 matched=1 updated=1 failed=0" in capsys.readouterr().out


def test_extra_key_is_deleted_alongside_the_built_in_keys():
    transport, seen = _fake_openai((POLLUTED,))

    _run(["--apply", "--extra-key", "trace_id"], transport)

    assert _posts(seen)[0][1]["metadata"]["trace_id"] is None
    assert "team" not in _posts(seen)[0][1]["metadata"]


def test_listing_follows_has_more_with_the_last_store_id(capsys):
    transport, seen = _fake_openai((CLEAN, POLLUTED, NO_METADATA), page_size=2)

    _run([], transport)

    assert [request.url.params.get("after") for request in seen] == [None, "vs_1"]
    assert "scanned=3 matched=1" in capsys.readouterr().out


def test_listing_walks_more_pages_than_the_interpreter_recursion_limit(capsys):
    stores: Final = tuple({"id": f"vs_{index}", "metadata": None} for index in range(sys.getrecursionlimit() + 50))
    transport, _ = _fake_openai(stores, page_size=1)

    assert _run([], transport) == 0
    assert f"scanned={len(stores)} matched=0" in capsys.readouterr().out


def test_non_string_metadata_values_are_scanned_without_printing_them(capsys):
    transport, _ = _fake_openai(({"id": "vs_9", "metadata": {"user_api_key": "secret-value", "count": 7}},))

    exit_code: Final = _run([], transport)

    out: Final = capsys.readouterr().out
    assert exit_code == 0
    assert "vs_9: would delete 1 keys: user_api_key" in out
    assert "secret-value" not in out


def test_an_unexpected_response_shape_exits_non_zero_without_echoing_metadata(capsys):
    def malformed(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"metadata": {"user_api_key": "secret-value"}}], "has_more": False})

    exit_code: Final = _run([], httpx.MockTransport(malformed))

    captured: Final = capsys.readouterr()
    assert exit_code == 2
    assert "unexpected vector store response shape" in captured.err
    assert "secret-value" not in captured.out + captured.err


def test_vector_store_id_retrieves_only_the_named_stores_without_listing(capsys):
    transport, seen = _fake_openai((POLLUTED, CLEAN, NO_METADATA))

    _run(["--vector-store-id", "vs_2", "--vector-store-id", "vs_1"], transport)

    assert [request.url.path for request in seen] == ["/v1/vector_stores/vs_2", "/v1/vector_stores/vs_1"]
    assert "scanned=2 matched=1" in capsys.readouterr().out


def test_a_failed_update_is_reported_and_exits_non_zero(capsys):
    transport, _ = _fake_openai((POLLUTED,), post_status=400)

    exit_code: Final = _run(["--apply"], transport)

    out: Final = capsys.readouterr().out
    assert exit_code == 1
    assert "vs_1: failed to delete 5 keys (HTTP 400: upstream rejected the update)" in out
    assert "updated=0 failed=1" in out


def test_a_listing_error_exits_non_zero_without_updating_anything():
    seen: Final[list[httpx.Request]] = []

    def reject(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    assert _run(["--apply"], httpx.MockTransport(reject)) == 2
    assert [request.method for request in seen] == ["GET"]


def test_an_unreachable_api_exits_non_zero():
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    assert _run([], httpx.MockTransport(unreachable)) == 2


def test_an_api_key_is_required(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exit_info:
        strip_script.main([])

    assert exit_info.value.code == 2
