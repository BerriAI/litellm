import gzip
import json
import threading
import time
import uuid
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.upstream import delete_scenario, register_scenario
from integration._support.wire import Reply, Request, wire_server
from integration.cost_calculation.cost_tracking_case import JsonResponse
from openai import AsyncOpenAI, NotFoundError, OpenAI
from pydantic import JsonValue

_UPSTREAM_ERROR: Final[dict[str, JsonValue]] = {
    "error": {
        "code": 404,
        "message": "Publisher Model `publishers/anthropic/models/claude-nope-9` was not found or your project does not have access to it. Please ensure you are using a valid model version.",
        "status": "NOT_FOUND",
    }
}


def test_gemini_passthrough_upstream_error_body_reaches_proxy_log_and_spend_row(
    gateway: Gateway, tmp_path: Path
) -> None:
    def respond(request: Request) -> Reply:
        return Reply(status=404, body=json.dumps(_UPSTREAM_ERROR).encode())

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path: Final = tmp_path / "gemini-passthrough.yaml"
    with wire_server(respond) as wire:
        config["environment_variables"] = {"GEMINI_API_BASE": wire.url, "GEMINI_API_KEY": "scripted"}
        path.write_text(yaml.safe_dump(config))
        with owned_proxy_process(gateway, tmp_path, {}, config=path) as owned:
            candidate: Final = owned.gateway
            response: Final = candidate.request(
                "POST",
                "/gemini/v1beta/models/claude-nope-9:generateContent",
                {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
                headers={"x-goog-api-key": candidate.key},
            )
            assert response.status_code == 404, response.text
            assert response.json() == _UPSTREAM_ERROR, response.text
            try:
                eventually(
                    lambda: owned.log.read_text(),
                    lambda text: "was not found or your project" in text,
                    seconds=30,
                )
            except AssertionError:
                pytest.fail(
                    f"upstream 404 body never reached the proxy log after {response.status_code} passthrough; "
                    f"log tail: {owned.log.read_text()[-2000:]}"
                )
            rows: Final = eventually(
                lambda: read_rows(
                    'SELECT metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                    (response.headers["x-litellm-call-id"],),
                ),
                lambda values: len(values) == 1,
                seconds=70,
            )
            metadata: Final = rows[0]["metadata"]
            parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
            error_information: Final = object_value(parsed["error_information"])
            assert "was not found or your project" in str(error_information["error_message"]), response.text
            assert error_information["error_code"] == "404", response.text


_GEMINI_MODEL_PATH: Final = "/gemini/v1beta/models/claude-nope-9:generateContent"
_GEMINI_STREAM_PATH: Final = "/gemini/v1beta/models/claude-nope-9:streamGenerateContent"
_GENERATE_CONTENT: Final[dict[str, JsonValue]] = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
_UPSTREAM_500_BODY: Final = (
    '{"error":{"code":500,"message":"' + "chunked upstream failure body " * 200 + '","status":"INTERNAL"}}'
).encode()


def _gemini_config(path: Path, wire_url: str) -> None:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["environment_variables"] = {"GEMINI_API_BASE": wire_url, "GEMINI_API_KEY": "scripted"}
    path.write_text(yaml.safe_dump(config))


def _gemini_headers(candidate: Gateway) -> dict[str, str]:
    return {"Authorization": f"Bearer {candidate.key}", "x-goog-api-key": candidate.key}


def _spend_error_information(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    metadata: Final = rows[0]["metadata"]
    parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
    return object_value(parsed["error_information"])


def _spend_error_information_or_none(call_id: str, seconds: float = 20) -> dict[str, JsonValue] | None:
    rows: Final = eventually(
        lambda: read_rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,)),
        lambda values: len(values) == 1,
        seconds=seconds,
    )
    metadata: Final = rows[0]["metadata"]
    parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
    error_information: Final = parsed.get("error_information")
    return None if error_information is None else object_value(error_information)


def _spend_status(call_id: str) -> str:
    rows: Final = eventually(
        lambda: read_rows('SELECT status FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return str(rows[0]["status"])


def _upstream_warning(log: Path, needle: str = "pass_through_endpoint: upstream") -> str:
    text: Final = eventually(lambda: log.read_text(), lambda content: needle in content, seconds=30)
    return next(line for line in text.splitlines() if needle in line)


def _upstream_warnings(log: Path, needle: str = "pass_through_endpoint: upstream") -> tuple[str, ...]:
    return tuple(line for line in log.read_text().splitlines() if needle in line)


async def test_gemini_passthrough_async_client_404_body_reaches_proxy_log_and_spend_row(
    gateway: Gateway, tmp_path: Path
) -> None:
    def respond(request: Request) -> Reply:
        return Reply(status=404, body=json.dumps(_UPSTREAM_ERROR).encode())

    path: Final = tmp_path / "gemini-async.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            async with httpx.AsyncClient(
                base_url=str(candidate.client.base_url), timeout=15, trust_env=False
            ) as async_client:
                response: Final = await async_client.post(
                    _GEMINI_MODEL_PATH, json=_GENERATE_CONTENT, headers=_gemini_headers(candidate)
                )
            assert response.status_code == 404, response.text
            assert response.json() == _UPSTREAM_ERROR, response.text
            warning: Final = _upstream_warning(owned.log)
            assert "was not found or your project" in warning, warning
            error_information: Final = _spend_error_information(response.headers["x-litellm-call-id"])
            assert "was not found or your project" in str(error_information["error_message"]), response.text
            assert error_information["error_code"] == "404", response.text


def test_gemini_passthrough_streaming_500_relays_full_body_and_logs_bounded_preview(
    gateway: Gateway, tmp_path: Path
) -> None:
    body: Final = _UPSTREAM_500_BODY
    assert len(body) == 6055
    chunks: Final = tuple(body[index * 512 : (index + 1) * 512] for index in range(11)) + (body[5632:],)

    def respond(request: Request) -> Reply:
        return Reply(status=500, chunks=chunks)

    path: Final = tmp_path / "gemini-stream-500.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            with candidate.client.stream(
                "POST",
                _GEMINI_STREAM_PATH,
                params={"alt": "sse"},
                json=_GENERATE_CONTENT,
                headers=_gemini_headers(candidate),
            ) as response:
                assert response.status_code == 500, response.text
                streamed: Final = response.read()
            assert streamed == body
            warning: Final = _upstream_warning(owned.log)
            assert warning.endswith("... (truncated at 4096 chars)"), warning
            error_information: Final = _spend_error_information(response.headers["x-litellm-call-id"])
            error_message: Final = str(error_information["error_message"])
            assert error_message.endswith("... (truncated at 4096 chars)"), error_message
            assert error_information["error_code"] == "500", error_message


def test_gemini_passthrough_success_logs_nothing_and_spend_row_is_success(gateway: Gateway, tmp_path: Path) -> None:
    upstream_ok: Final = {
        "candidates": [{"content": {"parts": [{"text": "hello"}], "role": "model"}}],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
    }

    def respond(request: Request) -> Reply:
        return Reply(status=200, body=json.dumps(upstream_ok).encode())

    path: Final = tmp_path / "gemini-200.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            response: Final = candidate.request(
                "POST", _GEMINI_MODEL_PATH, _GENERATE_CONTENT, headers={"x-goog-api-key": candidate.key}
            )
            assert response.status_code == 200, response.text
            assert response.json() == upstream_ok, response.text
            assert _spend_status(response.headers["x-litellm-call-id"]) == "success"
            assert not _upstream_warnings(owned.log), owned.log.read_text()[-2000:]


def test_gemini_passthrough_streaming_200_relays_every_chunk(gateway: Gateway, tmp_path: Path) -> None:
    chunks: Final = tuple(f"data: chunk-{index}\n\n".encode() for index in range(5))

    def respond(request: Request) -> Reply:
        return Reply(status=200, chunks=chunks, content_type="text/event-stream")

    path: Final = tmp_path / "gemini-stream-200.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            with candidate.client.stream(
                "POST",
                _GEMINI_STREAM_PATH,
                params={"alt": "sse"},
                json=_GENERATE_CONTENT,
                headers=_gemini_headers(candidate),
            ) as response:
                assert response.status_code == 200
                streamed: Final = response.read()
            assert streamed == b"".join(chunks)
            assert not _upstream_warnings(owned.log), owned.log.read_text()[-2000:]


def test_config_pass_through_route_logs_body_and_strips_query(gateway: Gateway, tmp_path: Path) -> None:
    upstream_error: Final = {"error": {"message": "max budget reached for this deployment"}}

    def respond(request: Request) -> Reply:
        return Reply(status=403, body=json.dumps(upstream_error).encode())

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path: Final = tmp_path / "config-route.yaml"
    with wire_server(respond) as wire:
        config["general_settings"]["pass_through_endpoints"] = [
            {
                "path": "/audit-pt",
                "target": f"{wire.url}/upstream?trace=secret-q",
                "include_subpath": True,
                "headers": {"Authorization": "Bearer scripted"},
            }
        ]
        path.write_text(yaml.safe_dump(config))
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            response: Final = candidate.request("POST", "/audit-pt", _GENERATE_CONTENT)
            assert response.status_code == 403, response.text
            assert response.json() == upstream_error, response.text
            warning: Final = _upstream_warning(owned.log)
            assert "max budget reached for this deployment" in warning, warning
            assert "?" not in warning and "secret-q" not in warning, warning
            error_information: Final = _spend_error_information(response.headers["x-litellm-call-id"])
            assert error_information["normalized_error"] == "500_UPSTREAM_PASSTHROUGH", response.text
            assert "max budget reached for this deployment" in str(error_information["error_message"]), response.text


_OPENAI_UPSTREAM_404: Final[dict[str, JsonValue]] = {
    "error": {"message": "The model `nope-9` does not exist", "type": "invalid_request_error"}
}


def _openai_config(path: Path, wire_url: str) -> None:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["environment_variables"] = {"OPENAI_API_BASE": wire_url, "OPENAI_API_KEY": "scripted"}
    path.write_text(yaml.safe_dump(config))


def test_openai_passthrough_sdk_error_body_reaches_proxy_log_and_spend_row(gateway: Gateway, tmp_path: Path) -> None:
    def respond(request: Request) -> Reply:
        return Reply(status=404, body=json.dumps(_OPENAI_UPSTREAM_404).encode())

    path: Final = tmp_path / "openai-404.yaml"
    with wire_server(respond) as wire:
        _openai_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            with OpenAI(
                api_key=candidate.key,
                base_url=f"{str(candidate.client.base_url).rstrip('/')}/openai",
                max_retries=0,
                http_client=httpx.Client(timeout=15, trust_env=False),
            ) as sdk:
                with pytest.raises(NotFoundError) as raised:
                    sdk.chat.completions.create(model="nope-9", messages=[{"role": "user", "content": "hi"}])
            assert "does not exist" in str(raised.value), raised.value
            warning: Final = _upstream_warning(owned.log)
            assert "does not exist" in warning, warning
            error_information: Final = _spend_error_information(raised.value.response.headers["x-litellm-call-id"])
            assert "does not exist" in str(error_information["error_message"])


async def test_openai_passthrough_async_sdk_error_body_reaches_proxy_log_and_spend_row(
    gateway: Gateway, tmp_path: Path
) -> None:
    def respond(request: Request) -> Reply:
        return Reply(status=404, body=json.dumps(_OPENAI_UPSTREAM_404).encode())

    path: Final = tmp_path / "openai-async-404.yaml"
    with wire_server(respond) as wire:
        _openai_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            async with AsyncOpenAI(
                api_key=candidate.key,
                base_url=f"{str(candidate.client.base_url).rstrip('/')}/openai",
                max_retries=0,
                http_client=httpx.AsyncClient(timeout=15, trust_env=False),
            ) as sdk:
                with pytest.raises(NotFoundError) as raised:
                    await sdk.chat.completions.create(model="nope-9", messages=[{"role": "user", "content": "hi"}])
            assert "does not exist" in str(raised.value), raised.value
            warning: Final = _upstream_warning(owned.log)
            assert "does not exist" in warning, warning
            error_information: Final = _spend_error_information(raised.value.response.headers["x-litellm-call-id"])
            assert "does not exist" in str(error_information["error_message"])


def test_gemini_passthrough_control_characters_cannot_forge_log_lines(gateway: Gateway, tmp_path: Path) -> None:
    forged: Final = b'{"error": "line one"}\n2026-01-01 FAKE LOG LINE\x1b[31m\r' + b"x" * 4943 + b"\x00tail"
    assert len(forged) == 5000

    def respond(request: Request) -> Reply:
        return Reply(status=502, body=forged, content_type="text/html")

    path: Final = tmp_path / "gemini-forged.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            response: Final = candidate.request(
                "POST", _GEMINI_MODEL_PATH, _GENERATE_CONTENT, headers={"x-goog-api-key": candidate.key}
            )
            assert response.status_code == 502, response.text
            assert response.content == forged, response.text
            warning: Final = _upstream_warning(owned.log)
            assert "\n" not in warning and "\x1b" not in warning, warning
            assert "line one" in warning and "FAKE LOG LINE" in warning, warning
            assert warning.endswith("... (truncated at 4096 chars)"), warning
            error_information: Final = _spend_error_information(response.headers["x-litellm-call-id"])
            assert error_information["error_code"] == "502", response.text


def test_gemini_passthrough_empty_error_body_still_logged_and_proxy_serves(gateway: Gateway, tmp_path: Path) -> None:
    def respond(request: Request) -> Reply:
        if "claude-nope-9" in request.target:
            return Reply(status=404, body=b"")
        return Reply(status=200, body=b'{"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}')

    path: Final = tmp_path / "gemini-empty.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            response: Final = candidate.request(
                "POST", _GEMINI_MODEL_PATH, _GENERATE_CONTENT, headers={"x-goog-api-key": candidate.key}
            )
            assert response.status_code == 404, response.text
            assert response.content == b"", response.text
            warning: Final = _upstream_warning(owned.log)
            assert "returned 404" in warning, warning
            error_information: Final = _spend_error_information(response.headers["x-litellm-call-id"])
            assert error_information["error_code"] == "404", response.text
            follow_up: Final = candidate.request(
                "POST",
                "/gemini/v1beta/models/healthy-model:generateContent",
                _GENERATE_CONTENT,
                headers={"x-goog-api-key": candidate.key},
            )
            assert follow_up.status_code == 200, follow_up.text


def test_gemini_passthrough_gzip_error_body_decoded_for_log_and_client(gateway: Gateway, tmp_path: Path) -> None:
    upstream_error: Final = {"error": {"message": "gzipped upstream says the model is gone"}}

    def respond(request: Request) -> Reply:
        return Reply(
            status=400,
            body=gzip.compress(json.dumps(upstream_error).encode()),
            headers={"content-encoding": "gzip"},
        )

    path: Final = tmp_path / "gemini-gzip.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            response: Final = candidate.request(
                "POST", _GEMINI_MODEL_PATH, _GENERATE_CONTENT, headers={"x-goog-api-key": candidate.key}
            )
            assert response.status_code == 400, response.text
            assert response.json() == upstream_error, response.text
            warning: Final = _upstream_warning(owned.log)
            assert "gzipped upstream says the model is gone" in warning, warning


def test_gemini_passthrough_streaming_gzip_error_body_decoded_for_log_and_client(
    gateway: Gateway, tmp_path: Path
) -> None:
    upstream_error: Final = {"error": {"message": "streamed gzip upstream denies the deployment"}}
    compressed: Final = gzip.compress(json.dumps(upstream_error).encode())
    third: Final = len(compressed) // 3

    def respond(request: Request) -> Reply:
        return Reply(
            status=403,
            chunks=(compressed[:third], compressed[third : 2 * third], compressed[2 * third :]),
            headers={"content-encoding": "gzip"},
        )

    path: Final = tmp_path / "gemini-stream-gzip.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            with candidate.client.stream(
                "POST",
                _GEMINI_STREAM_PATH,
                params={"alt": "sse"},
                json=_GENERATE_CONTENT,
                headers=_gemini_headers(candidate),
            ) as response:
                assert response.status_code == 403
                streamed: Final = response.read()
            assert json.loads(streamed) == upstream_error, streamed
            warning: Final = _upstream_warning(owned.log)
            assert "streamed gzip upstream denies the deployment" in warning, warning


def test_gemini_passthrough_error_body_redacted_when_message_logging_off(gateway: Gateway, tmp_path: Path) -> None:
    upstream_error: Final = {"error": {"message": "sensitive upstream explanation"}}

    def respond(request: Request) -> Reply:
        return Reply(status=404, body=json.dumps(upstream_error).encode())

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path: Final = tmp_path / "gemini-redacted.yaml"
    with wire_server(respond) as wire:
        config["environment_variables"] = {"GEMINI_API_BASE": wire.url, "GEMINI_API_KEY": "scripted"}
        config["litellm_settings"]["turn_off_message_logging"] = True
        path.write_text(yaml.safe_dump(config))
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            response: Final = candidate.request(
                "POST", _GEMINI_MODEL_PATH, _GENERATE_CONTENT, headers={"x-goog-api-key": candidate.key}
            )
            assert response.status_code == 404, response.text
            assert response.json() == upstream_error, response.text
            warning: Final = _upstream_warning(owned.log)
            assert "redacted-by-litellm" in warning, warning
            assert "sensitive upstream explanation" not in warning, warning
            error_information: Final = _spend_error_information(response.headers["x-litellm-call-id"])
            error_message: Final = str(error_information["error_message"])
            assert "redacted-by-litellm" in error_message, error_message
            assert "sensitive upstream explanation" not in error_message, error_message


def test_gemini_passthrough_exact_4096_byte_body_logged_without_marker(gateway: Gateway, tmp_path: Path) -> None:
    body: Final = b'{"error": "' + b"y" * 4083 + b'"}'
    assert len(body) == 4096

    def respond(request: Request) -> Reply:
        return Reply(status=404, body=body)

    path: Final = tmp_path / "gemini-exact.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            response: Final = candidate.request(
                "POST", _GEMINI_MODEL_PATH, _GENERATE_CONTENT, headers={"x-goog-api-key": candidate.key}
            )
            assert response.status_code == 404, response.text
            warning: Final = _upstream_warning(owned.log)
            assert body[:512].decode() in warning, warning
            assert "(truncated at 4096 chars)" not in warning, warning


def test_gemini_passthrough_4097_byte_body_truncated_with_marker(gateway: Gateway, tmp_path: Path) -> None:
    body: Final = b'{"error": "' + b"y" * 4084 + b'"}'
    assert len(body) == 4097

    def respond(request: Request) -> Reply:
        return Reply(status=404, body=body)

    path: Final = tmp_path / "gemini-over.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            response: Final = candidate.request(
                "POST", _GEMINI_MODEL_PATH, _GENERATE_CONTENT, headers={"x-goog-api-key": candidate.key}
            )
            assert response.status_code == 404, response.text
            warning: Final = _upstream_warning(owned.log)
            assert body[:512].decode() in warning, warning
            assert warning.endswith("... (truncated at 4096 chars)"), warning


def test_gemini_passthrough_one_byte_stream_chunks_reassembled_and_logged(gateway: Gateway, tmp_path: Path) -> None:
    body: Final = json.dumps(_UPSTREAM_ERROR).encode()

    def respond(request: Request) -> Reply:
        return Reply(status=404, chunks=tuple(bytes([byte]) for byte in body))

    path: Final = tmp_path / "gemini-one-byte.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            with candidate.client.stream(
                "POST",
                _GEMINI_STREAM_PATH,
                params={"alt": "sse"},
                json=_GENERATE_CONTENT,
                headers=_gemini_headers(candidate),
            ) as response:
                assert response.status_code == 404
                streamed: Final = response.read()
            assert streamed == body
            warning: Final = _upstream_warning(owned.log)
            assert "was not found or your project" in warning, warning


def test_gemini_passthrough_repeated_errors_each_get_row_and_log_line(gateway: Gateway, tmp_path: Path) -> None:
    def respond(request: Request) -> Reply:
        return Reply(status=404, body=json.dumps(_UPSTREAM_ERROR).encode())

    path: Final = tmp_path / "gemini-twice.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            responses: Final = tuple(
                candidate.request(
                    "POST", _GEMINI_MODEL_PATH, _GENERATE_CONTENT, headers={"x-goog-api-key": candidate.key}
                )
                for _ in range(2)
            )
            call_ids: Final = tuple(response.headers["x-litellm-call-id"] for response in responses)
            assert len(set(call_ids)) == 2
            for response in responses:
                assert response.status_code == 404, response.text
                error_information: Final = _spend_error_information(response.headers["x-litellm-call-id"])
                assert "was not found or your project" in str(error_information["error_message"]), response.text
            eventually(
                lambda: _upstream_warnings(owned.log, "returned 404"),
                lambda lines: len(lines) == 2,
                seconds=30,
            )


def test_budget_rejected_call_keeps_budget_normalized_error(gateway: Gateway, tmp_path: Path) -> None:
    path: Final = tmp_path / "budget.yaml"
    path.write_text(Path("tests/integration/proxy_config.yaml").read_text())
    with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
        candidate: Final = owned.gateway
        with candidate.scenario() as scenario:
            model: Final = scenario.model()
            key: Final = scenario.key(max_budget=0.000001)
            first: Final = candidate.chat(model, key=key)
            assert "id" in first, first
            rejected: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "over budget"}]},
                key=key,
            )
            assert rejected.status_code == 422 and "budget_exceeded" in rejected.text, rejected.text
            digest: Final = sha256(key.encode()).hexdigest()
            rows: Final = eventually(
                lambda: read_rows(
                    'SELECT metadata FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
                    (digest,),
                ),
                lambda values: any(
                    "BUDGET_EXCEEDED"
                    in str(
                        object_value(
                            json.loads(row["metadata"])
                            if isinstance(row["metadata"], str)
                            else object_value(row["metadata"])
                        )["error_information"]
                    )
                    for row in values
                ),
                seconds=70,
            )
            budget_rows: Final = tuple(
                row
                for row in rows
                if "BUDGET_EXCEEDED"
                in str(
                    object_value(
                        json.loads(row["metadata"])
                        if isinstance(row["metadata"], str)
                        else object_value(row["metadata"])
                    )["error_information"]
                )
            )
            assert len(budget_rows) == 1, budget_rows


_LEAKED_UPSTREAM_KEY: Final = "sk-" + "leak0" * 8


def test_gemini_passthrough_streaming_429_first_frame_reaches_client_while_upstream_holds(
    gateway: Gateway, tmp_path: Path
) -> None:
    gate: Final = threading.Event()
    frames: Final = (b'data: {"error":"rate limited"}\n\n', b"data: [DONE]\n\n")

    def respond(request: Request) -> Reply:
        return Reply(status=429, content_type="text/event-stream", chunks=frames, gate_after_first=gate)

    path: Final = tmp_path / "gemini-stream-429.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            with candidate.client.stream(
                "POST",
                _GEMINI_STREAM_PATH,
                params={"alt": "sse"},
                json=_GENERATE_CONTENT,
                headers=_gemini_headers(candidate),
                timeout=httpx.Timeout(3, connect=5),
            ) as response:
                assert response.status_code == 429, response.text
                iterator: Final = response.iter_bytes()
                first: Final = next(iterator)
                assert first.startswith(b'data: {"error":"rate limited"}'), first
                gate.set()
                rest: Final = b"".join(iterator)
            assert first + rest == b"".join(frames), first + rest
            warning: Final = _upstream_warning(owned.log)
            assert "rate limited" in warning, warning
            error_information: Final = _spend_error_information(response.headers["x-litellm-call-id"])
            assert error_information["error_code"] == "429", error_information


def test_gemini_passthrough_quota_wording_in_upstream_body_keeps_passthrough_normalized_error(
    gateway: Gateway, tmp_path: Path
) -> None:
    body: Final[dict[str, JsonValue]] = {
        "error": {
            "message": "You exceeded your current quota, please check your plan and billing details. "
            f"Budget for Key={_LEAKED_UPSTREAM_KEY} is spent",
            "type": "insufficient_quota",
        }
    }

    def respond(request: Request) -> Reply:
        return Reply(status=429, body=json.dumps(body).encode())

    path: Final = tmp_path / "gemini-quota-429.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            response: Final = candidate.request(
                "POST", _GEMINI_MODEL_PATH, _GENERATE_CONTENT, headers=_gemini_headers(candidate)
            )
            assert response.status_code == 429, response.text
            assert response.json() == body, response.text
            error_information: Final = _spend_error_information(response.headers["x-litellm-call-id"])
            assert error_information["normalized_error"] == "500_UPSTREAM_PASSTHROUGH", error_information
            assert error_information["error_code"] == "429", error_information
            assert "exceeded your current quota" in str(error_information["error_message"]), error_information
            assert _LEAKED_UPSTREAM_KEY not in str(error_information["error_message"]), error_information
            warning: Final = _upstream_warning(owned.log)
            assert "exceeded your current quota" in warning, warning
            assert _LEAKED_UPSTREAM_KEY not in warning, warning


def test_gemini_passthrough_streaming_429_client_disconnect_still_logs_failure(
    gateway: Gateway, tmp_path: Path
) -> None:
    gate: Final = threading.Event()
    frames: Final = (b'data: {"error":"rate limited"}\n\n', b"data: [DONE]\n\n")

    def respond(request: Request) -> Reply:
        return Reply(status=429, content_type="text/event-stream", chunks=frames, gate_after_first=gate)

    path: Final = tmp_path / "gemini-stream-429-disconnect.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            try:
                with candidate.client.stream(
                    "POST",
                    _GEMINI_STREAM_PATH,
                    params={"alt": "sse"},
                    json=_GENERATE_CONTENT,
                    headers=_gemini_headers(candidate),
                    timeout=httpx.Timeout(3, connect=5),
                ) as response:
                    assert response.status_code == 429, response.text
                    first: Final = next(response.iter_bytes())
                    assert first.startswith(b'data: {"error":"rate limited"}'), first
                    call_id: Final = response.headers["x-litellm-call-id"]
            finally:
                gate.set()
            error_information: Final = _spend_error_information_or_none(call_id)
            assert error_information is not None, f"no spend row for {call_id} after client disconnect"
            assert error_information["error_code"] == "429", error_information
            assert error_information["normalized_error"] == "500_UPSTREAM_PASSTHROUGH", error_information
            warnings: Final = _upstream_warnings(owned.log, "returned 429")
            assert len(warnings) == 1, warnings


async def test_gemini_passthrough_async_streaming_429_first_frame_reaches_client_while_upstream_holds(
    gateway: Gateway, tmp_path: Path
) -> None:
    gate: Final = threading.Event()
    frames: Final = (b'data: {"error":"rate limited"}\n\n', b"data: [DONE]\n\n")

    def respond(request: Request) -> Reply:
        return Reply(status=429, content_type="text/event-stream", chunks=frames, gate_after_first=gate)

    path: Final = tmp_path / "gemini-async-stream-429.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            try:
                async with httpx.AsyncClient(
                    base_url=str(candidate.client.base_url), timeout=httpx.Timeout(3, connect=5), trust_env=False
                ) as async_client:
                    async with async_client.stream(
                        "POST",
                        _GEMINI_STREAM_PATH,
                        params={"alt": "sse"},
                        json=_GENERATE_CONTENT,
                        headers=_gemini_headers(candidate),
                    ) as response:
                        assert response.status_code == 429, response.text
                        iterator = response.aiter_bytes()
                        first: Final = await iterator.__anext__()
                        assert first.startswith(b'data: {"error":"rate limited"}'), first
                        gate.set()
                        rest: Final = b""
                        async for chunk in iterator:
                            rest += chunk
                    assert first + rest == b"".join(frames), first + rest
            finally:
                gate.set()
            warning: Final = _upstream_warning(owned.log)
            assert "rate limited" in warning, warning
            error_information: Final = _spend_error_information(response.headers["x-litellm-call-id"])
            assert error_information["error_code"] == "429", error_information


_PROVIDER_STREAM_SPECS: Final = (
    (
        "anthropic",
        "/anthropic/v1/messages",
        {"ANTHROPIC_API_BASE": None, "ANTHROPIC_API_KEY": "scripted"},
        "/v1/messages",
    ),
    (
        "cohere",
        "/cohere/v2/chat",
        {"COHERE_API_BASE": None, "COHERE_API_KEY": "scripted"},
        "/v2/chat",
    ),
    (
        "mistral",
        "/mistral/v1/chat/completions",
        {"MISTRAL_API_BASE": None, "MISTRAL_API_KEY": "scripted"},
        "/v1/chat/completions",
    ),
    (
        "openai",
        "/openai/v1/chat/completions",
        {"OPENAI_API_BASE": None, "OPENAI_API_KEY": "scripted"},
        "/v1/chat/completions",
    ),
    (
        "azure",
        "/azure/openai/deployments/nope-9/chat/completions",
        {"AZURE_API_BASE": None, "AZURE_API_KEY": "scripted"},
        "/openai/deployments/nope-9/chat/completions",
    ),
    ("config-route", "/audit-pt", None, "/upstream"),
)


@pytest.mark.parametrize(
    ("name", "proxy_path", "environment", "upstream_suffix"),
    _PROVIDER_STREAM_SPECS,
    ids=tuple(spec[0] for spec in _PROVIDER_STREAM_SPECS),
)
def test_provider_passthrough_streaming_429_first_frame_reaches_client_while_upstream_holds(
    gateway: Gateway,
    tmp_path: Path,
    name: str,
    proxy_path: str,
    environment: dict[str, str | None] | None,
    upstream_suffix: str,
) -> None:
    gate: Final = threading.Event()
    frames: Final = (b'data: {"error":"rate limited"}\n\n', b"data: [DONE]\n\n")

    def respond(request: Request) -> Reply:
        return Reply(status=429, content_type="text/event-stream", chunks=frames, gate_after_first=gate)

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path: Final = tmp_path / f"{name}-stream-429.yaml"
    with wire_server(respond) as wire:
        if environment is None:
            config["general_settings"]["pass_through_endpoints"] = [
                {"path": proxy_path, "target": f"{wire.url}{upstream_suffix}", "include_subpath": True}
            ]
        else:
            config["environment_variables"] = {
                key: (wire.url if value is None else value) for key, value in environment.items()
            }
        path.write_text(yaml.safe_dump(config))
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            with candidate.client.stream(
                "POST",
                proxy_path,
                json={"model": "nope-9", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                headers={"Authorization": f"Bearer {candidate.key}"},
                timeout=httpx.Timeout(3, connect=5),
            ) as response:
                assert response.status_code == 429, response.text
                iterator: Final = response.iter_bytes()
                first: Final = next(iterator)
                assert first.startswith(b'data: {"error":"rate limited"}'), first
                gate.set()
                rest: Final = b"".join(iterator)
            assert first + rest == b"".join(frames), first + rest
            received: Final = wire.drain()
            assert any(request.target.endswith(upstream_suffix) for request in received), received
            warning: Final = _upstream_warning(owned.log)
            assert "rate limited" in warning, warning
            error_information: Final = _spend_error_information(response.headers["x-litellm-call-id"])
            assert error_information["error_code"] == "429", error_information


async def test_gemini_passthrough_async_quota_wording_keeps_passthrough_normalized_error(
    gateway: Gateway, tmp_path: Path
) -> None:
    body: Final[dict[str, JsonValue]] = {
        "error": {
            "message": "You exceeded your current quota, please check your plan and billing details. "
            f"Budget for Key={_LEAKED_UPSTREAM_KEY} is spent",
            "type": "insufficient_quota",
        }
    }

    def respond(request: Request) -> Reply:
        return Reply(status=429, body=json.dumps(body).encode())

    path: Final = tmp_path / "gemini-async-quota-429.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            async with httpx.AsyncClient(
                base_url=str(candidate.client.base_url), timeout=15, trust_env=False
            ) as async_client:
                response: Final = await async_client.post(
                    _GEMINI_MODEL_PATH, json=_GENERATE_CONTENT, headers=_gemini_headers(candidate)
                )
            assert response.status_code == 429, response.text
            assert response.json() == body, response.text
            error_information: Final = _spend_error_information(response.headers["x-litellm-call-id"])
            assert error_information["normalized_error"] == "500_UPSTREAM_PASSTHROUGH", error_information
            assert _LEAKED_UPSTREAM_KEY not in str(error_information["error_message"]), error_information
            warning: Final = _upstream_warning(owned.log)
            assert _LEAKED_UPSTREAM_KEY not in warning, warning


def test_gemini_passthrough_near_miss_pattern_body_keeps_liveliness_and_bounded_preview(
    gateway: Gateway, tmp_path: Path
) -> None:
    body: Final = (b"Upstream passthrough request faile " * 1873)[: 64 * 1024]
    assert len(body) == 64 * 1024

    def respond(request: Request) -> Reply:
        return Reply(
            status=429,
            content_type="text/event-stream",
            chunks=tuple(body[index : index + 512] for index in range(0, len(body), 512)),
            pause_between_chunks=0.01,
        )

    liveliness_seconds: list[float] = []
    liveliness_status: list[int] = []

    def probe_liveliness(client: httpx.Client) -> None:
        started: Final = time.monotonic()
        probe: Final = client.get("/health/liveliness")
        liveliness_seconds.append(time.monotonic() - started)
        liveliness_status.append(probe.status_code)

    path: Final = tmp_path / "gemini-near-miss-64k.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            probe_thread: Final = threading.Thread(target=probe_liveliness, args=(candidate.client,))
            with candidate.client.stream(
                "POST",
                _GEMINI_STREAM_PATH,
                params={"alt": "sse"},
                json=_GENERATE_CONTENT,
                headers=_gemini_headers(candidate),
                timeout=httpx.Timeout(30, connect=5),
            ) as response:
                assert response.status_code == 429, response.text
                iterator: Final = response.iter_bytes()
                first: Final = next(iterator)
                probe_thread.start()
                rest: Final = b"".join(iterator)
            streamed: Final = first + rest
            probe_thread.join(timeout=10)
            assert sha256(streamed).hexdigest() == sha256(body).hexdigest(), "relayed body differs"
            assert liveliness_status == [200], liveliness_status
            assert liveliness_seconds[0] < 2, liveliness_seconds
            warning: Final = _upstream_warning(owned.log)
            assert warning.endswith("... (truncated at 4096 chars)"), warning
            error_information: Final = _spend_error_information(response.headers["x-litellm-call-id"])
            assert error_information["error_code"] == "429", error_information
            assert error_information["normalized_error"] == "500_UPSTREAM_PASSTHROUGH", error_information


def test_gemini_passthrough_streaming_429_upstream_abort_after_first_frame_still_logs_once(
    gateway: Gateway, tmp_path: Path
) -> None:
    frames: Final = (b'data: {"error":"rate limited"}\n\n', b"data: second\n\n", b"data: [DONE]\n\n")

    def respond(request: Request) -> Reply:
        if "streamGenerateContent" in request.target:
            return Reply(status=429, content_type="text/event-stream", chunks=frames, abort_after=1)
        return Reply(status=200, body=json.dumps({"ok": True}).encode())

    path: Final = tmp_path / "gemini-stream-429-abort.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            received: Final = bytearray()
            call_ids: Final[list[str]] = []
            try:
                with candidate.client.stream(
                    "POST",
                    _GEMINI_STREAM_PATH,
                    params={"alt": "sse"},
                    json=_GENERATE_CONTENT,
                    headers=_gemini_headers(candidate),
                    timeout=httpx.Timeout(15, connect=5),
                ) as response:
                    assert response.status_code == 429, response.text
                    call_ids.append(response.headers["x-litellm-call-id"])
                    for chunk in response.iter_bytes():
                        received += chunk
            except httpx.HTTPError:
                pass
            assert len(call_ids) == 1, call_ids
            assert bytes(received).startswith(b'data: {"error":"rate limited"}'), bytes(received)
            eventually(
                lambda: _upstream_warnings(owned.log),
                lambda lines: (
                    any("returned 429" in line for line in lines) and any("read failed" in line for line in lines)
                ),
                seconds=30,
            )
            returned: Final = tuple(line for line in _upstream_warnings(owned.log) if "returned 429" in line)
            read_failures: Final = tuple(line for line in _upstream_warnings(owned.log) if "read failed" in line)
            assert len(returned) == 1, returned
            assert len(read_failures) == 1, read_failures
            error_information: Final = _spend_error_information(call_ids[0])
            assert error_information["error_code"] == "429", error_information
            follow_up: Final = candidate.request(
                "POST", _GEMINI_MODEL_PATH, _GENERATE_CONTENT, headers=_gemini_headers(candidate)
            )
            assert follow_up.status_code == 200, follow_up.text


def test_upstream_abort_after_preview_budget_reaches_client_as_truncated(gateway: Gateway, tmp_path: Path) -> None:
    frames: Final = tuple(b"d" * 1000 for _ in range(5)) + (b"data: tail\n\n",)

    def respond(request: Request) -> Reply:
        if "streamGenerateContent" in request.target:
            return Reply(status=500, content_type="text/event-stream", chunks=frames, abort_after=5)
        return Reply(status=200, body=json.dumps({"ok": True}).encode())

    path: Final = tmp_path / "gemini-stream-500-abort-past-preview.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            received: Final = bytearray()

            def consume_error_stream() -> None:
                with candidate.client.stream(
                    "POST",
                    _GEMINI_STREAM_PATH,
                    params={"alt": "sse"},
                    json=_GENERATE_CONTENT,
                    headers=_gemini_headers(candidate),
                    timeout=httpx.Timeout(15, connect=5),
                ) as response:
                    assert response.status_code == 500, response.text
                    for chunk in response.iter_bytes():
                        received.extend(chunk)

            with pytest.raises(httpx.HTTPError):
                consume_error_stream()
            assert bytes(received) == b"d" * 5000, bytes(received)[-64:]
            eventually(
                lambda: _upstream_warnings(owned.log),
                lambda lines: (
                    any("returned 500" in line for line in lines) and any("read failed" in line for line in lines)
                ),
                seconds=30,
            )
            returned: Final = tuple(line for line in _upstream_warnings(owned.log) if "returned 500" in line)
            read_failures: Final = tuple(line for line in _upstream_warnings(owned.log) if "read failed" in line)
            assert len(returned) == 1, returned
            assert len(read_failures) == 1, read_failures


def test_gemini_passthrough_empty_streaming_429_still_logged(gateway: Gateway, tmp_path: Path) -> None:
    def respond(request: Request) -> Reply:
        return Reply(status=429, content_type="text/event-stream", chunks=())

    path: Final = tmp_path / "gemini-stream-429-empty.yaml"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            with candidate.client.stream(
                "POST",
                _GEMINI_STREAM_PATH,
                params={"alt": "sse"},
                json=_GENERATE_CONTENT,
                headers=_gemini_headers(candidate),
            ) as response:
                assert response.status_code == 429, response.text
                streamed: Final = response.read()
            assert streamed == b"", streamed
            warning: Final = _upstream_warning(owned.log)
            assert "returned 429" in warning, warning
            error_information: Final = _spend_error_information(response.headers["x-litellm-call-id"])
            assert error_information["normalized_error"] == "500_UPSTREAM_PASSTHROUGH", error_information


_LLM_429_ENDPOINTS: Final = (
    (
        "/v1/chat/completions",
        "openai/gpt-4o-mini",
        lambda model, stream: {"model": model, "messages": [{"role": "user", "content": "audit"}], "stream": stream},
    ),
    (
        "/v1/messages",
        "anthropic/claude-sonnet-4-5-20250929",
        lambda model, stream: {
            "model": model,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "audit"}],
            "stream": stream,
        },
    ),
    (
        "/v1/responses",
        "openai/gpt-4o-mini",
        lambda model, stream: {"model": model, "input": "audit", "stream": stream},
    ),
)
_LLM_429_BODY: Final[dict[str, JsonValue]] = {
    "error": {
        "message": "You exceeded your current quota, please check your plan and billing details.",
        "type": "insufficient_quota",
    }
}
_LLM_429_NORMALIZED: Final = "429_RATE_LIMIT_EXCEEDED"


@pytest.mark.parametrize(
    ("endpoint", "litellm_model", "build_body"),
    _LLM_429_ENDPOINTS,
    ids=tuple(spec[0] for spec in _LLM_429_ENDPOINTS),
)
@pytest.mark.parametrize("stream", [False, True])
def test_llm_endpoint_upstream_quota_429_normalized_error_unchanged(
    gateway: Gateway, endpoint: str, litellm_model: str, build_body: Callable[..., dict[str, JsonValue]], stream: bool
) -> None:
    scenario_id: Final = f"quota429-{uuid.uuid4().hex}"
    handle: Final = register_scenario(
        scenario_id, JsonResponse(content_type="application/json", body=_LLM_429_BODY, status=429)
    )
    try:
        with gateway.scenario() as scenario:
            api_base: Final = handle.api_base() if litellm_model.startswith("anthropic/") else f"{handle.api_base()}/v1"
            model: Final = scenario.model(model=litellm_model, api_base=api_base)
            if stream:
                with gateway.client.stream(
                    "POST",
                    endpoint,
                    json=build_body(model, True),
                    headers={"Authorization": f"Bearer {gateway.key}"},
                ) as response:
                    assert response.status_code == 429, response.text
                    response.read()
            else:
                response = gateway.request("POST", endpoint, build_body(model, False))
                assert response.status_code == 429, response.text
            error_information: Final = _spend_error_information(response.headers["x-litellm-call-id"])
            assert error_information["normalized_error"] == _LLM_429_NORMALIZED, error_information
    finally:
        delete_scenario(handle)
