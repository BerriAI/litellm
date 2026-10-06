import gzip
import json
from hashlib import sha256
from pathlib import Path
from typing import Final
from urllib.parse import parse_qsl, urlsplit

from google import genai
from google.genai import types
import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
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
                "auth": True,
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


_GEMINI_REPLY: Final[dict[str, JsonValue]] = {
    "candidates": [{"content": {"parts": [{"text": "scripted gemini"}], "role": "model"}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
}
_GEMINI_SSE: Final = (f"data: {json.dumps(_GEMINI_REPLY)}\r\n\r\n".encode(),)
_GEMINI_BODY: Final[dict[str, JsonValue]] = {
    "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
    "systemInstruction": {"parts": [{"text": "be terse"}]},
    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 32, "thinkingConfig": {"thinkingBudget": 0}},
}
_GEMINI_SDK_BODY: Final[dict[str, JsonValue]] = {
    **_GEMINI_BODY,
    "systemInstruction": {"parts": [{"text": "be terse"}], "role": "user"},
}


def _upstream_query(request: Request) -> tuple[str, list[tuple[str, str]]]:
    split: Final = urlsplit(request.target)
    return split.path, parse_qsl(split.query, keep_blank_values=True)


def test_gemini_passthrough_both_auth_spellings_swap_the_virtual_key_for_the_proxy_key(
    gateway: Gateway, tmp_path: Path
) -> None:
    def respond(request: Request) -> Reply:
        if ":streamGenerateContent" in request.target:
            return Reply(chunks=_GEMINI_SSE, content_type="text/event-stream")
        return Reply(body=json.dumps(_GEMINI_REPLY).encode())

    path: Final = tmp_path / "gemini-auth-spellings.yaml"
    model_path: Final = "/v1beta/models/gemini-2.5-flash"
    with wire_server(respond) as wire:
        _gemini_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            with candidate.scenario() as scenario:
                key: Final = scenario.key()
                sent: Final[list[httpx.Request]] = []
                client: Final = genai.Client(
                    api_key=key,
                    http_options=types.HttpOptions(
                        base_url=f"{str(candidate.client.base_url).rstrip('/')}/gemini",
                        client_args={"event_hooks": {"request": [sent.append]}, "timeout": 30, "trust_env": False},
                    ),
                )
                generated: Final = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents="hi",
                    config=types.GenerateContentConfig(
                        system_instruction="be terse",
                        temperature=0.2,
                        max_output_tokens=32,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                streamed_text: Final = "".join(
                    chunk.text or ""
                    for chunk in client.models.generate_content_stream(
                        model="gemini-2.5-flash",
                        contents="hi",
                        config=types.GenerateContentConfig(
                            system_instruction="be terse",
                            temperature=0.2,
                            max_output_tokens=32,
                            thinking_config=types.ThinkingConfig(thinking_budget=0),
                        ),
                    )
                )
                assert generated.text == "scripted gemini", generated
                assert streamed_text == "scripted gemini", streamed_text
                assert len(sent) == 2, sent
                assert all(request.headers.get("x-goog-api-key") == key for request in sent), sent
                assert parse_qsl(sent[1].url.query.decode(), keep_blank_values=True) == [("alt", "sse")], sent[1].url
                query_spelling: Final = candidate.client.post(
                    f"/gemini{model_path}:generateContent", params={"key": key}, json=_GEMINI_BODY
                )
                assert query_spelling.status_code == 200, query_spelling.text
                assert query_spelling.json() == _GEMINI_REPLY, query_spelling.text
                with candidate.client.stream(
                    "POST",
                    f"/gemini{model_path}:streamGenerateContent",
                    params={"key": key, "alt": "sse"},
                    json=_GEMINI_BODY,
                ) as streamed_response:
                    streamed: Final = streamed_response.read()
                    assert streamed_response.status_code == 200, streamed
                assert streamed == b"".join(_GEMINI_SSE), streamed
                received: Final = wire.drain()
                assert [(request.method, *_upstream_query(request)) for request in received] == [
                    ("POST", f"{model_path}:generateContent", [("key", "scripted")]),
                    ("POST", f"{model_path}:streamGenerateContent", [("alt", "sse"), ("key", "scripted")]),
                    ("POST", f"{model_path}:generateContent", [("key", "scripted")]),
                    ("POST", f"{model_path}:streamGenerateContent", [("key", "scripted"), ("alt", "sse")]),
                ], received
                for upstream, sdk_request in zip(received[:2], sent, strict=True):
                    sdk_body: Final = json.loads(sdk_request.content)
                    assert json.loads(upstream.body) == sdk_body, upstream.body
                    assert sdk_body == _GEMINI_SDK_BODY, sdk_request.content
                for request in received[2:]:
                    assert json.loads(request.body) == _GEMINI_BODY, request.body
                for request in received:
                    assert {name: value for name, value in request.headers.items() if key in value} == {}, (
                        request.headers
                    )
                    assert "x-goog-api-key" not in request.headers, request.headers
                    assert key not in request.target, request.target


_CHAT_REPLY: Final[dict[str, JsonValue]] = {
    "id": "chatcmpl-scripted",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "scripted chat"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}
_RESPONSE_OBJECT: Final[dict[str, JsonValue]] = {
    "id": "resp_scripted",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": "gpt-4o-mini",
    "output": [
        {
            "id": "msg_scripted",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "scripted response", "annotations": []}],
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "usage": {
        "input_tokens": 5,
        "output_tokens": 3,
        "total_tokens": 8,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}
_RESPONSE_EVENTS: Final[tuple[dict[str, JsonValue], ...]] = (
    {
        "type": "response.created",
        "sequence_number": 0,
        "response": {**_RESPONSE_OBJECT, "status": "in_progress", "output": []},
    },
    {
        "type": "response.output_text.delta",
        "sequence_number": 1,
        "item_id": "msg_scripted",
        "output_index": 0,
        "content_index": 0,
        "delta": "scripted response",
        "logprobs": [],
    },
    {"type": "response.completed", "sequence_number": 2, "response": _RESPONSE_OBJECT},
)
_RESPONSE_SSE: Final = tuple(
    f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in _RESPONSE_EVENTS
)
_ASSISTANTS_PAGE: Final[dict[str, JsonValue]] = {
    "object": "list",
    "data": [
        {
            "id": "asst_scripted",
            "object": "assistant",
            "created_at": 1,
            "name": "scripted",
            "description": None,
            "model": "gpt-4o-mini",
            "instructions": None,
            "tools": [],
            "metadata": {},
        }
    ],
    "first_id": "asst_scripted",
    "last_id": "asst_scripted",
    "has_more": False,
}


def _openai_peer(request: Request) -> Reply:
    if request.target.startswith("/v1/assistants"):
        return Reply(body=json.dumps(_ASSISTANTS_PAGE).encode())
    if request.target == "/v1/chat/completions":
        return Reply(body=json.dumps(_CHAT_REPLY).encode())
    if json.loads(request.body).get("stream") is True:
        return Reply(chunks=_RESPONSE_SSE, content_type="text/event-stream")
    return Reply(body=json.dumps(_RESPONSE_OBJECT).encode())


def _openai_sdk(candidate: Gateway, key: str, prefix: str, sent: list[httpx.Request]) -> OpenAI:
    return OpenAI(
        api_key=key,
        base_url=f"{str(candidate.client.base_url).rstrip('/')}{prefix}/v1",
        max_retries=0,
        http_client=httpx.Client(timeout=30, trust_env=False, event_hooks={"request": [sent.append]}),
    )


def _assert_openai_upstream_auth(request: Request, key: str) -> None:
    assert request.headers["authorization"] == "Bearer scripted", request.headers
    assert {name: value for name, value in request.headers.items() if key in value} == {}, request.headers
    assert key.encode() not in request.body, request.body


def _assert_openai_sdk_request(
    wire: Wire,
    sent: list[httpx.Request],
    key: str,
    method: str,
    target: str,
    expected_body: dict[str, JsonValue] | None,
) -> Request:
    received: Final = wire.drain()
    assert [(request.method, request.target) for request in received] == [(method, target)], received
    assert len(sent) == 1, sent
    client_request: Final = sent[0]
    upstream: Final = received[0]
    _assert_openai_upstream_auth(upstream, key)
    if expected_body is None:
        assert client_request.content == b"", client_request.content
        assert upstream.body == b"", upstream.body
    else:
        assert json.loads(client_request.content) == expected_body, client_request.content
        assert json.loads(upstream.body) == expected_body, upstream.body
    return upstream


_OPENAI_CHAT_BODY: Final[dict[str, JsonValue]] = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "hi"}],
    "user": "end-user-1",
    "store": True,
    "temperature": 0.2,
}
_OPENAI_RESPONSES_BODY: Final[dict[str, JsonValue]] = {
    "model": "gpt-4o-mini",
    "input": "hi",
    "store": False,
    "user": "end-user-1",
    "temperature": 0.2,
}
_OPENAI_RESPONSES_STREAM_BODY: Final[dict[str, JsonValue]] = {
    **_OPENAI_RESPONSES_BODY,
    "stream": True,
}


@pytest.mark.parametrize(
    ("base_path", "surface"),
    [
        pytest.param("/openai", "chat", id="openai-chat"),
        pytest.param("/openai_passthrough", "chat", id="openai-passthrough-chat"),
        pytest.param("/openai_passthrough", "responses", id="openai-passthrough-responses"),
        pytest.param("/openai_passthrough", "responses-stream", id="openai-passthrough-responses-stream"),
        pytest.param("/openai", "assistants", id="openai-assistants"),
        pytest.param("/openai_passthrough", "assistants", id="openai-passthrough-assistants"),
    ],
)
def test_openai_passthrough_sdk_success_on_both_aliases_forwards_the_native_request(
    gateway: Gateway, tmp_path: Path, base_path: str, surface: str
) -> None:
    path: Final = tmp_path / "openai-success.yaml"
    with wire_server(_openai_peer) as wire:
        _openai_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            with candidate.scenario() as scenario:
                key: Final = scenario.key()
                sent: Final[list[httpx.Request]] = []
                with _openai_sdk(candidate, key, base_path, sent) as sdk:
                    if surface == "chat":
                        chat: Final = sdk.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": "hi"}],
                            user="end-user-1",
                            store=True,
                            temperature=0.2,
                        )
                        assert chat.model_dump(exclude_unset=True) == _CHAT_REPLY, chat
                        _assert_openai_sdk_request(wire, sent, key, "POST", "/v1/chat/completions", _OPENAI_CHAT_BODY)
                    elif surface == "responses":
                        response: Final = sdk.responses.create(
                            model="gpt-4o-mini",
                            input="hi",
                            store=False,
                            user="end-user-1",
                            temperature=0.2,
                        )
                        assert response.output_text == "scripted response", response
                        assert response.id == "resp_scripted", response
                        _assert_openai_sdk_request(wire, sent, key, "POST", "/v1/responses", _OPENAI_RESPONSES_BODY)
                    elif surface == "responses-stream":
                        events: Final = tuple(
                            sdk.responses.create(
                                model="gpt-4o-mini",
                                input="hi",
                                store=False,
                                user="end-user-1",
                                temperature=0.2,
                                stream=True,
                            )
                        )
                        streamed_text: Final = "".join(
                            event.delta for event in events if event.type == "response.output_text.delta"
                        )
                        final: Final = next(event.response for event in events if event.type == "response.completed")
                        assert [event.type for event in events] == [event["type"] for event in _RESPONSE_EVENTS], events
                        assert streamed_text == "scripted response", events
                        assert final.output_text == "scripted response", final
                        _assert_openai_sdk_request(
                            wire, sent, key, "POST", "/v1/responses", _OPENAI_RESPONSES_STREAM_BODY
                        )
                    else:
                        assistants: Final = sdk.beta.assistants.list(limit=2, order="desc")
                        assert [assistant.id for assistant in assistants.data] == ["asst_scripted"], assistants
                        upstream: Final = _assert_openai_sdk_request(
                            wire,
                            sent,
                            key,
                            "GET",
                            "/v1/assistants?limit=2&order=desc",
                            None,
                        )
                        assert upstream.headers["openai-beta"] == "assistants=v2", upstream.headers


def test_openai_passthrough_responses_metadata_reaches_upstream(gateway: Gateway, tmp_path: Path) -> None:
    pytest.skip("BUG: /openai_passthrough/v1/responses drops the native metadata field before forwarding upstream")
    path: Final = tmp_path / "openai-metadata.yaml"
    with wire_server(_openai_peer) as wire:
        _openai_config(path, wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            with candidate.scenario() as scenario:
                sent: Final[list[httpx.Request]] = []
                with _openai_sdk(candidate, scenario.key(), "/openai_passthrough", sent) as sdk:
                    sdk.responses.create(model="gpt-4o-mini", input="hi", metadata={"session": "scripted"})
            received: Final = wire.drain()
            assert [json.loads(request.body) for request in received] == [json.loads(sent[0].content)], received
