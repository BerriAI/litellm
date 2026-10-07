from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import pytest
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(180)

_PROJECT: Final = "scripted-gemini-batch-project"
_MODEL: Final = "gemini-nano-banana-2.1"
_BATCH_ID: Final = "scripted-batch"
_CALLBACK_PREFIX: Final = "CALLBACK:"
_REGISTRY_PREFIX: Final = "REGISTRY:"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_IMAGE_PART: Final = {
    "inlineData": {
        "mimeType": "image/png",
        "data": "aGVsbG8=",
    }
}


def _pipe(source: socket.socket, destination: socket.socket) -> None:
    try:
        for payload in iter(lambda: source.recv(65536), b""):
            destination.sendall(payload)
    except OSError:
        return
    finally:
        try:
            destination.shutdown(socket.SHUT_WR)
        except OSError:
            pass


@contextmanager
def _redirect_https_host(host: str, port: int, *, to_port: int) -> Generator[str, None, None]:
    """Yields an HTTPS_PROXY url that sends host:port to 127.0.0.1:to_port and refuses anything else."""
    authority: Final = f"{host}:{port}"

    class _RedirectHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 30

        def do_CONNECT(self) -> None:
            if self.path != authority:
                self.send_error(403)
                return

            self.send_response(200, "Connection Established")
            self.end_headers()
            try:
                with socket.create_connection(("127.0.0.1", to_port), timeout=10) as upstream:
                    client_to_upstream: Final = threading.Thread(
                        target=_pipe, args=(self.connection, upstream), daemon=True
                    )
                    upstream_to_client: Final = threading.Thread(
                        target=_pipe, args=(upstream, self.connection), daemon=True
                    )
                    client_to_upstream.start()
                    upstream_to_client.start()
                    client_to_upstream.join(timeout=30)
                    upstream_to_client.join(timeout=30)
            except OSError:
                return

        def log_message(self, format: str, *args: object) -> None:
            pass

    class _RedirectProxy(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 16

    with _RedirectProxy(("127.0.0.1", 0), _RedirectHandler) as server:
        thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join(timeout=6)
            assert not thread.is_alive(), "CONNECT tunnel server survived cleanup"
            server.server_close()


@dataclass(frozen=True, slots=True)
class _BatchScenario:
    name: str
    prompt_tokens: int
    prompt_details: tuple[tuple[str, int], ...]
    candidate_tokens: int
    candidate_details: tuple[tuple[str, int], ...]
    thoughts_tokens: int
    expected_prompt_cost: float
    expected_completion_cost: float
    use_explicit_image_batch_rate_override: bool = False


_SCENARIOS: Final = (
    _BatchScenario(
        name="batch_input_text_and_image",
        prompt_tokens=1680,
        prompt_details=(("TEXT", 560), ("IMAGE", 1120)),
        candidate_tokens=100,
        candidate_details=(("TEXT", 100),),
        thoughts_tokens=0,
        expected_prompt_cost=0.00126,
        expected_completion_cost=0.000375,
    ),
    _BatchScenario(
        name="batch_text_and_thinking_output",
        prompt_tokens=100,
        prompt_details=(("TEXT", 100),),
        candidate_tokens=400,
        candidate_details=(("TEXT", 400),),
        thoughts_tokens=600,
        expected_prompt_cost=0.000075,
        expected_completion_cost=0.00375,
    ),
    _BatchScenario(
        name="batch_image_output_1k",
        prompt_tokens=100,
        prompt_details=(("TEXT", 100),),
        candidate_tokens=1120,
        candidate_details=(("IMAGE", 1120),),
        thoughts_tokens=0,
        expected_prompt_cost=0.000075,
        expected_completion_cost=0.0168,
    ),
    _BatchScenario(
        name="batch_image_output_2k",
        prompt_tokens=100,
        prompt_details=(("TEXT", 100),),
        candidate_tokens=1680,
        candidate_details=(("IMAGE", 1680),),
        thoughts_tokens=0,
        expected_prompt_cost=0.000075,
        expected_completion_cost=0.0252,
    ),
    _BatchScenario(
        name="batch_image_output_4k",
        prompt_tokens=100,
        prompt_details=(("TEXT", 100),),
        candidate_tokens=3780,
        candidate_details=(("IMAGE", 3780),),
        thoughts_tokens=0,
        expected_prompt_cost=0.000075,
        expected_completion_cost=0.0567,
    ),
    _BatchScenario(
        name="batch_mixed_text_image_output",
        prompt_tokens=100,
        prompt_details=(("TEXT", 100),),
        candidate_tokens=1220,
        candidate_details=(("TEXT", 100), ("IMAGE", 1120)),
        thoughts_tokens=300,
        expected_prompt_cost=0.000075,
        expected_completion_cost=0.0183,
    ),
    _BatchScenario(
        name="batch_image_rate_uses_explicit_override",
        prompt_tokens=100,
        prompt_details=(("TEXT", 100),),
        candidate_tokens=1120,
        candidate_details=(("IMAGE", 1120),),
        thoughts_tokens=0,
        expected_prompt_cost=0.000075,
        expected_completion_cost=0.0224,
        use_explicit_image_batch_rate_override=True,
    ),
)

_SDK_SCRIPT: Final = textwrap.dedent(
    """
    import asyncio, json, os
    from typing import Final

    import litellm
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from litellm.types.utils import ModelInfo

    model: Final = "gemini-nano-banana-2.1"

    class CaptureLogger(CustomLogger):
        async def async_log_success_event(
            self,
            kwargs: dict[str, object],
            response_obj: object,
            start_time: object,
            end_time: object,
        ) -> None:
            standard: Final = kwargs.get("standard_logging_object")
            hidden: Final = getattr(response_obj, "_hidden_params", None)
            if not isinstance(standard, dict):
                raise RuntimeError("success callback omitted standard_logging_object")
            if standard.get("call_type") != "aretrieve_batch":
                return
            print(
                "CALLBACK:"
                + json.dumps(
                    {
                        "response_cost": standard.get("response_cost"),
                        "cost_breakdown": standard.get("cost_breakdown"),
                        "batch_response_cost": hidden.get("response_cost")
                        if isinstance(hidden, dict)
                        else None,
                    }
                ),
                flush=True,
            )

    async def main() -> None:
        has_explicit_image_batch_rate_override: Final = (
            os.environ["USE_EXPLICIT_IMAGE_BATCH_RATE_OVERRIDE"] == "1"
        )
        if has_explicit_image_batch_rate_override:
            pricing: Final[ModelInfo] = {
                **litellm.model_cost["vertex_ai/gemini-nano-banana-2.1"],
                "output_cost_per_image_token_batches": 2e-5,
            }
            litellm.register_model({f"vertex_ai/{model}": pricing}, persist_across_reloads=False)
        litellm.user_url_validation = False
        litellm.disable_vertex_batch_output_transformation = os.environ["TRANSFORM_OUTPUT"] != "1"
        info: Final = litellm.get_model_info(model=model, custom_llm_provider="vertex_ai")
        assert info["key"] == f"vertex_ai/{model}", info
        assert info["litellm_provider"] == "vertex_ai-language-models", info
        print(
            "REGISTRY:"
            + json.dumps({"key": info["key"], "provider": info["litellm_provider"]}),
            flush=True,
        )
        logger: Final = CaptureLogger()
        litellm.logging_callback_manager.add_litellm_async_success_callback(logger)
        await litellm.aretrieve_batch(
            "scripted-batch",
            custom_llm_provider="vertex_ai",
            model=model,
            api_base=os.environ["VERTEX_API_BASE"],
            vertex_project=os.environ["VERTEX_PROJECT"],
            vertex_location="us-central1",
            vertex_credentials=os.environ["VERTEX_CREDENTIALS"],
            gcs_bucket_name="scripted-bucket",
            num_retries=0,
        )
        await asyncio.sleep(0)
        await GLOBAL_LOGGING_WORKER.flush()

    asyncio.run(main())
    """
)


def _prediction_jsonl(scenario: _BatchScenario) -> bytes:
    prompt_details: Final = tuple(
        {"modality": modality, "tokenCount": count} for modality, count in scenario.prompt_details
    )
    candidate_details: Final = tuple(
        {"modality": modality, "tokenCount": count} for modality, count in scenario.candidate_details
    )
    candidate_parts: Final = tuple(
        {"text": "scripted batch output"} if modality == "TEXT" else _IMAGE_PART
        for modality, _ in scenario.candidate_details
    )
    usage_metadata: Final[dict[str, JsonValue]] = {
        "promptTokenCount": scenario.prompt_tokens,
        "candidatesTokenCount": scenario.candidate_tokens,
        "totalTokenCount": scenario.prompt_tokens + scenario.candidate_tokens + scenario.thoughts_tokens,
        "promptTokensDetails": prompt_details,
        "candidatesTokensDetails": candidate_details,
        **({"thoughtsTokenCount": scenario.thoughts_tokens} if scenario.thoughts_tokens else {}),
    }
    row: Final[dict[str, JsonValue]] = {
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "Scripted Vertex Gemini batch request"}],
                }
            ]
        },
        "status": "",
        "response": {
            "candidates": [
                {
                    "content": {"parts": candidate_parts, "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": usage_metadata,
            "modelVersion": _MODEL,
        },
    }
    return json.dumps(row, separators=(",", ":")).encode("utf-8") + b"\n"


def _api_reply(request: Request) -> Reply:
    if request.method == "POST" and request.target == "/_oauth/token":
        return Reply(body=b'{"access_token":"scripted-token","expires_in":3600,"token_type":"Bearer"}')
    if request.method == "GET" and request.target.endswith(f"/batchPredictionJobs/{_BATCH_ID}"):
        return Reply(
            body=json.dumps(
                {
                    "name": (f"projects/{_PROJECT}/locations/us-central1/batchPredictionJobs/{_BATCH_ID}"),
                    "state": "JOB_STATE_SUCCEEDED",
                    "outputInfo": {
                        "gcsOutputDirectory": (
                            "gs://scripted-bucket/litellm-vertex-files/"
                            "publishers/google/models/gemini-nano-banana-2.1/scripted-prefix"
                        )
                    },
                    "createTime": "2026-10-06T00:00:00.000Z",
                }
            ).encode("utf-8")
        )
    return Reply(status=404, body=b"{}")


def _gcs_reply(content: bytes):
    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target.endswith("predictions.jsonl?alt=media"):
            return Reply(body=content, content_type="application/jsonl")
        return Reply(status=404, body=b"{}")

    return respond


def _subprocess_environment(
    api: Wire,
    certificate: Path,
    credentials: str,
    proxy_url: str,
    scenario: _BatchScenario,
    transform_output: bool,
) -> dict[str, str]:
    repo_root: Final = Path(__file__).resolve().parents[3]
    python_path: Final = os.pathsep.join(path for path in (str(repo_root), os.environ.get("PYTHONPATH")) if path)
    return {
        **os.environ,
        "HTTPS_PROXY": proxy_url,
        "https_proxy": proxy_url,
        "HTTP_PROXY": "",
        "http_proxy": "",
        "SSL_CERT_FILE": str(certificate),
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "VERTEX_API_BASE": api.url,
        "VERTEX_CREDENTIALS": credentials,
        "VERTEX_PROJECT": _PROJECT,
        "USE_EXPLICIT_IMAGE_BATCH_RATE_OVERRIDE": (
            "1" if scenario.use_explicit_image_batch_rate_override else "0"
        ),
        "TRANSFORM_OUTPUT": "1" if transform_output else "0",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "PYTHONPATH": python_path,
    }


@pytest.mark.parametrize("transform_output", (False, True), ids=("untransformed", "transformed"))
@pytest.mark.parametrize("scenario", _SCENARIOS, ids=tuple(case.name for case in _SCENARIOS))
def test_aretrieve_batch_costs_native_gemini_image_tokens(
    scenario: _BatchScenario, transform_output: bool, tmp_path: Path
) -> None:
    certificate, key = write_self_signed_cert(tmp_path, names=("storage.googleapis.com",))
    tls: Final = server_context(certificate, key)
    row: Final = _prediction_jsonl(scenario)

    with wire_server(_api_reply) as api:
        credentials: Final = service_account_json(_PROJECT, api.url)
        with wire_server(_gcs_reply(row), tls=tls) as gcs:
            gcs_port: Final = urlsplit(gcs.url).port
            assert gcs_port is not None
            with _redirect_https_host("storage.googleapis.com", 443, to_port=gcs_port) as proxy_url:
                environment: Final = _subprocess_environment(
                    api, certificate, credentials, proxy_url, scenario, transform_output
                )
                outcome: Final = subprocess.run(
                    [sys.executable, "-P", "-c", _SDK_SCRIPT],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )

                assert outcome.returncode == 0, (outcome.stdout, outcome.stderr)
                registry_events: Final = tuple(
                    _JSON_OBJECT.validate_json(line[len(_REGISTRY_PREFIX) :])
                    for line in outcome.stdout.splitlines()
                    if line.startswith(_REGISTRY_PREFIX)
                )
                callback_events: Final = tuple(
                    _JSON_OBJECT.validate_json(line[len(_CALLBACK_PREFIX) :])
                    for line in outcome.stdout.splitlines()
                    if line.startswith(_CALLBACK_PREFIX)
                )
                expected_registry_key: Final = f"vertex_ai/{_MODEL}"
                assert registry_events == (
                    {"key": expected_registry_key, "provider": "vertex_ai-language-models"},
                ), outcome.stdout
                assert len(callback_events) == 1, (outcome.stdout, outcome.stderr)
                event: Final = callback_events[0]
                assert event["response_cost"] == pytest.approx(
                    scenario.expected_prompt_cost + scenario.expected_completion_cost
                ), event
                assert event["batch_response_cost"] == pytest.approx(
                    scenario.expected_prompt_cost + scenario.expected_completion_cost
                ), event
                breakdown: Final = event["cost_breakdown"]
                assert isinstance(breakdown, dict), event
                assert breakdown["input_cost"] == pytest.approx(scenario.expected_prompt_cost), event
                assert breakdown["output_cost"] == pytest.approx(scenario.expected_completion_cost), event
                assert breakdown["total_cost"] == pytest.approx(
                    scenario.expected_prompt_cost + scenario.expected_completion_cost
                ), event

                gcs_requests: Final = gcs.drain()
                assert any(
                    request.method == "GET" and request.target.endswith("predictions.jsonl?alt=media")
                    for request in gcs_requests
                ), tuple((request.method, request.target) for request in gcs_requests)
                api_requests: Final = api.drain()
                assert any(
                    request.method == "GET"
                    and request.target.endswith(f"/batchPredictionJobs/{_BATCH_ID}")
                    and request.headers.get("authorization") == "Bearer scripted-token"
                    for request in api_requests
                ), tuple((request.method, request.target) for request in api_requests)
