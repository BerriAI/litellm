import asyncio
import contextlib
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final

import pytest

import litellm
from litellm.evals.main import cancel_run, create_run, get_run

FAKE_KEY: Final = "sk-fake"
FAKE_AWS: Final[Mapping[str, str]] = MappingProxyType(
    {"aws_access_key_id": "AKIAFAKE", "aws_secret_access_key": "fake-secret", "aws_region_name": "us-east-1"}
)
MESSAGES: Final = ({"role": "user", "content": "hi"},)
GATEWAY_PATH: Final = "/gw/"


@dataclass(frozen=True, slots=True)
class RecordingServer:
    port: int
    requests: SimpleQueue[str]

    def received_paths(self) -> tuple[str, ...]:
        return tuple(self.requests.get_nowait() for _ in range(self.requests.qsize()))


def _recording_handler(requests: SimpleQueue[str]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _answer_not_found(self) -> None:
            requests.put(self.path.split("?", 1)[0])
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "not found", "code": 404}}')

        def do_GET(self) -> None:
            self._answer_not_found()

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self._answer_not_found()

        def do_DELETE(self) -> None:
            self._answer_not_found()

    return Handler


@pytest.fixture
def server() -> Iterator[RecordingServer]:
    requests: Final[SimpleQueue[str]] = SimpleQueue()
    httpd: Final = ThreadingHTTPServer(("127.0.0.1", 0), _recording_handler(requests))
    thread: Final = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield RecordingServer(port=httpd.server_address[1], requests=requests)
    httpd.shutdown()
    httpd.server_close()


def _drain(result: object) -> None:
    if isinstance(result, litellm.CustomStreamWrapper):
        for _ in result:
            pass


def _completion(model: str, stream: bool = False, **kwargs: object) -> Callable[[str], object]:
    return lambda api_base: _drain(
        litellm.completion(model=model, messages=list(MESSAGES), api_base=api_base, stream=stream, timeout=5, **kwargs)
    )


def _embedding(model: str, **kwargs: object) -> Callable[[str], object]:
    return lambda api_base: litellm.embedding(model=model, input=["hi"], api_base=api_base, timeout=5, **kwargs)


ROUTES: Final[Mapping[str, Callable[[str], object]]] = MappingProxyType(
    {
        "anthropic-chat": _completion("anthropic/claude-opus-5-5", api_key=FAKE_KEY),
        "bedrock-converse": _completion("bedrock/converse/anthropic.claude-sonnet-5-5", **FAKE_AWS),
        "bedrock-converse-stream": _completion("bedrock/converse/anthropic.claude-sonnet-5-5", stream=True, **FAKE_AWS),
        "gemini-chat": _completion("gemini/gemini-pro-latest", api_key=FAKE_KEY),
        "gemini-chat-stream": _completion("gemini/gemini-pro-latest", stream=True, api_key=FAKE_KEY),
        "mistral-chat": _completion("mistral/mistral-large-latest", api_key=FAKE_KEY),
        "gigachat-chat": _completion("gigachat/GigaChat-2-Max", api_key=FAKE_KEY),
        "bedrock-titan-embedding": _embedding("bedrock/amazon.titan-embed-text-v2:0", **FAKE_AWS),
        "gemini-embedding": _embedding("gemini/gemini-embedding-2", api_key=FAKE_KEY),
        "gigachat-embedding": _embedding("gigachat/Embeddings-2", api_key=FAKE_KEY),
        "volcengine-embedding": _embedding("volcengine/doubao-embedding-text-240715", api_key=FAKE_KEY),
        "ollama-embedding": _embedding("ollama/nomic-embed-text"),
        "oobabooga-chat": _completion("oobabooga/local-model"),
        "oobabooga-embedding": _embedding("oobabooga/local-model"),
        "bedrock-rerank": lambda api_base: litellm.rerank(
            model="bedrock/arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0",
            query="hi",
            documents=["a"],
            api_base=api_base,
            **FAKE_AWS,
        ),
        "bedrock-image-generation": lambda api_base: litellm.image_generation(
            model="bedrock/amazon.nova-canvas-v1:0", prompt="a cat", api_base=api_base, timeout=5, **FAKE_AWS
        ),
        "gemini-interaction-get": lambda api_base: litellm.interactions.get(
            interaction_id="interaction-1", api_base=api_base, api_key=FAKE_KEY
        ),
        "gemini-interaction-delete": lambda api_base: litellm.interactions.delete(
            interaction_id="interaction-1", api_base=api_base, api_key=FAKE_KEY
        ),
        "gemini-interaction-cancel": lambda api_base: litellm.interactions.cancel(
            interaction_id="interaction-1", api_base=api_base, api_key=FAKE_KEY
        ),
        "openai-eval-run-create": lambda api_base: create_run(
            eval_id="eval-1", data_source={"type": "jsonl"}, api_base=api_base, api_key=FAKE_KEY
        ),
        "openai-eval-run-get": lambda api_base: get_run(
            eval_id="eval-1", run_id="run-1", api_base=api_base, api_key=FAKE_KEY
        ),
        "openai-eval-run-cancel": lambda api_base: cancel_run(
            eval_id="eval-1", run_id="run-1", api_base=api_base, api_key=FAKE_KEY
        ),
        "edenai-model-list": lambda api_base: litellm.get_valid_models(
            check_provider_endpoint=True, custom_llm_provider="edenai", api_key=FAKE_KEY, api_base=api_base
        ),
        "gemini-rag-ingest": lambda api_base: asyncio.run(
            litellm.aingest(
                ingest_options={
                    "vector_store": {"custom_llm_provider": "gemini", "api_base": api_base, "api_key": FAKE_KEY}
                },
                file_data=("notes.txt", b"hello", "text/plain"),
            )
        ),
    }
)


@pytest.mark.parametrize("route", ROUTES.values(), ids=ROUTES.keys())
def test_api_base_with_a_trailing_slash_reaches_the_gateway_path_with_single_slashes(
    server: RecordingServer, route: Callable[[str], object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIGACHAT_ACCESS_TOKEN", FAKE_KEY)
    with contextlib.suppress(Exception):
        route(f"http://127.0.0.1:{server.port}{GATEWAY_PATH}")

    paths: Final = server.received_paths()
    assert paths, "the route never reached the api_base"
    assert all(path.startswith(GATEWAY_PATH) and "//" not in path for path in paths), paths
