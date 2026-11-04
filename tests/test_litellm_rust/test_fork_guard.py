import json
import os
import textwrap
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Final

import pytest

from tests.test_litellm_rust.support.child_interpreter import run_child_interpreter

pytestmark = pytest.mark.requires_rust_extension

_NATIVE_CONTRACT = textwrap.dedent(
    """
    import os
    from litellm.rust_bridge import _native
    from litellm.rust_bridge.fork_guard import reserve_process_for_forking

    def native_route_error():
        import asyncio

        async def call():
            await _native.ResponsesWebSocketConnection.connect("ws://127.0.0.1:1", {}, 0.2)

        try:
            asyncio.run(call())
        except Exception as error:
            return f"{type(error).__name__}: {error}"
        return ""

    assert _native.process_state_started() is False
    reserve_process_for_forking("the test master")
    assert native_route_error().startswith("ProcessReservedForForking: ")
    assert _native.process_state_started() is False

    pid = os.fork()
    if pid == 0:
        error = native_route_error()
        started = _native.process_state_started()
        os._exit(0 if started and "reserved" not in error and "forked" not in error else 1)
    assert os.waitpid(pid, 0)[1] == 0

    pid = os.fork()
    if pid == 0:
        native_route_error()
        grandchild = os.fork()
        if grandchild == 0:
            os._exit(0 if native_route_error().startswith("ForkedAfterNativeRuntimeStarted: ") else 1)
        os._exit(os.waitpid(grandchild, 0)[1])
    assert os.waitpid(pid, 0)[1] == 0
    """
)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork only")
def test_compiled_extension_forbids_the_master_and_frees_its_workers() -> None:
    env = {**os.environ, "OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES"}

    result = run_child_interpreter(_NATIVE_CONTRACT, env=env, timeout=60)

    assert result.returncode == 0, result.stderr


_SDK_CONTRACT = textwrap.dedent(
    """
    import asyncio, json, multiprocessing, os, threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import litellm
    from litellm.rust_bridge.fork_guard import ForkedAfterNativeRuntimeStarted
    from litellm.litellm_core_utils.add_retry_fallback_headers import get_hidden_params_dict

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({
                "pages": [{"index": 0, "markdown": "native", "images": [], "dimensions": None}],
                "model": "mistral-ocr-latest",
                "usage_info": {"pages_processed": 1, "doc_size_bytes": 3},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    arguments = {
        "model": "mistral/mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
        "api_key": "test-key",
        "api_base": f"http://127.0.0.1:{server.server_port}",
        "num_retries": 0,
    }
    SERVED, REFUSED, OTHER = 0, 3, 4

    def outcome(asynchronous):
        try:
            response = asyncio.run(litellm.aocr(**arguments)) if asynchronous else litellm.ocr(**arguments)
        except ForkedAfterNativeRuntimeStarted:
            return REFUSED
        except Exception:
            return OTHER
        return SERVED if (
            response.pages[0].markdown == "native"
            and get_hidden_params_dict(response).get("additional_headers", {}).get("x-litellm-rust") == "true"
        ) else OTHER

    def forked(asynchronous):
        pid = os.fork()
        if pid == 0:
            os._exit(outcome(asynchronous))
        return os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])

    def pooled(asynchronous):
        with multiprocessing.get_context("fork").Pool(1) as pool:
            return pool.apply(outcome, (asynchronous,))

    # Forking before the first native call is fine: the child starts its own runtime.
    assert [forked(False), forked(True)] == [SERVED, SERVED]

    assert outcome(False) == SERVED
    # After it, a forked child is told so instead of hanging on threads that do not exist.
    assert [forked(False), forked(True)] == [REFUSED, REFUSED]
    assert [pooled(False), pooled(True)] == [REFUSED, REFUSED]
    # The parent is not poisoned by any of it.
    assert [outcome(False), outcome(True)] == [SERVED, SERVED]
    """
)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork only")
def test_sdk_call_in_a_child_forked_after_native_use_raises_instead_of_hanging() -> None:
    env = {
        **os.environ,
        "OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }

    result = run_child_interpreter(_SDK_CONTRACT, env=env, timeout=120)

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork only")
@pytest.mark.parametrize("warm_fast_counter", (False, True))
def test_tokenizers_share_the_native_process_guard(warm_fast_counter: bool) -> None:
    script: Final = """
import asyncio
import os
import litellm
from litellm.proxy.spend_tracking.input_tokens import count_input_tokens
from litellm.rust_bridge import _native, catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from litellm.litellm_core_utils.tokenizer import HuggingFaceTokenizer
from litellm.utils import claude_json_str

catalog.RULES = (
    RouteRule(Route.TOKENIZER, Rollout.RUST_REQUIRED),
    RouteRule(Route.TOKEN_COUNTER, Rollout.RUST_REQUIRED),
    *catalog.RULES,
)
litellm.anthropic_models = {*litellm.anthropic_models, "tokenizer-fork-fixture"}
_native.reserve_process_for_forking()
for create in (
    lambda: _native.Tokenizer.from_tiktoken("cl100k_base"),
    lambda: _native.Tokenizer.from_json(claude_json_str),
    lambda: litellm.token_counter(model="tokenizer-fork-fixture", text="hello"),
):
    try:
        create()
    except _native.ProcessReservedForForking:
        pass
    else:
        raise AssertionError("reserved parent ran a native tokenizer")
assert not _native.process_state_started()

pid = os.fork()
if pid == 0:
    tokenizer = HuggingFaceTokenizer.from_str(claude_json_str)
    encoding = _native.Tokenizer.from_tiktoken("cl100k_base")
    if os.environ["WARM_FAST_COUNTER"] == "True":
        _native.TokenCounter.from_tokenizer(encoding, fast=True)
    expected = [item.ids for item in tokenizer.encode_batch(["hello", "world"])]
    assert _native.process_state_started()
    grandchild = os.fork()
    if grandchild == 0:
        for call in (
            lambda: tokenizer.encode_batch(["hello", "world"]),
            lambda: tokenizer.encode("hello"),
            lambda: encoding.count("hello"),
            lambda: encoding.count("hello", fast=True),
            lambda: _native.TokenCounter.from_tokenizer(encoding),
            lambda: _native.TokenCounter.from_tokenizer(encoding, fast=True),
            lambda: _native.Tokenizer.from_tiktoken("cl100k_base"),
            lambda: asyncio.run(
                count_input_tokens({"prompt": "hello"}, b'{"prompt": "hello"}', ("counter-fork-fixture",))
            ),
        ):
            try:
                call()
            except _native.ForkedAfterNativeRuntimeStarted:
                pass
            else:
                os._exit(1)
        os._exit(0)
    assert os.waitpid(grandchild, 0)[1] == 0
    assert [item.ids for item in tokenizer.encode_batch(["hello", "world"])] == expected
    os._exit(0)
assert os.waitpid(pid, 0)[1] == 0
"""
    result: Final = run_child_interpreter(
        script,
        env={
            **os.environ,
            "OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "WARM_FAST_COUNTER": str(warm_fast_counter),
        },
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


_IMPORT_REFRESH_CONTRACT = textwrap.dedent(
    """
    import asyncio, os, time
    import litellm
    from litellm.rust_bridge import _native

    def native_route_error():
        async def call():
            await _native.ResponsesWebSocketConnection.connect("ws://127.0.0.1:1", {}, 0.2)

        try:
            asyncio.run(call())
        except Exception as error:
            return f"{type(error).__name__}: {error}"
        return ""

    def adopted(deadline):
        while "fork-remote-only-model" not in litellm.model_cost and time.monotonic() < deadline:
            time.sleep(0.05)
        return "fork-remote-only-model" in litellm.model_cost

    assert _native.process_state_started() is False, "import litellm started the native runtime"
    assert "fork-remote-only-model" not in litellm.model_cost
    time.sleep(float(os.environ["FORK_AFTER_SECONDS"]))
    pid = os.fork()
    if pid == 0:
        child_adopted = adopted(time.monotonic() + 30)
        error = native_route_error()
        native = _native.process_state_started() and "reserved" not in error and "forked" not in error
        os._exit(0 if child_adopted and native else 1)
    assert os.waitpid(pid, 0)[1] == 0, "the forked child did not end up with the remote map or could not run natively"
    """
)


class _ScriptedCostMap(BaseHTTPRequestHandler):
    """Answers request n with ``statuses[n]``, the last one repeating, each after ``delay`` seconds"""

    body: bytes = b""
    statuses: tuple[int, ...] = (200,)
    delay: float = 0.0
    requests: list[str] = []  # mutable-ok: the server thread records each fetch

    def do_GET(self) -> None:
        handler: Final = type(self)
        handler.requests.append(self.path)
        time.sleep(handler.delay)
        status: Final = handler.statuses[min(len(handler.requests), len(handler.statuses)) - 1]
        body: Final = handler.body if status == 200 else b""
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def cost_map_server() -> Iterator[Callable[[tuple[int, ...], float], tuple[str, type[_ScriptedCostMap]]]]:
    remote: Final = json.loads(files("litellm").joinpath("model_prices_and_context_window_backup.json").read_bytes())
    remote["fork-remote-only-model"] = {"litellm_provider": "openai", "mode": "chat"}
    servers: Final[list[ThreadingHTTPServer]] = []  # mutable-ok: shut down after the test

    def serve(statuses: tuple[int, ...], delay: float) -> tuple[str, type[_ScriptedCostMap]]:
        attributes: Final = {"body": json.dumps(remote).encode(), "statuses": statuses, "delay": delay, "requests": []}
        handler: Final = type("Handler", (_ScriptedCostMap,), attributes)
        server: Final = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}/model_prices.json", handler

    yield serve
    for server in servers:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork only")
@pytest.mark.parametrize(
    ("statuses", "delay", "fork_after", "fetches"),
    [
        pytest.param((200,), 2.0, 0.0, 1, id="a_fork_mid_refresh_waits_and_the_child_inherits_the_map"),
        pytest.param((404, 200), 0.0, 1.0, 2, id="a_child_of_a_parent_whose_refresh_failed_fetches_its_own"),
    ],
)
def test_import_starts_no_runtime_and_a_forked_child_ends_up_with_the_remote_map(
    cost_map_server: Callable[[tuple[int, ...], float], tuple[str, type[_ScriptedCostMap]]],
    statuses: tuple[int, ...],
    delay: float,
    fork_after: float,
    fetches: int,
) -> None:
    url, handler = cost_map_server(statuses, delay)
    env: Final = {
        **{name: value for name, value in os.environ.items() if name != "LITELLM_LOCAL_MODEL_COST_MAP"},
        "LITELLM_MODEL_COST_MAP_URL": url,
        "FORK_AFTER_SECONDS": str(fork_after),
        "OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES",
    }

    result: Final = run_child_interpreter(_IMPORT_REFRESH_CONTRACT, env=env, timeout=120)

    assert result.returncode == 0, result.stderr
    assert handler.requests == ["/model_prices.json"] * fetches
