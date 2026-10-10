import os
import textwrap
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

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if self.headers.get("User-Agent", "").startswith("python-httpx"):
                self.send_response(418)
                self.end_headers()
                return
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
    litellm.rust(True)

    SERVED, REFUSED, OTHER = 0, 3, 4

    def outcome(asynchronous):
        try:
            response = asyncio.run(litellm.aocr(**arguments)) if asynchronous else litellm.ocr(**arguments)
        except ForkedAfterNativeRuntimeStarted:
            return REFUSED
        except Exception:
            return OTHER
        return SERVED if response.pages[0].markdown == "native" else OTHER

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
        "LITELLM_RUST": "1",
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
    RouteRule(Route.TOKENIZER, Rollout.RUST_OPT_IN),
    RouteRule(Route.TOKEN_COUNTER, Rollout.RUST_OPT_IN),
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
            lambda: asyncio.run(count_input_tokens({"prompt": "hello"}, b'{"prompt": "hello"}', ("counter-fork-fixture",))),
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
