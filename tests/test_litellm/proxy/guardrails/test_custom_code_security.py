import asyncio
import http.server
import threading
import time
from http.server import ThreadingHTTPServer

import pytest
from fastapi import HTTPException

import litellm
from litellm.exceptions import ModifyResponseException
from litellm.proxy.guardrails.guardrail_hooks.custom_code.custom_code_guardrail import (
    DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    CustomCodeCompilationError,
    CustomCodeExecutionError,
    CustomCodeGuardrail,
)
from litellm.proxy.guardrails.guardrail_registry import InMemoryGuardrailHandler
from litellm.types.guardrails import SupportedGuardrailIntegrations

# str.mro() + generator gi_code + code.replace(co_names=...) + __setattr__
# to swap a function's bytecode and read http_get's real builtins dict.
BYTECODE_REWRITE_PAYLOAD = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    "    obj = str.mro()[1]\n"
    "    def g(fn):\n"
    "        yield fn.placeholder\n"
    "    c = g(None).gi_code\n"
    '    gn = "_"+"_gl"+"ob"+"als"+"_"+"_"\n'
    '    cn = "_"+"_co"+"de_"+"_"\n'
    "    obj.__setattr__(g, cn, c.replace(co_names=(gn,)))\n"
    "    for v in g(http_get):\n"
    "        gd = v\n"
    "        break\n"
    '    bn = "_"+"_bu"+"ilt"+"ins"+"_"+"_"\n'
    '    imp = gd[bn]["_"+"_im"+"po"+"rt_"+"_"]\n'
    '    return {"rce": imp("os").popen("id").read()}\n'
)


def _compile(code: str) -> CustomCodeGuardrail:
    return CustomCodeGuardrail(custom_code=code, guardrail_name="t")


def test_bytecode_rewrite_rejected_at_compile():
    with pytest.raises(CustomCodeCompilationError):
        _compile(BYTECODE_REWRITE_PAYLOAD)


# Call the async http_get primitive without awaiting, then pull f_builtins off
# the returned coroutine's cr_frame. INSPECT_ATTRIBUTES covers cr_frame and
# f_builtins so this is rejected at compile time.
CR_FRAME_PAYLOAD = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    '    co = http_get("http://x")\n'
    "    b = co.cr_frame.f_builtins\n"
    "    co.close()\n"
    '    imp = b["_" + "_imp" + "ort_" + "_"]\n'
    '    return block(imp("os").popen("id").read())\n'
)


def test_cr_frame_rejected_at_compile():
    with pytest.raises(CustomCodeCompilationError):
        _compile(CR_FRAME_PAYLOAD)


# NFKC homoglyph: U+FF47 'ｇ' normalizes to 'g' at parse time, so "__ｇlobals__"
# arrives at the AST as "__globals__" and trips the underscore-prefix rule.
NFKC_PAYLOAD = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    '    b_key = "buil" + "tins"\n'
    '    i_key = "im" + "port"\n'
    "    b = allow.__\uff47lobals__[b_key]\n"
    "    import_fn = b[i_key]\n"
    '    o = import_fn("o" + "s")\n'
    '    return block(o.popen("id").read())\n'
)


def test_nfkc_homoglyph_rejected_at_compile():
    with pytest.raises(CustomCodeCompilationError):
        _compile(NFKC_PAYLOAD)


@pytest.mark.parametrize(
    "snippet",
    [
        # Literal dunder attribute access.
        "def apply_guardrail(i, r, t):\n    return str.__class__\n",
        "def apply_guardrail(i, r, t):\n    return ().__class__.__bases__[0].__subclasses__()\n",
        # gi_code — on the transformer's restricted-names list.
        "def apply_guardrail(i, r, t):\n    def g():\n        yield 1\n    return g().gi_code\n",
        # Import forms.
        "import os\ndef apply_guardrail(i, r, t):\n    return allow()\n",
        "from subprocess import call\ndef apply_guardrail(i, r, t):\n    return allow()\n",
        # __import__ is rejected as an underscore-prefixed name.
        'def apply_guardrail(i, r, t):\n    return __import__("os")\n',
    ],
)
def test_compile_time_rejections(snippet: str):
    with pytest.raises(CustomCodeCompilationError):
        _compile(snippet)


@pytest.mark.parametrize(
    "snippet",
    [
        # getattr is not in the sandbox builtins — NameError at call time.
        'def apply_guardrail(i, r, t):\n    return getattr(str, "_"+"_class_"+"_")\n',
        # setattr is guarded_setattr + full_write_guard — setting any attribute
        # on a user-defined object raises TypeError, whether the name is a
        # dunder or not.
        "def apply_guardrail(i, r, t):\n"
        "    def f():\n        pass\n"
        '    name = "_" + "_bad_" + "_"\n'
        "    setattr(f, name, None)\n"
        "    return allow()\n",
    ],
)
def test_runtime_rejections(snippet: str):
    guardrail = _compile(snippet)
    fn = guardrail._compiled_function
    assert fn is not None
    with pytest.raises((NameError, TypeError, AttributeError, SyntaxError)):
        fn({"texts": []}, {}, "request")


def test_documented_ssn_example_compiles_and_runs():
    code = (
        "def apply_guardrail(inputs, request_data, input_type):\n"
        '    for text in inputs["texts"]:\n'
        '        if regex_match(text, r"\\d{3}-\\d{2}-\\d{4}"):\n'
        '            return block("SSN detected")\n'
        "    return allow()\n"
    )
    guardrail = _compile(code)
    fn = guardrail._compiled_function
    assert fn is not None
    assert fn({"texts": ["hello"]}, {}, "request") == {"action": "allow"}
    blocked = fn({"texts": ["my ssn 123-45-6789"]}, {}, "request")
    assert blocked["action"] == "block"
    assert blocked["reason"] == "SSN detected"


@pytest.mark.asyncio
async def test_async_guardrail_compiles_and_runs():
    code = "async def apply_guardrail(inputs, request_data, input_type):\n    return allow()\n"
    guardrail = _compile(code)
    from litellm.types.utils import GenericGuardrailAPIInputs

    result = await guardrail.apply_guardrail(
        inputs=GenericGuardrailAPIInputs(texts=["test"]),
        request_data={},
        input_type="request",
    )
    assert result["texts"][0] == "test"


@pytest.mark.asyncio
async def test_custom_code_pre_call_block_uses_passthrough():
    code = 'def apply_guardrail(inputs, request_data, input_type):\n    return block("blocked by test")\n'
    guardrail = _compile(code)

    with pytest.raises(ModifyResponseException) as exc_info:
        await guardrail.apply_guardrail(
            inputs={"texts": ["test"]},
            request_data={"model": "test-model"},
            input_type="request",
        )

    assert exc_info.value.message == "blocked by test"
    assert exc_info.value.model == "test-model"
    assert exc_info.value.guardrail_name == "t"


@pytest.mark.asyncio
async def test_custom_code_post_call_block_raises_http_400():
    code = 'def apply_guardrail(inputs, request_data, input_type):\n    return block("blocked by test")\n'
    guardrail = _compile(code)

    with pytest.raises(HTTPException) as exc_info:
        await guardrail.apply_guardrail(
            inputs={"texts": ["test"]},
            request_data={"model": "test-model"},
            input_type="response",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "error": "blocked by test",
        "guardrail": "t",
        "detection_info": {},
    }


FLAG_CODE = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    '    return flag("audit hit", metadata={"category": "topic"})\n'
)


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type", ["request", "response"])
async def test_custom_code_flag_passes_content_through_and_records_flagged_entry(input_type):
    """LIT-6894: flag() must not raise, must return the content unchanged and must log
    exactly one guardrail_flagged entry (the decorator must not add a second "success")."""
    guardrail = CustomCodeGuardrail(custom_code=FLAG_CODE, guardrail_name="t", event_hook=["pre_call", "post_call"])
    request_data = {"model": "test-model", "litellm_metadata": {}}

    result = await guardrail.apply_guardrail(
        inputs={"texts": ["hello"]},
        request_data=request_data,
        input_type=input_type,
    )

    assert result == {"texts": ["hello"]}
    entries = request_data["litellm_metadata"]["standard_logging_guardrail_information"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["guardrail_status"] == "guardrail_flagged"
    assert entry["guardrail_name"] == "t"
    assert entry["guardrail_mode"] == ["pre_call", "post_call"]
    assert entry["guardrail_response"] == {
        "action": "flag",
        "reason": "audit hit",
        "input_type": input_type,
        "metadata": {"category": "topic"},
    }
    assert entry["duration"] is not None and entry["duration"] >= 0


@pytest.mark.asyncio
async def test_custom_code_flag_default_reason_and_empty_metadata():
    code = "def apply_guardrail(inputs, request_data, input_type):\n    return flag('just a note')\n"
    guardrail = _compile(code)
    request_data = {"model": "m", "litellm_metadata": {}}

    await guardrail.apply_guardrail(inputs={"texts": ["x"]}, request_data=request_data, input_type="request")

    entry = request_data["litellm_metadata"]["standard_logging_guardrail_information"][0]
    assert entry["guardrail_response"] == {
        "action": "flag",
        "reason": "just a note",
        "input_type": "request",
        "metadata": {},
    }


IDENTITY_ECHO_CODE = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    "    return flag('identity', metadata={\n"
    "        'ids': [request_data['user_id'], request_data['team_id'], request_data['end_user_id']],\n"
    "        'metadata_keys': sorted(request_data['metadata'].keys()),\n"
    "    })\n"
)
CALLER_IDENTITY = {
    "user_api_key_user_id": "someone@example.com",
    "user_api_key_team_id": "team-1",
    "user_api_key_end_user_id": "end-user-1",
    "user_api_key_alias": "guardrail-repro-key",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
async def test_custom_code_sandbox_sees_caller_identity_from_proxy_metadata_bucket(metadata_key):
    """LIT-6609: the proxy writes user_api_key_* into `metadata` (chat) or `litellm_metadata`
    (/v1/messages, responses, batches, files); the sandbox must resolve ids from either."""
    guardrail = _compile(IDENTITY_ECHO_CODE)
    request_data = {"model": "m", metadata_key: dict(CALLER_IDENTITY)}

    await guardrail.apply_guardrail(inputs={"texts": ["x"]}, request_data=request_data, input_type="request")

    entry = request_data[metadata_key]["standard_logging_guardrail_information"][0]
    assert entry["guardrail_response"]["metadata"] == {
        "ids": ["someone@example.com", "team-1", "end-user-1"],
        "metadata_keys": sorted(CALLER_IDENTITY),
    }


@pytest.mark.asyncio
async def test_custom_code_sandbox_merges_caller_metadata_with_litellm_metadata():
    """On litellm_metadata routes the caller's own `metadata` field must stay visible next to
    the proxy identity block, and the proxy block wins on key collisions."""
    guardrail = _compile(IDENTITY_ECHO_CODE)
    request_data = {
        "model": "m",
        "metadata": {"trace_id": "abc", "user_api_key_user_id": "forged"},
        "litellm_metadata": dict(CALLER_IDENTITY),
    }

    await guardrail.apply_guardrail(inputs={"texts": ["x"]}, request_data=request_data, input_type="request")

    entry = request_data["litellm_metadata"]["standard_logging_guardrail_information"][0]
    assert entry["guardrail_response"]["metadata"] == {
        "ids": ["someone@example.com", "team-1", "end-user-1"],
        "metadata_keys": sorted([*CALLER_IDENTITY, "trace_id"]),
    }


@pytest.mark.asyncio
async def test_custom_code_sandbox_ignores_top_level_identity_fields():
    """Only the proxy-owned metadata buckets carry identity; user_api_key_* keys at the top level
    of the request body are caller-controlled on ordinary routes and must never become ids."""
    code = (
        "def apply_guardrail(inputs, request_data, input_type):\n"
        "    ids = [request_data['user_id'], request_data['team_id'], request_data['end_user_id']]\n"
        "    return flag('identity', metadata={'ids': str(ids)})\n"
    )
    guardrail = _compile(code)
    request_data = {"model": "m", **CALLER_IDENTITY, "metadata": {"headers": {}}}

    await guardrail.apply_guardrail(inputs={"texts": ["x"]}, request_data=request_data, input_type="request")

    entry = request_data["metadata"]["standard_logging_guardrail_information"][0]
    assert entry["guardrail_response"]["metadata"]["ids"] == "[None, None, None]"


@pytest.mark.asyncio
async def test_custom_code_allow_still_records_success_not_flagged():
    code = "def apply_guardrail(inputs, request_data, input_type):\n    return allow()\n"
    guardrail = _compile(code)
    request_data = {"model": "m", "litellm_metadata": {}}

    await guardrail.apply_guardrail(inputs={"texts": ["x"]}, request_data=request_data, input_type="request")

    entries = request_data["litellm_metadata"]["standard_logging_guardrail_information"]
    assert [e["guardrail_status"] for e in entries] == ["success"]


def test_typical_sync_guardrail_still_works():
    code = "def apply_guardrail(inputs, request_data, input_type):\n    return allow()\n"
    guardrail = _compile(code)
    assert guardrail._compiled_function is not None


def test_augmented_assignment_works():
    # The transformer rewrites `n += 1` into `n = _inplacevar_("+=", n, 1)`,
    # so the sandbox must bind `_inplacevar_`.
    code = (
        "def apply_guardrail(inputs, request_data, input_type):\n"
        "    count = 0\n"
        '    for _ in inputs["texts"]:\n'
        "        count += 1\n"
        '    return {"action": "allow", "n": count}\n'
    )
    guardrail = _compile(code)
    fn = guardrail._compiled_function
    assert fn is not None
    assert fn({"texts": ["a", "b", "c"]}, {}, "request") == {
        "action": "allow",
        "n": 3,
    }


def test_missing_apply_guardrail_raises():
    with pytest.raises(CustomCodeCompilationError, match="apply_guardrail"):
        _compile("x = 1\n")


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request: object, client_address: object) -> None:
        return


def _guardrail_worker_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("guardrail-code:")]


class _LocalServer:
    """Loopback HTTP server that records every request it receives."""

    def __init__(self) -> None:
        self.hits: list[tuple[str, str]] = []
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                server.hits.append(("GET", self.path))
                if self.path.startswith("/redirect-to/"):
                    self._redirect()
                    return
                if self.path == "/slow":
                    time.sleep(2)
                self._reply(b"marker")

            def do_POST(self) -> None:
                server.hits.append(("POST", self.path))
                if self.path.startswith("/redirect-to/"):
                    self._redirect()
                    return
                self._reply(b"posted")

            def _redirect(self) -> None:
                target_port = self.path.rsplit("/", 1)[1]
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{target_port}/marker")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _reply(self, body: bytes) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                return

        self.httpd = _QuietServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def local_server():
    server = _LocalServer()
    yield server
    server.close()


@pytest.fixture
def second_server():
    server = _LocalServer()
    yield server
    server.close()


@pytest.fixture(autouse=True)
def _fresh_http_client_and_url_policy(monkeypatch):
    litellm.in_memory_llm_clients_cache.flush_cache()
    monkeypatch.setattr(litellm, "user_url_validation", True)
    monkeypatch.setattr(litellm, "user_url_allowed_hosts", [])


def _reporting_guardrail(call: str) -> CustomCodeGuardrail:
    code = (
        "async def apply_guardrail(inputs, request_data, input_type):\n"
        f"    r = await {call}\n"
        '    return block("status=" + str(r["status_code"]) + " body=" + str(r["body"])'
        ' + " error=" + str(r["error"]))\n'
    )
    return _compile(code)


async def _block_reason(guardrail: CustomCodeGuardrail) -> str:
    with pytest.raises(ModifyResponseException) as exc_info:
        await guardrail.apply_guardrail(inputs={"texts": ["x"]}, request_data={"model": "m"}, input_type="request")
    return exc_info.value.message


@pytest.mark.asyncio
async def test_http_get_refuses_loopback_by_default(local_server):
    guardrail = _reporting_guardrail(f'http_get("http://127.0.0.1:{local_server.port}/marker")')

    reason = await _block_reason(guardrail)

    assert "status=0" in reason
    assert "error=Blocked URL" in reason
    assert "user_url_allowed_hosts" in reason
    assert local_server.hits == []


@pytest.mark.asyncio
async def test_http_post_refuses_loopback_by_default(local_server):
    guardrail = _reporting_guardrail(f'http_post("http://127.0.0.1:{local_server.port}/hook", body={{"a": 1}})')

    reason = await _block_reason(guardrail)

    assert "error=Blocked URL" in reason
    assert local_server.hits == []


@pytest.mark.asyncio
async def test_http_get_reaches_an_allowlisted_host(local_server, monkeypatch):
    monkeypatch.setattr(litellm, "user_url_allowed_hosts", [f"127.0.0.1:{local_server.port}"])
    guardrail = _reporting_guardrail(f'http_get("http://127.0.0.1:{local_server.port}/marker")')

    reason = await _block_reason(guardrail)

    assert "status=200 body=marker error=None" in reason
    assert local_server.hits == [("GET", "/marker")]


@pytest.mark.asyncio
async def test_http_get_refuses_a_redirect_into_a_blocked_host(local_server, second_server, monkeypatch):
    monkeypatch.setattr(litellm, "user_url_allowed_hosts", [f"127.0.0.1:{local_server.port}"])
    guardrail = _reporting_guardrail(
        f'http_get("http://127.0.0.1:{local_server.port}/redirect-to/{second_server.port}")'
    )

    reason = await _block_reason(guardrail)

    assert "error=Blocked URL" in reason
    assert local_server.hits == [("GET", f"/redirect-to/{second_server.port}")]
    assert second_server.hits == []


@pytest.mark.asyncio
async def test_http_post_does_not_follow_redirects(local_server, second_server, monkeypatch):
    monkeypatch.setattr(litellm, "user_url_allowed_hosts", [f"127.0.0.1:{local_server.port}"])
    guardrail = _reporting_guardrail(
        f'http_post("http://127.0.0.1:{local_server.port}/redirect-to/{second_server.port}", body={{"a": 1}})'
    )

    reason = await _block_reason(guardrail)

    assert "status=302" in reason
    assert second_server.hits == []


@pytest.mark.asyncio
async def test_http_get_is_unvalidated_when_url_validation_is_disabled(local_server, monkeypatch):
    monkeypatch.setattr(litellm, "user_url_validation", False)
    guardrail = _reporting_guardrail(f'http_get("http://127.0.0.1:{local_server.port}/marker")')

    reason = await _block_reason(guardrail)

    assert "status=200 body=marker" in reason
    assert local_server.hits == [("GET", "/marker")]


BUSY_LOOP_GUARDRAIL = (
    "def apply_guardrail(inputs, request_data, input_type):\n    n = 0\n    while True:\n        n += 1\n"
)

SWALLOWING_BUSY_LOOP_GUARDRAIL = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    "    n = 0\n"
    "    while True:\n"
    "        try:\n"
    "            n += 1\n"
    "        except Exception:\n"
    "            n = 0\n"
)


async def _expect_execution_timeout(guardrail: CustomCodeGuardrail) -> float:
    started = time.monotonic()
    with pytest.raises(CustomCodeExecutionError, match=r"exceeded its 0\.3s execution timeout"):
        await guardrail.apply_guardrail(inputs={"texts": ["x"]}, request_data={"model": "m"}, input_type="request")
    return time.monotonic() - started


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [BUSY_LOOP_GUARDRAIL, SWALLOWING_BUSY_LOOP_GUARDRAIL])
async def test_sync_busy_loop_is_stopped_at_the_execution_timeout(code):
    guardrail = CustomCodeGuardrail(custom_code=code, guardrail_name="busy", execution_timeout=0.3)

    elapsed = await _expect_execution_timeout(guardrail)

    assert elapsed < 2.0
    await asyncio.sleep(0.2)
    assert _guardrail_worker_threads() == []


@pytest.mark.asyncio
async def test_sync_busy_loop_does_not_stall_the_event_loop():
    guardrail = CustomCodeGuardrail(custom_code=BUSY_LOOP_GUARDRAIL, guardrail_name="busy", execution_timeout=0.3)
    ticks = 0

    async def tick_forever() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    ticker = asyncio.create_task(tick_forever())
    try:
        await _expect_execution_timeout(guardrail)
    finally:
        ticker.cancel()

    assert ticks >= 5


@pytest.mark.asyncio
async def test_async_guardrail_is_stopped_at_the_execution_timeout(local_server, monkeypatch):
    monkeypatch.setattr(litellm, "user_url_allowed_hosts", [f"127.0.0.1:{local_server.port}"])
    code = (
        "async def apply_guardrail(inputs, request_data, input_type):\n"
        f'    await http_get("http://127.0.0.1:{local_server.port}/slow")\n'
        "    return allow()\n"
    )
    guardrail = CustomCodeGuardrail(custom_code=code, guardrail_name="busy", execution_timeout=0.3)

    elapsed = await _expect_execution_timeout(guardrail)

    assert elapsed < 1.5


@pytest.mark.asyncio
async def test_async_guardrail_that_swallows_cancellation_is_stopped_at_the_execution_timeout(
    local_server, monkeypatch
):
    monkeypatch.setattr(litellm, "user_url_allowed_hosts", [f"127.0.0.1:{local_server.port}"])
    code = (
        "async def apply_guardrail(inputs, request_data, input_type):\n"
        "    attempts = 0\n"
        "    while attempts < 3:\n"
        "        try:\n"
        f'            await http_get("http://127.0.0.1:{local_server.port}/slow")\n'
        "        except BaseException:\n"
        "            attempts += 1\n"
        "    return allow()\n"
    )
    guardrail = CustomCodeGuardrail(custom_code=code, guardrail_name="stubborn", execution_timeout=0.3)

    elapsed = await _expect_execution_timeout(guardrail)

    assert elapsed < 1.5


LOOP_SHAPES_GUARDRAIL = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    "    n = 0\n"
    "    while n < 3:\n"
    "        n += 1\n"
    "    else:\n"
    "        n += 10\n"
    "    pairs = [(k, v) for k, v in request_data['metadata'].items()]\n"
    "    for k, v in pairs:\n"
    "        n += v\n"
    "    for i, (k, v) in zip(range(len(pairs)), pairs):\n"
    "        n += i\n"
    "    keys = sorted(k for k, v in pairs)\n"
    "    return block(reason=str(n) + ' ' + ' '.join(keys))\n"
)


@pytest.mark.asyncio
async def test_budget_checks_keep_every_loop_shape_working():
    guardrail = CustomCodeGuardrail(custom_code=LOOP_SHAPES_GUARDRAIL, guardrail_name="loops")

    with pytest.raises(ModifyResponseException) as exc_info:
        await guardrail.apply_guardrail(
            inputs={"texts": ["x"]}, request_data={"model": "m", "metadata": {"b": 2, "a": 5}}, input_type="request"
        )

    assert exc_info.value.message == "21 a b"


@pytest.mark.asyncio
@pytest.mark.timeout(10)
@pytest.mark.parametrize(
    "code",
    [
        "async def apply_guardrail(inputs, request_data, input_type):\n    while True:\n        pass\n",
        (
            "async def apply_guardrail(inputs, request_data, input_type):\n"
            "    for a in range(500):\n"
            "        for b in range(500):\n"
            "            for c in range(500):\n"
            "                pass\n"
            "    return allow()\n"
        ),
        (
            "async def apply_guardrail(inputs, request_data, input_type):\n"
            "    try:\n"
            "        while True:\n"
            "            pass\n"
            "    except BaseException:\n"
            "        pass\n"
            "    return allow()\n"
        ),
    ],
)
async def test_async_loop_that_never_yields_is_stopped_at_the_execution_timeout(code):
    guardrail = CustomCodeGuardrail(custom_code=code, guardrail_name="spin", execution_timeout=0.3)

    elapsed = await _expect_execution_timeout(guardrail)

    assert elapsed < 1.5
    assert await asyncio.sleep(0, result="loop still running") == "loop still running"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    [
        "def apply_guardrail(inputs, request_data, input_type):\n    raise SystemExit('bye')\n",
        "async def apply_guardrail(inputs, request_data, input_type):\n    raise SystemExit('bye')\n",
    ],
)
async def test_system_exit_from_guardrail_code_is_an_execution_error_not_a_timeout(code):
    guardrail = CustomCodeGuardrail(custom_code=code, guardrail_name="exit", execution_timeout=5.0)
    started = time.monotonic()

    with pytest.raises(CustomCodeExecutionError, match="execution failed: SystemExit: bye"):
        await guardrail.apply_guardrail(inputs={"texts": ["x"]}, request_data={"model": "m"}, input_type="request")

    assert time.monotonic() - started < 1.0


def test_module_level_busy_loop_fails_compilation_at_the_execution_timeout():
    code = "n = 0\nwhile True:\n    n += 1\n" + BUSY_LOOP_GUARDRAIL
    started = time.monotonic()

    with pytest.raises(CustomCodeCompilationError, match=r"exceeded the 0\.3s execution timeout"):
        CustomCodeGuardrail(custom_code=code, guardrail_name="busy", execution_timeout=0.3)

    assert time.monotonic() - started < 2.0


@pytest.mark.parametrize("execution_timeout", [0, -1.0])
def test_execution_timeout_must_be_positive(execution_timeout):
    with pytest.raises(ValueError, match="execution_timeout must be positive"):
        CustomCodeGuardrail(
            custom_code="def apply_guardrail(i, r, t):\n    return allow()\n", execution_timeout=execution_timeout
        )


def _initialize_from_config(guardrail_name: str, litellm_params: dict[str, object]) -> CustomCodeGuardrail:
    InMemoryGuardrailHandler().initialize_guardrail(
        guardrail={
            "guardrail_name": guardrail_name,
            "litellm_params": {
                "guardrail": SupportedGuardrailIntegrations.CUSTOM_CODE.value,
                "mode": "pre_call",
                "custom_code": "def apply_guardrail(inputs, request_data, input_type):\n    return allow()\n",
                **litellm_params,
            },
        }
    )
    initialized = [
        callback
        for callback in litellm.callbacks
        if isinstance(callback, CustomCodeGuardrail) and callback.guardrail_name == guardrail_name
    ]
    assert initialized, f"{guardrail_name} was not registered as a callback"
    return initialized[-1]


def test_config_timeout_reaches_the_guardrail():
    assert _initialize_from_config("custom-code-timeout", {"timeout": 0.2}).execution_timeout == 0.2


def test_config_without_timeout_uses_the_default():
    assert (
        _initialize_from_config("custom-code-default-timeout", {}).execution_timeout
        == DEFAULT_EXECUTION_TIMEOUT_SECONDS
    )
