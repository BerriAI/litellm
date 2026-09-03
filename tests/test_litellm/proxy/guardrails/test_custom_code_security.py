import inspect
import socket

import httpx
import pytest
from fastapi import HTTPException

import litellm
from litellm.exceptions import ModifyResponseException
from litellm.proxy.guardrails.guardrail_hooks.custom_code.custom_code_guardrail import (
    CustomCodeCompilationError,
    CustomCodeGuardrail,
)

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


# --- SSRF protection on the HTTP primitives ---------------------------------
#
# The primitives run inside the sandbox but talk to the network with the
# proxy process's privileges. http_request/http_get/http_post must therefore
# refuse loopback, private, and cloud-metadata targets and must not follow
# redirects (a 302 to an internal address would otherwise bypass the check).


class _FakeAsyncClient:
    """Mimics the REAL AsyncHTTPHandler method signatures.

    If the primitives pass a kwarg the production handler does not accept,
    these fakes raise TypeError exactly like production would.
    """

    def __init__(self, response=None):
        self.response = response
        self.calls = []

    async def _record(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    async def get(
        self,
        url,
        params=None,
        headers=None,
        follow_redirects=None,
        timeout=None,
    ):
        return await self._record(
            url=url,
            params=params,
            headers=headers,
            follow_redirects=follow_redirects,
            timeout=timeout,
        )

    async def post(
        self,
        url,
        data=None,
        json=None,
        params=None,
        headers=None,
        timeout=None,
        stream=False,
        logging_obj=None,
        files=None,
        content=None,
        follow_redirects=None,
    ):
        return await self._record(
            url=url,
            data=data,
            json=json,
            params=params,
            headers=headers,
            timeout=timeout,
            follow_redirects=follow_redirects,
        )

    async def put(
        self,
        url,
        data=None,
        json=None,
        params=None,
        headers=None,
        timeout=None,
        stream=False,
        content=None,
        follow_redirects=None,
    ):
        return await self._record(
            url=url,
            data=data,
            json=json,
            params=params,
            headers=headers,
            timeout=timeout,
            follow_redirects=follow_redirects,
        )

    async def patch(
        self,
        url,
        data=None,
        json=None,
        params=None,
        headers=None,
        timeout=None,
        stream=False,
        content=None,
        follow_redirects=None,
    ):
        return await self._record(
            url=url,
            data=data,
            json=json,
            params=params,
            headers=headers,
            timeout=timeout,
            follow_redirects=follow_redirects,
        )

    async def delete(
        self,
        url,
        data=None,
        json=None,
        params=None,
        headers=None,
        timeout=None,
        stream=False,
        content=None,
        follow_redirects=None,
    ):
        return await self._record(
            url=url,
            data=data,
            json=json,
            params=params,
            headers=headers,
            timeout=timeout,
            follow_redirects=follow_redirects,
        )


@pytest.mark.parametrize("method", ["get", "post", "put", "patch", "delete"])
def test_async_http_handler_accepts_follow_redirects(method):
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    params = inspect.signature(getattr(AsyncHTTPHandler, method)).parameters
    assert "follow_redirects" in params, (
        f"AsyncHTTPHandler.{method} must accept follow_redirects or the "
        "guardrail HTTP primitives cannot disable redirects"
    )
    assert params["follow_redirects"].default is None


def _ok_response(status_code=200):
    return httpx.Response(status_code, request=httpx.Request("GET", "http://ok"))


@pytest.fixture
def _public_dns(monkeypatch):
    """Point every hostname at a globally routable IP so tests stay hermetic."""

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr("litellm.litellm_core_utils.url_utils.socket.getaddrinfo", fake_getaddrinfo)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8971/",
        "http://localhost/admin",
        "http://10.1.2.3/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        "http://[::1]/",
        # Malformed port: is_valid_url accepts it (scheme + netloc), but
        # urlparse raises ValueError when validate_url reads parsed.port.
        # Must surface as a structured error, not an escaped exception.
        "http://example.com:99999/",
        "http://example.com:notaport/",
    ],
)
async def test_http_primitives_block_internal_targets(url, monkeypatch):
    from litellm.proxy.guardrails.guardrail_hooks.custom_code import primitives

    client = _FakeAsyncClient()
    monkeypatch.setattr(primitives, "get_async_httpx_client", lambda **kwargs: client)

    result = await primitives.http_request(url)

    assert result["success"] is False
    assert result["status_code"] == 0
    assert "Blocked" in result["error"]
    assert client.calls == []


@pytest.mark.asyncio
async def test_http_primitives_allow_public_url(monkeypatch, _public_dns):
    from litellm.proxy.guardrails.guardrail_hooks.custom_code import primitives

    client = _FakeAsyncClient(_ok_response())
    monkeypatch.setattr(primitives, "get_async_httpx_client", lambda **kwargs: client)

    result = await primitives.http_get("http://moderation.example.com/v1/check", headers={"Authorization": "Bearer t"})

    assert result["success"] is True
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["url"] == "http://93.184.216.34/v1/check"
    assert call["headers"]["Host"] == "moderation.example.com"
    assert call["headers"]["Authorization"] == "Bearer t"
    assert call["follow_redirects"] is False


@pytest.mark.asyncio
async def test_http_primitives_honor_allowlisted_internal_host(monkeypatch, _public_dns):
    from litellm.proxy.guardrails.guardrail_hooks.custom_code import primitives

    monkeypatch.setattr(litellm, "user_url_allowed_hosts", ["internal-moderation.corp"], raising=False)
    client = _FakeAsyncClient(_ok_response())
    monkeypatch.setattr(primitives, "get_async_httpx_client", lambda **kwargs: client)

    result = await primitives.http_get("http://internal-moderation.corp/check")

    assert result["success"] is True
    assert client.calls[0]["follow_redirects"] is False


@pytest.mark.asyncio
async def test_http_primitives_validation_can_be_disabled(monkeypatch, _public_dns):
    from litellm.proxy.guardrails.guardrail_hooks.custom_code import primitives

    monkeypatch.setattr(litellm, "user_url_validation", False, raising=False)
    client = _FakeAsyncClient(_ok_response())
    monkeypatch.setattr(primitives, "get_async_httpx_client", lambda **kwargs: client)

    result = await primitives.http_get("http://10.1.2.3/internal")

    assert result["success"] is True
    assert client.calls[0]["url"] == "http://10.1.2.3/internal"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
async def test_http_primitives_do_not_follow_redirects(method, monkeypatch, _public_dns):
    from litellm.proxy.guardrails.guardrail_hooks.custom_code import primitives

    client = _FakeAsyncClient(_ok_response(status_code=302))
    monkeypatch.setattr(primitives, "get_async_httpx_client", lambda **kwargs: client)

    result = await primitives.http_request("http://moderation.example.com/v1/check", method=method, body={"text": "hi"})

    assert len(client.calls) == 1
    assert client.calls[0]["follow_redirects"] is False
    assert result["status_code"] == 302
    assert result["success"] is False
