import json
from collections.abc import Callable, Mapping

import httpx
import pytest
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.llms.base_llm.sandbox.transformation import SANDBOX_MAX_OUTPUT_BYTES
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.kosa8.sandbox.transformation import Kosa8SandboxConfig


_BODY = TypeAdapter(dict[str, object])


class FakeKosa8:
    """A kosa8 Cloud API in memory: sandboxes, a Python 'kernel' that counts cells, and every request seen."""

    def __init__(self, *, code_response: Callable[[dict[str, object]], httpx.Response] | None = None) -> None:
        self.requests: list[tuple[str, str, dict[str, str], object]] = []
        self.live: set[str] = set()
        self.cells = 0
        self.code_response = code_response

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = _BODY.validate_python(json.loads(request.content)) if request.content else {}
        path = request.url.raw_path.decode()
        self.requests.append((request.method, path, dict(request.headers), body))
        if request.headers.get("authorization") != "Bearer k8c_live_test":
            return httpx.Response(401, json={"error": "unauthorized", "message": "bad key"})
        parts = path.strip("/").split("/")
        if request.method == "POST" and parts == ["v1", "sandboxes"]:
            sandbox_id = f"sbx_{len(self.live) + 1}"
            self.live.add(sandbox_id)
            return httpx.Response(
                201, json={"sandbox": {"id": sandbox_id, "state": "running", "template": body["template"]}}
            )
        if len(parts) >= 3 and parts[:2] == ["v1", "sandboxes"] and parts[2] not in self.live:
            return httpx.Response(404, json={"error": "not_found", "message": f"no such sandbox: {parts[2]}"})
        if request.method == "POST" and parts[3:] == ["code"]:
            if self.code_response is not None:
                return self.code_response(body)
            self.cells += 1
            return httpx.Response(
                200,
                json={
                    "context": "default",
                    "execution_count": self.cells,
                    "results": [{"type": "text", "data": "42"}],
                    "stdout": "42\n",
                    "stderr": "warn\n",
                },
            )
        if request.method == "DELETE" and len(parts) == 3:
            self.live.discard(parts[2])
            return httpx.Response(200, json={"ok": True, "id": parts[2]})
        return httpx.Response(404, json={"error": "not_found"})

    def client(self) -> AsyncHTTPHandler:
        return AsyncHTTPHandler(transport=httpx.MockTransport(self.handle))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("template", "timeout", "internet", "expected_body"),
    [
        (None, None, None, {"template": "python", "ttl_s": 300, "persistent": True}),
        ("base", 900, True, {"template": "base", "ttl_s": 900, "persistent": True, "egress": {"internet": True}}),
        (None, 5, False, {"template": "python", "ttl_s": 60, "persistent": True, "egress": {"internet": False}}),
    ],
)
async def test_acreate_sandbox_sends_the_kosa8_create_body(
    template: str | None, timeout: int | None, internet: bool | None, expected_body: Mapping[str, object]
):
    fake = FakeKosa8()

    handle = await Kosa8SandboxConfig().acreate_sandbox(
        template=template,
        timeout=timeout,
        allow_internet_access=internet,
        api_key="k8c_live_test",
        client=fake.client(),
    )

    method, path, headers, body = fake.requests[0]
    assert (method, path, body) == ("POST", "/v1/sandboxes", expected_body)
    assert headers["authorization"] == "Bearer k8c_live_test"
    assert (handle.id, handle.provider) == ("sbx_1", "kosa8")
    assert handle.hidden_params == {"api_key": "k8c_live_test", "api_base": "https://api.kosa8.com"}


@pytest.mark.asyncio
async def test_api_base_comes_from_the_argument_then_the_environment(monkeypatch: pytest.MonkeyPatch):
    fake = FakeKosa8()
    monkeypatch.setenv("KOSA8_API_BASE", "https://kosa8.internal.example")

    from_env = await Kosa8SandboxConfig().acreate_sandbox(api_key="k8c_live_test", client=fake.client())
    from_arg = await Kosa8SandboxConfig().acreate_sandbox(
        api_key="k8c_live_test", api_base="https://other.example/", client=fake.client()
    )

    assert from_env.hidden_params["api_base"] == "https://kosa8.internal.example"
    assert from_arg.hidden_params["api_base"] == "https://other.example/"
    assert [str(r[1]) for r in fake.requests] == ["/v1/sandboxes", "/v1/sandboxes"]


def test_validate_environment_reads_kosa8_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("KOSA8_API_KEY", raising=False)
    with pytest.raises(ValueError, match="KOSA8_API_KEY"):
        Kosa8SandboxConfig().validate_environment()

    monkeypatch.setenv("KOSA8_API_KEY", "k8c_live_env")
    assert Kosa8SandboxConfig().validate_environment() == "k8c_live_env"
    assert Kosa8SandboxConfig().validate_environment(api_key="k8c_live_arg") == "k8c_live_arg"


@pytest.mark.asyncio
async def test_arun_code_returns_the_kernel_output_and_reuses_the_handle_credentials():
    fake = FakeKosa8()
    config = Kosa8SandboxConfig()
    handle = await config.acreate_sandbox(api_key="k8c_live_test", client=fake.client())

    first = await config.arun_code(container=handle, code="x = 6 * 7\nx", client=fake.client())
    second = await config.arun_code(container=handle, code="x", client=fake.client())

    assert first.model_dump() == {
        "stdout": "42\n",
        "stderr": "warn\n",
        "results": [{"type": "text", "data": "42"}],
        "error": None,
        "execution_count": 1,
        "object": "code_execution",
    }
    assert second.execution_count == 2
    method, path, headers, body = fake.requests[1]
    assert (method, path, body) == ("POST", "/v1/sandboxes/sbx_1/code", {"code": "x = 6 * 7\nx", "language": "python"})
    assert headers["authorization"] == "Bearer k8c_live_test"


@pytest.mark.asyncio
async def test_arun_code_passes_a_python_error_through():
    error = {"name": "ZeroDivisionError", "value": "division by zero", "traceback": "Traceback ..."}
    fake = FakeKosa8(code_response=lambda body: httpx.Response(200, json={"stdout": "", "error": error}))
    fake.live.add("sbx_9")

    result = await Kosa8SandboxConfig().arun_code(
        container="sbx_9", code="1/0", api_key="k8c_live_test", client=fake.client()
    )

    assert (result.error, result.stdout, result.stderr, result.results) == (error, "", "", [])


@pytest.mark.asyncio
async def test_arun_code_refuses_output_over_the_cap():
    oversized = httpx.Response(200, content=b'{"stdout": "' + b"a" * SANDBOX_MAX_OUTPUT_BYTES + b'"}')
    fake = FakeKosa8(code_response=lambda body: oversized)
    fake.live.add("sbx_9")

    with pytest.raises(ValueError, match="exceeded"):
        await Kosa8SandboxConfig().arun_code(container="sbx_9", code="1", api_key="k8c_live_test", client=fake.client())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b'["secret-output"]',
        b'"secret-output"',
        b'{"stdout": ["secret-output"]}',
        b'{"stderr": {"secret-output": 1}}',
        b'{"results": ["secret-output"]}',
        b'{"error": "secret-output"}',
        b'{"execution_count": "secret-output"}',
    ],
)
async def test_arun_code_rejects_malformed_responses_without_echoing_them(body: bytes):
    fake = FakeKosa8(code_response=lambda _: httpx.Response(200, content=body))
    fake.live.add("sbx_9")

    with pytest.raises(ValidationError) as exc_info:
        await Kosa8SandboxConfig().arun_code(container="sbx_9", code="1", api_key="k8c_live_test", client=fake.client())

    assert "secret-output" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_a_bare_sandbox_id_is_path_encoded():
    fake = FakeKosa8()

    assert not await Kosa8SandboxConfig().adelete_sandbox(
        container="../v1/api-keys", api_key="k8c_live_test", client=fake.client()
    )

    assert fake.requests[0][1] == "/v1/sandboxes/..%2Fv1%2Fapi-keys"


@pytest.mark.asyncio
async def test_adelete_sandbox_is_true_once_then_false_for_a_gone_sandbox():
    fake = FakeKosa8()
    config = Kosa8SandboxConfig()
    handle = await config.acreate_sandbox(api_key="k8c_live_test", client=fake.client())

    assert await config.adelete_sandbox(container=handle, client=fake.client()) is True
    assert await config.adelete_sandbox(container=handle, client=fake.client()) is False
    assert fake.live == set()


@pytest.mark.asyncio
async def test_a_refused_key_raises():
    fake = FakeKosa8()

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await Kosa8SandboxConfig().acreate_sandbox(api_key="k8c_live_wrong", client=fake.client())

    assert exc_info.value.response.status_code == 401


@pytest.mark.asyncio
async def test_code_interpreter_tool_routes_to_kosa8_and_deletes_the_sandbox(monkeypatch: pytest.MonkeyPatch):
    fake = FakeKosa8()
    monkeypatch.setenv("KOSA8_API_KEY", "k8c_live_test")

    result = await litellm.acode_interpreter_tool(provider="kosa8", code="print(42)", client=fake.client())

    assert result.stdout == "42\n"
    assert [(m, p) for m, p, _, _ in fake.requests] == [
        ("POST", "/v1/sandboxes"),
        ("POST", "/v1/sandboxes/sbx_1/code"),
        ("DELETE", "/v1/sandboxes/sbx_1"),
    ]
    assert fake.live == set()
