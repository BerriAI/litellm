"""
Tests for the e2b code execution sandbox primitive.

Unit tests inject a fake async HTTP client (dependency injection, no
monkeypatching) and assert request shapes and result mapping. Real-network
integration tests live in tests/integration/sandbox/test_e2b_sandbox.py.
"""

import json

import httpx
import pytest

import litellm
from litellm.llms.base_llm.sandbox.transformation import ContainerHandle
from litellm.llms.e2b.sandbox.transformation import (
    MAX_OUTPUT_BYTES,
    E2BSandboxConfig,
)


class FakeResponse:
    def __init__(self, *, json_data=None, lines=None, status_code=200):
        self._json = json_data
        self._lines = lines or []
        self.status_code = status_code

    def json(self):
        return self._json

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class FakeHTTPClient:
    """Records outbound requests and returns canned responses keyed by URL."""

    def __init__(
        self,
        *,
        create_json=None,
        execute_lines=None,
        delete_status=204,
        execute_raises=None,
    ):
        self.create_json = create_json or {
            "sandboxID": "sbx_123",
            "domain": "e2b.app",
            "envdAccessToken": "tok_abc",
        }
        self.execute_lines = execute_lines or []
        self.delete_status = delete_status
        self.execute_raises = execute_raises
        self.calls = []

    async def post(self, url, headers=None, json=None, stream=False, **kwargs):
        self.calls.append(("POST", url, headers, json))
        if url.endswith("/sandboxes"):
            return FakeResponse(json_data=self.create_json)
        if url.endswith("/execute"):
            if self.execute_raises is not None:
                raise self.execute_raises
            return FakeResponse(lines=self.execute_lines)
        raise AssertionError(f"unexpected POST {url}")

    async def delete(self, url, headers=None, **kwargs):
        self.calls.append(("DELETE", url, headers, None))
        if not (200 <= self.delete_status < 300):
            raise httpx.HTTPStatusError(
                f"status {self.delete_status}",
                request=httpx.Request("DELETE", url),
                response=httpx.Response(self.delete_status),
            )
        return FakeResponse(status_code=self.delete_status)


# ---------- pure parser ----------


def test_parse_lines_stdout_and_count():
    lines = [
        json.dumps({"type": "stdout", "text": "6\n", "timestamp": 1}),
        json.dumps({"type": "number_of_executions", "execution_count": 1}),
    ]
    result = E2BSandboxConfig._parse_lines(lines)
    assert result.stdout == "6\n"
    assert result.execution_count == 1
    assert result.error is None


def test_parse_lines_error_surfaces_name_and_traceback():
    lines = [
        json.dumps(
            {
                "type": "error",
                "name": "ZeroDivisionError",
                "value": "division by zero",
                "traceback": "Traceback (most recent call last): ...",
            }
        )
    ]
    result = E2BSandboxConfig._parse_lines(lines)
    assert result.error["name"] == "ZeroDivisionError"
    assert "Traceback" in result.error["traceback"]


def test_parse_lines_result_carries_png():
    lines = [
        json.dumps({"type": "result", "png": "BASE64DATA", "is_main_result": True})
    ]
    result = E2BSandboxConfig._parse_lines(lines)
    assert result.results and result.results[0]["png"] == "BASE64DATA"
    assert "type" not in result.results[0]


# ---------- request shapes ----------


@pytest.mark.asyncio
async def test_template_flows_into_create_request_as_templateID():
    client = FakeHTTPClient()
    cfg = E2BSandboxConfig()
    handle = await cfg.acreate_sandbox(
        template="my-custom-template", api_key="e2b_key", client=client
    )

    method, url, headers, body = client.calls[0]
    assert method == "POST"
    assert url.endswith("/sandboxes")
    assert body["templateID"] == "my-custom-template"  # not "template"
    assert body["secure"] is True
    assert headers["X-API-Key"] == "e2b_key"
    assert handle.id == "sbx_123"
    assert handle._hidden_params["envd_access_token"] == "tok_abc"


@pytest.mark.asyncio
async def test_create_defaults_template_when_omitted():
    client = FakeHTTPClient()
    await E2BSandboxConfig().acreate_sandbox(api_key="e2b_key", client=client)
    _, _, _, body = client.calls[0]
    assert body["templateID"] == "code-interpreter-v1"


@pytest.mark.asyncio
async def test_run_code_targets_jupyter_host_with_access_token():
    client = FakeHTTPClient(
        execute_lines=[json.dumps({"type": "stdout", "text": "42\n", "timestamp": 1})]
    )
    handle = ContainerHandle(id="sbx_xyz", provider="e2b", domain="e2b.app")
    handle._hidden_params = {"envd_access_token": "tok_run"}

    result = await E2BSandboxConfig().arun_code(
        container=handle, code="print(6*7)", client=client
    )

    method, url, headers, body = client.calls[0]
    assert url == "https://49999-sbx_xyz.e2b.app/execute"
    assert headers["X-Access-Token"] == "tok_run"
    assert body["code"] == "print(6*7)"
    assert result.stdout.strip() == "42"


@pytest.mark.asyncio
async def test_delete_issues_delete_to_sandbox_id():
    client = FakeHTTPClient(delete_status=204)
    handle = ContainerHandle(id="sbx_del", provider="e2b", domain="e2b.app")
    handle._hidden_params = {"api_key": "e2b_key"}

    ok = await E2BSandboxConfig().adelete_sandbox(container=handle, client=client)

    method, url, headers, _ = client.calls[0]
    assert method == "DELETE"
    assert url.endswith("/sandboxes/sbx_del")
    assert ok is True


@pytest.mark.asyncio
async def test_delete_returns_false_on_404():
    client = FakeHTTPClient(delete_status=404)
    handle = ContainerHandle(id="sbx_gone", provider="e2b", domain="e2b.app")
    handle._hidden_params = {"api_key": "e2b_key"}
    ok = await E2BSandboxConfig().adelete_sandbox(container=handle, client=client)
    assert ok is False


# ---------- ephemeral teardown ----------


@pytest.mark.asyncio
async def test_code_interpreter_tool_deletes_even_when_run_raises():
    client = FakeHTTPClient(execute_raises=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await litellm.acode_interpreter_tool(
            provider="e2b", code="1/0", api_key="e2b_key", client=client
        )

    methods = [c[0] for c in client.calls]
    urls = [c[1] for c in client.calls]
    assert methods == ["POST", "POST", "DELETE"]  # create, run(raises), delete
    assert urls[0].endswith("/sandboxes")
    assert urls[1].endswith("/execute")
    assert urls[2].endswith("/sandboxes/sbx_123")


# ---------- correctness guards ----------


@pytest.mark.asyncio
async def test_delete_reraises_non_404_http_error():
    client = FakeHTTPClient(delete_status=500)
    handle = ContainerHandle(id="sbx_err", provider="e2b", domain="e2b.app")
    handle._hidden_params = {"api_key": "e2b_key"}
    with pytest.raises(httpx.HTTPStatusError):
        await E2BSandboxConfig().adelete_sandbox(container=handle, client=client)


@pytest.mark.asyncio
async def test_create_preserves_explicit_zero_timeout():
    client = FakeHTTPClient()
    await E2BSandboxConfig().acreate_sandbox(
        timeout=0, api_key="e2b_key", client=client
    )
    _, _, _, body = client.calls[0]
    assert body["timeout"] == 0


@pytest.mark.asyncio
async def test_run_code_rejects_bare_id_without_access_token():
    client = FakeHTTPClient()
    with pytest.raises(ValueError, match="access token"):
        await E2BSandboxConfig().arun_code(
            container="sbx_no_token", code="print(1)", client=client
        )
    assert client.calls == []  # never reached the network


def test_parse_lines_skips_non_json_lines():
    lines = [
        "not-json-heartbeat",
        json.dumps({"type": "stdout", "text": "ok\n"}),
        "",
        "{partial",
    ]
    result = E2BSandboxConfig._parse_lines(lines)
    assert result.stdout == "ok\n"
    assert result.error is None


@pytest.mark.asyncio
async def test_run_code_aborts_on_output_over_cap():
    big_line = "x" * (MAX_OUTPUT_BYTES + 1)
    client = FakeHTTPClient(execute_lines=[big_line])
    handle = ContainerHandle(id="sbx_big", provider="e2b", domain="e2b.app")
    handle._hidden_params = {"envd_access_token": "tok"}
    with pytest.raises(ValueError, match="exceeded"):
        await E2BSandboxConfig().arun_code(
            container=handle, code="print('x'*999)", client=client
        )


# ---------- public entrypoints ----------


@pytest.mark.asyncio
async def test_public_lifecycle_create_run_delete():
    client = FakeHTTPClient(
        execute_lines=[json.dumps({"type": "stdout", "text": "42\n"})]
    )
    container = await litellm.acreate_sandbox(
        provider="e2b", api_key="e2b_key", client=client
    )
    assert container.id == "sbx_123"

    result = await litellm.arun_code(
        provider="e2b",
        container=container,
        api_key="e2b_key",
        code="print(6*7)",
        client=client,
    )
    assert result.stdout.strip() == "42"

    assert (
        await litellm.adelete_sandbox(
            provider="e2b", container=container, api_key="e2b_key", client=client
        )
        is True
    )


@pytest.mark.asyncio
async def test_unsupported_provider_raises():
    with pytest.raises(ValueError):
        await litellm.acreate_sandbox(provider="not-a-provider")


# ---------- api_base override ----------


@pytest.mark.asyncio
async def test_create_uses_api_base_override():
    client = FakeHTTPClient()
    await E2BSandboxConfig().acreate_sandbox(
        api_base="http://my-sandbox:8080", api_key="k", client=client
    )
    _, url, _, _ = client.calls[0]
    assert url == "http://my-sandbox:8080/sandboxes"


@pytest.mark.asyncio
async def test_create_defaults_to_e2b_api_base():
    client = FakeHTTPClient()
    await E2BSandboxConfig().acreate_sandbox(api_key="k", client=client)
    _, url, _, _ = client.calls[0]
    assert url == "https://api.e2b.app/sandboxes"
