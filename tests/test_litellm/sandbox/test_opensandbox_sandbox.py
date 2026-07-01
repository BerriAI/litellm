import json

import httpx
import pytest

import litellm
from litellm.llms.base_llm.sandbox.transformation import ContainerHandle
from litellm.llms.opensandbox.sandbox.transformation import (
    MAX_OUTPUT_BYTES,
    OPEN_SANDBOX_DEFAULT_TEMPLATE,
    OpenSandboxSandboxConfig,
)
from litellm.utils import ProviderConfigManager

TEST_API_BASE = "https://sandbox.test/v1"


def http_status_error(status_code, url="http://test"):
    return httpx.HTTPStatusError(
        f"status {status_code}",
        request=httpx.Request("GET", url),
        response=httpx.Response(status_code),
    )


def sse(data):
    return f"data: {json.dumps(data)}"


class FakeResponse:
    def __init__(self, *, json_data=None, lines=None, status_code=200):
        self._json = json_data
        self._lines = lines or []
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise http_status_error(self.status_code)

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class FakeHTTPClient:
    def __init__(
        self,
        *,
        create_json=None,
        sandbox_states=None,
        endpoint_json=None,
        endpoint_responses=None,
        execute_lines=None,
        delete_status=204,
        execute_raises=None,
    ):
        self.create_json = create_json or {
            "id": "osb_123",
            "status": {"state": "Running"},
            "createdAt": "2026-01-01T00:00:00Z",
            "entrypoint": ["/opt/code-interpreter/code-interpreter.sh"],
        }
        self.sandbox_states = list(
            sandbox_states
            or [
                {
                    "id": "osb_123",
                    "status": {"state": "Running"},
                    "createdAt": "2026-01-01T00:00:00Z",
                    "entrypoint": ["/opt/code-interpreter/code-interpreter.sh"],
                }
            ]
        )
        self.endpoint_json = endpoint_json or {
            "endpoint": "execd.local:44772",
            "headers": {"X-EXECD-ACCESS-TOKEN": "execd-token"},
        }
        self.endpoint_responses = (
            list(endpoint_responses) if endpoint_responses is not None else None
        )
        self.execute_lines = execute_lines or []
        self.delete_status = delete_status
        self.execute_raises = execute_raises
        self.calls = []

    async def post(self, url, headers=None, json=None, stream=False, **kwargs):
        self.calls.append(("POST", url, headers, json, {"stream": stream}))
        if url.endswith("/sandboxes"):
            return FakeResponse(json_data=self.create_json)
        if url.endswith("/code"):
            if self.execute_raises is not None:
                raise self.execute_raises
            return FakeResponse(lines=self.execute_lines)
        raise AssertionError(f"unexpected POST {url}")

    async def get(self, url, headers=None, params=None, **kwargs):
        self.calls.append(("GET", url, headers, None, params))
        if "/endpoints/44772" in url:
            if self.endpoint_responses is not None and self.endpoint_responses:
                response = self.endpoint_responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                if isinstance(response, FakeResponse):
                    return response
                return FakeResponse(json_data=response)
            return FakeResponse(json_data=self.endpoint_json)
        if "/sandboxes/" in url:
            state = self.sandbox_states.pop(0)
            return FakeResponse(json_data=state)
        raise AssertionError(f"unexpected GET {url}")

    async def delete(self, url, headers=None, **kwargs):
        self.calls.append(("DELETE", url, headers, None, None))
        if not (200 <= self.delete_status < 300):
            raise http_status_error(self.delete_status, url)
        return FakeResponse(status_code=self.delete_status)


def test_parse_sse_lines_maps_output_result_count_and_error():
    lines = [
        sse({"type": "stdout", "text": "hello\n"}),
        sse({"type": "stderr", "text": "warn\n"}),
        sse({"type": "result", "results": {"text/plain": "4"}}),
        sse({"type": "execution_count", "execution_count": 7}),
        sse(
            {
                "type": "error",
                "error": {
                    "ename": "ValueError",
                    "evalue": "bad",
                    "traceback": ["Traceback"],
                },
            }
        ),
    ]

    result = OpenSandboxSandboxConfig._parse_lines(lines)

    assert result.stdout == "hello\n"
    assert result.stderr == "warn\n"
    assert result.results == [{"text/plain": "4"}]
    assert result.execution_count == 7
    assert result.error == {
        "name": "ValueError",
        "value": "bad",
        "traceback": ["Traceback"],
    }


def test_parse_sse_lines_skips_non_json_and_control_lines():
    lines = [
        "event: message",
        "not-json",
        "",
        sse({"type": "stdout", "text": "ok\n"}),
    ]

    result = OpenSandboxSandboxConfig._parse_lines(lines)

    assert result.stdout == "ok\n"
    assert result.error is None


def test_parse_sse_lines_maps_fallback_shapes():
    lines = [
        "data:",
        sse(["not-a-dict"]),
        sse({"code": "BadRequest", "message": "nope"}),
        sse({"type": "result", "text/plain": "4"}),
        sse({"type": "error", "name": "RuntimeError", "text": "boom"}),
        sse({"type": "execution_count", "execution_count": "8"}),
    ]

    result = OpenSandboxSandboxConfig._parse_lines(lines)

    assert result.results == [{"text/plain": "4"}]
    assert result.execution_count == 8
    assert result.error == {
        "name": "BadRequest",
        "value": "nope",
        "traceback": [],
    }
    fallback_error = OpenSandboxSandboxConfig._parse_lines(
        [sse({"type": "error", "name": "RuntimeError", "text": "boom"})]
    )
    assert fallback_error.error == {
        "name": "RuntimeError",
        "value": "boom",
        "traceback": [],
    }
    empty_string_error = OpenSandboxSandboxConfig._parse_lines(
        [
            sse(
                {
                    "type": "error",
                    "error": {
                        "ename": "",
                        "name": "FallbackName",
                        "evalue": "",
                        "value": "fallback value",
                        "traceback": [],
                    },
                }
            )
        ]
    )
    assert empty_string_error.error == {
        "name": "",
        "value": "",
        "traceback": [],
    }


def test_static_helpers_cover_defaults_and_fallbacks(monkeypatch):
    def fake_secret(key):
        if key == "OPEN_SANDBOX_API_KEY":
            return "env-key"
        if key == "OPEN_SANDBOX_API_BASE":
            return TEST_API_BASE
        return None

    monkeypatch.setattr(
        "litellm.llms.opensandbox.sandbox.transformation.get_secret_str",
        fake_secret,
    )
    config = OpenSandboxSandboxConfig()
    handle = ContainerHandle(id="osb", provider="opensandbox", domain="http://x/v1")

    assert config.validate_environment() == "env-key"
    assert config.validate_environment(api_key="") == ""
    assert config._api_key(api_key=None, handle=handle) == "env-key"

    handle._hidden_params = {"api_key": "stored-key"}
    assert config._api_key(api_key=None, handle=handle) == "stored-key"
    assert config._http(None) is not None

    body = config._create_body(
        template=None,
        timeout=None,
        allow_internet_access=False,
        metadata=None,
        env_vars=None,
        resource_limits=None,
        resource_requests=None,
        entrypoint=None,
        network_policy={"egress": [{"domain": "example.com"}]},
        secure_access=True,
    )
    assert body["networkPolicy"] == {"egress": [{"domain": "example.com"}]}
    assert body["secureAccess"] is True

    other_body = config._create_body(
        template=None,
        timeout=None,
        allow_internet_access=False,
        metadata=None,
        env_vars=None,
        resource_limits=None,
        resource_requests=None,
        entrypoint=None,
        network_policy=None,
        secure_access=False,
    )
    assert body["resourceLimits"] is not other_body["resourceLimits"]

    assert config._sandbox_state(None) is None
    assert config._sandbox_state({"status": "Running"}) is None
    assert config._as_str_dict(None) == {}
    assert config._endpoint_base_url("http://execd.local", "https://api/v1") == (
        "http://execd.local"
    )
    assert config._api_base(None) == TEST_API_BASE
    assert config._api_base("https://direct.test/v1/") == "https://direct.test/v1"
    assert config._as_int("9") == 9
    assert config._as_int("nope") is None
    assert config._as_int(None) is None
    assert isinstance(
        ProviderConfigManager.get_provider_sandbox_config("opensandbox"),
        OpenSandboxSandboxConfig,
    )


def test_api_base_requires_kwarg_or_env(monkeypatch):
    monkeypatch.setattr(
        "litellm.llms.opensandbox.sandbox.transformation.get_secret_str",
        lambda key: None,
    )

    with pytest.raises(ValueError, match="api_base is required"):
        OpenSandboxSandboxConfig._api_base(None)


@pytest.mark.asyncio
async def test_create_posts_default_body_and_omits_empty_api_key():
    client = FakeHTTPClient()

    handle = await OpenSandboxSandboxConfig().acreate_sandbox(
        api_key="", api_base=TEST_API_BASE, client=client
    )

    method, url, headers, body, _ = client.calls[0]
    assert method == "POST"
    assert url == f"{TEST_API_BASE}/sandboxes"
    assert "OPEN-SANDBOX-API-KEY" not in headers
    assert body["image"] == {"uri": OPEN_SANDBOX_DEFAULT_TEMPLATE}
    assert body["entrypoint"] == ["/opt/code-interpreter/code-interpreter.sh"]
    assert body["timeout"] == 300
    assert body["resourceLimits"] == {"cpu": "1", "memory": "2Gi"}
    assert body["networkPolicy"] == {"defaultAction": "deny", "egress": []}
    assert handle.id == "osb_123"
    assert handle._hidden_params["execd_endpoint"] == "execd.local:44772"


@pytest.mark.asyncio
async def test_create_can_opt_into_internet_access():
    client = FakeHTTPClient()

    await OpenSandboxSandboxConfig().acreate_sandbox(
        api_key="",
        api_base=TEST_API_BASE,
        allow_internet_access=True,
        client=client,
    )

    _, _, _, body, _ = client.calls[0]
    assert "networkPolicy" not in body


@pytest.mark.asyncio
async def test_create_custom_options_poll_and_endpoint_resolution():
    client = FakeHTTPClient(
        create_json={
            "id": "osb_pending",
            "status": {"state": "Pending"},
            "createdAt": "2026-01-01T00:00:00Z",
            "entrypoint": ["/bin/sh"],
        },
        sandbox_states=[
            {
                "id": "osb_pending",
                "status": {"state": "Running"},
                "createdAt": "2026-01-01T00:00:00Z",
                "entrypoint": ["/bin/sh"],
            }
        ],
    )

    handle = await OpenSandboxSandboxConfig().acreate_sandbox(
        template="custom/image:latest",
        timeout=600,
        allow_internet_access=False,
        api_key="osb-key",
        api_base="https://sandbox.example/v1",
        metadata={"suite": "unit"},
        env_vars={"PYTHONUNBUFFERED": "1"},
        resource_limits={"cpu": "500m", "memory": "512Mi"},
        resource_requests={"cpu": "250m", "memory": "256Mi"},
        entrypoint=["/bin/sh", "-lc", "sleep 3600"],
        use_server_proxy=True,
        client=client,
    )

    _, create_url, create_headers, body, _ = client.calls[0]
    _, poll_url, poll_headers, _, _ = client.calls[1]
    _, endpoint_url, endpoint_headers, _, endpoint_params = client.calls[2]

    assert create_url == "https://sandbox.example/v1/sandboxes"
    assert create_headers["OPEN-SANDBOX-API-KEY"] == "osb-key"
    assert body["image"] == {"uri": "custom/image:latest"}
    assert body["entrypoint"] == ["/bin/sh", "-lc", "sleep 3600"]
    assert body["metadata"] == {"suite": "unit"}
    assert body["env"] == {"PYTHONUNBUFFERED": "1"}
    assert body["resourceLimits"] == {"cpu": "500m", "memory": "512Mi"}
    assert body["resourceRequests"] == {"cpu": "250m", "memory": "256Mi"}
    assert body["networkPolicy"] == {"defaultAction": "deny", "egress": []}
    assert poll_url == "https://sandbox.example/v1/sandboxes/osb_pending"
    assert poll_headers["OPEN-SANDBOX-API-KEY"] == "osb-key"
    assert endpoint_url.endswith("/sandboxes/osb_pending/endpoints/44772")
    assert endpoint_headers["OPEN-SANDBOX-API-KEY"] == "osb-key"
    assert endpoint_params == {"use_server_proxy": True}
    assert handle.id == "osb_pending"


@pytest.mark.asyncio
async def test_create_waits_across_pending_state(monkeypatch):
    client = FakeHTTPClient(
        create_json={
            "id": "osb_pending",
            "status": {"state": "Pending"},
            "createdAt": "2026-01-01T00:00:00Z",
        },
        sandbox_states=[
            {"id": "osb_pending", "status": {"state": "Pending"}},
            {"id": "osb_pending", "status": {"state": "Running"}},
        ],
    )
    sleeps = []

    async def fake_sleep(interval):
        sleeps.append(interval)

    monkeypatch.setattr(
        "litellm.llms.opensandbox.sandbox.transformation.asyncio.sleep", fake_sleep
    )

    handle = await OpenSandboxSandboxConfig().acreate_sandbox(
        api_key="",
        api_base=TEST_API_BASE,
        ready_timeout=1,
        poll_interval=0.01,
        client=client,
    )

    assert handle.id == "osb_pending"
    assert sleeps == [0.01]


@pytest.mark.asyncio
async def test_create_raises_for_terminal_state():
    client = FakeHTTPClient(
        create_json={"id": "osb_failed", "status": {"state": "Pending"}},
        sandbox_states=[
            {"id": "osb_failed", "status": {"state": "Failed"}},
        ],
    )

    with pytest.raises(ValueError, match="entered Failed"):
        await OpenSandboxSandboxConfig().acreate_sandbox(
            api_key="", api_base=TEST_API_BASE, client=client
        )


@pytest.mark.asyncio
async def test_create_times_out_waiting_for_running():
    client = FakeHTTPClient(
        create_json={"id": "osb_slow", "status": {"state": "Pending"}},
        sandbox_states=[
            {"id": "osb_slow", "status": {"state": "Pending"}},
        ],
    )

    with pytest.raises(TimeoutError, match="was not Running"):
        await OpenSandboxSandboxConfig().acreate_sandbox(
            api_key="",
            api_base=TEST_API_BASE,
            ready_timeout=0,
            poll_interval=0,
            client=client,
        )


@pytest.mark.asyncio
async def test_create_waits_for_endpoint_resolution(monkeypatch):
    client = FakeHTTPClient(
        endpoint_responses=[
            http_status_error(404, f"{TEST_API_BASE}/sandboxes/osb_123"),
            {
                "endpoint": "execd.local:44772",
                "headers": {"X-EXECD-ACCESS-TOKEN": "execd-token"},
            },
        ],
    )
    sleeps = []

    async def fake_sleep(interval):
        sleeps.append(interval)

    monkeypatch.setattr(
        "litellm.llms.opensandbox.sandbox.transformation.asyncio.sleep", fake_sleep
    )

    handle = await OpenSandboxSandboxConfig().acreate_sandbox(
        api_key="",
        api_base=TEST_API_BASE,
        ready_timeout=1,
        poll_interval=0.01,
        client=client,
    )

    endpoint_calls = [call for call in client.calls if "/endpoints/44772" in call[1]]
    assert handle._hidden_params["execd_endpoint"] == "execd.local:44772"
    assert len(endpoint_calls) == 2
    assert sleeps == [0.01]


@pytest.mark.asyncio
async def test_create_raises_when_endpoint_is_missing():
    client = FakeHTTPClient(endpoint_json={"headers": {"X": "y"}})

    with pytest.raises(TimeoutError, match="execd endpoint.*not ready"):
        await OpenSandboxSandboxConfig().acreate_sandbox(
            api_key="", api_base=TEST_API_BASE, ready_timeout=0, client=client
        )


@pytest.mark.asyncio
async def test_create_reraises_non_404_endpoint_error():
    client = FakeHTTPClient(endpoint_responses=[http_status_error(500)])

    with pytest.raises(httpx.HTTPStatusError):
        await OpenSandboxSandboxConfig().acreate_sandbox(
            api_key="", api_base=TEST_API_BASE, client=client
        )


@pytest.mark.asyncio
async def test_run_code_resolves_bare_id_and_posts_sse_request():
    client = FakeHTTPClient(
        execute_lines=[
            sse({"type": "stdout", "text": "42\n"}),
        ]
    )

    result = await OpenSandboxSandboxConfig().arun_code(
        container="osb_bare",
        code="print(6*7)",
        language="python",
        api_key="",
        api_base="http://sandbox.local/v1",
        client=client,
    )

    endpoint_call = client.calls[0]
    run_call = client.calls[1]
    assert endpoint_call[0] == "GET"
    assert (
        endpoint_call[1] == "http://sandbox.local/v1/sandboxes/osb_bare/endpoints/44772"
    )
    assert run_call[0] == "POST"
    assert run_call[1] == "http://execd.local:44772/code"
    assert run_call[2]["X-EXECD-ACCESS-TOKEN"] == "execd-token"
    assert run_call[3] == {
        "code": "print(6*7)",
        "context": {"language": "python"},
    }
    assert run_call[4] == {"stream": True}
    assert result.stdout == "42\n"


@pytest.mark.asyncio
async def test_run_code_uses_https_for_scheme_less_endpoint_when_api_base_is_https():
    client = FakeHTTPClient()
    handle = ContainerHandle(
        id="osb_https", provider="opensandbox", domain="https://sandbox.example/v1"
    )
    handle._hidden_params = {
        "execd_endpoint": "execd.example/route/44772",
        "execd_headers": {},
    }

    await OpenSandboxSandboxConfig().arun_code(
        container=handle, code="print(1)", client=client
    )

    assert client.calls[0][1] == "https://execd.example/route/44772/code"


@pytest.mark.asyncio
async def test_run_code_aborts_on_output_over_cap():
    client = FakeHTTPClient(execute_lines=["x" * (MAX_OUTPUT_BYTES + 1)])
    handle = ContainerHandle(id="osb_big", provider="opensandbox", domain="http://x/v1")
    handle._hidden_params = {"execd_endpoint": "execd.local:44772", "execd_headers": {}}

    with pytest.raises(ValueError, match="exceeded"):
        await OpenSandboxSandboxConfig().arun_code(
            container=handle, code="print('x')", client=client
        )


@pytest.mark.asyncio
async def test_delete_returns_false_on_404():
    client = FakeHTTPClient(delete_status=404)

    ok = await OpenSandboxSandboxConfig().adelete_sandbox(
        container="osb_gone",
        api_key="",
        api_base="http://sandbox.local/v1",
        client=client,
    )

    assert ok is False


@pytest.mark.asyncio
async def test_delete_reraises_non_404_http_error():
    client = FakeHTTPClient(delete_status=500)

    with pytest.raises(httpx.HTTPStatusError):
        await OpenSandboxSandboxConfig().adelete_sandbox(
            container="osb_err",
            api_key="",
            api_base="http://sandbox.local/v1",
            client=client,
        )


@pytest.mark.asyncio
async def test_public_lifecycle_create_run_delete():
    client = FakeHTTPClient(
        execute_lines=[
            sse({"type": "stdout", "text": "42\n"}),
        ]
    )

    container = await litellm.acreate_sandbox(
        provider="opensandbox", api_key="", api_base=TEST_API_BASE, client=client
    )
    result = await litellm.arun_code(
        provider="opensandbox",
        container=container,
        code="print(6*7)",
        api_key="",
        client=client,
    )
    ok = await litellm.adelete_sandbox(
        provider="opensandbox",
        container=container,
        api_key="",
        client=client,
    )

    assert container.id == "osb_123"
    assert result.stdout == "42\n"
    assert ok is True


@pytest.mark.asyncio
async def test_code_interpreter_tool_deletes_even_when_run_raises():
    client = FakeHTTPClient(execute_raises=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await litellm.acode_interpreter_tool(
            provider="opensandbox",
            code="1/0",
            api_key="",
            api_base=TEST_API_BASE,
            client=client,
        )

    assert [call[0] for call in client.calls] == ["POST", "GET", "POST", "DELETE"]
    assert client.calls[0][1].endswith("/sandboxes")
    assert client.calls[1][1].endswith("/endpoints/44772")
    assert client.calls[2][1].endswith("/code")
    assert client.calls[3][1].endswith("/sandboxes/osb_123")
