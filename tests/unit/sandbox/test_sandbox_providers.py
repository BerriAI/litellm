"""
Tests for the Boat, Tensorlake, Blaxel and Modal sandbox providers and for sandbox cost.

A fake async HTTP client stands in for the provider, so no network is used.
Real-network tests live in tests/integration/sandbox/test_sandbox_providers.py.
"""

import sys
import types
from collections.abc import Callable
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.base_llm.sandbox.transformation import (
    ContainerHandle,
    estimate_sandbox_cost,
    get_sandbox_prices,
)
from litellm.llms.blaxel.sandbox.transformation import MIN_TTL_SECONDS


class FakeResponse:
    def __init__(self, json_data: object = None, status_code: int = 200) -> None:
        self._json = json_data
        self.status_code = status_code

    def json(self) -> object:
        return self._json


class FakeHTTPClient:
    """Answers each request from `route(method, url, json)` and records what was sent."""

    def __init__(self, route: Callable[[str, str, object], object]) -> None:
        self.route = route
        self.calls: list[tuple[str, str, dict, object]] = []

    async def _send(self, method: str, url: str, headers: dict | None, json: object) -> FakeResponse:
        self.calls.append((method, url, headers or {}, json))
        answer = self.route(method, url, json)
        if isinstance(answer, int):
            raise httpx.HTTPStatusError(
                f"status {answer}", request=httpx.Request(method, url), response=httpx.Response(answer)
            )
        return FakeResponse(answer)

    async def post(self, url, headers=None, json=None, **kwargs):
        return await self._send("POST", url, headers, json)

    async def get(self, url, headers=None, **kwargs):
        return await self._send("GET", url, headers, None)

    async def delete(self, url, headers=None, **kwargs):
        return await self._send("DELETE", url, headers, None)


def _boat(method: str, url: str, json: object) -> object:
    if method == "POST" and url.endswith("/sandboxes"):
        return {"sandbox": {"id": "bx_1", "vcpu": 2, "memoryGB": 4}}
    if url.endswith("/commands"):
        return {"stdout": "out\n", "stderr": "err\n", "exitCode": 3}
    if url.endswith("/usage"):
        return {"dollars": 0.5, "seconds": 20}
    return {}


def _tensorlake_route() -> Callable[[str, str, object], object]:
    polls = iter(("running", "exited"))

    def route(method: str, url: str, json: object) -> object:
        if method == "POST" and url.endswith("/sandboxes"):
            return {"sandbox_id": "tl_1", "sandbox_url": "https://tl_1.sandbox.example"}
        if method == "POST":
            return {"pid": 7, "status": "running"}
        if url.endswith("/stdout"):
            return {"lines": ["a", "b"]}
        if url.endswith("/stderr"):
            return {"lines": ["err"]}
        if url.endswith("/processes/7"):
            return {"pid": 7, "status": next(polls), "exit_code": 3}
        return {}

    return route


def _blaxel(method: str, url: str, json: object) -> object:
    if method == "POST" and url.endswith("/sandboxes"):
        return {"metadata": {"name": json["metadata"]["name"], "url": "https://sbx.example"}}
    if url.endswith("/process"):
        return {"stdout": "out\n", "stderr": "err\n", "exitCode": 3}
    return {}


ROUTES: Final = {"boat": lambda: _boat, "tensorlake": _tensorlake_route, "blaxel": lambda: _blaxel}


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", sorted(ROUTES))
async def test_lifecycle_runs_the_command_and_reports_a_failing_exit_code(provider):
    client = FakeHTTPClient(ROUTES[provider]())
    handle = await litellm.acreate_sandbox(provider=provider, api_key="k", client=client)
    result = await litellm.arun_code(
        provider=provider, container=handle, code="echo out; exit 3", language="bash", client=client
    )
    deleted = await litellm.adelete_sandbox(provider=provider, container=handle, client=client)

    assert result.stderr == "err\n"
    assert result.exit_code == 3
    assert result.error["name"] == "ExitCode"
    assert deleted is True
    sent_commands = [str(json) for method, _, _, json in client.calls if method == "POST"]
    assert any("echo out; exit 3" in sent for sent in sent_commands)
    delete_call = next(call for call in client.calls if call[0] == "DELETE")
    assert delete_call[1].endswith(f"/sandboxes/{handle.id}")
    assert all(headers["Authorization"] == "Bearer k" for _, _, headers, _ in client.calls)


@pytest.mark.asyncio
async def test_tensorlake_waits_for_the_process_and_joins_output_lines():
    client = FakeHTTPClient(_tensorlake_route())
    handle = await litellm.acreate_sandbox(provider="tensorlake", api_key="k", client=client)
    result = await litellm.arun_code(provider="tensorlake", container=handle, code="print(1)", client=client)

    assert result.stdout == "a\nb\n"
    polls = [url for method, url, _, _ in client.calls if method == "GET" and url.endswith("/processes/7")]
    assert len(polls) == 2
    assert all(url.startswith("https://tl_1.sandbox.example/") for url in polls)


@pytest.mark.asyncio
async def test_python_code_is_quoted_into_one_shell_argument():
    client = FakeHTTPClient(_boat)
    handle = await litellm.acreate_sandbox(provider="boat", api_key="k", client=client)
    await litellm.arun_code(provider="boat", container=handle, code="print('a b'); print(\"c\")", client=client)

    assert client.calls[-1][3] == {"command": "python3 -c 'print('\"'\"'a b'\"'\"'); print(\"c\")'"}


@pytest.mark.asyncio
async def test_unknown_language_is_rejected_before_any_request():
    client = FakeHTTPClient(_boat)
    with pytest.raises(ValueError, match="Unsupported sandbox language"):
        await litellm.arun_code(
            provider="boat", container="bx_1", code="1", language="ruby", api_key="k", client=client
        )
    assert client.calls == []


@pytest.mark.asyncio
async def test_boat_delete_confirms_the_sandbox_id_and_a_missing_sandbox_is_not_deleted():
    client = FakeHTTPClient(lambda method, url, json: 404)
    deleted = await litellm.adelete_sandbox(provider="boat", container="bx_gone", api_key="k", client=client)

    assert deleted is False
    assert client.calls[0][2]["X-Ascii-Confirm-Delete"] == "bx_gone"


@pytest.mark.asyncio
async def test_boat_create_sends_a_fresh_idempotency_key_each_time():
    client = FakeHTTPClient(_boat)
    await litellm.acreate_sandbox(provider="boat", api_key="k", client=client)
    await litellm.acreate_sandbox(provider="boat", api_key="k", client=client)

    first, second = (headers["Idempotency-Key"] for _, _, headers, _ in client.calls)
    assert first and second and first != second


@pytest.mark.asyncio
async def test_blaxel_raises_a_short_timeout_to_the_provider_minimum():
    client = FakeHTTPClient(_blaxel)
    await litellm.acreate_sandbox(provider="blaxel", api_key="k", timeout=1, client=client)
    await litellm.acreate_sandbox(provider="blaxel", api_key="k", timeout=MIN_TTL_SECONDS + 60, client=client)

    ttls = [json["spec"]["runtime"]["ttl"] for _, _, _, json in client.calls]
    assert ttls == [f"{MIN_TTL_SECONDS}s", f"{MIN_TTL_SECONDS + 60}s"]


@pytest.mark.asyncio
async def test_blaxel_cannot_run_code_from_a_bare_name():
    with pytest.raises(ValueError, match="ContainerHandle"):
        await litellm.arun_code(provider="blaxel", container="name", code="1", api_key="k")


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["boat", "tensorlake", "blaxel", "modal"])
async def test_missing_credentials_fail_with_a_clear_error(provider, monkeypatch):
    for name in ("BOAT_API_KEY", "TENSORLAKE_API_KEY", "BL_API_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="not set"):
        await litellm.acreate_sandbox(provider=provider)


# ---------- cost ----------


@pytest.mark.asyncio
async def test_boat_cost_is_the_dollars_the_provider_reports_and_survives_delete():
    client = FakeHTTPClient(_boat)
    handle = await litellm.acreate_sandbox(provider="boat", api_key="k", client=client)

    running = await litellm.asandbox_cost(provider="boat", container=handle, client=client)
    await litellm.adelete_sandbox(provider="boat", container=handle, client=client)
    requests_before: Final = len(client.calls)
    final = await litellm.asandbox_cost(provider="boat", container=handle, client=client)

    assert (running.cost_usd, running.source, running.seconds) == (0.5, "provider", 20)
    assert final == running
    assert len(client.calls) == requests_before


@pytest.mark.asyncio
async def test_cost_without_provider_dollars_is_an_estimate_that_stops_growing_after_delete():
    client = FakeHTTPClient(_blaxel)
    handle = await litellm.acreate_sandbox(provider="blaxel", api_key="k", client=client)
    handle.created_at -= 100

    running = await litellm.asandbox_cost(provider="blaxel", container=handle)
    await litellm.adelete_sandbox(provider="blaxel", container=handle, client=client)
    final = await litellm.asandbox_cost(provider="blaxel", container=handle)
    handle.created_at -= 1000

    assert running.source == final.source == "estimate"
    assert 0 < running.cost_usd <= final.cost_usd
    assert final.seconds == pytest.approx(100, abs=5)
    assert await litellm.asandbox_cost(provider="blaxel", container=handle) == final


def test_estimate_scales_with_time_and_with_each_priced_resource():
    price = get_sandbox_prices()["tensorlake"]
    base = ContainerHandle(id="s", provider="tensorlake", created_at=0, vcpus=1, memory_gib=1, disk_gib=1)
    more = base.model_copy(update={"vcpus": 2, "memory_gib": 3, "disk_gib": 5})

    assert estimate_sandbox_cost(base, now=10).cost_usd == pytest.approx(
        10 * (price.cost_per_vcpu_second + price.cost_per_gib_second + price.cost_per_disk_gib_second)
    )
    assert estimate_sandbox_cost(base, now=20).cost_usd == pytest.approx(
        2 * estimate_sandbox_cost(base, now=10).cost_usd
    )
    assert estimate_sandbox_cost(more, now=10).cost_usd == pytest.approx(
        10 * (2 * price.cost_per_vcpu_second + 3 * price.cost_per_gib_second + 5 * price.cost_per_disk_gib_second)
    )


@pytest.mark.parametrize(
    "container",
    [
        "bare-id",
        ContainerHandle(id="s", provider="opensandbox", created_at=0, vcpus=1, memory_gib=1),
        ContainerHandle(id="s", provider="e2b", created_at=0),
        ContainerHandle(id="s", provider="e2b", vcpus=1, memory_gib=1),
    ],
)
def test_cost_that_cannot_be_priced_is_unknown_not_zero(container):
    cost = estimate_sandbox_cost(container, now=10)

    assert cost.cost_usd is None
    assert cost.source == "unknown"


@pytest.mark.asyncio
async def test_ephemeral_run_reports_the_sandbox_cost_as_response_cost():
    client = FakeHTTPClient(_boat)
    result = await litellm.acode_interpreter_tool(provider="boat", code="print(1)", api_key="k", client=client)

    assert result._hidden_params["response_cost"] == 0.5
    assert client.calls[-1][0] == "DELETE"


@pytest.mark.asyncio
async def test_ephemeral_run_still_deletes_the_sandbox_when_the_command_request_fails():
    def route(method: str, url: str, json: object) -> object:
        return 500 if url.endswith("/commands") else _boat(method, url, json)

    client = FakeHTTPClient(route)
    with pytest.raises(httpx.HTTPStatusError):
        await litellm.acode_interpreter_tool(provider="boat", code="print(1)", api_key="k", client=client)

    assert client.calls[-1][0] == "DELETE"


# ---------- modal ----------


class _FakeAio:
    def __init__(self, fn: Callable[..., object]) -> None:
        self.fn = fn

    async def aio(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class _FakeModalSandbox:
    object_id = "sb-1"

    def __init__(self, record: dict) -> None:
        self.record = record
        self.exec = _FakeAio(self._exec)
        self.terminate = _FakeAio(lambda: record.__setitem__("terminated", True))

    def _exec(self, *args: str):
        self.record["exec"] = args
        stream = lambda text: types.SimpleNamespace(read=_FakeAio(lambda: text))  # noqa: E731  # tiny local factory
        return types.SimpleNamespace(stdout=stream("out\n"), stderr=stream("err\n"), wait=_FakeAio(lambda: 3))


@pytest.mark.asyncio
async def test_modal_lifecycle_goes_through_the_sdk(monkeypatch):
    record: dict = {}

    def create(**kwargs):
        record["create"] = kwargs
        return _FakeModalSandbox(record)

    fake_modal = types.SimpleNamespace(
        Client=types.SimpleNamespace(from_credentials=_FakeAio(lambda token_id, secret: (token_id, secret))),
        App=types.SimpleNamespace(lookup=_FakeAio(lambda name, **kwargs: name)),
        Image=types.SimpleNamespace(debian_slim=lambda: "slim", from_registry=lambda tag: tag),
        Sandbox=types.SimpleNamespace(create=_FakeAio(create)),
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    handle = await litellm.acreate_sandbox(
        provider="modal", api_key="id:secret", timeout=60, allow_internet_access=False, app_name="my-app"
    )
    result = await litellm.arun_code(provider="modal", container=handle, code="exit 3", language="bash")
    deleted = await litellm.adelete_sandbox(provider="modal", container=handle)

    assert handle.id == "sb-1"
    assert record["create"]["client"] == ("id", "secret")
    assert record["create"]["app"] == "my-app"
    assert record["create"]["timeout"] == 60
    assert record["create"]["block_network"] is True
    assert record["exec"][-1] == "exit 3"
    assert (result.stdout, result.stderr, result.exit_code) == ("out\n", "err\n", 3)
    assert deleted is True and record["terminated"] is True
    assert (await litellm.asandbox_cost(provider="modal", container=handle)).source == "estimate"
