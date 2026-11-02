"""
The factory-built endpoints of ``litellm/router_endpoint_table.py`` on each core: every row is
served, answers sync or async as its dispatch says, and reaches the host through ``host.invoke``
with the row's host op. The host here names every function it serves ``renamed``, so the routing
decisions that depend on the endpoint (stream-timeout split, video-id pin, the responses socket's
turn admission) are shown to key on the endpoint, never on the host function's ``__name__``.
"""

import inspect
from collections.abc import Callable, Mapping
from typing import Final

import pytest

from litellm.router_core import RouterCore
from litellm.router_endpoint_table import DISPATCHES, ENDPOINTS, EndpointSpec, dispatch_for
from litellm.types.videos.utils import encode_video_id_with_provider
from tests.unit.test_router.core.endpoints_helpers import recording_endpoint
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment

MakeCore = Callable[..., RouterCore]
HOST_OPS: Final = frozenset(spec.host_op for spec in ENDPOINTS.values())
SYNC_HOST_OPS: Final = frozenset(spec.host_op for spec in ENDPOINTS.values() if spec.sync)
SEARCH_TOOLS: Final = [{"search_tool_name": "s", "litellm_params": {"search_provider": "tavily", "api_key": "tk"}}]


class RenamingHost(ScriptedHost):
    """Serves every host op of the endpoint table with a recording function named ``renamed``,
    recorded under the op it was invoked as."""

    __slots__ = ()

    def invoke(self, op: str) -> Callable[..., object]:
        if op not in HOST_OPS:
            return super().invoke(op)
        endpoint: Final = recording_endpoint(self, op, is_async=op not in SYNC_HOST_OPS)
        endpoint.__name__ = "renamed"
        endpoint.__qualname__ = "renamed"
        return endpoint


@pytest.fixture
def host() -> RenamingHost:
    return RenamingHost()


async def _settled(result: object) -> object:
    return await result if inspect.isawaitable(result) else result


def _request(spec: EndpointSpec) -> Mapping[str, object]:
    match spec.dispatch:
        case "assistants":
            return {"custom_llm_provider": "openai"}
        case "search" | "search_sync":
            return {"model": "s", "query": "q"}
        case "managed_agent" | "managed_agent_sync":
            return {"name": "a"}
        case "vector_store_file_sync":
            return {"vector_store_id": "v"}
        case _:
            return {
                "model": "g",
                "input": "x",
                "video_id": "v",
                "container_id": "cntr_native",
                "response_id": "resp_plain",
                "interaction_id": "i",
            }


@pytest.mark.parametrize("name", sorted(ENDPOINTS))
async def test_every_table_endpoint_reaches_the_host_with_its_op(make_core: MakeCore, host: RenamingHost, name: str):
    spec: Final = ENDPOINTS[name]
    core: Final = make_core(model_list=[deployment("g", "d1")], search_tools=SEARCH_TOOLS)
    result: Final[object] = getattr(core, name)(**_request(spec))
    assert inspect.isawaitable(result) is not spec.sync
    assert await _settled(result) is not None
    assert [call.op for call in host.calls] == [spec.host_op]


def test_every_table_row_has_a_known_dispatch_and_every_call_type_one_dispatch():
    assert {spec.dispatch for spec in ENDPOINTS.values()} <= DISPATCHES
    assert all(dispatch_for(spec.call_type) == spec.dispatch for spec in ENDPOINTS.values())
    with pytest.raises(ValueError, match="no router endpoint dispatches the call type 'nope'"):
        dispatch_for("nope")


def test_every_table_endpoint_is_declared_on_the_python_core_for_type_checkers():
    from litellm.router_python_core import PythonCore, TableEndpoint

    declared: Final = {name for name, kind in PythonCore.__annotations__.items() if kind == TableEndpoint}
    assert declared == set(ENDPOINTS)


async def test_a_renamed_host_function_keeps_the_stream_timeout_split(make_core: MakeCore, host: RenamingHost):
    core: Final = make_core(model_list=[deployment("g", "d1", timeout=11, stream_timeout=3)])
    await getattr(core, "aresponses")(model="g", input="hi", stream=True)
    sent: Final = host.calls[0].kwargs
    assert (sent["timeout"], sent["stream_timeout"]) == (11.0, 3.0)


@pytest.mark.parametrize("endpoint", ["avideo_status", "video_status"])
async def test_a_renamed_host_function_keeps_the_video_id_pin(make_core: MakeCore, host: RenamingHost, endpoint: str):
    video_id: Final = encode_video_id_with_provider("video_up", "openai", model_id="d2", deployment_id="d2")
    core: Final = make_core(model_list=[deployment("g", "d1"), deployment("g", "d2")])
    for _ in range(6):
        await _settled(getattr(core, endpoint)(model="g", video_id=video_id))
    assert host.deployments_called() == ["d2"] * 6


async def test_a_renamed_host_function_keeps_the_responses_socket_turn_admission(
    make_core: MakeCore, host: RenamingHost
):
    core: Final = make_core(model_list=[deployment("g", "d1")])
    await getattr(core, "_aresponses_websocket")(model="g")
    assert callable(host.calls[0].kwargs["turn_admission"])
