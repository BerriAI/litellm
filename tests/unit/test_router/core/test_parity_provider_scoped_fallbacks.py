"""
Parity of the Rust core with ``router_python_core.py`` on fallbacks for a request that creates or
names a resource only one provider holds (``run_async_fallback``'s ``same_model_group_only``): a file
upload, a batch on an uploaded file, a batch cancel and the generic file and fine-tuning ops fall
back within their own model group only, while the same op naming no such resource still falls back
to another group.
"""

import io
from collections.abc import Awaitable, Callable
from typing import Final, cast

import pytest

from litellm.router_core import RouterCore
from litellm.router_python_host import Endpoint
from tests.unit.test_router.core.endpoints_helpers import EndpointHost
from tests.unit.test_router.core.scripted_host import deployment, server_error

MakeCore = Callable[..., RouterCore]

_BATCH: Final = {"endpoint": "/v1/chat/completions", "completion_window": "24h"}
_TWO_GROUPS: Final = [deployment("a", "a1"), deployment("b", "b1")]
_TO_B: Final = [{"a": ["b"]}]


class _NamedHost(EndpointHost):
    """``EndpointHost`` whose endpoints carry the op's name, as litellm's own functions do: the Python
    core reads a generic op's handler name to decide it names a provider-scoped resource."""

    __slots__ = ()

    def invoke(self, op: str) -> Endpoint:
        endpoint: Final = super().invoke(op)
        endpoint.__name__ = op.rsplit(".", 1)[-1]
        return endpoint


@pytest.fixture
def host() -> _NamedHost:
    return _NamedHost()


def _async(core: object, name: str) -> Callable[..., Awaitable[object]]:
    """An endpoint both cores serve outside the ``RouterCore`` protocol."""
    return cast("Callable[..., Awaitable[object]]", getattr(core, name))  # cast-ok: the member is a coroutine function


def _upload() -> dict[str, object]:  # mutable-ok: a fresh request
    return {"file": io.BytesIO(b'{"custom_id": "1", "body": {"model": "a", "messages": []}}\n'), "purpose": "batch"}


def _failing_a(
    make_core: MakeCore, host: EndpointHost, model_list: list[dict[str, object]] | None = None
) -> RouterCore:
    host.answer("openai/a1", server_error())
    return make_core(model_list=list(model_list or _TWO_GROUPS), num_retries=0)


# --- no fallback to another group ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_file_upload_does_not_fall_back_to_another_group(make_core: MakeCore, host: _NamedHost) -> None:
    core: Final = _failing_a(make_core, host)

    with pytest.raises(type(server_error())):
        await _async(core, "acreate_file")(model="a", fallbacks=_TO_B, **_upload())

    assert host.deployments_called("acreate_file") == ["a1"]


@pytest.mark.asyncio
async def test_a_batch_on_an_uploaded_file_does_not_fall_back_to_another_group(
    make_core: MakeCore, host: _NamedHost
) -> None:
    core: Final = _failing_a(make_core, host)

    with pytest.raises(type(server_error())):
        await _async(core, "acreate_batch")(model="a", input_file_id="file-1", fallbacks=_TO_B, **_BATCH)

    assert host.deployments_called("acreate_batch") == ["a1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("op", "request_fields"),
    [
        ("acancel_batch", {"batch_id": "batch-1"}),
        ("afile_content", {"file_id": "file-1"}),
        ("afile_delete", {"file_id": "file-1"}),
        ("acreate_fine_tuning_job", {"training_file": "file-1"}),
        ("acancel_fine_tuning_job", {"fine_tuning_job_id": "ftjob-1"}),
    ],
    ids=["cancel-batch", "file-content", "file-delete", "create-fine-tuning-job", "cancel-fine-tuning-job"],
)
async def test_an_op_naming_a_provider_resource_does_not_fall_back_to_another_group(
    make_core: MakeCore, host: _NamedHost, op: str, request_fields: dict[str, str]
) -> None:
    core: Final = _failing_a(make_core, host)

    with pytest.raises(type(server_error())):
        await _async(core, op)(model="a", fallbacks=_TO_B, **request_fields)

    assert host.deployments_called(op) == ["a1"]


@pytest.mark.asyncio
async def test_a_dict_fallback_without_a_model_is_skipped_for_a_scoped_batch(
    make_core: MakeCore, host: _NamedHost
) -> None:
    core: Final = _failing_a(make_core, host)

    with pytest.raises(type(server_error())):
        await _async(core, "acreate_batch")(
            model="a", input_file_id="file-1", fallbacks=[{"a": [{"user": "u"}]}], **_BATCH
        )

    assert host.deployments_called("acreate_batch") == ["a1"]


# --- what still falls back ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_batch_without_an_input_file_still_falls_back_to_another_group(
    make_core: MakeCore, host: _NamedHost
) -> None:
    core: Final = _failing_a(make_core, host)

    await _async(core, "acreate_batch")(model="a", fallbacks=_TO_B, **_BATCH)

    assert host.deployments_called("acreate_batch") == ["a1", "b1"]


@pytest.mark.asyncio
async def test_a_file_op_naming_no_file_still_falls_back_to_another_group(
    make_core: MakeCore, host: _NamedHost
) -> None:
    core: Final = _failing_a(make_core, host)

    await _async(core, "afile_content")(model="a", fallbacks=_TO_B)

    assert host.deployments_called("afile_content") == ["a1", "b1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["acreate_file", "acreate_batch"], ids=["file-upload", "batch-on-file"])
async def test_a_scoped_request_still_falls_back_within_its_own_group(
    make_core: MakeCore, host: _NamedHost, op: str
) -> None:
    core: Final = _failing_a(make_core, host)
    request: Final = _upload() if op == "acreate_file" else {"input_file_id": "file-1", **_BATCH}
    # A dict target naming the failed group with a param of its own is a distinct target, not the failed one.
    same_group: Final = [{"a": [{"model": "a", "user": "u"}, "b"]}]

    with pytest.raises(type(server_error())):
        await _async(core, op)(model="a", fallbacks=same_group, **request)

    assert host.deployments_called(op) == ["a1", "a1"]


@pytest.mark.asyncio
async def test_a_batch_on_an_uploaded_file_still_falls_back_to_the_next_order_level(
    make_core: MakeCore, host: _NamedHost
) -> None:
    tiered: Final = [deployment("a", "a1", order=1), deployment("a", "a2", order=2), deployment("b", "b1")]
    core: Final = _failing_a(make_core, host, tiered)

    await _async(core, "acreate_batch")(model="a", input_file_id="file-1", fallbacks=_TO_B, **_BATCH)

    assert host.deployments_called("acreate_batch") == ["a1", "a2"]


@pytest.mark.asyncio
async def test_a_file_upload_moves_to_the_next_order_level(make_core: MakeCore, host: _NamedHost) -> None:
    tiered: Final = [deployment("a", "a1", order=1), deployment("a", "a2", order=2)]
    core: Final = _failing_a(make_core, host, tiered)

    await _async(core, "acreate_file")(model="a", **_upload())

    assert host.deployments_called("acreate_file") == ["a1", "a2"]
