from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import Final

import pytest
from pydantic import JsonValue

from litellm._v2 import routing

pytestmark = pytest.mark.requires_rust_extension


@pytest.fixture
def model_list() -> tuple[Mapping[str, JsonValue], ...]:
    return (
        {
            "model_name": "primary",
            "model_info": {"id": "first"},
            "litellm_params": {"model": "provider/a", "api_key": "private-key"},
        },
        {
            "model_name": "primary",
            "model_info": {"id": "second"},
            "litellm_params": {"model": "provider/b"},
        },
    )


def test_native_handle_reconfiguration_returns_detached_snapshots(model_list: tuple[Mapping[str, JsonValue], ...]) -> None:
    router: Final = routing.create(model_list=model_list, retries=2, timeout=30.0)
    before: Final = routing.snapshot(router)
    routing.reconfigure(router, config=routing.RouterConfig(model_list=()))
    after: Final = routing.snapshot(router)
    routing.close(router)

    assert before.deployments == (
        routing.Candidate("first", "primary", "provider/a"),
        routing.Candidate("second", "primary", "provider/b"),
    )
    assert after.deployments == ()
    assert after.generation == before.generation + 1
    assert routing.snapshot(router).closed
    with pytest.raises(FrozenInstanceError):
        setattr(before.deployments[0], "model", "changed")
    with pytest.raises(TypeError):
        type("Subclass", (routing.RouterHandle,), {})
    with pytest.raises(AttributeError):
        setattr(router, "model_list", ())


@pytest.mark.asyncio
async def test_close_is_idempotent_and_reconfigure_cannot_reopen_a_handle(
    model_list: tuple[Mapping[str, JsonValue], ...],
) -> None:
    router: Final = routing.create(model_list=model_list)
    await routing.aclose(router)
    routing.close(router)

    assert routing.snapshot(router).closed
    with pytest.raises(RuntimeError, match="closed"):
        routing.reconfigure(router, config=routing.RouterConfig(model_list=()))


def test_invalid_reconfiguration_keeps_the_original_snapshot(model_list: tuple[Mapping[str, JsonValue], ...]) -> None:
    router: Final = routing.create(model_list=model_list)
    before: Final = routing.snapshot(router)
    config: Final = routing.RouterConfig.model_validate({"model_list": (model_list[0], model_list[0])})

    with pytest.raises(ValueError, match="duplicated"):
        routing.reconfigure(router, config=config)
    assert routing.snapshot(router) == before
    routing.close(router)
