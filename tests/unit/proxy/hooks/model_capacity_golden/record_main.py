import asyncio
import subprocess
import sys
import types
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

import pytest

MAIN: Final = "cd2e4b6b60"
ROOT: Final = Path(__file__).resolve().parents[5]
NODE: Final = "tests/unit/proxy/hooks/test_model_capacity_native.py::"
MODULES: Final = (
    ("litellm.proxy.hooks.parallel_request_limiter_v3", "litellm/proxy/hooks/parallel_request_limiter_v3.py"),
    ("litellm.proxy.hooks.dynamic_rate_limiter_v3", "litellm/proxy/hooks/dynamic_rate_limiter_v3.py"),
)


def _install(name: str, path: str) -> None:
    source: Final = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{MAIN}:{path}"], check=True, capture_output=True, text=True
    ).stdout
    module: Final = types.ModuleType(name)
    module.__file__ = str(ROOT / path)
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)


async def _record() -> dict[str, object]:
    import litellm.proxy.hooks

    for name, path in MODULES:
        _install(name, path)

    from litellm import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.hooks.dynamic_rate_limiter_v3 import PROXY_DynamicRateLimitHandlerV3
    from tests.unit.proxy.hooks.model_capacity_golden import golden, scenarios

    @runtime_checkable
    class PythonHandler(scenarios.CapacityHandler, Protocol):
        async def async_pre_call_hook(
            self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: str
        ) -> object: ...

    def build(cache: DualCache) -> scenarios.CapacityHandler:
        handler: Final[object] = PROXY_DynamicRateLimitHandlerV3(cache)
        assert isinstance(handler, PythonHandler), litellm.proxy.hooks.__name__
        return handler

    async def pre_call(handler: scenarios.CapacityHandler, auth: UserAPIKeyAuth, data: dict[str, object]) -> object:
        assert isinstance(handler, PythonHandler)
        return await handler.async_pre_call_hook(auth, DualCache(), data, "acompletion")

    recorded: Final[dict[str, object]] = {}  # mutable-ok: each scenario's recorded python value
    for name, scenario in scenarios.SCENARIOS.items():
        with pytest.MonkeyPatch.context() as monkeypatch:
            recorded[NODE + name] = [golden.encode(await scenario(monkeypatch, build, pre_call))]
    return recorded


def main() -> int:
    from tests.unit.proxy.hooks.model_capacity_golden import golden

    golden.record(asyncio.run(_record()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
