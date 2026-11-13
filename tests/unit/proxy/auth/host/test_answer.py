import re
from typing import Final, NoReturn, TypeGuard

import pytest

from litellm.proxy._types import LiteLLM_JWTAuth
from litellm.proxy.auth import core
from litellm.proxy.auth.host.answer import answer
from litellm.proxy.auth.host.python import PythonAuthHost
from litellm.rust_bridge._native import RustAuth, RustAuthSettings, RustAuthStep

_FIELD_VALUES: Final = {
    "upsert": True,
    "explicit_team": True,
    "ttl_seconds": 1.0,
    "add": ["a"],
    "remove": ["r"],
    "allowed_team_ids": ["t"],
    "membership": None,
    "owner": None,
    "is_llm_api_route": True,
}


def _is_step(value: object) -> TypeGuard[type[RustAuthStep]]:
    return isinstance(value, type) and issubclass(value, RustAuthStep) and value is not RustAuthStep.Done


_ATTRIBUTES: Final[tuple[object, ...]] = tuple(vars(RustAuthStep).values())
STEPS: Final = tuple(value for value in _ATTRIBUTES if _is_step(value))


class _Asked(Exception):
    def __init__(self, method: str) -> None:
        super().__init__(method)
        self.method: Final = method


class _AskingHost:
    """Fails on the first method `answer` reaches for, naming it."""

    def __getattr__(self, method: str) -> object:
        async def ask(*args: object) -> NoReturn:
            raise _Asked(method)

        return ask


def _step(variant: type[RustAuthStep]) -> RustAuthStep:
    fields: Final[tuple[str, ...]] = getattr(variant, "__match_args__", ())
    return variant(**{field: _FIELD_VALUES.get(field, "x") for field in fields})


def _call() -> RustAuth:
    settings: Final = RustAuthSettings(
        {"enable_jwt_auth": True}, LiteLLM_JWTAuth(), core.HostFacts(master_key=None, premium=True, has_db=False)
    )
    return RustAuth.verify_jwt("token", 0.0, settings)


def test_the_steps_are_found_on_the_native_class() -> None:
    assert RustAuthStep.CachedKey in STEPS


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", STEPS, ids=[step.__name__ for step in STEPS])
async def test_every_step_the_core_parks_on_asks_the_python_host_method_of_its_name(
    variant: type[RustAuthStep],
) -> None:
    method: Final = re.sub(r"(?<!^)(?=[A-Z])", "_", variant.__name__).lower()

    with pytest.raises(_Asked) as asked:
        await answer(
            _call(),
            _step(variant),
            _AskingHost(),  # pyright: ignore[reportArgumentType]  # records which host method answer asks
        )

    assert asked.value.method == method
    assert callable(getattr(PythonAuthHost, method, None))
