import datetime
from collections.abc import Iterator
from typing import Final

import pytest

import litellm
from litellm import utils
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.rust_bridge.loader import native_bridge_available

pytestmark = pytest.mark.skipif(not native_bridge_available(), reason="requires the Rust extension")

REGISTRIES: Final = (
    "input_callback",
    "_async_input_callback",
    "success_callback",
    "_async_success_callback",
    "failure_callback",
    "_async_failure_callback",
    "callbacks",
)


@pytest.fixture
def clean_registries(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in REGISTRIES:
        monkeypatch.setattr(litellm, name, [])
    monkeypatch.setattr(utils, "callback_list", [])
    monkeypatch.setattr(utils, "user_logger_fn", None)
    monkeypatch.setenv("OPENMETER_API_KEY", "test")
    monkeypatch.setenv("OPENMETER_API_ENDPOINT", "http://127.0.0.1:9")
    yield


class SyncLogger(CustomLogger):
    pass


class AsyncOnly(CustomLogger):
    pass


def sync_fn(*args: object, **kwargs: object) -> None:
    del args, kwargs


async def async_fn(*args: object, **kwargs: object) -> None:
    del args, kwargs


def snapshot() -> dict[str, list[object]]:
    return {name: list(getattr(litellm, name)) for name in REGISTRIES}


def run_legacy(kwargs: dict[str, object]) -> tuple[Logging, dict[str, object]]:
    logger, prepared = utils.function_setup(
        "ocr",
        utils.Rules(),
        datetime.datetime.now(),
        is_async_call=False,
        **{"litellm_call_id": "legacy", **kwargs},
    )
    assert isinstance(logger, Logging)
    return logger, prepared


def run_native(kwargs: dict[str, object]) -> tuple[Logging, dict[str, object]]:
    from litellm.rust_bridge import _native

    logger, prepared = _native._debug_setup(
        "ocr", (), {"litellm_call_id": "native", **kwargs}, datetime.datetime.now(), False
    )
    assert isinstance(logger, Logging)
    return logger, prepared


@pytest.mark.parametrize(
    "globals_before, kwargs",
    [
        ({}, {}),
        ({"callbacks": [SyncLogger()]}, {}),
        ({"callbacks": [async_fn]}, {}),
        ({}, {"callbacks": [SyncLogger(), sync_fn]}),
        ({"success_callback": [async_fn, "openmeter", sync_fn]}, {}),
        ({"failure_callback": [async_fn, sync_fn]}, {}),
        ({"input_callback": [async_fn, sync_fn]}, {}),
        ({}, {"success_callback": [sync_fn, async_fn, "s3", "dynamodb"], "failure_callback": [sync_fn]}),
        ({"callbacks": [SyncLogger()]}, {"callbacks": [SyncLogger()], "success_callback": [async_fn]}),
    ],
    ids=[
        "empty",
        "global-custom-logger",
        "global-async-callable",
        "dynamic-callbacks",
        "success-safety-net",
        "failure-safety-net",
        "input-safety-net",
        "per-call-success-failure-split",
        "mixed",
    ],
)
def test_native_setup_registry_side_effects_match_function_setup(
    clean_registries: None, globals_before: dict[str, list[object]], kwargs: dict[str, object]
) -> None:
    for name, values in globals_before.items():
        getattr(litellm, name).extend(values)
    legacy_logger, legacy_kwargs = run_legacy(
        {key: list(value) if isinstance(value, list) else value for key, value in kwargs.items()}
    )
    legacy_snapshot: Final = snapshot()
    legacy_bootstrap: Final = list(utils.callback_list or [])

    for name in REGISTRIES:
        getattr(litellm, name).clear()
    utils.callback_list = []
    for name, values in globals_before.items():
        getattr(litellm, name).extend(values)
    native_logger, native_kwargs = run_native(
        {key: list(value) if isinstance(value, list) else value for key, value in kwargs.items()}
    )

    assert snapshot() == legacy_snapshot
    assert sorted(map(repr, utils.callback_list or [])) == sorted(map(repr, legacy_bootstrap))
    assert set(native_kwargs) - {"litellm_call_id"} == set(legacy_kwargs) - {"litellm_call_id"}
    for attribute in (
        "dynamic_success_callbacks",
        "dynamic_async_success_callbacks",
        "dynamic_failure_callbacks",
        "dynamic_async_failure_callbacks",
        "call_type",
        "stream",
        "model",
    ):
        assert getattr(native_logger, attribute) == getattr(legacy_logger, attribute), attribute


def test_native_setup_honours_caller_supplied_logging_object(clean_registries: None) -> None:
    class Supplied(Logging):
        pass

    supplied: Final = Supplied(
        model="mistral/mistral-ocr-latest",
        messages=[],
        stream=False,
        call_type="ocr",
        start_time=datetime.datetime.now(),
        litellm_call_id="supplied",
        function_id="",
    )
    litellm.callbacks.append(SyncLogger())
    logger, kwargs = run_native({"litellm_logging_obj": supplied, "callbacks": [SyncLogger()]})
    assert logger is supplied
    assert kwargs["callbacks"] is not None
    assert litellm.success_callback == []


def test_native_setup_records_logger_fn_and_metadata(clean_registries: None) -> None:
    def logger_fn(details: object) -> None:
        del details

    logger, kwargs = run_native(
        {"model": "mistral/mistral-ocr-latest", "logger_fn": logger_fn, "metadata": {"source": "test"}}
    )
    assert utils.user_logger_fn is logger_fn
    assert logger.litellm_params["metadata"] == {"source": "test"}
    assert logger.model == "mistral/mistral-ocr-latest"
    assert kwargs["metadata"] == {"source": "test"}
