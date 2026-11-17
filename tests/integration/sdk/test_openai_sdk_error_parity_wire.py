import asyncio
import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from queue import SimpleQueue
from typing import Final, Literal

import openai
import pytest
from integration._support.openai_wire import chat_reply
from integration._support.ports import allocate_port
from integration._support.wire import Reply, Request, wire_server

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.router import AllowedFailsPolicy, RetryPolicy

MODELS: Final = ("openai/gpt-5.5", "groq/openai/gpt-oss-120b", "hosted_vllm/qwen3-coder")
MODES: Final = ("sync", "async")
REFUSED: Final = None
Mode = Literal["sync", "async"]


@dataclass(frozen=True, slots=True)
class Row:
    status: int | None
    sdk_classes: tuple[type[BaseException], ...]
    kept_classes: tuple[type[BaseException], ...]
    kept_status_code: int
    kept_prefix: str
    logged_class: str

    @property
    def id(self) -> str:
        return "connection" if self.status is None else str(self.status)


ROWS: Final = (
    Row(409, (openai.ConflictError, openai.APIStatusError), (litellm.APIError,), 409, "litellm.APIError: ", "APIError"),
    Row(
        422,
        (openai.UnprocessableEntityError, litellm.UnprocessableEntityError),
        (litellm.BadRequestError, openai.BadRequestError),
        422,
        "litellm.BadRequestError: ",
        "BadRequestError",
    ),
    Row(
        502,
        (openai.InternalServerError,),
        (litellm.BadGatewayError,),
        502,
        "litellm.BadGatewayError: ",
        "BadGatewayError",
    ),
    Row(
        503,
        (openai.InternalServerError,),
        (litellm.ServiceUnavailableError,),
        503,
        "litellm.ServiceUnavailableError: ",
        "ServiceUnavailableError",
    ),
    Row(
        504,
        (openai.InternalServerError, openai.APIStatusError),
        (litellm.Timeout, openai.APITimeoutError),
        504,
        "litellm.Timeout: ",
        "Timeout",
    ),
    Row(
        REFUSED,
        (openai.APIConnectionError,),
        (litellm.InternalServerError, openai.InternalServerError),
        500,
        "litellm.InternalServerError: ",
        "InternalServerError",
    ),
)
ROW_IDS: Final = tuple(row.id for row in ROWS)
INVALID_REQUEST_MODELS: Final = ("openai/gpt-5.5", "groq/openai/gpt-oss-120b")
INVALID_REQUEST_ROWS: Final = (
    Row(
        409,
        (openai.ConflictError,),
        (litellm.BadRequestError, openai.BadRequestError),
        409,
        "litellm.BadRequestError: ",
        "BadRequestError",
    ),
    Row(
        422,
        (openai.UnprocessableEntityError, litellm.UnprocessableEntityError),
        (litellm.BadRequestError, openai.BadRequestError),
        422,
        "litellm.BadRequestError: ",
        "BadRequestError",
    ),
    *(
        Row(
            status,
            (openai.InternalServerError, openai.APIStatusError),
            (litellm.BadRequestError, openai.BadRequestError),
            status,
            "litellm.BadRequestError: ",
            "BadRequestError",
        )
        for status in (502, 503, 504)
    ),
)
INVALID_REQUEST_ROW_IDS: Final = tuple(row.id for row in INVALID_REQUEST_ROWS)
MESSAGES: Final = ({"role": "user", "content": "hi"},)


def _error(status: int) -> Reply:
    return Reply(
        status=status,
        body=json.dumps({"error": {"message": f"upstream {status}", "type": "x", "code": None}}).encode(),
    )


def _invalid_request_error(status: int) -> Reply:
    return Reply(
        status=status,
        body=json.dumps(
            {"error": {"message": f"upstream {status}", "type": "invalid_request_error", "param": None, "code": None}}
        ).encode(),
    )


def _anthropic_error(status: int) -> Reply:
    return Reply(
        status=status,
        body=json.dumps({"type": "error", "error": {"type": "api_error", "message": f"upstream {status}"}}).encode(),
    )


@contextmanager
def _upstream(
    status: int | None, reply: Callable[[int], Reply] = _error, target: str = "/chat/completions"
) -> Iterator[str]:
    if status is None:
        yield f"http://127.0.0.1:{allocate_port()}"
        return
    with wire_server(lambda _request: reply(status)) as wire:
        yield wire.url
        posted: Final = tuple(request.target for request in wire.drain() if request.method == "POST")
    assert posted == (target,), posted


async def _raised(model: str, api_base: str, mode: Mode) -> Exception:
    with pytest.raises(Exception) as caught:
        if mode == "sync":
            litellm.completion(
                model=model, messages=list(MESSAGES), api_base=api_base, api_key="sk-fake-parity", max_retries=0
            )
        else:
            await litellm.acompletion(
                model=model, messages=list(MESSAGES), api_base=api_base, api_key="sk-fake-parity", max_retries=0
            )
    return caught.value


def _qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _assert_instance_of_all(error: Exception, classes: tuple[type[BaseException], ...]) -> None:
    missing: Final = tuple(_qualified(cls) for cls in classes if not isinstance(error, cls))
    lineage: Final = tuple(_qualified(cls) for cls in type(error).__mro__)
    assert missing == (), f"{type(error).__name__} is not {missing}; lineage {lineage}"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("row", ROWS, ids=ROW_IDS)
async def test_regression_error_is_what_the_openai_sdk_raises(row: Row, model: str, mode: Mode) -> None:
    with _upstream(row.status) as api_base:
        error: Final = await _raised(model, api_base, mode)
    _assert_instance_of_all(error, row.sdk_classes)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("row", ROWS, ids=ROW_IDS)
async def test_guard_error_keeps_its_litellm_contract(row: Row, model: str, mode: Mode) -> None:
    with _upstream(row.status) as api_base:
        error: Final = await _raised(model, api_base, mode)
    _assert_instance_of_all(error, row.kept_classes)
    assert getattr(error, "status_code", None) == row.kept_status_code
    assert str(error).startswith(row.kept_prefix), str(error)


class _ErrorClassSink(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.logged: Final[SimpleQueue[str]] = SimpleQueue()

    def _record(self, kwargs: Mapping[str, object]) -> None:
        payload: Final = kwargs["standard_logging_object"]
        assert isinstance(payload, Mapping)
        information: Final = payload["error_information"]
        assert isinstance(information, Mapping)
        self.logged.put(str(information["error_class"]))

    def log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record(kwargs)

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record(kwargs)


async def _logged_error_class(
    row: Row, reply: Callable[[int], Reply], mode: Mode, monkeypatch: pytest.MonkeyPatch
) -> str:
    sink: Final = _ErrorClassSink()
    monkeypatch.setattr(litellm, "callbacks", [sink])
    with _upstream(row.status, reply) as api_base:
        await _raised("openai/gpt-5.5", api_base, mode)
    async with asyncio.timeout(10):
        while sink.logged.empty():
            await asyncio.sleep(0.05)
    return sink.logged.get_nowait()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("row", ROWS, ids=ROW_IDS)
async def test_guard_logged_error_class_label_is_unchanged(
    row: Row, mode: Mode, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert await _logged_error_class(row, _error, mode, monkeypatch) == row.logged_class


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model", INVALID_REQUEST_MODELS)
@pytest.mark.parametrize("row", INVALID_REQUEST_ROWS, ids=INVALID_REQUEST_ROW_IDS)
async def test_regression_invalid_request_error_is_what_the_openai_sdk_raises(row: Row, model: str, mode: Mode) -> None:
    with _upstream(row.status, _invalid_request_error) as api_base:
        error: Final = await _raised(model, api_base, mode)
    _assert_instance_of_all(error, row.sdk_classes)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model", INVALID_REQUEST_MODELS)
@pytest.mark.parametrize("row", INVALID_REQUEST_ROWS, ids=INVALID_REQUEST_ROW_IDS)
async def test_guard_invalid_request_error_keeps_its_litellm_contract(row: Row, model: str, mode: Mode) -> None:
    with _upstream(row.status, _invalid_request_error) as api_base:
        error: Final = await _raised(model, api_base, mode)
    _assert_instance_of_all(error, row.kept_classes)
    assert getattr(error, "status_code", None) == row.kept_status_code
    assert str(error).startswith(row.kept_prefix), str(error)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("row", INVALID_REQUEST_ROWS, ids=INVALID_REQUEST_ROW_IDS)
async def test_guard_invalid_request_logged_error_class_label_is_unchanged(
    row: Row, mode: Mode, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert await _logged_error_class(row, _invalid_request_error, mode, monkeypatch) == row.logged_class


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("status", "kept_class", "prefix"),
    (
        (502, litellm.BadGatewayError, "litellm.BadGatewayError: "),
        (503, litellm.ServiceUnavailableError, "litellm.ServiceUnavailableError: "),
    ),
    ids=("502", "503"),
)
async def test_guard_anthropic_server_error_is_not_an_openai_internal_server_error(
    status: int, kept_class: type[Exception], prefix: str, mode: Mode
) -> None:
    with _upstream(status, _anthropic_error, "/v1/messages") as api_base:
        error: Final = await _raised("anthropic/claude-opus-5", api_base, mode)
    assert not isinstance(error, openai.InternalServerError), _qualified(type(error))
    _assert_instance_of_all(error, (kept_class,))
    assert getattr(error, "status_code", None) == status
    assert str(error).startswith(prefix), str(error)


def _router(api_base: str, **settings: object) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "parity",
                "litellm_params": {
                    "model": "openai/gpt-5.5",
                    "api_key": "sk-fake-parity",
                    "api_base": api_base,
                    "max_retries": 0,
                },
            }
        ],
        num_retries=2,
        retry_after=0,
        **settings,
    )


async def _router_attempts(status: int, reply: Callable[[int], Reply] = _error, **settings: object) -> int:
    with wire_server(lambda _request: reply(status)) as wire:
        with pytest.raises(Exception):
            await _router(wire.url, **settings).acompletion(model="parity", messages=list(MESSAGES))
        return sum(1 for request in wire.drain() if request.method == "POST")


@pytest.mark.parametrize(("status", "attempts"), ((409, 3), (422, 1), (502, 3), (503, 3), (504, 3)))
async def test_guard_router_retry_attempts_reach_the_upstream(status: int, attempts: int) -> None:
    assert await _router_attempts(status) == attempts


@pytest.mark.parametrize("status", (502, 503, 504))
async def test_guard_internal_server_error_retry_policy_does_not_apply(status: int) -> None:
    assert await _router_attempts(status, retry_policy=RetryPolicy(InternalServerErrorRetries=0)) == 3


async def test_guard_timeout_retry_policy_still_applies_to_504() -> None:
    assert await _router_attempts(504, retry_policy=RetryPolicy(TimeoutErrorRetries=0)) == 1


async def test_guard_router_retries_an_invalid_request_typed_502() -> None:
    assert await _router_attempts(502, _invalid_request_error) == 3


def _primary_fails(status: int) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target.startswith("/primary/"):
            return _error(status)
        return chat_reply("chatcmpl-parity-backup", "gpt-5.5", "backup", stream=False)

    return respond


async def _targets_after_three_calls(status: int, policy: AllowedFailsPolicy) -> tuple[str, ...]:
    with wire_server(_primary_fails(status)) as wire:
        router: Final = Router(
            model_list=[
                {
                    "model_name": "parity",
                    "litellm_params": {
                        "model": "openai/gpt-5.5",
                        "api_key": "sk-fake-parity",
                        "api_base": f"{wire.url}/{name}",
                        "max_retries": 0,
                        "order": order,
                    },
                }
                for name, order in (("primary", 1), ("backup", 2))
            ],
            num_retries=0,
            allowed_fails_policy=policy,
        )
        answers: Final = [
            (await router.acompletion(model="parity", messages=list(MESSAGES))).choices[0].message.content
            for _ in range(3)
        ]
        assert answers == ["backup"] * 3
        return tuple(request.target for request in wire.drain() if request.method == "POST")


PRIMARY_KEPT: Final = ("/primary/chat/completions", "/backup/chat/completions") * 3
PRIMARY_COOLED: Final = ("/primary/chat/completions",) + ("/backup/chat/completions",) * 3


async def test_guard_bad_gateway_allowed_fails_policy_cools_down_at_zero() -> None:
    policy: Final = AllowedFailsPolicy(BadGatewayErrorAllowedFails=0)
    assert await _targets_after_three_calls(502, policy) == PRIMARY_COOLED


async def test_guard_bad_gateway_allowed_fails_wins_over_internal_server_error() -> None:
    policy: Final = AllowedFailsPolicy(InternalServerErrorAllowedFails=0, BadGatewayErrorAllowedFails=100)
    assert await _targets_after_three_calls(502, policy) == PRIMARY_KEPT


@pytest.mark.parametrize("status", (502, 503, 504))
async def test_guard_internal_server_error_allowed_fails_does_not_apply(status: int) -> None:
    policy: Final = AllowedFailsPolicy(InternalServerErrorAllowedFails=0)
    assert await _targets_after_three_calls(status, policy) == PRIMARY_KEPT
