"""
Router behavior the router-code differential fuzz run found changed once the Rust core became the
only core: count settings below zero, an allowed_fails_policy write that is no policy, a model
response object as a deployment's mock_response, and endpoints read or patched on the Router class.
"""

import asyncio
import gc
import json
import socket
import threading
import weakref
from collections.abc import Iterator, Mapping
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final
from unittest import mock

import pytest

import litellm
from litellm import Router
from litellm.router import RetryPolicy
from litellm.types.router import Deployment
from litellm.types.utils import ModelResponse

MESSAGES: Final = [{"role": "user", "content": "hi"}]


def _deployment(name: str, reply: object) -> dict[str, object]:
    return {"model_name": name, "litellm_params": {"model": "openai/gpt-4o", "api_key": "k", "mock_response": reply}}


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", ["num_retries", "max_fallbacks", "allowed_fails"])
async def test_a_negative_count_setting_builds_and_serves(setting: str) -> None:
    router: Final = Router(model_list=[_deployment("g", "ok")], **{setting: -1})

    response: Final = await router.acompletion(model="g", messages=MESSAGES)

    assert response.choices[0].message.content == "ok"


@pytest.mark.asyncio
async def test_a_negative_max_fallbacks_takes_no_fallback() -> None:
    router: Final = Router(
        model_list=[_deployment("g", Exception("boom")), _deployment("fb", "fb")],
        fallbacks=[{"g": ["fb"]}],
        max_fallbacks=-1,
        num_retries=0,
    )

    with pytest.raises(litellm.InternalServerError):
        await router.acompletion(model="g", messages=MESSAGES)


@pytest.mark.asyncio
async def test_a_negative_num_retries_still_falls_back() -> None:
    router: Final = Router(
        model_list=[_deployment("g", Exception("boom")), _deployment("fb", "fb")],
        fallbacks=[{"g": ["fb"]}],
        num_retries=-1,
    )

    response: Final = await router.acompletion(model="g", messages=MESSAGES)

    assert response.choices[0].message.content == "fb"


@pytest.mark.asyncio
async def test_an_allowed_fails_policy_written_as_a_list_reads_as_no_policy() -> None:
    router: Final = Router(model_list=[_deployment("g", "ok")])

    with pytest.warns(DeprecationWarning, match="allowed_fails_policy"):
        router.allowed_fails_policy = []  # pyright: ignore[reportAttributeAccessIssue]  # the old attribute write
    response: Final = await router.acompletion(model="g", messages=MESSAGES)

    assert response.choices[0].message.content == "ok"
    assert router.settings().allowed_fails_policy == []


def test_a_model_response_mock_reads_back_from_the_model_list_as_a_dict() -> None:
    reply: Final = ModelResponse(id="chatcmpl-fixed", choices=[{"message": {"role": "assistant", "content": "fb"}}])
    router: Final = Router(model_list=[_deployment("g", reply)])

    params: Final = router.get_model_list()[0]["litellm_params"]

    assert isinstance(params["mock_response"], dict)
    assert params["mock_response"]["id"] == "chatcmpl-fixed"


def test_an_endpoint_the_core_serves_by_name_is_readable_on_the_class() -> None:
    endpoint: Final = Router.aretrieve_batch

    assert callable(endpoint)
    assert endpoint is Router.aretrieve_batch
    assert endpoint.__name__ == "aretrieve_batch"


@pytest.mark.asyncio
async def test_a_class_level_patch_of_an_endpoint_reaches_the_instance_and_unwinds() -> None:
    router: Final = Router(model_list=[_deployment("g", "ok")])

    with mock.patch.object(Router, "asearch", new=mock.AsyncMock(return_value="patched")) as patched:
        assert await router.asearch(query="q", search_tool_name="t") == "patched"
    patched.assert_awaited_once()

    assert router.asearch is not patched


def test_an_assigned_affinity_ttl_builds_the_affinity_check_added_afterwards() -> None:
    from litellm.router_utils.pre_call_checks.deployment_affinity_check import DeploymentAffinityCheck

    router: Final = Router(model_list=[_deployment("g", "ok")])

    router.deployment_affinity_ttl_seconds = 7
    router.add_optional_pre_call_checks(["deployment_affinity"])

    checks: Final = [c for c in litellm.callbacks if isinstance(c, DeploymentAffinityCheck)]
    assert [check.ttl_seconds for check in checks][-1:] == [7]


@pytest.mark.asyncio
async def test_a_deleted_timeout_leaves_get_settings_and_still_serves() -> None:
    router: Final = Router(model_list=[_deployment("g", "ok")], num_retries=0)

    del router.timeout

    assert "timeout" not in router.get_settings()
    response: Final = await router.acompletion(model="g", messages=MESSAGES)
    assert response.choices[0].message.content == "ok"


def test_litellm_params_json_cannot_carry_read_back_as_given_and_hash_like_python() -> None:
    from datetime import datetime

    from litellm.router_api import deployment_id

    params: Final = {"model": "openai/gpt-4o", "api_key": "k", "cfg": {"s": {"a"}}, "when": datetime(2026, 1, 1, 12)}
    router: Final = Router(model_list=[{"model_name": "g", "litellm_params": dict(params)}])

    served: Final = router.get_model_list()[0]

    assert served["litellm_params"]["cfg"] == {"s": {"a"}}
    assert served["litellm_params"]["when"] == datetime(2026, 1, 1, 12)
    assert served["model_info"]["id"] == deployment_id("g", params)


def test_a_bad_deployment_before_a_secret_reference_raises_its_own_validation_error() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Router(
            model_list=[
                {"model_name": "t", "litellm_params": {"api_key": "k"}},
                {"model_name": "u", "litellm_params": {"model": "openai/gpt-4o", "api_key": "os.environ/NOPE_KEY"}},
            ]
        )


@pytest.mark.asyncio
async def test_a_failed_vector_store_create_names_no_failed_deployment() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "vs",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "k", "api_base": "http://127.0.0.1:9/v1"},
            }
        ],
        num_retries=0,
    )

    with pytest.raises(Exception) as raised:  # noqa: PT011  # the provider's connection error, whatever class maps it
        await router.avector_store_create(model="vs", name="x")

    assert getattr(raised.value, "failed_deployment_id", None) is None


class _Upstream(BaseHTTPRequestHandler):
    """An OpenAI upstream whose path's first segment names how it answers: ``cut`` sends two
    Responses stream events then cuts the chunked body, ``filtered`` answers a chat completion the
    provider blocked, ``down`` answers 502, and anything else a plain chat completion."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002  # the base class's name
        del format, args

    def _json(self, status: int, body: Mapping[str, object]) -> None:
        data: Final = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        _ = self.wfile.write(data)

    def _chunk(self, event: Mapping[str, object]) -> None:
        data: Final = f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
        _ = self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _cut(self) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        response: Final = {"id": "r1", "object": "response", "created_at": 1, "status": "in_progress", "output": []}
        self._chunk({"type": "response.created", "sequence_number": 0, "response": {**response, "model": "gpt-4o"}})
        self._chunk(
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": "m1",
                "output_index": 0,
                "content_index": 0,
                "delta": "hello ",
            }
        )
        self.connection.shutdown(socket.SHUT_RDWR)
        self.close_connection = True

    def do_POST(self) -> None:
        _ = self.rfile.read(int(self.headers.get("content-length") or 0))
        mode: Final = self.path.split("/")[1]
        if mode == "cut":
            self._cut()
            return
        if mode == "down":
            self._json(502, {"error": {"message": "provider error", "type": "api_error"}})
            return
        finish: Final = "content_filter" if mode == "filtered" else "stop"
        message: Final = {"role": "assistant", "content": mode}
        choice: Final = {"index": 0, "finish_reason": finish, "message": message}
        usage: Final = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        self._json(200, {"id": "c1", "object": "chat.completion", "created": 1, "choices": [choice], "usage": usage})


@pytest.fixture
def upstream() -> Iterator[str]:
    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join()


def _served(name: str, deployment_id: str, model: str, api_base: str) -> dict[str, object]:
    return {
        "model_name": name,
        "litellm_params": {"model": model, "api_key": "k", "api_base": api_base},
        "model_info": {"id": deployment_id},
    }


def _request_logging(call_type: str, *, stream: bool) -> object:
    from litellm.litellm_core_utils.litellm_logging import Logging

    return Logging(
        model="g",
        messages=MESSAGES,
        stream=stream,
        call_type=call_type,
        start_time=datetime.now(),  # noqa: DTZ005  # litellm's own naive start time
        litellm_call_id="call-1",
        function_id="f",
    )


async def _recorded() -> None:
    """Let the background recordings of a stream's outcome run."""
    for _ in range(50):
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_a_responses_stream_cut_after_it_opened_counts_one_fail_with_a_shared_logging_object(
    upstream: str,
) -> None:
    router: Final = Router(
        model_list=[_served("g", "d1", "openai/gpt-4o", f"{upstream}/cut/v1")], num_retries=0, disable_cooldowns=True
    )

    stream: Final = await router.aresponses(
        model="g", input="hi", stream=True, litellm_logging_obj=_request_logging("aresponses", stream=True)
    )
    with pytest.raises(Exception):  # noqa: B017, PT011  # the transport's read error, whatever class maps it
        async for _ in stream:  # pyright: ignore[reportGeneralTypeIssues]  # a stream when stream=True
            pass
    await _recorded()

    assert router.health().store.get_cache(key="d1:fails") == 1


@pytest.mark.asyncio
async def test_a_blocked_response_raised_for_a_fallback_counts_no_success_with_a_shared_logging_object(
    upstream: str,
) -> None:
    router: Final = Router(
        model_list=[
            _served("a", "a1", "openai/gpt-4o", f"{upstream}/filtered/v1"),
            _served("safe", "s1", "openai/gpt-4o", f"{upstream}/ok/v1"),
        ],
        content_policy_fallbacks=[{"a": ["safe"]}],
        num_retries=0,
        disable_cooldowns=True,
    )

    response: Final = await router.acompletion(
        model="a", messages=MESSAGES, litellm_logging_obj=_request_logging("acompletion", stream=False)
    )
    await _recorded()

    assert response.choices[0].message.content == "ok"
    assert router.health().store.get_cache(key="a1:successes") is None
    assert router.health().store.get_cache(key="s1:successes") == 1


@pytest.mark.asyncio
async def test_a_text_completion_stream_that_fails_before_its_first_chunk_counts_nothing(upstream: str) -> None:
    router: Final = Router(
        model_list=[_served("g", "d1", "openai/gpt-3.5-turbo-instruct", f"{upstream}/down/v1")],
        num_retries=0,
        disable_cooldowns=True,
    )

    stream: Final = await router.atext_completion(model="g", prompt="hi", stream=True)
    with pytest.raises(litellm.BadGatewayError):
        async for _ in stream:  # pyright: ignore[reportGeneralTypeIssues]  # a stream when stream=True
            pass
    await _recorded()

    assert router.health().store.get_cache(key="d1:successes") is None
    assert router.health().store.get_cache(key="d1:fails") is None


@pytest.mark.asyncio
async def test_a_text_completion_without_a_prompt_raises_type_error_before_routing() -> None:
    router: Final = Router(model_list=[_deployment("g", "ok")], num_retries=0)

    with pytest.raises(TypeError, match="'prompt'"):
        await router.atext_completion(model="g")  # pyright: ignore[reportCallIssue]  # the missing argument
    with pytest.raises(TypeError, match="'prompt'"):
        router.text_completion(model="g")  # pyright: ignore[reportCallIssue]  # the missing argument


@pytest.mark.asyncio
async def test_a_default_priority_assigned_after_construction_reads_back_and_queues_calls() -> None:
    router: Final = Router(model_list=[_deployment("g", "ok")], num_retries=0)

    router.default_priority = 5
    response: Final = await router.acompletion(model="g", messages=MESSAGES)

    assert router.default_priority == 5
    assert response._hidden_params["additional_headers"]["x-litellm-request-prioritization-used"] is True


@pytest.mark.asyncio
async def test_a_prioritized_call_on_an_unknown_group_raises_the_calls_own_error() -> None:
    router: Final = Router(model_list=[_deployment("g", "ok")], num_retries=0)

    with pytest.raises(litellm.InternalServerError):
        await router.schedule_acompletion(
            model="nope", messages=MESSAGES, priority=0, mock_testing_fallbacks=True, num_retries=0
        )


def test_a_registry_lookup_of_a_record_with_no_model_name_raises_key_error() -> None:
    router: Final = Router(model_list=[_deployment("g", "ok")])

    with pytest.raises(KeyError):
        router.registry().includes({"litellm_params": {}}, model_name="g")  # pyright: ignore[reportArgumentType]  # the record with no name


@pytest.mark.asyncio
async def test_a_refused_pick_by_deployment_id_names_the_deployments_group() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "t2-pub",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "k", "mock_response": "p2"},
                "model_info": {"team_id": "T2", "team_public_model_name": "pub", "id": "p2"},
            }
        ],
        num_retries=0,
    )

    with pytest.raises(Exception, match="model=t2-pub"):  # noqa: PT011  # the refusal, whatever class names it
        await router.async_get_available_deployment(model="p2", messages=MESSAGES, request_kwargs={})


@pytest.mark.asyncio
async def test_a_request_refused_before_the_provider_counts_against_no_deployment(upstream: str) -> None:
    router: Final = Router(
        model_list=[_served("g", "d1", "openai/gpt-4o", f"{upstream}/ok/v1")], num_retries=0, disable_cooldowns=True
    )

    with pytest.raises(litellm.UnsupportedParamsError):
        await router.acompletion(model="g", messages=MESSAGES, reasoning_effort="low")
    await _recorded()

    assert router.health().store.get_cache(key="d1:fails") is None
    assert not [key for key in router.health().store.in_memory_cache.cache_dict if ":rpm" in key]


@pytest.mark.asyncio
async def test_the_async_pick_read_on_the_class_is_a_function_taking_the_router() -> None:
    router: Final = Router(model_list=[_deployment("g", "ok")])

    pick: Final = Router.async_get_available_deployment

    assert pick.__name__ == "async_get_available_deployment"
    assert (await pick(router, "g", {}))["model_name"] == "g"


def _sharing_ids(*ids: str) -> Router:
    return Router(
        model_list=[{**_deployment("g", f"from-{index}"), "model_info": {"id": id}} for index, id in enumerate(ids)]
    )


@pytest.mark.asyncio
async def test_async_picks_split_over_deployments_sharing_an_id() -> None:
    router: Final = _sharing_ids("d1", "d1", "b")

    served: Final = {
        (await router.acompletion(model="g", messages=MESSAGES)).choices[0].message.content for _ in range(60)
    }

    assert served == {"from-0", "from-1", "from-2"}


def test_sync_picks_split_over_deployments_sharing_an_id() -> None:
    router: Final = _sharing_ids("d1", "d1")

    served: Final = {router.completion(model="g", messages=MESSAGES).choices[0].message.content for _ in range(40)}

    assert served == {"from-0", "from-1"}


@pytest.mark.asyncio
async def test_no_model_is_served_by_the_first_default_fallback() -> None:
    router: Final = Router(model_list=[_deployment("g", "ok")], default_fallbacks=["g"])

    async_served: Final = await router.acompletion(model=None, messages=MESSAGES)
    sync_served: Final = router.completion(model=None, messages=MESSAGES)

    assert (async_served.choices[0].message.content, sync_served.choices[0].message.content) == ("ok", "ok")


@pytest.mark.asyncio
async def test_no_model_is_served_by_a_catch_all_wildcard() -> None:
    router: Final = Router(
        model_list=[
            _deployment("g", "ok"),
            {
                **_deployment("*", "star"),
                "litellm_params": {"model": "openai/*", "api_key": "k", "mock_response": "star"},
            },
        ]
    )

    served: Final = await router.acompletion(model=None, messages=MESSAGES)

    assert served.choices[0].message.content == "star"


@pytest.mark.asyncio
async def test_no_model_is_retried_by_the_retry_policy_and_raised_without_a_fallback_note() -> None:
    router: Final = Router(
        model_list=[_deployment("g", "ok")],
        default_fallbacks=["h"],
        retry_after=0,
        retry_policy=RetryPolicy(BadRequestErrorRetries=1),
    )

    with pytest.raises(litellm.BadRequestError) as raised:
        await router.acompletion(model=None, messages=MESSAGES)

    assert raised.value.num_retries == 1
    assert "You passed in model=h. There are no healthy deployments for this model" in str(raised.value)
    assert "failed with the error above" not in str(raised.value)


class _TokenProvider:
    def __call__(self) -> str:
        return "token"


def test_a_deleted_deployment_releases_the_callable_it_was_built_with() -> None:
    router: Final = Router(model_list=[_deployment("g", "ok")])
    provider = _TokenProvider()
    released: Final = weakref.ref(provider)
    router.add_deployment(
        Deployment(
            model_name="g",
            litellm_params={"model": "openai/gpt-4o", "api_key": "k", "azure_ad_token_provider": provider},
            model_info={"id": "rotated"},
        )
    )
    del provider

    router.delete_deployment("rotated")
    gc.collect()

    assert released() is None


@pytest.mark.asyncio
async def test_a_pick_keeps_model_info_keys_that_are_not_strings() -> None:
    router: Final = Router(model_list=[{**_deployment("g", "ok"), "model_info": {"id": "d1", "tiers": {1: "a"}}}])

    picked: Final = await router.async_get_available_deployment(model="g", messages=MESSAGES, request_kwargs={})

    assert picked["model_info"]["tiers"] == {1: "a"}
