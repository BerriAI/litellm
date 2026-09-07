import asyncio
import copy
import gzip
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import Literal
from unittest import TestCase

import httpx

import litellm
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.integrations.custom_guardrail import CustomGuardrail, ModifyResponseException
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.datadog.datadog import DataDogLogger
from litellm.integrations.gcs_bucket.gcs_bucket import GCSBucketLogger
from litellm.integrations.literal_ai import LiteralAILogger
from litellm.integrations.rubrik import RubrikLogger
from litellm.litellm_core_utils.litellm_logging import Logging, create_dummy_standard_logging_payload
from litellm.llms.openai.chat.guardrail_translation.handler import OpenAIChatCompletionsHandler
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.crowdstrike_aidr.crowdstrike_aidr import CrowdStrikeAIDRHandler
from litellm.proxy.guardrails.guardrail_hooks.microsoft_purview.purview_dlp import MicrosoftPurviewDLPGuardrail
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import ModelResponse


async def integration_invoke(owners, callback, *args, **kwargs):
    owner = owners.prepare(callback, args, kwargs, awaited=True)
    pending = owner.invoke()
    owner.close()
    return await pending


def integration_response(url, body, status=200, headers=None):
    return httpx.Response(status, json=body, headers=headers, request=httpx.Request("POST", url))


def integration_callback_scope(scenario):
    @wraps(scenario)
    async def run(owners):
        callbacks = tuple(litellm.callbacks)
        try:
            return await scenario(owners)
        finally:
            litellm.callbacks[:] = callbacks

    return run


@integration_callback_scope
async def real_logging_queue_chain(owners):
    return await integration_logging_queue_case(owners)


@dataclass(frozen=True, slots=True)
class QueueObservation:
    gcs_model_parameters: str
    datadog_snapshot: str
    literal_prepared_settings: str


@integration_callback_scope
async def real_logging_queue_copy_control(owners):
    baseline = await integration_logging_queue_case(owners)
    copied = await integration_logging_queue_case(owners, literal_copy="payload")
    envelope = await integration_logging_queue_case(owners, literal_copy="envelope")
    assert envelope == baseline
    assert json.loads(baseline.gcs_model_parameters) == {"stream": True, "temperature": 0.25}
    assert copied.datadog_snapshot == baseline.datadog_snapshot
    assert copied.literal_prepared_settings == baseline.literal_prepared_settings
    assert json.loads(copied.literal_prepared_settings) == {"stream": True}
    assert json.loads(copied.gcs_model_parameters) == {
        **json.loads(baseline.gcs_model_parameters),
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }
    return copied


def queue_loggers(entered, release, uploads):
    class VertexTransport:
        async def _ensure_access_token_async(self, **kwargs):
            entered.set()
            await release.wait()
            return "fixture-token", "fixture-project"

        def _get_token_and_url(self, **kwargs):
            return kwargs["auth_header"], None

    class Transport:
        async def post(self, url, **kwargs):
            wire = {**kwargs, "json": json.loads(json.dumps(kwargs["json"]))} if "json" in kwargs else kwargs
            uploads.append((url, wire))
            return integration_response(url, {}, 202 if "datadog" in url else 200)

    datadog = DataDogLogger.__new__(DataDogLogger)
    CustomBatchLogger.__init__(datadog, batch_size=100, flush_lock=asyncio.Lock())
    datadog.intake_url, datadog.DD_API_KEY, datadog.is_mock_mode = "https://datadog.invalid/logs", "test", False
    datadog.async_client = Transport()
    gcs = GCSBucketLogger.__new__(GCSBucketLogger)
    CustomBatchLogger.__init__(gcs, batch_size=100)
    gcs.log_queue = asyncio.Queue()
    gcs.BUCKET_NAME, gcs.path_service_account_json = "fixture-bucket", None
    gcs.vertex_instances = {"IAM_AUTH": VertexTransport()}
    gcs.use_batched_logging = True
    gcs.async_httpx_client = Transport()
    literal = LiteralAILogger.__new__(LiteralAILogger)
    CustomBatchLogger.__init__(literal, batch_size=100, flush_lock=asyncio.Lock())
    literal.literalai_api_url, literal.headers = "https://literal.invalid", {}
    literal.async_httpx_client = Transport()
    return datadog, gcs, literal


async def integration_logging_queue_case(owners, *, literal_copy: Literal["direct", "envelope", "payload"] = "direct"):
    entered, release = asyncio.Event(), asyncio.Event()
    uploads = []
    datadog, gcs, literal = queue_loggers(entered, release, uploads)
    copy_payload = literal_copy == "payload"

    class LiteralCallback(CustomLogger):
        def __init__(self, delegate):
            super().__init__()
            self.delegate = delegate
            self.calls = self.completed = 0

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            self.calls += 1
            self.received = kwargs
            self.forwarded = {
                **kwargs,
                "standard_logging_object": (
                    copy.deepcopy(kwargs["standard_logging_object"])
                    if copy_payload
                    else kwargs["standard_logging_object"]
                ),
            }
            await self.delegate.async_log_failure_event(self.forwarded, response_obj, start_time, end_time)
            self.completed += 1

    literal_callback = literal if literal_copy == "direct" else LiteralCallback(literal)

    payload = create_dummy_standard_logging_payload()
    payload.update(status="failure", error_str="x" * 10001)
    messages, settings, metadata = payload["messages"], payload["model_parameters"], payload["metadata"]
    completion = payload["response"]["choices"][0]["message"]
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    settings["tools"] = tools
    now = datetime.now()
    logging = Logging(
        model="fixture-model",
        messages=messages,
        stream=False,
        call_type="acompletion",
        start_time=now,
        litellm_call_id="fixture-queue",
        function_id="fixture",
        dynamic_async_failure_callbacks=[datadog, gcs, literal_callback],
    )
    error = RuntimeError("fixture failure")
    kwargs = logging.model_call_details
    kwargs.update(standard_logging_object=payload, model="fixture-model", exception=error, end_time=now)
    await integration_invoke(owners, logging.async_failure_handler, error, "fixture traceback", now, now)
    if literal_copy != "direct":
        assert literal_callback.calls == literal_callback.completed == 1
        assert literal_callback.received is kwargs and literal_callback.forwarded is not kwargs
        assert (literal_callback.forwarded["standard_logging_object"] is payload) is (not copy_payload)
    assert len(datadog.log_queue) == gcs.log_queue.qsize() == len(literal.log_queue) == 1
    assert kwargs["standard_logging_object"] is payload
    assert payload["messages"] is messages and payload["model_parameters"] is settings
    assert payload["error_str"].endswith("truncated by litellm, this logger does not support large content")
    assert ("tools" in settings) is copy_payload
    dd_snapshot = json.loads(datadog.log_queue[0]["message"])
    assert dd_snapshot["model_parameters"]["tools"] == tools
    queued = gcs.log_queue.get_nowait()
    assert queued["payload"] is payload and queued["kwargs"] is kwargs and queued["response_obj"] is None
    gcs.log_queue.put_nowait(queued)
    generation = literal.log_queue[0]["generation"]
    prepared_settings = json.dumps(generation["settings"], sort_keys=True)
    assert "tools" not in generation["settings"] and generation["tools"] == tools
    if copy_payload:
        assert generation["settings"] is not settings and generation["tools"] is not tools
        assert generation["messages"] is not messages and generation["messageCompletion"] is not completion
        assert literal.log_queue[0]["metadata"] is not metadata
    else:
        assert generation["settings"] is settings and generation["tools"] is tools
        assert generation["messages"] is messages and generation["messageCompletion"] is completion
        assert literal.log_queue[0]["metadata"] is metadata

    flush = asyncio.create_task(integration_invoke(owners, gcs.flush_queue))
    try:
        await entered.wait()
        assert not uploads and not flush.done() and gcs.log_queue.empty()
        messages[0]["content"] = "mutated before serialization"
        settings["temperature"] = 0.25
        completion["content"] = "late completion"
        payload["messages"] = [{"role": "user", "content": "replacement field"}]
        kwargs["standard_logging_object"] = {"replacement": True}
        release.set()
        await flush
    finally:
        release.set()
        if not flush.done():
            flush.cancel()
        await asyncio.gather(flush, return_exceptions=True)
    assert len(uploads) == 1
    gcs_snapshot = json.loads(uploads[0][1]["data"])
    assert gcs_snapshot["messages"] == payload["messages"]
    assert gcs_snapshot["model_parameters"] == settings
    assert "replacement" not in gcs_snapshot
    await integration_invoke(owners, datadog.flush_queue)
    await integration_invoke(owners, literal.flush_queue)
    assert len(uploads) == 3 and not datadog.log_queue and not literal.log_queue
    sent_dd = json.loads(gzip.decompress(uploads[1][1]["data"]))
    assert json.loads(sent_dd[0]["message"]) == dd_snapshot
    literal_wire = uploads[2][1]["json"]
    sent_generation = literal_wire["variables"]["generation_0"]
    assert sent_generation["messages"] != gcs_snapshot["messages"]
    if copy_payload:
        assert sent_generation["messages"] == dd_snapshot["messages"]
        assert json.dumps(sent_generation["settings"], sort_keys=True) == prepared_settings
        assert sent_generation["messageCompletion"] == dd_snapshot["response"]["choices"][0]["message"]
    else:
        assert sent_generation["messages"] == messages
        assert sent_generation["settings"]["temperature"] == 0.25
        assert sent_generation["messageCompletion"]["content"] == "late completion"
    messages[0]["content"] = "after serialization"
    assert sent_generation["messages"][0]["content"] == (
        "Hello, world!" if copy_payload else "mutated before serialization"
    )
    assert dd_snapshot["messages"][0]["content"] == "Hello, world!"
    return QueueObservation(
        gcs_model_parameters=json.dumps(gcs_snapshot["model_parameters"], sort_keys=True),
        datadog_snapshot=json.dumps(dd_snapshot, sort_keys=True),
        literal_prepared_settings=prepared_settings,
    )


@integration_callback_scope
async def real_crowdstrike_translator_identity(owners):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    class Transport:
        async def post(self, url, json, **kwargs):
            calls.append(json)
            entered.set()
            await release.wait()
            return integration_response(
                url,
                {
                    "result": {
                        "blocked": False,
                        "transformed": True,
                        "guard_output": {"messages": [{"role": "user", "content": "redacted"}]},
                    }
                },
            )

    guardrail = CrowdStrikeAIDRHandler.__new__(CrowdStrikeAIDRHandler)
    CustomGuardrail.__init__(guardrail, guardrail_name="fixture-crowdstrike", event_hook=GuardrailEventHooks.pre_call)
    guardrail.api_base, guardrail.api_key, guardrail.fail_on_error = "https://crowdstrike.invalid", "test", True
    guardrail.skip_system_message_in_guardrail = True
    guardrail.async_handler = Transport()
    system = {"role": "system", "content": "internal policy"}
    user = {"role": "user", "content": "private text", "extra": {"retained": True}}
    messages = [system, user]
    data = {"model": "fixture-model", "messages": messages}
    task = asyncio.create_task(
        integration_invoke(
            owners,
            OpenAIChatCompletionsHandler().process_input_messages,
            data,
            guardrail,
        )
    )
    try:
        await entered.wait()
        assert data["messages"] is messages and not task.done()
        assert calls[0]["guard_input"]["messages"] == [{"role": "user", "content": "private text"}]
        user["extra"]["during_http"] = True
        release.set()
        assert await task is data
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert data["messages"] is not messages
    assert data["messages"][0] is system
    assert data["messages"][1] is not user
    assert data["messages"][1]["extra"] is user["extra"]
    assert data["messages"][1]["content"] == "redacted"
    assert user["content"] == "private text" and messages[1] is user
    detached = copy.deepcopy(messages[1:])
    inputs = {"texts": ["private text"], "structured_messages": detached}
    control = await integration_invoke(owners, guardrail.apply_guardrail, inputs, {"messages": messages}, "request")
    assert len(calls) == 2
    assert control["structured_messages"] is detached and detached[0] is not user
    assert control["texts"] == ["redacted"]
    assert detached[0]["content"] == user["content"] == "private text"


@integration_callback_scope
async def real_rubrik_block_lifecycle(owners):
    for input_type, populated in (("request", False), ("response", True)):
        entered, release = asyncio.Event(), asyncio.Event()
        moderation, uploads = [], []

        class Transport:
            def __init__(self, moderation, uploads, entered, release):
                self.moderation, self.uploads = moderation, uploads
                self.entered, self.release = entered, release

            async def post(self, url, json, **kwargs):
                if url.endswith("/batch"):
                    self.uploads.append(json)
                    return integration_response(url, {})
                self.moderation.append(json)
                self.entered.set()
                await self.release.wait()
                return integration_response(url, {"choices": [{"message": {"content": "blocked by policy"}}]})

        rubrik = RubrikLogger.__new__(RubrikLogger)
        CustomGuardrail.__init__(
            rubrik,
            guardrail_name="fixture-rubrik",
            event_hook=GuardrailEventHooks.post_call,
            flush_lock=asyncio.Lock(),
            batch_size=100,
        )
        rubrik._periodic_flush_task = None
        rubrik.sampling_rate, rubrik._headers = 1.0, {}
        rubrik._dropped_since_warning, rubrik._last_drop_warning_time = 0, 0.0
        rubrik.prompt_moderation_endpoint = "https://rubrik.invalid/before"
        rubrik.response_moderation_endpoint = "https://rubrik.invalid/after"
        rubrik.logging_endpoint = "https://rubrik.invalid/batch"
        rubrik.moderation_client = rubrik.async_httpx_client = Transport(moderation, uploads, entered, release)
        other = RubrikLogger.__new__(RubrikLogger)
        CustomGuardrail.__init__(other, guardrail_name="fixture-other-rubrik", event_hook=GuardrailEventHooks.post_call)
        logging = Logging(
            model="fixture-model",
            messages=[{"role": "user", "content": "original prompt"}],
            stream=False,
            call_type="acompletion",
            start_time=datetime.now(),
            litellm_call_id="fixture-correlation",
            function_id="fixture",
        )
        details = logging.model_call_details
        details.update(messages=logging.messages, model="fixture-model", litellm_call_id="fixture-correlation")
        details["system"] = "system scaffold"
        if populated:
            details["standard_logging_object"] = create_dummy_standard_logging_payload()
        messages = details["messages"]
        request = {"model": "fixture-model", "litellm_call_id": "fixture-correlation", "messages": messages}
        inputs = {"texts": ["original response"], "structured_messages": messages}
        success = owners.prepare(rubrik.async_log_success_event, (details, None, None, None), awaited=True)
        task = asyncio.create_task(
            integration_invoke(owners, rubrik.apply_guardrail, inputs, request, input_type, logging)
        )
        try:
            await entered.wait()
            assert not task.done() and "_rubrik_logging_obj" not in request
            assert "_rubrik_blocked" not in details
            if input_type == "request":
                assert moderation[0]["correlation_key"] == "fixture-correlation"
                assert moderation[0]["messages"][0]["content"] == "original prompt"
            else:
                assert moderation[0]["request"]["messages"] is messages
                assert moderation[0]["response"]["id"] == "fixture-correlation"
            release.set()
            with TestCase().assertRaises(ModifyResponseException) as caught:
                await task
            error = caught.exception
            assert error.request_data is request and error.message == "blocked by policy"
            assert request["_rubrik_logging_obj"] is logging and details["_rubrik_blocked"] is True
            await integration_invoke(
                owners, other.async_post_call_failure_hook, request, error, UserAPIKeyAuth(user_id="fixture-user")
            )
            assert request["_rubrik_logging_obj"] is logging and details["_rubrik_blocked"] is True
            assert not other.log_queue and not rubrik.log_queue and not uploads
            await integration_invoke(
                owners, rubrik.async_post_call_failure_hook, request, error, UserAPIKeyAuth(user_id="fixture-user")
            )
            assert "_rubrik_logging_obj" not in request and details["_rubrik_blocked"] is True
            assert len(rubrik.log_queue) == 1
            queued = rubrik.log_queue[0]
            assert queued["id"] == "fixture-correlation"
            assert queued["response"] == "ModifyResponseException: blocked by policy"
            assert queued["messages"][0] == {"role": "system", "content": "system scaffold"}
            assert messages[0] == {"role": "user", "content": "original prompt"}
            if populated:
                base = details["standard_logging_object"]
                assert queued["metadata"] is not base["metadata"]
                assert queued["messages"][1] is not base["messages"][0]
                assert isinstance(base["response"], dict)
            else:
                assert queued["messages"][1] is messages[0]
                assert queued["status"] == "failure"
                assert queued["metadata"]["user_api_key_user_id"] == "fixture-user"
            pending = success.invoke()
            success.close()
            assert await pending is None
            assert len(rubrik.log_queue) == 1 and rubrik.log_queue[0] is queued
            await integration_invoke(owners, rubrik.flush_queue)
            assert not rubrik.log_queue and len(uploads) == 1 and uploads[0][0] is queued
        finally:
            success.close()
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await rubrik.aclose()
            if rubrik._periodic_flush_task is not None:
                await asyncio.gather(rubrik._periodic_flush_task, return_exceptions=True)


@integration_callback_scope
async def real_parallel_guardrail_snapshots(owners):
    original_mode = litellm.safe_memory_mode
    try:
        for safe_memory_mode in (False, True):
            litellm.safe_memory_mode = safe_memory_mode
            await integration_parallel_snapshot_case(owners)
    finally:
        litellm.safe_memory_mode = original_mode


async def integration_parallel_snapshot_case(owners):
    from litellm.caching.dual_cache import DualCache
    from litellm.litellm_core_utils.core_helpers import independent_snapshot
    from litellm.proxy.utils import ProxyLogging

    class Uncopyable:
        def __init__(self):
            self.attempts = 0
            self.observed = []

        def __deepcopy__(self, memo):
            self.attempts += 1
            raise TypeError("fixture cannot be copied")

    arrived, release, mutated = asyncio.Event(), asyncio.Event(), asyncio.Event()
    observations = {}
    sentinel = Uncopyable()
    live = {"messages": [{"role": "user", "content": "original"}], "uncopyable": sentinel}
    raw = independent_snapshot(live)
    assert raw is not live and raw["messages"][0] is not live["messages"][0]
    assert raw["uncopyable"] is sentinel and sentinel.attempts == 1
    live["messages"][0]["content"] = "masked"

    class Inspect(CustomGuardrail):
        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            observations[self.guardrail_name] = data
            if len(observations) == 4:
                arrived.set()
            await release.wait()
            if self.guardrail_name == "fixture-live-writer":
                data["messages"][0]["content"] = "shared mutation"
                mutated.set()
            await mutated.wait()
            if self.scan_raw_request:
                assert data["messages"][0]["content"] == "original"
                data["messages"][0]["content"] = self.guardrail_name
                assert data["uncopyable"] is sentinel
                data["uncopyable"].observed.append(self.guardrail_name)
            else:
                assert data is live and data["messages"][0]["content"] == "shared mutation"
            return {"discarded": self.guardrail_name}

    guardrails = tuple(
        Inspect(
            guardrail_name=name,
            event_hook=GuardrailEventHooks.pre_call,
            default_on=True,
            run_in_parallel=True,
            scan_raw_request="raw" in name,
        )
        for name in ("fixture-live-writer", "fixture-live-reader", "fixture-raw-a", "fixture-raw-b")
    )
    proxy = ProxyLogging.__new__(ProxyLogging)
    proxy.call_details = {"user_api_key_cache": DualCache()}
    task = asyncio.create_task(
        integration_invoke(
            owners, proxy._run_parallel_pre_call_guardrails, guardrails, live, raw, UserAPIKeyAuth(), "acompletion"
        )
    )
    try:
        await arrived.wait()
        assert not task.done()
        assert observations["fixture-live-writer"] is observations["fixture-live-reader"] is live
        first, second = observations["fixture-raw-a"], observations["fixture-raw-b"]
        assert first is not second and first is not raw and second is not raw
        assert first["messages"][0] is not second["messages"][0]
        assert first["messages"][0] is not raw["messages"][0]
        assert first["uncopyable"] is second["uncopyable"] is raw["uncopyable"] is sentinel
        assert sentinel.attempts == 3 and not sentinel.observed
        release.set()
        assert await task is None
        assert raw["messages"] == [{"role": "user", "content": "original"}]
        assert live["messages"][0]["content"] == "shared mutation"
        assert set(sentinel.observed) == {"fixture-raw-a", "fixture-raw-b"} and len(sentinel.observed) == 2
        assert "discarded" not in live
        assert first["messages"][0]["content"] == "fixture-raw-a"
        assert second["messages"][0]["content"] == "fixture-raw-b"
        assert all(guardrail._pre_call_hook_already_ran(live) for guardrail in guardrails if guardrail.scan_raw_request)
    finally:
        release.set()
        mutated.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@integration_callback_scope
async def real_purview_sync_background(owners):
    entered, release = threading.Event(), threading.Event()
    calls, workers = [], []
    main_thread = threading.get_ident()

    class Transport:
        async def post(self, url, **kwargs):
            workers.append(threading.current_thread())
            calls.append((url, kwargs))
            if url.endswith("/token"):
                entered.set()
                assert release.wait(5), "background audit was not released"
                return integration_response(url, {"access_token": "fixture-token", "expires_in": 3600})
            assert kwargs["headers"]["Authorization"] == "Bearer fixture-token"
            if url.endswith("/compute"):
                return integration_response(url, {}, headers={"etag": "fixture-etag"})
            assert url.endswith("/processContent")
            assert kwargs["headers"]["If-None-Match"] == "fixture-etag"
            return integration_response(
                url, {"policyActions": [{"action": "restrictAccess", "restrictionAction": "block"}]}
            )

    purview = MicrosoftPurviewDLPGuardrail.__new__(MicrosoftPurviewDLPGuardrail)
    CustomGuardrail.__init__(purview, guardrail_name="fixture-purview", event_hook=GuardrailEventHooks.logging_only)
    purview.async_handler = Transport()
    purview.tenant_id, purview.client_id, purview.client_secret = "fixture-tenant", "fixture-client", "test"
    purview.purview_app_name, purview.user_id_field, purview.guardrail_provider = (
        "fixture",
        "user_id",
        "microsoft_purview",
    )
    purview._token_cache, purview._scope_cache = None, OrderedDict()
    purview._scope_cache_maxsize, purview._cache_lock = 1000, threading.Lock()
    metadata = {"user_api_key_user_id": "fixture-user"}
    kwargs = {
        "messages": [{"role": "user", "content": "prompt at dispatch"}],
        "litellm_params": {"metadata": metadata},
        "litellm_call_id": "before-http",
    }
    result = ModelResponse(model="fixture-model", choices=[{"message": {"role": "assistant", "content": "before"}}])
    owner = owners.prepare(purview.logging_hook, (kwargs, result, "completion"), awaited=False)
    try:
        returned = await asyncio.to_thread(owner.invoke)
        owner.close()
        assert returned[0] is kwargs and returned[1] is result
        assert await asyncio.to_thread(entered.wait, 5)
        assert len(calls) == 1 and workers[0].ident != main_thread and workers[0].daemon
        assert workers[0].is_alive()
        kwargs["messages"][0]["content"] = "too late for prompt extraction"
        kwargs["litellm_call_id"] = "after-http"
        result.choices[0].message.content = "response mutated while audit waits"
        release.set()
        await asyncio.to_thread(workers[0].join, 5)
        assert not workers[0].is_alive() and all(worker is workers[0] for worker in workers)
        assert len(calls) == 4
        entries = [call[1]["json"]["contentToProcess"] for call in calls[2:]]
        assert [entry["activityMetadata"]["activity"] for entry in entries] == ["uploadText", "downloadText"]
        assert entries[0]["contentEntries"][0]["content"]["data"] == "prompt at dispatch"
        assert entries[1]["contentEntries"][0]["content"]["data"] == "response mutated while audit waits"
        assert all(entry["contentEntries"][0]["correlationId"] == "after-http" for entry in entries)
        assert kwargs["litellm_params"]["metadata"] is metadata
        info = kwargs["metadata"]["standard_logging_guardrail_information"]
        assert len(info) == 2 and all(item["guardrail_status"] == "guardrail_intervened" for item in info)
        assert all(item["end_time"] >= item["start_time"] and item["duration"] >= 0 for item in info)
    finally:
        owner.close()
        release.set()
        if workers:
            await asyncio.to_thread(workers[0].join, 5)
