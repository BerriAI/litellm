import asyncio
import json
from typing import Final

import pytest
from fastapi import HTTPException

import litellm
from litellm.integrations.custom_guardrail import ModifyResponseException
from litellm.integrations.literal_ai import LiteralAILogger
from litellm.integrations.rubrik import RubrikLogger
from litellm.proxy.guardrails.guardrail_hooks.azure.text_moderation import AzureContentSafetyTextModerationGuardrail
from litellm.proxy.guardrails.guardrail_hooks.crowdstrike_aidr.crowdstrike_aidr import CrowdStrikeAIDRHandler
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import ContentFilterGuardrail
from litellm.rust_bridge.provenance import has_rust_response_marker
from litellm.types.guardrails import BlockedWord, ContentFilterAction, GuardrailEventHooks
from litellm.types.utils import CallTypes
from tests.test_litellm_rust.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.conftest import Backend, isolated_backend
from tests.test_litellm_rust.contracts import CHAT_MESSAGES, CHAT_MODEL, CHAT_RESPONSE
from tests.test_litellm_rust.integrations import (
    MESSAGES_ROUTE,
    NON_STREAM_ASYNC_ROUTES,
    GenericAPIExportHarness,
    OtelHarness,
    ReviewGuardrail,
    Route,
    provider_response,
    purview_harness,
    route_id,
    wait_for_audits,
    wait_for_callback,
)
from tests.test_litellm_rust.recording_server import RecordingServer, ResponseSpec, recording_service

pytestmark = pytest.mark.requires_rust_extension

AZURE_MODERATION_ALLOW_RESPONSE: Final = {
    "blocklistsMatch": [],
    "categoriesAnalysis": [
        {"category": "Hate", "severity": 0},
        {"category": "Sexual", "severity": 0},
        {"category": "SelfHarm", "severity": 0},
        {"category": "Violence", "severity": 0},
    ],
}
AZURE_MODERATION_BLOCK_RESPONSE: Final = {
    **AZURE_MODERATION_ALLOW_RESPONSE,
    "categoriesAnalysis": [{"category": "Violence", "severity": 6}],
}


def azure_text_moderation(server: RecordingServer) -> AzureContentSafetyTextModerationGuardrail:
    return AzureContentSafetyTextModerationGuardrail(
        guardrail_name="azure-text-review",
        api_key="test-azure-key",
        api_base=server.base_url,
        event_hook=GuardrailEventHooks.post_call,
    )


@pytest.mark.asyncio
async def test_crowdstrike_redaction_reaches_native_chat_provider_and_exporter(
    generic_api_export: GenericAPIExportHarness,
) -> None:
    with recording_service() as provider, recording_service() as guard:
        provider.default_response = ResponseSpec(body=CHAT_RESPONSE)
        guard.default_response = ResponseSpec(
            body={
                "result": {
                    "blocked": False,
                    "transformed": True,
                    "guard_output": {
                        "messages": [
                            {"role": "system", "content": "Keep the answer short"},
                            {"role": "user", "content": "Employee SSN: <US_SSN>"},
                        ]
                    },
                }
            }
        )
        guardrail: Final = CrowdStrikeAIDRHandler(
            guardrail_name="crowdstrike-redaction",
            api_key="test-crowdstrike-key",
            api_base=guard.base_url,
            event_hook=GuardrailEventHooks.pre_call,
        )
        recorder: Final = RecordingLogger()
        litellm.callbacks.append(guardrail)

        response: Final = await litellm.acompletion(
            model=CHAT_MODEL,
            messages=CHAT_MESSAGES,
            api_key="test-key",
            api_base=provider.base_url,
            callbacks=[generic_api_export.logger, recorder],
            guardrails=[guardrail.guardrail_name],
        )
        await recorder.wait_for_async("async_log_success_event")

        provider_messages: Final = provider.requests[0].body["messages"]
        guard_messages: Final = guard.requests[0].body["guard_input"]["messages"]
        exported_messages: Final = generic_api_export.exports[0].body["messages"]
        assert response.choices[0].message.content == "Handled safely"
        assert guard_messages == [CHAT_MESSAGES[0], CHAT_MESSAGES[3]]
        assert provider_messages == [
            {"role": "user", "content": [{"type": "text", "text": "Earlier safe question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Earlier safe answer"}]},
            {"role": "user", "content": [{"type": "text", "text": "Employee SSN: <US_SSN>"}]},
        ]
        assert provider.requests[0].body["system"] == [{"type": "text", "text": "Keep the answer short"}]
        assert exported_messages == CHAT_MESSAGES
        assert has_rust_response_marker(response)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("python", "rust"))
async def test_rubrik_block_preserves_context_for_error_exporters(
    backend: Backend,
    monkeypatch: pytest.MonkeyPatch,
    generic_api_export: GenericAPIExportHarness,
) -> None:
    monkeypatch.setenv("RUBRIK_BATCH_SIZE", "1")
    async with isolated_backend(backend):
        with recording_service() as provider, recording_service() as rubrik_service:
            provider.default_response = ResponseSpec(body=CHAT_RESPONSE)
            rubrik_service.expected_requests = None
            rubrik_service.default_response = ResponseSpec(
                body={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Response blocked by policy",
                                "tool_calls": [],
                            }
                        }
                    ]
                }
            )
            rubrik: Final = RubrikLogger(
                api_key="test-rubrik-key",
                api_base=rubrik_service.base_url,
                guardrail_name="rubrik-block",
                event_hook=GuardrailEventHooks.post_call,
                default_on=True,
            )
            recorder: Final = RecordingLogger()
            litellm.callbacks.append(rubrik)

            try:
                with pytest.raises(ModifyResponseException, match="Response blocked by policy") as raised:
                    await litellm.acompletion(
                        model=CHAT_MODEL,
                        messages=CHAT_MESSAGES,
                        api_key="test-key",
                        api_base=provider.base_url,
                        callbacks=[generic_api_export.logger, recorder],
                        guardrails=[rubrik.guardrail_name],
                    )
                assert has_rust_response_marker(raised.value.original_response) is (backend == "rust")
                await drain_logging()
                await rubrik.flush_queue()

                moderation_requests: Final = tuple(
                    request for request in rubrik_service.requests if request.path == "/v1/after_completion/openai/v1"
                )
                batch_requests: Final = tuple(
                    request for request in rubrik_service.requests if request.path == "/v1/litellm/batch"
                )
                assert len(provider.requests) == 1
                assert len(moderation_requests) == 1
                assert moderation_requests[0].body["request"]["messages"] == CHAT_MESSAGES
                assert moderation_requests[0].body["response"]["choices"][0]["message"]["content"] == "Handled safely"
                assert len(batch_requests) == 1
                assert "Response blocked by policy" in json.dumps(batch_requests[0].body)
                assert len(generic_api_export.exports) == 1
                exported: Final = generic_api_export.exports[0].body
                assert exported["status"] == "failure"
                assert exported["error_information"]["error_class"] == "ModifyResponseException"
                assert "Response blocked by policy" in exported["error_information"]["error_message"]
            finally:
                await rubrik.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("python", "rust"))
async def test_purview_audit_retains_payload_after_sdk_response(
    backend: Backend,
    generic_api_export: GenericAPIExportHarness,
) -> None:
    async with isolated_backend(backend):
        with recording_service() as provider:
            provider.default_response = ResponseSpec(body=CHAT_RESPONSE)
            purview: Final = purview_harness("purview-audit")
            recorder: Final = RecordingLogger()
            litellm.callbacks.append(purview.guardrail)
            request_task: Final = asyncio.create_task(
                litellm.acompletion(
                    model=CHAT_MODEL,
                    messages=CHAT_MESSAGES,
                    api_key="test-key",
                    api_base=provider.base_url,
                    metadata={"user_api_key_user_id": "audit-user"},
                    callbacks=[generic_api_export.logger, recorder],
                    guardrails=[purview.guardrail.guardrail_name],
                )
            )
            try:
                accepted: Final = await asyncio.to_thread(purview.graph.accepted.wait, 3)
                assert accepted
                response: Final = await asyncio.wait_for(asyncio.shield(request_task), 2)
                assert response.choices[0].message.content == "Handled safely"
                assert len(await wait_for_audits(purview.graph, count=1)) == 1

                purview.graph.release.set()
                await recorder.wait_for_async("async_log_success_event")
                audits: Final = await wait_for_audits(purview.graph)
                activities: Final = tuple(
                    post.body["contentToProcess"]["activityMetadata"]["activity"] for post in audits
                )
                assert activities == ("uploadText", "downloadText")
                assert CHAT_MESSAGES[-1]["content"] in json.dumps(audits[0].body)
                assert "Handled safely" in json.dumps(audits[1].body)
                assert generic_api_export.exports[0].body["status"] == "success"
                assert has_rust_response_marker(response) is (backend == "rust")
            finally:
                purview.graph.release.set()
                if not request_task.done():
                    request_task.cancel()
                await asyncio.gather(request_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("python", "rust"))
async def test_purview_sync_audit_runs_without_caller_event_loop(
    backend: Backend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITERAL_BATCH_SIZE", "1")
    async with isolated_backend(backend):
        with recording_service() as provider, recording_service() as sink:
            provider.default_response = ResponseSpec(body=CHAT_RESPONSE)
            purview: Final = purview_harness("purview-sync-audit")
            exporter: Final = LiteralAILogger(literalai_api_key="test-key", literalai_api_url=sink.base_url)
            litellm.callbacks.append(purview.guardrail)

            try:
                response: Final = await asyncio.to_thread(
                    litellm.completion,
                    model=CHAT_MODEL,
                    messages=CHAT_MESSAGES,
                    api_key="test-key",
                    api_base=provider.base_url,
                    metadata={"user_api_key_user_id": "audit-user"},
                    callbacks=[exporter],
                    guardrails=[purview.guardrail.guardrail_name],
                )
                accepted: Final = await asyncio.to_thread(purview.graph.accepted.wait, 3)
                assert accepted
                assert response.choices[0].message.content == "Handled safely"
                assert len(await wait_for_audits(purview.graph, count=1)) == 1

                purview.graph.release.set()
                audits: Final = await wait_for_audits(purview.graph)
                assert CHAT_MESSAGES[-1]["content"] in json.dumps(audits[0].body)
                assert "Handled safely" in json.dumps(audits[1].body)
                assert sink.requests[0].path == "/api/graphql"
                assert "Handled safely" in json.dumps(sink.requests[0].body["variables"]["generation_0"])
                assert has_rust_response_marker(response) is (backend == "rust")
            finally:
                purview.graph.release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", NON_STREAM_ASYNC_ROUTES, ids=route_id)
async def test_content_filter_post_call_blocks_provider_response(route: Route, provider: RecordingServer) -> None:
    guardrail: Final = ContentFilterGuardrail(
        guardrail_name="enforced-content-review",
        event_hook=GuardrailEventHooks.post_call,
        blocked_words=[BlockedWord(keyword=route.expected_text, action=ContentFilterAction.BLOCK)],
    )
    litellm.callbacks.append(guardrail)
    with pytest.raises(HTTPException, match="Content blocked") as blocked:
        await route.invoke(provider, guardrails=["enforced-content-review"])
    assert blocked.value.status_code == 400


@pytest.mark.asyncio
async def test_azure_text_moderation_allows_messages_response_over_http(
    recording_server: RecordingServer, otel: OtelHarness
) -> None:
    recording_server.expected_requests = None
    recording_server.default_response = ResponseSpec(body=AZURE_MODERATION_ALLOW_RESPONSE)
    recording_server.enqueue(provider_response(MESSAGES_ROUTE))
    guardrail: Final = azure_text_moderation(recording_server)
    recorder: Final = RecordingLogger()
    litellm.callbacks.append(guardrail)

    response: Final = await MESSAGES_ROUTE.invoke(
        recording_server, callbacks=[otel.logger, recorder], guardrails=[guardrail.guardrail_name]
    )
    await wait_for_callback(MESSAGES_ROUTE, recorder)

    assert len(recording_server.requests) == 2
    moderation_request: Final = recording_server.requests[1]
    assert moderation_request.path == "/contentsafety/text:analyze?api-version=2024-09-01"
    assert moderation_request.headers["ocp-apim-subscription-key"] == "test-azure-key"
    assert moderation_request.body == {
        "text": MESSAGES_ROUTE.expected_text,
        "categories": ["Hate", "Sexual", "SelfHarm", "Violence"],
        "blocklistNames": None,
        "haltOnBlocklistHit": False,
        "outputType": "FourSeverityLevels",
    }
    assert response["content"][0]["text"] == MESSAGES_ROUTE.expected_text
    assert len(await otel.wait_for_spans()) == 1


@pytest.mark.asyncio
async def test_azure_text_moderation_blocks_messages_response_over_http(recording_server: RecordingServer) -> None:
    recording_server.expected_requests = None
    recording_server.default_response = ResponseSpec(body=AZURE_MODERATION_BLOCK_RESPONSE)
    recording_server.enqueue(provider_response(MESSAGES_ROUTE))
    guardrail: Final = azure_text_moderation(recording_server)
    litellm.callbacks.append(guardrail)
    with pytest.raises(HTTPException, match="Violence crossed severity 2") as blocked:
        await MESSAGES_ROUTE.invoke(recording_server, guardrails=[guardrail.guardrail_name])
    assert blocked.value.status_code == 400
    assert recording_server.requests[1].body["text"] == MESSAGES_ROUTE.expected_text


@pytest.mark.asyncio
@pytest.mark.parametrize("route", NON_STREAM_ASYNC_ROUTES, ids=route_id)
async def test_post_call_guardrail_replacement_is_what_loggers_see(
    route: Route, provider: RecordingServer, otel: OtelHarness
) -> None:
    async def review(response: object) -> object:
        match route.name:
            case "ocr-async":
                return response.model_copy(
                    update={"pages": [response.pages[0].model_copy(update={"markdown": "Reviewed OCR"})]}
                )
            case _:
                return {**response, "content": [{"type": "text", "text": "Reviewed Messages"}]}

    guardrail: Final = ReviewGuardrail(review)
    recorder: Final = RecordingLogger()
    litellm.callbacks.append(guardrail)
    response: Final = await route.invoke(provider, callbacks=[otel.logger, recorder], guardrails=["rust-review"])

    assert guardrail.call_types == [CallTypes(route.call_type)]
    event: Final = (await wait_for_callback(route, recorder))[0]
    span: Final = (await otel.wait_for_spans())[0]
    match route.name:
        case "ocr-async":
            assert response.pages[0].markdown == "Reviewed OCR"
            assert event.response.pages[0].markdown == "Reviewed OCR"
        case _:
            assert response["content"][0]["text"] == "Reviewed Messages"
            assert event.response.choices[0].message.content == "Reviewed Messages"
            assert "Reviewed Messages" in json.dumps(dict(span.attributes))
    assert "guardrails" not in provider.requests[0].body
