"""Differential parity: one native call reaches a legacy logger and its v1 port, each posting
to its own recording vendor, and every difference between the two vendor bodies is `Allowed`.

The native chain runs the legacy contract and then v1 on the same call, so nothing is
replayed or mocked: both sides see the one real call. Registration happens in fixtures only.
"""

from collections.abc import Generator
from datetime import datetime, timezone
from typing import Final

import pytest

import litellm
from litellm import callbacks_v1
from litellm.callbacks_v1.builtin import anthropic_cache_control, generic_api, openmeter
from litellm.callbacks_v1.builtin.manifest import entry
from litellm.callbacks_v1.builtin.runtime import HttpxTransport, Patcher, Sink
from litellm.integrations.generic_api.generic_api_callback import GenericAPILogger
from litellm.integrations.openmeter import OpenMeterLogger
from tests.test_litellm.callbacks_v1.builtin.support import Allowed, assert_parity
from tests.test_litellm_rust.support.callback_recorder import drain_logging
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec, recording_service
from tests.test_litellm_rust.support.requests import MESSAGES, MESSAGES_MODEL, MESSAGES_RESPONSE

pytestmark = pytest.mark.requires_rust_extension

METADATA: Final = {"user_api_key_user_id": "user-1"}


@pytest.fixture
def provider(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    return recording_server


@pytest.fixture
def legacy_vendor() -> Generator[RecordingServer]:
    with recording_service() as server:
        yield server


@pytest.fixture
def port_vendor() -> Generator[RecordingServer]:
    with recording_service() as server:
        yield server


@pytest.fixture(autouse=True)
def clean_registry() -> Generator[None]:
    yield
    for subscriber in callbacks_v1.snapshot():
        callbacks_v1.unregister(subscriber)


def arguments(server: RecordingServer, **kwargs: object) -> dict[str, object]:
    # Identity rides in `metadata` here because call.started projects only that keyword; the proxy
    # sends it as `litellm_metadata` on this route, which v1 does not see yet (see entry("openmeter").gaps).
    return {
        "model": MESSAGES_MODEL,
        "messages": [dict(message) for message in MESSAGES],
        "max_tokens": 64,
        "api_key": "test-key",
        "api_base": server.base_url,
        "metadata": METADATA,
        **kwargs,
    }


OPENMETER_ALLOWED: Final = (
    Allowed("time", "each side stamps its own clock"),
    Allowed("data.cost", "no cost fact on call.succeeded", gap="cost"),
    Allowed("data.*_tokens", "no usage fact; legacy counts tokens off the OpenAI-normalised response", gap="usage"),
)


@pytest.mark.asyncio
async def test_openmeter_meters_a_native_call_like_its_legacy_twin(
    provider: RecordingServer,
    legacy_vendor: RecordingServer,
    port_vendor: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMETER_API_KEY", "test-key")
    monkeypatch.setenv("OPENMETER_API_ENDPOINT", legacy_vendor.base_url)
    port: Final = openmeter.OpenMeter(openmeter.Config(api_key="test-key", endpoint=port_vendor.base_url))
    sink: Final = Sink("openmeter", port, HttpxTransport(), lambda: datetime.now(timezone.utc))
    callbacks_v1.register(sink)

    await litellm.anthropic.messages.acreate(**arguments(provider, callbacks=[OpenMeterLogger()]))
    await drain_logging()
    assert sink.outbox.flush()

    (legacy_request,) = legacy_vendor.requests
    (port_request,) = port_vendor.requests
    assert port_request.path == legacy_request.path == "/api/v1/events"
    assert port_request.headers["authorization"] == legacy_request.headers["authorization"]
    assert port_request.headers["content-type"] == legacy_request.headers["content-type"]
    assert_parity(legacy_request.raw_body, port_request.raw_body, OPENMETER_ALLOWED, entry("openmeter").gaps)


# What the legacy StandardLoggingPayload has and v1 facts cannot fill yet, on any outcome.
GENERIC_API_GAP_ALLOWED: Final = (
    Allowed("0.stream", "legacy leaves stream null on a non-streaming call, v1 always says false"),
    Allowed("0.completionStartTime", "no first-token time in timing", gap="first_token_time"),
    Allowed("0.response_cost*", "no cost facts", gap="cost"),
    Allowed("0.cost_breakdown*", "no cost facts", gap="cost"),
    Allowed("0.saved_cache_cost", "no cost facts", gap="cost"),
    Allowed("0.autorouter_savings", "no cost facts", gap="cost"),
    Allowed("0.*_tokens", "no usage fact", gap="usage"),
    Allowed("0.cache_*", "no cache fact", gap="cache"),
    Allowed("0.model_id", "no routing fact", gap="routing"),
    Allowed("0.model_group", "no routing fact", gap="routing"),
    Allowed("0.model_map_information.*", "no routing fact", gap="routing"),
    Allowed("0.api_base", "no routing fact", gap="routing"),
    Allowed("0.metadata.*", "no identity fact", gap="identity"),
    Allowed("0.end_user", "no identity fact", gap="identity"),
    Allowed("0.requester_ip_address", "no identity fact", gap="identity"),
    Allowed("0.user_agent", "no identity fact", gap="identity"),
    Allowed("0.request_tags", "no tags fact", gap="tags"),
    Allowed("0.request_model_access_groups", "no tags fact", gap="tags"),
    Allowed("0.trace_id", "no correlation beyond call_id", gap="correlation"),
    Allowed("0.session_id", "no correlation beyond call_id", gap="correlation"),
    Allowed("0.hidden_params.*", "no equivalent", gap="no_equivalent"),
    Allowed("0.guardrail_information", "no equivalent", gap="no_equivalent"),
    Allowed("0.standard_built_in_tools_params.*", "no equivalent", gap="no_equivalent"),
    Allowed("0.status_fields.*", "derived from guardrail_information, which v1 has no fact for", gap="no_equivalent"),
    Allowed("0.messages.*", "no route-neutral prompt projection", gap="prompt"),
    Allowed(
        "0.model_parameters.metadata.*",
        "this test sends identity as `metadata`, which Anthropic also accepts; legacy keeps it out of model_parameters",
    ),
)

GENERIC_API_ALLOWED: Final = (
    *GENERIC_API_GAP_ALLOWED,
    Allowed(
        "0.response.*",
        "legacy logs the OpenAI-normalised response, v1 carries the route's own",
        gap="normalised_response",
    ),
    Allowed("0.error_str", "legacy sends null on success, the port omits the key"),
    Allowed("0.error_information.*", "legacy sends an empty shell on success, the port omits the key"),
)


@pytest.mark.asyncio
async def test_generic_api_logs_a_native_call_like_its_legacy_twin(
    provider: RecordingServer, legacy_vendor: RecordingServer, port_vendor: RecordingServer
) -> None:
    legacy: Final = GenericAPILogger(endpoint=legacy_vendor.base_url, batch_size=1)
    port: Final = generic_api.GenericApi(generic_api.Config(endpoint=port_vendor.base_url, batch_size=1))
    sink: Final = Sink("generic_api", port, HttpxTransport())
    callbacks_v1.register(sink)

    await litellm.anthropic.messages.acreate(**arguments(provider, callbacks=[legacy]))
    await drain_logging()
    assert sink.outbox.flush()

    (legacy_request,) = legacy_vendor.requests
    (port_request,) = port_vendor.requests
    assert port_request.headers["content-type"] == legacy_request.headers["content-type"]
    assert_parity(legacy_request.raw_body, port_request.raw_body, GENERIC_API_ALLOWED, entry("generic_api").gaps)


GENERIC_API_FAILURE_ALLOWED: Final = (
    *GENERIC_API_GAP_ALLOWED,
    Allowed("0.response", "legacy sends an empty object on failure, the port omits the key"),
    Allowed("0.error_information.error_budget_*", "ErrorFacts is class/message/status only", gap="error_detail"),
    Allowed("0.error_information.error_rate_limit_*", "ErrorFacts is class/message/status only", gap="error_detail"),
    Allowed(
        "0.error_information.error_provider_request_id", "ErrorFacts is class/message/status only", gap="error_detail"
    ),
    Allowed("0.error_information.llm_provider", "ErrorFacts is class/message/status only", gap="error_detail"),
    Allowed("0.error_information.traceback", "ErrorFacts is class/message/status only", gap="error_detail"),
)


@pytest.mark.asyncio
async def test_generic_api_logs_a_failed_native_call_like_its_legacy_twin(
    provider: RecordingServer, legacy_vendor: RecordingServer, port_vendor: RecordingServer
) -> None:
    provider.enqueue(
        ResponseSpec(body={"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}, status=400)
    )
    legacy: Final = GenericAPILogger(endpoint=legacy_vendor.base_url, batch_size=1)
    port: Final = generic_api.GenericApi(generic_api.Config(endpoint=port_vendor.base_url, batch_size=1))
    sink: Final = Sink("generic_api", port, HttpxTransport())
    callbacks_v1.register(sink)

    with pytest.raises(litellm.BadRequestError):
        await litellm.anthropic.messages.acreate(**arguments(provider, callbacks=[legacy]))
    await drain_logging()
    assert sink.outbox.flush()

    (legacy_request,) = legacy_vendor.requests
    (port_request,) = port_vendor.requests
    assert_parity(
        legacy_request.raw_body, port_request.raw_body, GENERIC_API_FAILURE_ALLOWED, entry("generic_api").gaps
    )


def test_the_cache_control_patch_survives_the_contract_and_reaches_the_provider(provider: RecordingServer) -> None:
    config: Final = anthropic_cache_control.Config(
        points=(anthropic_cache_control.InjectionPoint(role="system"), anthropic_cache_control.InjectionPoint(index=-1))
    )
    callbacks_v1.register(Patcher("anthropic_cache_control", anthropic_cache_control.AnthropicCacheControl(config)))

    litellm.anthropic.messages.create(**arguments(provider, system="be brief"))

    sent: Final = provider.requests[-1].body
    assert isinstance(sent, dict)
    assert sent["system"] == [{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral"}}]
    assert sent["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert sent["model"] == MESSAGES_MODEL.split("/", 1)[1]
