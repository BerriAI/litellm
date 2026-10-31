from dataclasses import dataclass
from typing import Final

import pytest

from litellm.telemetry.consent import (
    OFF,
    REQUIRES,
    ConsentGatedSink,
    MissingRequirement,
    TelemetryConsent,
    UnknownGroup,
    consent_of,
    parse_consent,
)
from litellm.telemetry.records import (
    AttemptRecord,
    BlockCounts,
    BlockType,
    InstanceInfo,
    RequestRecord,
    StatusClass,
    TelemetryGroup,
    TokenCounts,
    UIAction,
    UIEvent,
)


@dataclass
class _RecordingSink:
    instances: tuple[InstanceInfo, ...] = ()
    requests: tuple[RequestRecord, ...] = ()
    attempts: tuple[AttemptRecord, ...] = ()
    ui_events: tuple[UIEvent, ...] = ()
    flushes: int = 0

    def set_instance(self, info: InstanceInfo) -> None:
        self.instances = (*self.instances, info)

    def record_request(self, record: RequestRecord) -> None:
        self.requests = (*self.requests, record)

    def record_attempt(self, record: AttemptRecord) -> None:
        self.attempts = (*self.attempts, record)

    def record_ui_event(self, event: UIEvent) -> None:
        self.ui_events = (*self.ui_events, event)

    async def flush(self) -> None:
        self.flushes += 1


_REQUEST: Final = RequestRecord(
    endpoint="/chat/completions",
    stream=True,
    litellm_status=StatusClass.SUCCESS,
    provider="openai",
    deployment_hash="d1",
    provider_status=StatusClass.SUCCESS,
    litellm_cache_hit=True,
    handled_by_rust=True,
    provider_cache_hit=True,
    provider_attempts=2,
    tokens=TokenCounts(input=10, output=5, cache_read=3),
    latency_to_headers_ms=20.0,
    latency_to_first_byte_ms=40.0,
    blocks=BlockCounts(total=2, by_type=((BlockType.TEXT, 1), (BlockType.IMAGE, 1))),
    header_keys=frozenset({"Anthropic-Beta", "authorization", "x-customer-secret"}),
)
_SUCCESS_ONLY: Final = RequestRecord(
    endpoint="/chat/completions",
    stream=True,
    litellm_status=StatusClass.SUCCESS,
    provider_status=StatusClass.SUCCESS,
    litellm_cache_hit=True,
    handled_by_rust=True,
    provider_attempts=2,
    latency_to_headers_ms=20.0,
    latency_to_first_byte_ms=40.0,
)
_ATTEMPT: Final = AttemptRecord(provider="openai", provider_status=StatusClass.SUCCESS, stream=True)
_INSTANCE: Final = InstanceInfo(instance_id="abc", litellm_version="1.0.0", config_keys=frozenset({"cache"}))
_UI_EVENT: Final = UIEvent(page="models", action=UIAction.VIEW)


def _consent(*groups: TelemetryGroup) -> TelemetryConsent:
    consent: Final = consent_of(groups)
    assert isinstance(consent, TelemetryConsent)
    return consent


_CHAIN: Final = (
    TelemetryGroup.HEARTBEAT,
    TelemetryGroup.REQUEST_SUCCESS,
    TelemetryGroup.TOKEN_INFO,
    TelemetryGroup.REQUEST_TAXONOMY,
    TelemetryGroup.EVENT_DETAILS,
)


async def _send_everything(consent: TelemetryConsent) -> _RecordingSink:
    inner: Final = _RecordingSink()
    sink: Final = ConsentGatedSink(inner, consent)
    sink.set_instance(_INSTANCE)
    sink.record_request(_REQUEST)
    sink.record_attempt(_ATTEMPT)
    sink.record_ui_event(_UI_EVENT)
    await sink.flush()
    return inner


@pytest.mark.asyncio
async def test_off_forwards_nothing() -> None:
    assert await _send_everything(OFF) == _RecordingSink()


@pytest.mark.asyncio
async def test_heartbeat_alone_sends_only_the_instance_header() -> None:
    assert await _send_everything(_consent(TelemetryGroup.HEARTBEAT)) == _RecordingSink(
        instances=(InstanceInfo(instance_id="abc", litellm_version="1.0.0", groups=frozenset(_CHAIN[:1])),),
        flushes=1,
    )


@pytest.mark.asyncio
async def test_a_directly_built_consent_missing_its_heartbeat_forwards_nothing() -> None:
    assert await _send_everything(TelemetryConsent(frozenset({TelemetryGroup.REQUEST_SUCCESS}))) == _RecordingSink()


@pytest.mark.asyncio
async def test_a_directly_built_consent_skipping_a_middle_group_reports_and_forwards_only_the_connected_groups() -> (
    None
):
    inner: Final = await _send_everything(
        TelemetryConsent(frozenset({TelemetryGroup.HEARTBEAT, TelemetryGroup.EVENT_DETAILS}))
    )
    assert inner == _RecordingSink(
        instances=(InstanceInfo(instance_id="abc", litellm_version="1.0.0", groups=frozenset(_CHAIN[:1])),),
        flushes=1,
    )


@pytest.mark.asyncio
async def test_request_success_keeps_status_endpoint_and_latency_but_no_tokens_provider_or_details() -> None:
    inner: Final = await _send_everything(_consent(*_CHAIN[:2]))
    assert inner.requests == (_SUCCESS_ONLY,)
    assert inner.attempts == ()


@pytest.mark.asyncio
async def test_token_info_adds_token_sums_and_the_provider_cache_hit() -> None:
    inner: Final = await _send_everything(_consent(*_CHAIN[:3]))
    assert inner.requests[0].tokens == _REQUEST.tokens
    assert inner.requests[0].provider_cache_hit is True
    assert (inner.requests[0].provider, inner.requests[0].deployment_hash) == (None, None)


@pytest.mark.asyncio
async def test_request_taxonomy_adds_provider_deployment_and_attempt_rows() -> None:
    inner: Final = await _send_everything(_consent(*_CHAIN[:4]))
    assert (inner.requests[0].provider, inner.requests[0].deployment_hash) == ("openai", "d1")
    assert inner.attempts == (_ATTEMPT,)
    assert (inner.requests[0].blocks, inner.requests[0].header_keys) == (None, frozenset())


@pytest.mark.asyncio
async def test_request_taxonomy_without_token_info_keeps_the_provider_but_drops_tokens() -> None:
    inner: Final = await _send_everything(
        _consent(TelemetryGroup.HEARTBEAT, TelemetryGroup.REQUEST_SUCCESS, TelemetryGroup.REQUEST_TAXONOMY)
    )
    assert (inner.requests[0].provider, inner.requests[0].deployment_hash) == ("openai", "d1")
    assert (inner.requests[0].tokens, inner.requests[0].provider_cache_hit) == (TokenCounts(), False)
    assert inner.attempts == (_ATTEMPT,)


@pytest.mark.asyncio
async def test_event_details_adds_blocks_and_collapses_unlisted_header_keys_into_other() -> None:
    inner: Final = await _send_everything(_consent(*_CHAIN))
    assert inner.requests[0].blocks == _REQUEST.blocks
    assert inner.requests[0].header_keys == frozenset({"anthropic-beta", "other"})
    assert inner.instances[0].config_keys == frozenset()
    assert inner.ui_events == ()


@pytest.mark.asyncio
async def test_instance_configuration_and_page_navigation_need_only_the_heartbeat() -> None:
    inner: Final = await _send_everything(
        _consent(TelemetryGroup.HEARTBEAT, TelemetryGroup.INSTANCE_CONFIGURATION, TelemetryGroup.PAGE_NAVIGATION)
    )
    assert inner.instances[0].config_keys == frozenset({"cache"})
    assert inner.ui_events == (_UI_EVENT,)
    assert inner.requests == ()


def test_only_allowlisted_header_keys_adds_no_other_key() -> None:
    inner: Final = _RecordingSink()
    ConsentGatedSink(inner, _consent(*_CHAIN)).record_request(
        RequestRecord(
            endpoint="/v1/messages",
            stream=False,
            litellm_status=StatusClass.SUCCESS,
            header_keys=frozenset({"anthropic-version"}),
        )
    )
    assert inner.requests[0].header_keys == frozenset({"anthropic-version"})


@pytest.mark.parametrize("group", [group for group in TelemetryGroup if REQUIRES[group] is not None])
def test_every_group_but_heartbeat_is_refused_without_its_requirement(group: TelemetryGroup) -> None:
    required: Final = REQUIRES[group]
    assert required is not None
    everything_else: Final = frozenset(TelemetryGroup) - {required}
    assert consent_of(everything_else) == MissingRequirement(
        min((g for g in everything_else if REQUIRES[g] == required), key=tuple(TelemetryGroup).index), required
    )


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        ((), OFF),
        (("", " "), OFF),
        (
            (" Heartbeat ", "PAGE_NAVIGATION"),
            TelemetryConsent(frozenset(_CHAIN[:1]) | {TelemetryGroup.PAGE_NAVIGATION}),
        ),
        (("heartbeat", "token_info"), MissingRequirement(TelemetryGroup.TOKEN_INFO, TelemetryGroup.REQUEST_SUCCESS)),
        (("heartbeat", "full"), UnknownGroup("full")),
    ],
)
def test_parse_consent(names: tuple[str, ...], expected: object) -> None:
    assert parse_consent(names) == expected
