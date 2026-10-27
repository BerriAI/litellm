import itertools
import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
from _openinference_support import Rig, _chat_plain_response, _rig, _spans
from integration._support.client import Gateway, eventually
from integration._support.wire import Reply, Request, Wire
from pydantic import JsonValue

_OPENINFERENCE_CALLBACKS: Final = ("arize", "arize_phoenix", "weave_otel")
_GENAI_CALLBACKS: Final = ("otel", "langfuse_otel", "langtrace", "signoz", "levo", "newrelic", "agentops")
_UNUSED_ENDPOINT: Final[dict[str, JsonValue]] = {"otel": {"exporter": "http/protobuf", "endpoint": "unused"}}
_METADATA_ALLOWLIST: Final = "LITELLM_OTEL_BAGGAGE_METADATA_KEYS"
_LANGFUSE_CALLBACK: Final = "langfuse_otel"


def _settings(callback: str) -> Mapping[str, JsonValue] | None:
    if callback == "arize":
        return None
    if callback in _OPENINFERENCE_CALLBACKS:
        return _UNUSED_ENDPOINT
    return {"otel": {"exporter": "http/protobuf", "endpoint": "unused", "mapper_names": ["genai"]}}


def _environment(callback: str) -> dict[str, str]:
    return {
        "PHOENIX_PROJECT_NAME": "integration",
        "WANDB_BASE_URL": "http://127.0.0.1",
        "LANGFUSE_HOST": "http://127.0.0.1",
        **(
            {"HTTPS_PROXY": "http://127.0.0.1:0", "https_proxy": "http://127.0.0.1:0", "NO_PROXY": "", "no_proxy": ""}
            if callback == "agentops"
            else {}
        ),
    }


def _removed_environment(callback: str) -> tuple[str, ...]:
    if callback == "newrelic":
        return ("NEW_RELIC_LICENSE_KEY",)
    if callback == "agentops":
        return ("AGENTOPS_API_KEY",)
    return ()


def _upstream(request: Request) -> Reply:
    marker: Final = json.loads(request.body)["messages"][0]["content"]
    return _chat_plain_response(marker, "ok")


def _model_call_spans(destination: Wire, markers: tuple[str, ...]) -> Mapping[str, tuple[dict[str, str], ...]]:
    seen: Final[list[Request]] = []

    def by_marker() -> Mapping[str, tuple[dict[str, str], ...]]:
        seen.extend(destination.drain())
        spans: Final = tuple(attributes for attributes in _spans(tuple(seen)) if "gen_ai.response.id" in attributes)
        return {
            marker: tuple(attributes for attributes in spans if attributes["gen_ai.response.id"] == marker)
            for marker in markers
        }

    return eventually(by_marker, lambda found: all(found.values()), seconds=30)


def _carrying(spans: Mapping[str, tuple[dict[str, str], ...]], value: str) -> tuple[str, ...]:
    model_call_spans: Final = itertools.chain.from_iterable(spans.values())
    return tuple(key for span in model_call_spans for key, attribute in span.items() if value in str(attribute))


def _chat(rig: Rig, marker: str, key: str, headers: Mapping[str, str], extra: Mapping[str, JsonValue]) -> None:
    response: Final = rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": rig.model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}, **extra},
        key=key,
        headers=headers,
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("callback", (*_OPENINFERENCE_CALLBACKS, *_GENAI_CALLBACKS))
def test_otel_v2_callback_sends_the_request_user_as_user_id(callback: str, gateway: Gateway, tmp_path: Path) -> None:
    run: Final = uuid.uuid4().hex
    end_user: Final = f"end-user-{run}"
    caller: Final = f"caller-{run}"
    markers: Final = (f"end-{run}", f"trace-{run}", f"owner-{run}")
    with (
        _rig(
            gateway,
            tmp_path,
            _upstream,
            callbacks=(callback,),
            callback_settings=_settings(callback),
            environment=_environment(callback),
            remove_environment=(*_removed_environment(callback), _METADATA_ALLOWLIST),
            disabled_environment=(_METADATA_ALLOWLIST,),
        ) as rig,
        rig.proxy.scenario() as scenario,
    ):
        service_key: Final = scenario.key()
        owned_key: Final = scenario.key(user_id=scenario.user())
        sent: Final = (
            (markers[0], service_key, {"x-litellm-end-user-id": end_user}, {}),
            (markers[1], service_key, {"x-litellm-end-user-id": end_user}, {"metadata": {"trace_user_id": caller}}),
            (markers[2], owned_key, {}, {}),
        )
        for marker, key, headers, extra in sent:
            _chat(rig, marker, key, headers, extra)
        spans: Final = _model_call_spans(rig.destination, markers)
    users: Final = {marker: {attributes.get("user.id") for attributes in spans[marker]} for marker in markers}
    caller_user: Final = caller if callback == _LANGFUSE_CALLBACK else end_user
    assert users == {markers[0]: {end_user}, markers[1]: {caller_user}, markers[2]: {None}}, spans
    sessions: Final = {attributes.get("session.id") for attributes in itertools.chain.from_iterable(spans.values())}
    assert end_user not in sessions, spans
    assert callback == _LANGFUSE_CALLBACK or not _carrying(spans, caller), spans


@pytest.mark.parametrize("callback", ("arize", "otel", "langfuse_otel"))
def test_otel_v2_callback_withholds_an_end_user_the_metadata_allowlist_drops(
    callback: str, gateway: Gateway, tmp_path: Path
) -> None:
    run: Final = uuid.uuid4().hex
    end_user: Final = f"end-user-{run}"
    caller: Final = f"caller-{run}"
    markers: Final = (f"end-{run}", f"trace-{run}")
    with (
        _rig(
            gateway,
            tmp_path,
            _upstream,
            callbacks=(callback,),
            callback_settings=_settings(callback),
            environment={**_environment(callback), _METADATA_ALLOWLIST: "user_api_key_org_id"},
        ) as rig,
        rig.proxy.scenario() as scenario,
    ):
        service_key: Final = scenario.key()
        _chat(rig, markers[0], service_key, {"x-litellm-end-user-id": end_user}, {})
        _chat(
            rig, markers[1], service_key, {"x-litellm-end-user-id": end_user}, {"metadata": {"trace_user_id": caller}}
        )
        spans: Final = _model_call_spans(rig.destination, markers)
    users: Final = {marker: {attributes.get("user.id") for attributes in spans[marker]} for marker in markers}
    assert users == {markers[0]: {None}, markers[1]: {caller if callback == _LANGFUSE_CALLBACK else None}}, spans
    assert not _carrying(spans, end_user), spans
    assert callback == _LANGFUSE_CALLBACK or not _carrying(spans, caller), spans
