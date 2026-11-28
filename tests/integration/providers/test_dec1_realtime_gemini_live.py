"""Contract: ``/v1/realtime`` on a Gemini (AI Studio) deployment, translated to Gemini Live BidiGenerateContent.

The proxy speaks the OpenAI realtime protocol to the client and Gemini Live to the provider
(``GeminiRealtimeConfig``, litellm/llms/gemini/realtime/transformation.py, driven by
``BaseLLMHTTPHandler.async_realtime``):

- the provider URL is ``<api_base as ws>/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent
  ?key=<deployment api_key>``; the deployment key travels in the query, the client's virtual key never leaves;
- the client gets a synthetic ``session.created`` right after connect (Gemini builds its setup from the first
  ``session.update``, so nothing is sent at connect);
- the first ``session.update`` becomes the one ``setup`` frame (``model`` ``models/<model>``,
  ``inputAudioTranscription`` ``{}``); every later ``session.update`` is dropped because Gemini closes 1007 on a
  second setup;
- ``conversation.item.create`` with ``input_text`` becomes ``clientContent`` with one user turn and
  ``turnComplete: true``; ``input_audio_buffer.append`` becomes ``realtimeInput.audio`` with mime type
  ``audio/pcm;rate=24000``; ``input_audio_buffer.clear`` and unknown client events are not forwarded;
- ``serverContent.turnComplete`` with ``usageMetadata`` becomes ``response.done`` whose usage is billed.
"""

from __future__ import annotations

import base64
import uuid
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, object_value
from tests.integration._support.upstream import GEMINI_AI_STUDIO_LIVE_PATH, ScenarioHandle
from tests.integration.providers.dec1_realtime_support import (
    GEMINI_LIVE_MODEL,
    Await,
    Idle,
    Send,
    observations,
    realtime,
    scripted,
    spend_rows,
    upstream_frames,
)

pytestmark: Final = [pytest.mark.timeout(180), pytest.mark.shares_worker_state]

SESSION_UPDATE: Final = {"type": "session.update", "session": {"instructions": "be brief", "modalities": ["text"]}}
USER_TEXT: Final = {
    "type": "conversation.item.create",
    "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello gemini"}]},
}
TURN: Final[tuple[dict[str, JsonValue], ...]] = (
    {"serverContent": {"modelTurn": {"parts": [{"text": "hi there"}]}}},
    {
        "serverContent": {"turnComplete": True},
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 6, "totalTokenCount": 18},
    },
)


def _gemini(
    scenario: Scenario, events: tuple[dict[str, JsonValue], ...] = TURN, **prices: JsonValue
) -> tuple[str, ScenarioHandle]:
    handle: Final = scripted(scenario, events)
    model: Final = scenario.model(
        model=f"gemini/{GEMINI_LIVE_MODEL}",
        api_key=handle.scenario_id,
        api_base=scenario.gateway.upstream_url.rstrip("/"),
        **prices,
    )
    return model, handle


def _setups(gateway: Gateway, scenario_id: str) -> list[dict[str, JsonValue]]:
    return [object_value(frame["setup"]) for frame in upstream_frames(gateway, scenario_id) if "setup" in frame]


def test_the_client_gets_a_session_created_before_sending_anything(gateway: Gateway) -> None:
    """A client upgrades on a Gemini Live deployment and sends nothing. It still receives a ``session.created``
    first (the synthetic event from ``transform_session_created_event``) whose ``session.model`` is
    ``gemini-2.5-flash-native-audio-latest`` with the ``models/`` prefix stripped, and the provider has received
    no frame yet."""
    with gateway.scenario() as scenario:
        model, handle = _gemini(scenario)
        outcome: Final = realtime(gateway, f"model={model}", key=gateway.key, steps=(Await("session.created"), Idle(2)))
        assert outcome.types[:1] == ("session.created",), outcome
        assert object_value(outcome.frames[0]["session"]).get("model") == GEMINI_LIVE_MODEL, outcome
        assert upstream_frames(gateway, handle.scenario_id) == ()


def test_the_provider_connection_carries_the_deployment_key_in_the_query_and_not_the_virtual_key(
    gateway: Gateway,
) -> None:
    """A virtual key opens a Gemini session and sends a ``session.update``. The scripted AI Studio Live endpoint
    at ``/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent`` observes the handshake
    with ``?key=<deployment api_key>`` (it keys the scenario on that query value) and the virtual key appears in
    no observation."""
    with gateway.scenario() as scenario:
        model, handle = _gemini(scenario)
        key: Final = scenario.key()
        realtime(gateway, f"model={model}", key=key, steps=(Await("session.created"), Send(SESSION_UPDATE), Idle(2)))
        seen: Final = [entry for entry in observations(gateway) if entry.get("api_key") == handle.scenario_id]
        assert [entry["path"] for entry in seen if entry["method"] == "WEBSOCKET"] == [GEMINI_AI_STUDIO_LIVE_PATH], seen
        assert key not in str(observations(gateway)), "the virtual key reached the provider"


def test_the_first_session_update_becomes_the_single_setup_frame_with_the_models_prefix(gateway: Gateway) -> None:
    """The client's first ``session.update`` (instructions and text modality) reaches the provider as one
    ``{"setup": {...}}`` frame whose ``model`` is ``models/gemini-2.5-flash-native-audio-latest`` and whose
    ``inputAudioTranscription`` is ``{}`` (``_handle_session_update``). The proxy holds session updates and sends
    the setup right before the first conversation content (``_holds_session_updates_until_first_content`` and
    ``_send_setup_before_content``, litellm/litellm_core_utils/realtime_streaming.py:1144 and :2421-2427), so the
    client sends a user text turn after the update."""
    with gateway.scenario() as scenario:
        model, handle = _gemini(scenario)
        realtime(
            gateway,
            f"model={model}",
            key=gateway.key,
            steps=(Await("session.created"), Send(SESSION_UPDATE), Send(USER_TEXT), Await("response.done")),
        )
        setups: Final = _setups(gateway, handle.scenario_id)
        assert len(setups) == 1, setups
        assert setups[0]["model"] == f"models/{GEMINI_LIVE_MODEL}", setups
        assert setups[0]["inputAudioTranscription"] == {}, setups


def test_a_second_session_update_is_dropped_instead_of_sent_as_a_second_setup(gateway: Gateway) -> None:
    """The client sends two ``session.update`` frames and then a user text turn. The provider receives exactly one
    ``setup`` frame (Gemini Live closes with 1007 on a second setup, which would end the session): the proxy merges
    the held updates into the one setup it sends before the first content (realtime_streaming.py:2421-2427)."""
    with gateway.scenario() as scenario:
        model, handle = _gemini(scenario)
        realtime(
            gateway,
            f"model={model}",
            key=gateway.key,
            steps=(
                Await("session.created"),
                Send(SESSION_UPDATE),
                Idle(1),
                Send(SESSION_UPDATE),
                Send(USER_TEXT),
                Await("response.done"),
            ),
        )
        assert len(_setups(gateway, handle.scenario_id)) == 1


def test_user_text_becomes_client_content_with_one_completed_user_turn(gateway: Gateway) -> None:
    """After the setup, a ``conversation.item.create`` with ``input_text`` ``hello gemini`` reaches the provider
    as ``{"clientContent": {"turns": [{"role": "user", "parts": [{"text": "hello gemini"}]}],
    "turnComplete": true}}`` (``_handle_user_text_content``)."""
    with gateway.scenario() as scenario:
        model, handle = _gemini(scenario)
        realtime(
            gateway,
            f"model={model}",
            key=gateway.key,
            steps=(Await("session.created"), Send(SESSION_UPDATE), Send(USER_TEXT), Await("response.done")),
        )
        contents: Final = [f for f in upstream_frames(gateway, handle.scenario_id) if "clientContent" in f]
        assert contents == [
            {
                "clientContent": {
                    "turns": [{"role": "user", "parts": [{"text": "hello gemini"}]}],
                    "turnComplete": True,
                }
            }
        ], contents


def test_appended_audio_becomes_realtime_input_audio_with_the_pcm_24k_mime_type(gateway: Gateway) -> None:
    """An ``input_audio_buffer.append`` with base64 PCM reaches the provider as
    ``{"realtimeInput": {"audio": {"mimeType": "audio/pcm;rate=24000", "data": <same base64>}}}``
    (``get_audio_mime_type("pcm16")``)."""
    audio: Final = base64.b64encode(b"\x01\x02" * 400).decode()
    with gateway.scenario() as scenario:
        model, handle = _gemini(scenario)
        realtime(
            gateway,
            f"model={model}",
            key=gateway.key,
            steps=(
                Await("session.created"),
                Send(SESSION_UPDATE),
                Send({"type": "input_audio_buffer.append", "audio": audio}),
                Idle(3),
            ),
        )
        inputs: Final = [f for f in upstream_frames(gateway, handle.scenario_id) if "realtimeInput" in f]
        audio_inputs: Final = [object_value(f["realtimeInput"]).get("audio") for f in inputs]
        assert {"mimeType": "audio/pcm;rate=24000", "data": audio} in audio_inputs, inputs


def test_a_buffer_clear_and_an_unknown_client_event_are_not_forwarded(gateway: Gateway) -> None:
    """After the session update the client sends ``input_audio_buffer.clear`` and ``x.client.custom``, then a user
    text turn. Neither of the first two produces a provider frame (``transform_realtime_request`` returns ``[]``):
    the provider sees only the setup, sent before the first content, and the ``clientContent`` turn."""
    with gateway.scenario() as scenario:
        model, handle = _gemini(scenario)
        realtime(
            gateway,
            f"model={model}",
            key=gateway.key,
            steps=(
                Await("session.created"),
                Send(SESSION_UPDATE),
                Send({"type": "input_audio_buffer.clear"}),
                Send({"type": "x.client.custom", "n": 1}),
                Send(USER_TEXT),
                Await("response.done"),
            ),
        )
        frames: Final = upstream_frames(gateway, handle.scenario_id)
        assert [sorted(frame) for frame in frames] == [["setup"], ["clientContent"]], frames


def test_text_content_without_a_session_update_still_gets_a_default_setup_first(gateway: Gateway) -> None:
    """The client skips ``session.update`` and sends user text right away. The provider's first frame is a
    ``setup`` with ``model`` ``models/gemini-2.5-flash-native-audio-latest`` and the ``clientContent`` follows
    (``_send_setup_before_content``). UNVERIFIED: the held-content grace of
    ``REALTIME_LAZY_SETUP_CONTENT_GRACE_SECONDS`` (1 s) delays the setup by about a second."""
    with gateway.scenario() as scenario:
        model, handle = _gemini(scenario)
        realtime(gateway, f"model={model}", key=gateway.key, steps=(Await("session.created"), Send(USER_TEXT), Idle(4)))
        frames: Final = upstream_frames(gateway, handle.scenario_id)
        assert [next(iter(frame)) for frame in frames][:2] == ["setup", "clientContent"], frames


def test_a_completed_gemini_turn_reaches_the_client_as_response_done_with_mapped_usage(gateway: Gateway) -> None:
    """The provider answers the user turn with a text part and ``turnComplete`` carrying ``usageMetadata``
    (12 prompt, 6 candidates). The client receives a ``response.done`` whose ``response.usage.input_tokens`` is 12
    and ``output_tokens`` 6, and whose ``response.status`` is ``completed``."""
    with gateway.scenario() as scenario:
        model, _handle = _gemini(scenario)
        outcome: Final = realtime(
            gateway,
            f"model={model}",
            key=gateway.key,
            steps=(Await("session.created"), Send(SESSION_UPDATE), Send(USER_TEXT), Await("response.done")),
        )
        response: Final = object_value(outcome.of_type("response.done")[0]["response"])
        usage: Final = object_value(response["usage"])
        assert (usage.get("input_tokens"), usage.get("output_tokens")) == (12, 6), response
        assert response.get("status") == "completed", response


def test_a_gemini_session_is_billed_from_its_usage_metadata(gateway: Gateway) -> None:
    """With deployment rates 0.001 in and 0.002 out, the 12 / 6 token turn writes one ``_arealtime`` row with
    ``prompt_tokens`` 12, ``completion_tokens`` 6 and ``spend`` 12 * 0.001 + 6 * 0.002 = 0.024, and
    ``custom_llm_provider`` ``gemini``."""
    with gateway.scenario() as scenario:
        model, _handle = _gemini(scenario, input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key()
        realtime(
            gateway,
            f"model={model}",
            key=key,
            steps=(Await("session.created"), Send(SESSION_UPDATE), Send(USER_TEXT), Await("response.done")),
        )
        row: Final = spend_rows(gateway, key, 1)[0]
        assert row["call_type"] == "_arealtime", row
        assert (row["prompt_tokens"], row["completion_tokens"]) == (12, 6), row
        assert float(str(row["spend"])) == pytest.approx(0.024), row
        assert row["custom_llm_provider"] == "gemini", row


def test_an_unknown_gemini_scenario_key_closes_the_client_without_a_session_turn(gateway: Gateway) -> None:
    """A Gemini deployment whose ``api_key`` is not a registered scenario makes the scripted provider close the
    socket with 4404 before accepting (an HTTP 403 handshake answer). The client, which already got the
    synthetic ``session.created``, receives an ``error`` event ``Upstream realtime handshake rejected with HTTP
    403`` and close 1008. UNVERIFIED: whether the synthetic session.created is sent before the provider
    connect completes (the connect happens first in ``async_realtime``, so it is likely absent)."""
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"gemini/{GEMINI_LIVE_MODEL}",
            api_key=f"dec1-rt-unknown-{uuid.uuid4().hex[:8]}",
            api_base=gateway.upstream_url.rstrip("/"),
        )
        outcome: Final = realtime(gateway, f"model={model}", key=gateway.key, steps=(Await("never"),))
        assert "Upstream realtime handshake rejected with HTTP 403" in outcome.error_messages(), outcome
        assert outcome.close_code == 1008, outcome
