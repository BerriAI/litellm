import base64

from litellm.llms.xai.realtime.usage_meter import XAIRealtimeUsageMeter


def _b64(num_bytes: int) -> str:
    return base64.b64encode(b"x" * num_bytes).decode()


def test_text_messages_counts_only_billable_items():
    meter = XAIRealtimeUsageMeter()

    meter.observe_client_event(
        {
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        }
    )
    meter.observe_client_event(
        {"type": "conversation.item.create", "item": {"type": "function_call_output", "output": "{}"}}
    )
    meter.observe_client_event(
        {
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": [{"type": "input_audio", "audio": _b64(10)}]},
        }
    )
    meter.observe_client_event({"type": "response.create"})

    assert meter.snapshot().text_messages == 1


def test_input_audio_item_part_counts_as_sent_audio_not_text():
    meter = XAIRealtimeUsageMeter()

    meter.observe_client_event(
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_audio", "audio": _b64(48000)}],
            },
        }
    )

    snapshot = meter.snapshot()
    assert snapshot.audio_seconds_sent == 1.0
    assert snapshot.text_messages == 0


def test_audio_seconds_default_pcm24():
    meter = XAIRealtimeUsageMeter()

    meter.observe_client_event({"type": "input_audio_buffer.append", "audio": _b64(48000)})
    meter.observe_backend_event({"type": "response.output_audio.delta", "delta": _b64(24000)})
    meter.observe_backend_event({"type": "response.audio.delta", "delta": _b64(24000)})

    snapshot = meter.snapshot()
    assert snapshot.audio_seconds_sent == 1.0
    assert snapshot.audio_seconds_received == 1.0


def test_session_update_nested_pcm_rate_changes_seconds():
    meter = XAIRealtimeUsageMeter()

    meter.observe_client_event(
        {
            "type": "session.update",
            "session": {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}}},
        }
    )
    meter.observe_client_event({"type": "input_audio_buffer.append", "audio": _b64(32000)})

    assert meter.snapshot().audio_seconds_sent == 1.0


def test_g711_format_bills_at_8000_bytes_per_second():
    meter = XAIRealtimeUsageMeter()

    meter.observe_backend_event(
        {
            "type": "session.created",
            "session": {"audio": {"output": {"format": {"type": "audio/pcmu"}}}},
        }
    )
    meter.observe_backend_event({"type": "response.output_audio.delta", "delta": _b64(8000)})

    assert meter.snapshot().audio_seconds_received == 1.0


def test_beta_flat_g711_format_bills_at_8000_bytes_per_second():
    meter = XAIRealtimeUsageMeter()

    meter.observe_client_event({"type": "session.update", "session": {"input_audio_format": "g711_ulaw"}})
    meter.observe_client_event({"type": "input_audio_buffer.append", "audio": _b64(8000)})

    assert meter.snapshot().audio_seconds_sent == 1.0


def test_opus_format_contributes_no_audio_seconds():
    meter = XAIRealtimeUsageMeter()

    meter.observe_client_event(
        {
            "type": "session.update",
            "session": {"audio": {"input": {"format": {"type": "audio/opus"}}}},
        }
    )
    meter.observe_client_event({"type": "input_audio_buffer.append", "audio": _b64(48000)})

    assert meter.snapshot().audio_seconds_sent == 0.0


def test_null_turn_detection_marks_push_to_talk_flat_and_nested():
    flat = XAIRealtimeUsageMeter()
    flat.observe_client_event({"type": "session.update", "session": {"turn_detection": None}})
    assert flat.snapshot().push_to_talk is True

    nested = XAIRealtimeUsageMeter()
    nested.observe_client_event(
        {
            "type": "session.update",
            "session": {"audio": {"input": {"turn_detection": None}}},
        }
    )
    assert nested.snapshot().push_to_talk is True


def test_client_null_type_turn_detection_marks_push_to_talk():
    meter = XAIRealtimeUsageMeter()

    meter.observe_client_event({"type": "session.update", "session": {"turn_detection": {"type": None}}})

    assert meter.snapshot().push_to_talk is True


def test_backend_session_events_do_not_override_client_turn_detection():
    meter = XAIRealtimeUsageMeter()

    meter.observe_client_event({"type": "session.update", "session": {"turn_detection": None}})
    meter.observe_backend_event(
        {"type": "session.created", "session": {"turn_detection": {"type": None}}}
    )
    meter.observe_backend_event({"type": "session.updated", "session": {"modalities": ["audio"]}})

    assert meter.snapshot().push_to_talk is True


def test_later_client_server_vad_turn_detection_disables_push_to_talk():
    meter = XAIRealtimeUsageMeter()

    meter.observe_client_event({"type": "session.update", "session": {"turn_detection": None}})
    meter.observe_client_event(
        {"type": "session.update", "session": {"turn_detection": {"type": "server_vad"}}}
    )

    assert meter.snapshot().push_to_talk is False


def test_backend_turn_detection_alone_is_ignored():
    meter = XAIRealtimeUsageMeter()

    meter.observe_backend_event(
        {"type": "session.created", "session": {"turn_detection": {"type": None}}}
    )

    assert meter.snapshot().push_to_talk is False


def test_server_vad_turn_detection_stays_not_push_to_talk():
    meter = XAIRealtimeUsageMeter()

    meter.observe_client_event({"type": "session.update", "session": {"turn_detection": {"type": "server_vad"}}})

    assert meter.snapshot().push_to_talk is False


def test_non_string_audio_payloads_are_ignored():
    meter = XAIRealtimeUsageMeter()

    meter.observe_client_event({"type": "input_audio_buffer.append", "audio": 123})
    meter.observe_backend_event({"type": "response.output_audio.delta", "delta": None})

    snapshot = meter.snapshot()
    assert snapshot.audio_seconds_sent == 0.0
    assert snapshot.audio_seconds_received == 0.0
