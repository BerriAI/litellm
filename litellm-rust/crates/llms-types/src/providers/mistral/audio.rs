use serde_json::{Map, Value};

use super::MistralUsage;

#[macro_rules_attribute::apply(wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MistralTimestampGranularity {
    Segment,
    Word,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralTranscriptionFields {
    pub model: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub file_url: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub file_id: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub language: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub temperature: Option<Option<f64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub stream: Option<Option<bool>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub diarize: Option<Option<bool>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub context_bias: Option<Option<Vec<String>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub timestamp_granularities: Option<Option<Vec<MistralTimestampGranularity>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub response_format: Option<Option<String>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralTranscriptionSegment {
    pub text: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub start: Option<Option<f64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub end: Option<Option<f64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub score: Option<Option<f64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub speaker_id: Option<Option<String>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralTranscriptionResponse {
    pub model: String,
    pub text: String,
    pub usage: MistralUsage,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub language: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub segments: Option<Option<Vec<MistralTranscriptionSegment>>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type")]
pub enum MistralTranscriptionEvent {
    #[serde(rename = "transcription.text.delta")]
    TextDelta {
        text: String,
        #[serde(flatten)]
        extra_fields: Map<String, Value>,
    },
    #[serde(rename = "transcription.segment")]
    Segment {
        text: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "serde_with::rust::double_option::deserialize"
        )]
        start: Option<Option<f64>>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "serde_with::rust::double_option::deserialize"
        )]
        end: Option<Option<f64>>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "serde_with::rust::double_option::deserialize"
        )]
        speaker_id: Option<Option<String>>,
        #[serde(flatten)]
        extra_fields: Map<String, Value>,
    },
    #[serde(rename = "transcription.language")]
    Language {
        audio_language: String,
        #[serde(flatten)]
        extra_fields: Map<String, Value>,
    },
    #[serde(rename = "transcription.done")]
    Done {
        #[serde(flatten)]
        response: MistralTranscriptionResponse,
    },
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MistralSpeechFormat {
    Pcm,
    Wav,
    Mp3,
    Flac,
    Opus,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralSpeechRequest {
    pub input: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub model: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub voice_id: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub ref_audio: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub response_format: Option<Option<MistralSpeechFormat>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub stream: Option<Option<bool>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub metadata: Option<Option<Map<String, Value>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub prompt_cache_key: Option<Option<String>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralSpeechResponse {
    pub audio_data: String,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
#[serde(tag = "type")]
pub enum MistralSpeechEvent {
    #[serde(rename = "speech.audio.delta")]
    AudioDelta {
        audio_data: String,
        #[serde(flatten)]
        extra_fields: Map<String, Value>,
    },
    #[serde(rename = "speech.audio.done")]
    Done {
        usage: MistralUsage,
        #[serde(flatten)]
        extra_fields: Map<String, Value>,
    },
}

#[cfg(test)]
mod tests {
    use super::{
        MistralSpeechEvent, MistralSpeechRequest, MistralTranscriptionEvent,
        MistralTranscriptionFields, MistralTranscriptionResponse,
    };
    use crate::recognized::Recognized;
    use rstest::rstest;
    use serde_json::{Value, json};

    #[rstest]
    fn transcription_fields_preserve_provider_options() {
        let wire = json!({
            "model":"audio-model", "diarize":true, "context_bias":["term"],
            "timestamp_granularities":["word","segment"], "language":"en", "stream":false,
            "temperature":0.2, "future_field":null
        });
        let fields: MistralTranscriptionFields = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(fields.diarize, Some(Some(true)));
        assert_eq!(serde_json::to_value(fields).unwrap(), wire);
    }

    #[rstest]
    fn transcription_preserves_diarization_and_audio_usage() {
        let wire = json!({
            "model":"audio-model", "text":"hello", "language":"en",
            "segments":[{"type":"transcription_segment","text":"hello","start":0.0,"end":1.5,"speaker_id":"speaker","score":null}],
            "usage":{"prompt_audio_seconds":2, "future_usage":null}, "future_field":true
        });
        let response: MistralTranscriptionResponse = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(
            response.segments.as_ref().unwrap().as_ref().unwrap()[0]
                .speaker_id
                .as_ref()
                .unwrap()
                .as_deref(),
            Some("speaker")
        );
        assert_eq!(response.usage.prompt_audio_seconds, Some(Some(2)));
        let encoded = serde_json::to_value(response).unwrap();
        assert_eq!(encoded["segments"], wire["segments"]);
        assert_eq!(
            encoded["usage"]["future_usage"],
            wire["usage"]["future_usage"]
        );
        assert_eq!(encoded["future_field"], wire["future_field"]);
    }

    #[rstest]
    #[case::text(json!({"type":"transcription.text.delta","text":"hello","future":null}))]
    #[case::segment(json!({"type":"transcription.segment","text":"hello","start":null,"end":1.5,"speaker_id":"speaker"}))]
    #[case::language(json!({"type":"transcription.language","audio_language":"en"}))]
    fn transcription_events_round_trip(#[case] wire: Value) {
        let event: MistralTranscriptionEvent = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(event).unwrap(), wire);
    }

    #[rstest]
    fn speech_uses_native_voice_and_base64_audio() {
        let wire = json!({"model":"speech-model","input":"hello","voice_id":"voice-id","ref_audio":"AA==","response_format":"wav"});
        let request: MistralSpeechRequest = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(
            request.voice_id.as_ref().unwrap().as_deref(),
            Some("voice-id")
        );
        assert_eq!(serde_json::to_value(request).unwrap(), wire);
        let event_wire = json!({"type":"speech.audio.delta","audio_data":"AA==","future":null});
        let event: MistralSpeechEvent = serde_json::from_value(event_wire.clone()).unwrap();
        assert!(
            matches!(&event, MistralSpeechEvent::AudioDelta { audio_data, .. } if audio_data == "AA==")
        );
        assert_eq!(serde_json::to_value(event).unwrap(), event_wire);
    }

    #[rstest]
    fn terminal_audio_events_preserve_native_usage() {
        let speech_wire = json!({"type":"speech.audio.done","usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5,"request_count":1}});
        let speech: MistralSpeechEvent = serde_json::from_value(speech_wire.clone()).unwrap();
        assert!(
            matches!(&speech, MistralSpeechEvent::Done { usage, .. } if usage.total_tokens == Some(Some(5)))
        );
        assert_eq!(serde_json::to_value(speech).unwrap(), speech_wire);
        let transcript_wire = json!({"type":"transcription.done","model":"audio-model","text":"hello","language":null,"usage":{"prompt_audio_seconds":2}});
        let transcript: MistralTranscriptionEvent =
            serde_json::from_value(transcript_wire.clone()).unwrap();
        assert!(
            matches!(&transcript, MistralTranscriptionEvent::Done { response } if response.text == "hello")
        );
        assert_eq!(serde_json::to_value(transcript).unwrap(), transcript_wire);
    }

    #[rstest]
    fn unknown_audio_events_are_preserved_by_the_open_union() {
        let wire = json!({"type":"future.audio.event","payload":{"nested":null}});
        let event: Recognized<MistralSpeechEvent> = serde_json::from_value(wire.clone()).unwrap();
        assert!(event.known().is_none());
        assert_eq!(serde_json::to_value(event).unwrap(), wire);
    }

    #[rstest]
    #[case::delta(json!({"type":"transcription.text.delta","text":false}))]
    #[case::segment(json!({"type":"transcription.segment","text":"hello","start":"bad"}))]
    fn transcription_events_reject_malformed_known_payloads(#[case] wire: Value) {
        assert!(serde_json::from_value::<MistralTranscriptionEvent>(wire).is_err());
    }
}
