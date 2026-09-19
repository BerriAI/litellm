use litellm_host::event::{FailureOrigin, RequestContext, Timing, WireRequest};
use serde::Serialize;
use serde_json::{Map, Value};
use strum::{EnumString, IntoStaticStr, VariantArray};

use crate::redact;

pub const SCHEMA_V1: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Envelope {
    pub schema: u32,
    pub call_id: String,
    pub call_type: &'static str,
    pub seq: u32,
    pub event: Event,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum Event {
    #[serde(rename = "call.started")]
    CallStarted(CallFacts),
    #[serde(rename = "request.sending")]
    RequestSending(RequestFacts),
    #[serde(rename = "response.received")]
    ResponseReceived { body: String },
    #[serde(rename = "call.succeeded")]
    CallSucceeded {
        timing: TimingFacts,
        streamed: bool,
        response: Value,
        response_error: Option<String>,
    },
    #[serde(rename = "call.failed")]
    CallFailed {
        timing: TimingFacts,
        streamed: bool,
        origin: Origin,
        error: ErrorFacts,
    },
}

#[derive(
    EnumString,
    IntoStaticStr,
    VariantArray,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Debug,
)]
pub enum EventKind {
    #[strum(serialize = "call.started")]
    CallStarted,
    #[strum(serialize = "request.sending")]
    RequestSending,
    #[strum(serialize = "response.received")]
    ResponseReceived,
    #[strum(serialize = "call.succeeded")]
    CallSucceeded,
    #[strum(serialize = "call.failed")]
    CallFailed,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CallFacts {
    pub start_time: f64,
    pub asynchronous: bool,
    pub metadata: Map<String, Value>,
    pub metadata_dropped: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RequestFacts {
    pub model: String,
    pub custom_llm_provider: String,
    pub optional_params: Value,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Value,
}

impl RequestFacts {
    pub fn new(wire: &WireRequest, context: &RequestContext) -> Self {
        Self {
            model: context.model.clone(),
            custom_llm_provider: context.custom_llm_provider.clone(),
            optional_params: redact::params(&context.optional_params, &context.secret_fields),
            url: wire.url.clone(),
            headers: redact::headers(&wire.headers),
            body: wire.body.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct TimingFacts {
    pub start_time: f64,
    pub end_time: f64,
    pub duration: f64,
}

impl From<Timing> for TimingFacts {
    fn from(timing: Timing) -> Self {
        Self {
            start_time: timing.start_time,
            end_time: timing.end_time,
            duration: timing.end_time - timing.start_time,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Origin {
    Call,
    Host,
}

impl From<FailureOrigin> for Origin {
    fn from(origin: FailureOrigin) -> Self {
        match origin {
            FailureOrigin::Call => Self::Call,
            FailureOrigin::Host => Self::Host,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ErrorFacts {
    pub class: String,
    pub message: String,
    pub status_code: Option<u16>,
}

impl Event {
    pub fn kind(&self) -> EventKind {
        match self {
            Self::CallStarted(_) => EventKind::CallStarted,
            Self::RequestSending(_) => EventKind::RequestSending,
            Self::ResponseReceived { .. } => EventKind::ResponseReceived,
            Self::CallSucceeded { .. } => EventKind::CallSucceeded,
            Self::CallFailed { .. } => EventKind::CallFailed,
        }
    }
}

pub struct Sequencer {
    call_id: String,
    call_type: &'static str,
    next: u32,
}

impl Sequencer {
    pub fn new(call_id: String, call_type: &'static str) -> Self {
        Self {
            call_id,
            call_type,
            next: 0,
        }
    }

    pub fn envelope(&mut self, event: Event) -> Envelope {
        let envelope = Envelope {
            schema: SCHEMA_V1,
            call_id: self.call_id.clone(),
            call_type: self.call_type,
            seq: self.next,
            event,
        };
        self.next += 1;
        envelope
    }
}

#[cfg(test)]
mod tests {
    use litellm_auth::SecretValue;
    use proptest::prelude::*;
    use serde_json::json;
    use strum::VariantArray;

    use super::*;

    fn event(kind: EventKind) -> Event {
        match kind {
            EventKind::CallStarted => Event::CallStarted(CallFacts {
                start_time: 10.0,
                asynchronous: true,
                metadata: Map::from_iter([("team".to_string(), json!("core"))]),
                metadata_dropped: vec!["broken".to_string()],
            }),
            EventKind::RequestSending => Event::RequestSending(RequestFacts {
                model: "model".to_string(),
                custom_llm_provider: "provider".to_string(),
                optional_params: json!({"temperature": 0.2}),
                url: "https://provider.example/v1".to_string(),
                headers: vec![("Authorization".to_string(), redact::REDACTED.to_string())],
                body: json!({"input": "hello"}),
            }),
            EventKind::ResponseReceived => Event::ResponseReceived {
                body: "raw response".to_string(),
            },
            EventKind::CallSucceeded => Event::CallSucceeded {
                timing: TimingFacts {
                    start_time: 10.0,
                    end_time: 12.5,
                    duration: 2.5,
                },
                streamed: false,
                response: json!({"id": "response"}),
                response_error: None,
            },
            EventKind::CallFailed => Event::CallFailed {
                timing: TimingFacts {
                    start_time: 10.0,
                    end_time: 11.0,
                    duration: 1.0,
                },
                streamed: true,
                origin: Origin::Call,
                error: ErrorFacts {
                    class: "builtins.ValueError".to_string(),
                    message: "bad request".to_string(),
                    status_code: Some(400),
                },
            },
        }
    }

    fn golden(kind: EventKind, seq: u32) -> Envelope {
        Envelope {
            schema: SCHEMA_V1,
            call_id: "call-123".to_string(),
            call_type: "ocr",
            seq,
            event: event(kind),
        }
    }

    #[rstest::rstest]
    #[case(EventKind::CallStarted, 0, include_str!("../golden/v1/call.started.json"))]
    #[case(EventKind::RequestSending, 1, include_str!("../golden/v1/request.sending.json"))]
    #[case(EventKind::ResponseReceived, 2, include_str!("../golden/v1/response.received.json"))]
    #[case(EventKind::CallSucceeded, 3, include_str!("../golden/v1/call.succeeded.json"))]
    #[case(EventKind::CallFailed, 4, include_str!("../golden/v1/call.failed.json"))]
    fn serialized_v1_envelopes_match_golden_contract(
        #[case] kind: EventKind,
        #[case] seq: u32,
        #[case] expected: &str,
    ) {
        assert_eq!(
            serde_json::to_value(golden(kind, seq)).unwrap(),
            serde_json::from_str::<Value>(expected).unwrap()
        );
    }

    #[test]
    fn event_tags_equal_subscription_names() {
        for &kind in EventKind::VARIANTS {
            let value = serde_json::to_value(event(kind)).unwrap();
            let expected: &'static str = kind.into();
            assert_eq!(value["type"], expected);
        }
    }

    #[test]
    fn request_facts_never_serialize_credentials() {
        let sentinel = "sentinel-secret-78e4";
        let wire = WireRequest {
            url: "https://provider.example".to_string(),
            headers: vec![("Authorization".to_string(), sentinel.to_string())],
            body: json!({"input": "safe"}),
        };
        let context = RequestContext {
            model: "model".to_string(),
            custom_llm_provider: "provider".to_string(),
            optional_params: json!({"api_key": sentinel}),
            secret_fields: vec!["api_key".to_string()],
            api_key: Some(SecretValue::new(sentinel.to_string())),
        };
        assert!(
            !serde_json::to_string(&RequestFacts::new(&wire, &context))
                .unwrap()
                .contains(sentinel)
        );
    }

    proptest! {
        #[test]
        fn sequence_numbers_increase_from_zero(length in 0usize..100) {
            let mut sequencer = Sequencer::new("call".to_string(), "ocr");
            let sequence: Vec<u32> = (0..length)
                .map(|_| sequencer.envelope(event(EventKind::ResponseReceived)).seq)
                .collect();
            prop_assert_eq!(sequence, (0..length as u32).collect::<Vec<_>>());
        }
    }
}
