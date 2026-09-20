//! Frames: a 4-byte big-endian length, then that many bytes of JSON, one object tagged by
//! `type`. The worker only answers; there is no frame by which it calls into the gateway.
//! Everything a worker sends is untrusted: sizes are capped, unknown frames and fields are
//! errors, and a patch is only data until the session validates it.

use std::io::{self, Read, Write};

use litellm_callbacks_v1::{Envelope, RequestFacts, Subscription, WirePatch};
use serde::{Deserialize, Serialize};

pub const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// `subscriber` is an index into the hello's subscriptions.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToWorker<'a> {
    /// Fire and forget: nothing an observer does comes back, except a [`FromWorker::Report`].
    Event {
        subscriber: usize,
        envelope: &'a Envelope,
    },
    Intercept {
        id: u64,
        subscriber: usize,
        request: &'a RequestFacts,
    },
    /// Answered by `Flushed` once every event sent before it has been handled.
    Flush { id: u64 },
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum FromWorker {
    Hello {
        schema: u32,
        subscriptions: Vec<Subscription>,
    },
    /// `None` is an interceptor that returned nothing.
    Patch {
        id: u64,
        patch: Option<WirePatch>,
    },
    Error {
        id: u64,
        error_class: String,
        message: String,
    },
    /// An observer raised; the worker swallowed it, as the contract says.
    Report {
        subscriber: usize,
        event: Option<String>,
        message: String,
    },
    Flushed {
        id: u64,
    },
}

pub fn write_frame(writer: &mut impl Write, frame: &ToWorker<'_>) -> io::Result<()> {
    let body = serde_json::to_vec(frame)?;
    let length = u32::try_from(body.len())
        .ok()
        .filter(|_| body.len() <= MAX_FRAME_BYTES)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "frame too large"))?;
    writer.write_all(&length.to_be_bytes())?;
    writer.write_all(&body)?;
    writer.flush()
}

/// The next frame, or `None` when the worker closed the connection between frames.
pub fn read_frame(reader: &mut impl Read) -> io::Result<Option<FromWorker>> {
    let mut header = [0u8; 4];
    match reader.read_exact(&mut header) {
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        other => other?,
    }
    let length = u32::from_be_bytes(header) as usize;
    if length > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame of {length} bytes exceeds {MAX_FRAME_BYTES}"),
        ));
    }
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body)?;
    serde_json::from_slice(&body)
        .map(Some)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use litellm_callbacks_v1::{CallFacts, CallSession, EventKind, HeaderPatch};
    use litellm_host::event::{RequestContext, WireRequest};
    use serde_json::{Value, json};

    use super::*;

    fn subscriptions() -> Vec<Subscription> {
        vec![
            Subscription {
                name: "recorder".to_string(),
                events: BTreeSet::from([EventKind::CallFailed, EventKind::CallSucceeded]),
                observes: true,
                intercepts: false,
            },
            Subscription {
                name: "my_pkg.cb.patch".to_string(),
                events: BTreeSet::new(),
                observes: false,
                intercepts: true,
            },
        ]
    }

    fn golden(text: &str) -> Value {
        serde_json::from_str(text).unwrap()
    }

    /// The frames carry the contract's own golden envelope and request, so the two golden
    /// directories cannot drift apart.
    #[test]
    fn frames_to_the_worker_match_golden() {
        let facts = CallFacts {
            start_time: 10.0,
            asynchronous: true,
            metadata: json!({"team": "core"}).as_object().unwrap().clone(),
            metadata_dropped: vec!["broken".to_string()],
        };
        let (session, started) =
            CallSession::start("call-123".to_string(), "ocr", subscriptions(), facts);
        let interception = session.interception(
            WireRequest {
                url: "https://provider.example/v1".to_string(),
                headers: vec![("Authorization".to_string(), "secret".to_string())],
                body: json!({"input": "hello"}),
            },
            RequestContext {
                model: "model".to_string(),
                custom_llm_provider: "provider".to_string(),
                optional_params: json!({"temperature": 0.2}),
                secret_fields: Vec::new(),
                api_key: None,
            },
        );
        let turn = interception.turn().unwrap();

        let frames = [
            (
                ToWorker::Event {
                    subscriber: 0,
                    envelope: &started.envelope,
                },
                include_str!("../golden/to_worker/event.json"),
            ),
            (
                ToWorker::Intercept {
                    id: 1,
                    subscriber: turn.subscriber,
                    request: &turn.request,
                },
                include_str!("../golden/to_worker/intercept.json"),
            ),
            (
                ToWorker::Flush { id: 2 },
                include_str!("../golden/to_worker/flush.json"),
            ),
        ];
        for (frame, expected) in frames {
            assert_eq!(serde_json::to_value(&frame).unwrap(), golden(expected));
        }
    }

    #[test]
    fn frames_from_the_worker_parse_from_golden() {
        let parsed = |text: &str| serde_json::from_str::<FromWorker>(text).unwrap();

        assert_eq!(
            parsed(include_str!("../golden/from_worker/hello.json")),
            FromWorker::Hello {
                schema: 1,
                subscriptions: subscriptions()
            }
        );
        assert_eq!(
            parsed(include_str!("../golden/from_worker/patch.json")),
            FromWorker::Patch {
                id: 1,
                patch: Some(WirePatch {
                    headers: HeaderPatch {
                        remove: vec!["X-Trace".to_string()],
                        set: vec![("X-Team".to_string(), "core".to_string())],
                    },
                    body: Some(json!({"input": "patched"})),
                }),
            }
        );
        assert_eq!(
            parsed(include_str!("../golden/from_worker/patch_none.json")),
            FromWorker::Patch { id: 1, patch: None }
        );
        assert_eq!(
            parsed(include_str!("../golden/from_worker/error.json")),
            FromWorker::Error {
                id: 1,
                error_class: "builtins.PermissionError".to_string(),
                message: "not this model".to_string(),
            }
        );
        assert_eq!(
            parsed(include_str!("../golden/from_worker/report.json")),
            FromWorker::Report {
                subscriber: 0,
                event: Some("call.started".to_string()),
                message: "boom".to_string(),
            }
        );
        assert_eq!(
            parsed(include_str!("../golden/from_worker/flushed.json")),
            FromWorker::Flushed { id: 2 }
        );
    }

    fn framed(body: &[u8]) -> Vec<u8> {
        [&(body.len() as u32).to_be_bytes()[..], body].concat()
    }

    #[test]
    fn a_worker_cannot_say_anything_the_protocol_does_not_name() {
        for body in [
            &br#"{"type":"call_gateway","method":"get_secret"}"#[..],
            br#"{"type":"flushed","id":1,"extra":true}"#,
            br#"{"type":"patch","id":1,"patch":{"url":"https://evil.example"}}"#,
            br#"[1,2,3]"#,
        ] {
            let error = read_frame(&mut framed(body).as_slice()).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        }
    }

    #[test]
    fn an_oversized_frame_is_refused_before_it_is_read() {
        let header = ((MAX_FRAME_BYTES + 1) as u32).to_be_bytes();
        let error = read_frame(&mut header.as_slice()).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn a_closed_connection_between_frames_is_the_end_not_an_error() {
        assert_eq!(read_frame(&mut [].as_slice()).unwrap(), None);
    }
}
