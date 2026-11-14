use std::sync::{Arc, Mutex, PoisonError};

use base64::Engine;
use bytes::BytesMut;
use futures_util::{StreamExt, stream};
use litellm_framer::{
    EventStreamError,
    aws_event_stream::{AwsEventStreamCodec, HeaderValue, Message},
    frames,
};
use litellm_python_compat::{Value as PythonValue, json as python_json, repr};
use serde_json::Value;
use tokio_util::codec::Decoder;

use crate::{Error, ErrorDetail, base_llm::base_model_iterator::tunnelled_http};

const STREAM_HEAD_BYTES: usize = 200;
const PRELUDE_BYTES: usize = 12;
const MAX_HEADERS_BYTES: u32 = 128 * 1024;
const MAX_PAYLOAD_BYTES: i64 = 24 * 1024 * 1024;

const PROVIDER_REQUEST_ID_HEADERS: [&str; 7] = [
    "x-amzn-requestid",
    "x-request-id",
    "request-id",
    "x-ms-request-id",
    "apim-request-id",
    "x-goog-request-id",
    "cf-ray",
];

const STREAM_EVENT_STATUSES: [(&str, Option<u16>); 13] = [
    ("chunk", None),
    ("messageStart", None),
    ("contentBlockStart", None),
    ("contentBlockDelta", None),
    ("contentBlockStop", None),
    ("messageStop", None),
    ("metadata", None),
    ("internalServerException", Some(500)),
    ("modelStreamErrorException", Some(424)),
    ("validationException", Some(400)),
    ("throttlingException", Some(429)),
    ("modelTimeoutException", Some(408)),
    ("serviceUnavailableException", Some(503)),
];

const RESPONSE_STREAM_EXCEPTION_STATUSES: [(&str, u16); 6] = [
    ("internalServerException", 500),
    ("modelStreamErrorException", 424),
    ("validationException", 400),
    ("throttlingException", 429),
    ("modelTimeoutException", 408),
    ("serviceUnavailableException", 503),
];

/// An exception Python's Bedrock stream decoder raises as it is, unmapped: botocore's
/// `EventStreamBuffer` parser errors, `json.loads` failing on an event payload, and the
/// chunk parser's `pop` on a payload that is not a JSON object.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BedrockStreamRaise {
    #[error("Checksum mismatch: expected 0x{expected:08x}, calculated 0x{calculated:08x}")]
    ChecksumMismatch { expected: u32, calculated: u32 },
    #[error("Header length of {0} exceeded the maximum of {MAX_HEADERS_BYTES}")]
    InvalidHeadersLength(u32),
    #[error("Payload length of {0} exceeded the maximum of {MAX_PAYLOAD_BYTES}")]
    InvalidPayloadLength(i64),
    #[error("Duplicate header present: \"{0}\"")]
    DuplicateHeader(String),
    #[error("Bedrock event payload is not JSON: {0}")]
    InvalidJson(String),
    #[error("Bedrock chunk is not a JSON object: {0}")]
    NotAnObject(String),
}

pub(super) fn raised(raise: BedrockStreamRaise) -> Error {
    Error::InvalidResponse(ErrorDetail::failed("Bedrock event stream", raise))
}

#[derive(Default)]
struct Tally {
    bytes_received: usize,
    bytes_decoded: usize,
    events: usize,
    recognized_events: usize,
    unrecognized_event_types: std::collections::BTreeSet<String>,
    unrecognized_head: Vec<u8>,
    head: Vec<u8>,
}

type SharedTally = Arc<Mutex<Tally>>;

fn with_tally<T>(tally: &SharedTally, update: impl FnOnce(&mut Tally) -> T) -> T {
    update(&mut tally.lock().unwrap_or_else(PoisonError::into_inner))
}

#[derive(Debug, thiserror::Error)]
enum FrameError {
    #[error(transparent)]
    Raise(#[from] BedrockStreamRaise),
    #[error(transparent)]
    Framing(#[from] EventStreamError),
}

fn be_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn checked(data: &[u8], expected: u32, seed: u32) -> Result<(), BedrockStreamRaise> {
    let mut hasher = crc32fast::Hasher::new_with_initial(seed);
    hasher.update(data);
    let calculated = hasher.finalize();
    if calculated == expected {
        return Ok(());
    }
    Err(BedrockStreamRaise::ChecksumMismatch {
        expected,
        calculated,
    })
}

/// Validates each frame in the order botocore's `EventStreamBuffer` does, so a body that is not
/// an event stream fails with the parser error Python raises.
struct BotocoreCodec {
    tally: SharedTally,
}

impl BotocoreCodec {
    fn prelude(src: &[u8]) -> Result<usize, BedrockStreamRaise> {
        let total = be_u32(src, 0);
        let headers = be_u32(src, 4);
        checked(&src[..PRELUDE_BYTES - 4], be_u32(src, 8), 0)?;
        if headers > MAX_HEADERS_BYTES {
            return Err(BedrockStreamRaise::InvalidHeadersLength(headers));
        }
        let payload = i64::from(total) - i64::from(headers) - 16;
        if payload > MAX_PAYLOAD_BYTES {
            return Err(BedrockStreamRaise::InvalidPayloadLength(payload));
        }
        Ok(total as usize)
    }

    fn unique_headers(message: &Message) -> Result<(), BedrockStreamRaise> {
        let mut seen = std::collections::HashSet::new();
        message
            .headers()
            .iter()
            .map(|header| header.name().as_str())
            .find(|name| !seen.insert(*name))
            .map_or(Ok(()), |name| {
                Err(BedrockStreamRaise::DuplicateHeader(name.to_string()))
            })
    }
}

impl Decoder for BotocoreCodec {
    type Item = Message;
    type Error = FrameError;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Message>, FrameError> {
        if src.len() < PRELUDE_BYTES {
            return Ok(None);
        }
        let length = Self::prelude(src)?;
        if src.len() < length {
            return Ok(None);
        }
        if length >= 16 {
            checked(
                &src[PRELUDE_BYTES - 4..length - 4],
                be_u32(src, length - 4),
                be_u32(src, 8),
            )?;
        }
        let Some(message) = AwsEventStreamCodec.decode(src)? else {
            return Ok(None);
        };
        Self::unique_headers(&message)?;
        with_tally(&self.tally, |tally| tally.bytes_decoded += length);
        Ok(Some(message))
    }

    fn decode_eof(&mut self, src: &mut BytesMut) -> Result<Option<Message>, FrameError> {
        self.decode(src)
    }
}

impl From<std::io::Error> for FrameError {
    fn from(error: std::io::Error) -> Self {
        Self::Framing(EventStreamError::Body(error))
    }
}

fn header_text<'a>(message: &'a Message, name: &str) -> Option<&'a str> {
    message
        .headers()
        .iter()
        .find(|header| header.name().as_str() == name)
        .and_then(|header| match header.value() {
            HeaderValue::String(value) => Some(value.as_str()),
            _ => None,
        })
}

fn event_status(event_type: &str) -> Option<Option<u16>> {
    STREAM_EVENT_STATUSES
        .iter()
        .find(|(known, _)| *known == event_type)
        .map(|(_, status)| *status)
}

fn python_bytes_repr(bytes: &[u8]) -> String {
    repr::repr(&PythonValue::Bytes(bytes.to_vec()))
}

fn python_optional_str_repr(value: Option<&str>) -> String {
    repr::repr(&value.map_or(PythonValue::None, |text| PythonValue::Str(text.to_string())))
}

fn response_header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

fn request_id_headers(headers: &[(String, String)]) -> Vec<(String, String)> {
    headers
        .iter()
        .filter(|(name, value)| {
            PROVIDER_REQUEST_ID_HEADERS.contains(&name.to_lowercase().as_str()) && value.is_ascii()
        })
        .cloned()
        .collect()
}

fn bedrock_error(status: u16, message: String, headers: &[(String, String)]) -> Error {
    Error::Http(litellm_http::Error::upstream_with_headers(
        status,
        message,
        request_id_headers(headers),
    ))
}

fn record_event(tally: &SharedTally, message: &Message) {
    let message_type = header_text(message, ":message-type");
    let event_type = header_text(message, ":event-type");
    with_tally(tally, |tally| {
        tally.events += 1;
        if message_type != Some("event") || event_type.and_then(event_status).is_some() {
            tally.recognized_events += 1;
            return;
        }
        tally
            .unrecognized_event_types
            .insert(event_type.unwrap_or("<missing>").to_string());
        if tally.unrecognized_head.is_empty() {
            let payload = message.payload();
            tally.unrecognized_head = payload[..payload.len().min(STREAM_HEAD_BYTES)].to_vec();
        }
    });
}

fn undecoded_stream_error(tally: &Tally, headers: &[(String, String)]) -> Option<Error> {
    let undecoded = tally.bytes_received.saturating_sub(tally.bytes_decoded);
    if tally.recognized_events > 0 && undecoded == 0 {
        return None;
    }
    let detail = format!(
        "content-type={}, x-amzn-requestid={}, {} bytes received",
        python_optional_str_repr(response_header(headers, "content-type")),
        python_optional_str_repr(response_header(headers, "x-amzn-requestid")),
        tally.bytes_received
    );
    let message = if tally.events == 0 {
        format!(
            "Bedrock answered the stream with HTTP 200 but its body decoded to no events ({detail}, first bytes={})",
            python_bytes_repr(&tally.head)
        )
    } else if tally.recognized_events == 0 {
        let event_types = PythonValue::List(
            tally
                .unrecognized_event_types
                .iter()
                .cloned()
                .map(PythonValue::Str)
                .collect(),
        );
        format!(
            "Bedrock answered the stream with HTTP 200 but none of its {} events carried a known event type (event types={}, {detail}, first payload={})",
            tally.events,
            repr::repr(&event_types),
            python_bytes_repr(&tally.unrecognized_head)
        )
    } else {
        format!(
            "Bedrock stream ended with {undecoded} undecoded bytes after {} events ({detail})",
            tally.events
        )
    };
    Some(bedrock_error(502, message, headers))
}

fn framing(error: FrameError) -> Error {
    let error = match error {
        FrameError::Raise(raise) => return raised(raise),
        FrameError::Framing(error) => error,
    };
    match &error {
        EventStreamError::Body(body) if let Some(http) = tunnelled_http(body) => {
            Error::Http(http.clone())
        }
        _ => Error::InvalidResponse(ErrorDetail::failed("stream framing", error)),
    }
}

fn invalid_payload(error: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::InvalidResponse(ErrorDetail::invalid("Bedrock event payload", error))
}

fn chunk_text(payload: &[u8]) -> Result<Option<Vec<u8>>, Error> {
    let parsed: Value = serde_json::from_slice(payload).map_err(invalid_payload)?;
    let Some(chunk) = parsed.as_object().filter(|chunk| !chunk.is_empty()) else {
        return Ok(None);
    };
    let encoded = chunk
        .get("bytes")
        .and_then(Value::as_str)
        .ok_or(Error::InvalidResponse(ErrorDetail::MissingField {
            subject: "Bedrock chunk event",
            field: "bytes",
        }))?;
    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map(Some)
        .map_err(invalid_payload)
}

fn decoded_event(
    message: &Message,
    headers: &[(String, String)],
) -> Result<Option<PythonValue>, Error> {
    let body = String::from_utf8_lossy(message.payload());
    if header_text(message, ":message-type") != Some("event") {
        let error = match header_text(message, ":exception-type") {
            Some(exception_type) => {
                let status = RESPONSE_STREAM_EXCEPTION_STATUSES
                    .iter()
                    .find(|(known, _)| *known == exception_type)
                    .map_or(400, |(_, status)| *status);
                bedrock_error(status, format!("{exception_type} {body}"), headers)
            }
            None => bedrock_error(400, body.into_owned(), headers),
        };
        return Err(error);
    }
    let event_type = header_text(message, ":event-type");
    if let Some((event_type, status)) =
        event_type.and_then(|event_type| Some((event_type, event_status(event_type).flatten()?)))
    {
        return Err(bedrock_error(
            status,
            format!("{event_type} {body}"),
            headers,
        ));
    }
    let text = if event_type == Some("chunk") {
        chunk_text(message.payload())?
    } else {
        Some(message.payload().to_vec()).filter(|payload| !payload.is_empty())
    };
    text.filter(|text| !text.is_empty())
        .map(|text| {
            let text = String::from_utf8_lossy(&text);
            python_json::loads(&text)
                .map_err(|_| raised(BedrockStreamRaise::InvalidJson(text.into_owned())))
        })
        .transpose()
}

pub fn bedrock_invoke_chunk_stream(
    bytes: crate::base_llm::messages::streaming::ByteStream,
    headers: Vec<(String, String)>,
) -> impl futures_util::Stream<Item = Result<PythonValue, Error>> + Send {
    let tally = SharedTally::default();
    let counted = {
        let tally = tally.clone();
        bytes.inspect(move |chunk| {
            if let Ok(chunk) = chunk {
                with_tally(&tally, |tally| {
                    tally.bytes_received += chunk.len();
                    let room = STREAM_HEAD_BYTES.saturating_sub(tally.head.len());
                    tally
                        .head
                        .extend_from_slice(&chunk[..chunk.len().min(room)]);
                });
            }
        })
    };
    let headers = Arc::new(headers);
    let events = {
        let tally = tally.clone();
        let headers = headers.clone();
        frames(
            counted,
            BotocoreCodec {
                tally: tally.clone(),
            },
        )
        .map(move |message| {
            let message = message.map_err(framing)?;
            record_event(&tally, &message);
            decoded_event(&message, &headers)
        })
    };
    let end = stream::once(async move {
        with_tally(&tally, |tally| undecoded_stream_error(tally, &headers)).map_or(Ok(None), Err)
    });
    events
        .chain(end)
        .scan(false, |failed, item| {
            if *failed {
                return futures_util::future::ready(None);
            }
            *failed = item.is_err();
            futures_util::future::ready(Some(item))
        })
        .filter_map(|item| futures_util::future::ready(item.transpose()))
}

#[cfg(test)]
mod tests {
    use aws_smithy_eventstream::frame::write_message_to;
    use aws_smithy_types::event_stream::{
        Header, HeaderValue as WireHeaderValue, Message as WireMessage,
    };
    use base64::engine::general_purpose::STANDARD;
    use bytes::Bytes;
    use futures_util::StreamExt;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn frame(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
        let message = headers.iter().fold(
            WireMessage::new(Bytes::copy_from_slice(payload)),
            |message, (name, value)| {
                message.add_header(Header::new(
                    name.to_string(),
                    WireHeaderValue::String(value.to_string().into()),
                ))
            },
        );
        let mut wire = Vec::new();
        write_message_to(&message, &mut wire).unwrap();
        wire
    }

    fn chunk(event: &Value) -> Vec<u8> {
        let payload = json!({"bytes": STANDARD.encode(event.to_string())});
        frame(
            &[(":event-type", "chunk"), (":message-type", "event")],
            payload.to_string().as_bytes(),
        )
    }

    fn response_headers() -> Vec<(String, String)> {
        vec![
            (
                "content-type".into(),
                "application/vnd.amazon.eventstream".into(),
            ),
            ("x-amzn-requestid".into(), "rid-1".into()),
            ("x-other".into(), "dropped".into()),
        ]
    }

    async fn decoded(wire: Vec<u8>) -> (Vec<PythonValue>, Option<Error>) {
        let bytes = futures_util::stream::iter(
            wire.chunks(5)
                .map(|piece| Ok(Bytes::copy_from_slice(piece)))
                .collect::<Vec<_>>(),
        )
        .boxed();
        let items: Vec<Result<PythonValue, Error>> =
            bedrock_invoke_chunk_stream(bytes, response_headers())
                .collect()
                .await;
        let (events, errors): (Vec<_>, Vec<_>) = items.into_iter().partition(Result::is_ok);
        (
            events.into_iter().filter_map(Result::ok).collect(),
            errors.into_iter().find_map(Result::err),
        )
    }

    fn upstream(error: Option<Error>) -> (u16, String, Vec<(String, String)>) {
        let Some(Error::Http(error)) = error else {
            panic!("expected an upstream error, got {error:?}");
        };
        let (status, body) = error.upstream_response().unwrap();
        (status, body.to_string(), error.upstream_headers().to_vec())
    }

    #[tokio::test]
    async fn chunk_events_decode_to_anthropic_events() {
        let event = json!({"type": "message_stop"});
        let (events, error) = decoded(chunk(&event)).await;
        assert_eq!(
            (events, error.is_none()),
            (vec![python_json::loads(&event.to_string()).unwrap()], true)
        );
    }

    #[rstest]
    #[case::throttling("throttlingException", 429)]
    #[case::model_stream_error("modelStreamErrorException", 424)]
    #[case::validation("validationException", 400)]
    #[case::unmodeled("somethingNewException", 400)]
    #[tokio::test]
    async fn exception_frames_raise_their_modeled_status(
        #[case] exception_type: &str,
        #[case] status: u16,
    ) {
        let wire = [
            chunk(&json!({"type": "ping"})),
            frame(
                &[
                    (":message-type", "exception"),
                    (":exception-type", exception_type),
                ],
                br#"{"message":"slow down"}"#,
            ),
        ]
        .concat();
        let (events, error) = decoded(wire).await;
        assert_eq!(events.len(), 1);
        assert_eq!(
            upstream(error),
            (
                status,
                format!(r#"{exception_type} {{"message":"slow down"}}"#),
                vec![("x-amzn-requestid".to_string(), "rid-1".to_string())]
            )
        );
    }

    #[tokio::test]
    async fn an_error_event_type_raises_its_status() {
        let wire = frame(
            &[
                (":message-type", "event"),
                (":event-type", "internalServerException"),
            ],
            b"boom",
        );
        assert_eq!(upstream(decoded(wire).await.1).0, 500);
    }

    #[tokio::test]
    async fn an_empty_body_is_a_502() {
        let (_, body, _) = upstream(decoded(Vec::new()).await.1);
        assert_eq!(
            body,
            "Bedrock answered the stream with HTTP 200 but its body decoded to no events (content-type='application/vnd.amazon.eventstream', x-amzn-requestid='rid-1', 0 bytes received, first bytes=b'')"
        );
    }

    #[tokio::test]
    async fn unknown_event_types_are_a_502() {
        let wire = frame(
            &[(":message-type", "event"), (":event-type", "mystery")],
            b"{}",
        );
        let (status, body, _) = upstream(decoded(wire.clone()).await.1);
        assert_eq!(
            (status, body),
            (
                502,
                format!(
                    "Bedrock answered the stream with HTTP 200 but none of its 1 events carried a known event type (event types=['mystery'], content-type='application/vnd.amazon.eventstream', x-amzn-requestid='rid-1', {} bytes received, first payload=b'{{}}')",
                    wire.len()
                )
            )
        );
    }

    #[tokio::test]
    async fn trailing_undecoded_bytes_are_a_502() {
        let complete = chunk(&json!({"type": "ping"}));
        let truncated = chunk(&json!({"type": "message_stop"}));
        let wire = [complete, truncated[..20].to_vec()].concat();
        let (events, error) = decoded(wire).await;
        let (status, body, _) = upstream(error);
        assert_eq!(events.len(), 1);
        assert_eq!(status, 502);
        assert!(
            body.starts_with("Bedrock stream ended with 20 undecoded bytes after 1 events ("),
            "{body}"
        );
    }

    #[tokio::test]
    async fn decoding_stops_at_the_first_error() {
        let wire = [
            frame(
                &[
                    (":message-type", "exception"),
                    (":exception-type", "throttlingException"),
                ],
                b"x",
            ),
            chunk(&json!({"type": "ping"})),
        ]
        .concat();
        let bytes = futures_util::stream::iter([Ok(Bytes::from(wire))]).boxed();
        let items: Vec<Result<PythonValue, Error>> = bedrock_invoke_chunk_stream(bytes, Vec::new())
            .collect()
            .await;
        assert_eq!(items.len(), 1);
        assert!(items[0].is_err());
    }

    fn raise_of(error: Option<Error>) -> BedrockStreamRaise {
        litellm_error::find::<BedrockStreamRaise>(&error.expect("an error"))
            .cloned()
            .expect("a botocore parser error")
    }

    #[tokio::test]
    async fn a_json_body_fails_the_prelude_checksum_as_botocore_does() {
        let body = br#"{"id":"msg_1","type":"message","content":[]}"#;
        assert_eq!(
            raise_of(decoded(body.to_vec()).await.1),
            BedrockStreamRaise::ChecksumMismatch {
                expected: be_u32(body, 8),
                calculated: crc32fast::hash(&body[..8]),
            }
        );
    }

    fn with_prelude(total: u32, headers: u32) -> Vec<u8> {
        let mut prelude = [total.to_be_bytes(), headers.to_be_bytes()].concat();
        let crc = crc32fast::hash(&prelude);
        prelude.extend_from_slice(&crc.to_be_bytes());
        prelude
    }

    #[rstest]
    #[case::headers_too_long(
        with_prelude(200_000, 140_000),
        BedrockStreamRaise::InvalidHeadersLength(140_000)
    )]
    #[case::payload_too_long(
        with_prelude(30_000_000, 0),
        BedrockStreamRaise::InvalidPayloadLength(30_000_000 - 16)
    )]
    #[case::duplicate_header(
        frame(&[(":message-type", "event"), (":message-type", "event")], b"{}"),
        BedrockStreamRaise::DuplicateHeader(":message-type".into())
    )]
    #[tokio::test]
    async fn a_malformed_frame_raises_the_botocore_parser_error(
        #[case] wire: Vec<u8>,
        #[case] expected: BedrockStreamRaise,
    ) {
        assert_eq!(raise_of(decoded(wire).await.1), expected);
    }

    #[tokio::test]
    async fn a_corrupt_message_checksum_raises_a_checksum_mismatch() {
        let mut wire = chunk(&json!({"type": "ping"}));
        let last = wire.len() - 1;
        wire[last] ^= 0xff;
        assert!(matches!(
            raise_of(decoded(wire).await.1),
            BedrockStreamRaise::ChecksumMismatch { .. }
        ));
    }
}
