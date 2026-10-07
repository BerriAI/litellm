use base64::Engine;
use bytes::Buf;
use futures_util::{Stream, StreamExt};
use litellm_framer::{
    aws_event_stream::{AwsEventStreamCodec, Message},
    frames,
};
use litellm_llms_types::formats::messages::streaming::MessagesStreamEvent;
use serde::Deserialize;
use serde_json::Value;

use crate::{
    Error,
    anthropic::chat::handler::ModelResponseIterator,
    base_llm::{
        chat::streaming::{ChatStream, StreamShape},
        messages::streaming::{ByteStream, EventStream},
    },
};

#[derive(Deserialize)]
struct InvokeChunkPayload {
    bytes: String,
}

fn bedrock_stream_error(message: &Message) -> Option<crate::ErrorDetail> {
    let header = |name: &str| {
        message
            .headers()
            .iter()
            .find(|header| header.name().as_str() == name)
            .and_then(|header| header.value().as_string().ok())
            .map(|value| value.as_str())
    };
    if !matches!(header(":message-type"), Some("error" | "exception")) {
        return None;
    }
    let exception = header(":exception-type");
    let status = match exception {
        Some("internalServerException") => 500,
        Some("modelStreamErrorException") => 424,
        Some("modelTimeoutException") => 408,
        Some("serviceUnavailableException") => 503,
        Some("throttlingException") => 429,
        Some("validationException") | Some(_) | None => 400,
    };
    let payload = String::from_utf8_lossy(message.payload());
    let message = match exception {
        Some(exception) => format!("{exception} {payload}"),
        None => payload.into_owned(),
    };
    Some(crate::ErrorDetail::Http { status, message })
}

pub fn decode_invoke_chunk(message: Message) -> Result<Value, Error> {
    if let Some(error) = bedrock_stream_error(&message) {
        return Err(Error::InvalidResponse(error));
    }
    let payload: InvokeChunkPayload =
        serde_json::from_slice(message.payload()).map_err(|error| {
            Error::InvalidResponse(crate::ErrorDetail::invalid("Bedrock event payload", error))
        })?;
    let chunk = base64::engine::general_purpose::STANDARD
        .decode(payload.bytes)
        .map_err(|error| {
            Error::InvalidResponse(crate::ErrorDetail::invalid(
                "Bedrock event payload base64",
                error,
            ))
        })?;
    serde_json::from_slice(&chunk).map_err(|error| {
        Error::InvalidResponse(crate::ErrorDetail::invalid("Anthropic stream event", error))
    })
}

pub fn invoke_chunk_stream<S, B, E>(input: S) -> impl Stream<Item = Result<Value, Error>> + Send
where
    S: Stream<Item = Result<B, E>> + Send,
    B: Buf + Send,
    E: std::error::Error + Send + Sync + 'static,
{
    frames(input, AwsEventStreamCodec).map(|message| {
        decode_invoke_chunk(message.map_err(|error| {
            Error::InvalidResponse(crate::ErrorDetail::failed("stream framing", error))
        })?)
    })
}

pub fn decode_invoke_anthropic_chunk(chunk: Value) -> Result<MessagesStreamEvent, Error> {
    serde_json::from_value(chunk).map_err(|error| {
        Error::InvalidResponse(crate::ErrorDetail::invalid("Anthropic stream event", error))
    })
}

pub fn invoke_anthropic_event_stream(bytes: ByteStream) -> EventStream {
    Box::pin(invoke_chunk_stream(bytes).map(|chunk| decode_invoke_anthropic_chunk(chunk?)))
}

#[derive(Clone, Copy)]
enum InvokeProvider {
    Anthropic,
    DeepseekR1,
    Moonshot,
    Unsupported,
}

impl From<&str> for InvokeProvider {
    fn from(value: &str) -> Self {
        match value {
            "anthropic" => Self::Anthropic,
            "deepseek_r1" => Self::DeepseekR1,
            "moonshot" => Self::Moonshot,
            _ => Self::Unsupported,
        }
    }
}

pub fn invoke_chat_stream(invoke_provider: &str, shape: StreamShape) -> Result<ChatStream, Error> {
    match InvokeProvider::from(invoke_provider) {
        InvokeProvider::Anthropic => Ok(ChatStream::new(
            invoke_anthropic_event_stream,
            ModelResponseIterator::new(shape),
        )),
        InvokeProvider::DeepseekR1 | InvokeProvider::Moonshot => Err(Error::Unsupported(
            "Bedrock invoke streaming for this model family",
        )),
        InvokeProvider::Unsupported => Err(Error::Unsupported("Bedrock invoke streaming")),
    }
}

#[cfg(test)]
mod tests {
    use aws_smithy_eventstream::frame::write_message_to;
    use aws_smithy_types::event_stream::{Header, HeaderValue, Message};
    use base64::engine::general_purpose::STANDARD;
    use bytes::Bytes;
    use futures_util::TryStreamExt;
    use litellm_llms_types::formats::messages::streaming::MessagesContentBlockDelta;

    use super::*;
    use crate::base_llm::messages::streaming::anthropic_sse_event_stream;

    fn in_pieces(wire: &[u8]) -> ByteStream {
        let pieces: Vec<Bytes> = wire.chunks(3).map(Bytes::copy_from_slice).collect();
        futures_util::stream::iter(pieces.into_iter().map(Ok)).boxed()
    }

    const TEXT_DELTA: &str =
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}"#;

    fn aws_wire(chunk: &str) -> Vec<u8> {
        let payload = serde_json::json!({"bytes": STANDARD.encode(chunk)});
        let message = Message::new(Bytes::from(serde_json::to_vec(&payload).unwrap())).add_header(
            Header::new(":event-type", HeaderValue::String("chunk".into())),
        );
        let mut wire = Vec::new();
        write_message_to(&message, &mut wire).unwrap();
        wire
    }

    #[tokio::test]
    async fn aws_and_sse_framing_decode_to_the_same_anthropic_events() {
        let aws = aws_wire(TEXT_DELTA);
        let sse = format!("event: content_block_delta\ndata: {TEXT_DELTA}\n\n");

        let from_aws = invoke_anthropic_event_stream(in_pieces(&aws))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let from_sse = anthropic_sse_event_stream(in_pieces(sse.as_bytes()))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(
            from_aws,
            vec![MessagesStreamEvent::ContentBlockDelta {
                index: 0,
                delta: MessagesContentBlockDelta::TextDelta {
                    text: "hello".into(),
                    extra: Default::default(),
                },
                extra: Default::default(),
            }]
        );
        assert_eq!(from_aws, from_sse);
    }
    #[rstest::rstest]
    #[case::error("error", None, 400, "{\"message\":\"upstream failed\"}")]
    #[case::unknown(
        "exception",
        Some("somethingNotModeled"),
        400,
        "somethingNotModeled {\"message\":\"upstream failed\"}"
    )]
    // AWS InvokeModelWithResponseStream response elements, https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModelWithResponseStream.html, verified 2026-10-07
    #[case::throttled(
        "exception",
        Some("throttlingException"),
        429,
        "throttlingException {\"message\":\"upstream failed\"}"
    )]
    #[case::internal(
        "exception",
        Some("internalServerException"),
        500,
        "internalServerException {\"message\":\"upstream failed\"}"
    )]
    #[case::stream_failure(
        "exception",
        Some("modelStreamErrorException"),
        424,
        "modelStreamErrorException {\"message\":\"upstream failed\"}"
    )]
    #[case::timeout(
        "exception",
        Some("modelTimeoutException"),
        408,
        "modelTimeoutException {\"message\":\"upstream failed\"}"
    )]
    #[case::unavailable(
        "exception",
        Some("serviceUnavailableException"),
        503,
        "serviceUnavailableException {\"message\":\"upstream failed\"}"
    )]
    #[case::validation(
        "exception",
        Some("validationException"),
        400,
        "validationException {\"message\":\"upstream failed\"}"
    )]
    #[tokio::test]
    async fn test_build_bedrock_stream_error_resolves_status_from_the_exception_type(
        #[case] message_type: &str,
        #[case] exception: Option<&str>,
        #[case] status: u16,
        #[case] text: &str,
    ) {
        let message = Message::new(Bytes::from_static(b"{\"message\":\"upstream failed\"}"))
            .add_header(Header::new(
                ":message-type",
                HeaderValue::String(message_type.to_string().into()),
            ));
        let message = match exception {
            Some(exception) => message.add_header(Header::new(
                ":exception-type",
                HeaderValue::String(exception.to_string().into()),
            )),
            None => message,
        };
        let mut wire = Vec::new();
        write_message_to(&message, &mut wire).unwrap();
        let events = invoke_chunk_stream(in_pieces(&wire))
            .collect::<Vec<_>>()
            .await;
        assert_eq!(
            events,
            vec![Err(Error::InvalidResponse(crate::ErrorDetail::Http {
                status,
                message: text.into()
            }))]
        );
    }
}
