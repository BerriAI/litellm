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

pub fn decode_invoke_chunk(message: Message) -> Result<Value, Error> {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::EnumString)]
enum InvokeProvider {
    #[strum(serialize = "anthropic")]
    Anthropic,
    #[strum(serialize = "deepseek_r1")]
    DeepseekR1,
    #[strum(serialize = "moonshot")]
    Moonshot,
    #[strum(disabled)]
    Unsupported,
}

pub fn invoke_chat_stream(invoke_provider: &str, shape: StreamShape) -> Result<ChatStream, Error> {
    match invoke_provider
        .parse()
        .unwrap_or(InvokeProvider::Unsupported)
    {
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

    #[rstest::rstest]
    #[case::anthropic("anthropic", InvokeProvider::Anthropic)]
    #[case::deepseek_r1("deepseek_r1", InvokeProvider::DeepseekR1)]
    #[case::moonshot("moonshot", InvokeProvider::Moonshot)]
    #[case::unknown("qwen", InvokeProvider::Unsupported)]
    #[case::case_sensitive("Anthropic", InvokeProvider::Unsupported)]
    fn invoke_provider_maps_each_model_family_name(
        #[case] name: &str,
        #[case] expected: InvokeProvider,
    ) {
        assert_eq!(
            name.parse().unwrap_or(InvokeProvider::Unsupported),
            expected
        );
    }

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
                },
            }]
        );
        assert_eq!(from_aws, from_sse);
    }
}
