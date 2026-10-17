use base64::Engine;
use bytes::Buf;
use futures_util::{Stream, StreamExt, stream::BoxStream};
use litellm_core_utils::core_helpers::finish_reason_for;
use litellm_framer::{
    aws_event_stream::{AwsEventStreamCodec, HeaderValue, Message},
    frames,
};
use litellm_llms_types::formats::{
    chat_completions::{
        ChatCompletionChunk, ChatCompletionDelta, ChatCompletionStreamingChoice,
        ChatCompletionThinkingBlock, ChatCompletionToolCallChunk,
        ChatCompletionToolCallFunctionChunk, ChatCompletionsUsage, CompletionTokensDetails,
        PromptTokensDetails,
    },
    messages::streaming::MessagesStreamEvent,
};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{
    Error,
    anthropic::chat::handler::ModelResponseIterator,
    base_llm::{
        base_model_iterator::StreamTransformer,
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

/// The HTTP status Bedrock's ConverseStream models for each exception event; botocore
/// reports every one as a 400 on the wire.
const CONVERSE_STREAM_EXCEPTIONS: [(&str, u16); 5] = [
    ("internalServerException", 500),
    ("modelStreamErrorException", 424),
    ("validationException", 400),
    ("throttlingException", 429),
    ("serviceUnavailableException", 503),
];
const UNMODELED_EXCEPTION_STATUS: u16 = 400;
/// Python's status for a 200 stream whose body held no events.
const EMPTY_STREAM_STATUS: u16 = 502;

fn header<'a>(message: &'a Message, name: &str) -> Option<&'a str> {
    message
        .headers()
        .iter()
        .find(|header| header.name().as_str() == name)
        .and_then(|header| match header.value() {
            HeaderValue::String(value) => Some(value.as_str()),
            _ => None,
        })
}

/// One ConverseStream frame: its event payload, nothing for an empty one, or the
/// exception it carries as the provider's error.
pub fn decode_converse_message(message: &Message) -> Result<Option<Value>, Error> {
    let body = String::from_utf8_lossy(message.payload()).into_owned();
    match header(message, ":message-type") {
        Some("exception") => {
            let exception = header(message, ":exception-type").unwrap_or_default();
            let status = CONVERSE_STREAM_EXCEPTIONS
                .iter()
                .find(|(name, _)| *name == exception)
                .map_or(UNMODELED_EXCEPTION_STATUS, |(_, status)| *status);
            Err(Error::Provider {
                status,
                body: format!("{exception} {body}").trim().to_owned(),
            })
        }
        Some("error") => Err(Error::Provider {
            status: UNMODELED_EXCEPTION_STATUS,
            body: format!(
                "{} {}",
                header(message, ":error-code").unwrap_or_default(),
                header(message, ":error-message").unwrap_or(&body)
            )
            .trim()
            .to_owned(),
        }),
        _ if message.payload().is_empty() => Ok(None),
        _ => serde_json::from_slice(message.payload())
            .map(Some)
            .map_err(|error| {
                Error::InvalidResponse(crate::ErrorDetail::invalid("Bedrock event payload", error))
            }),
    }
}

/// ConverseStream's AWS event stream as the JSON payload of each event.
pub fn converse_event_stream(bytes: ByteStream) -> BoxStream<'static, Result<Value, Error>> {
    let events = frames(bytes, AwsEventStreamCodec).map(|message| {
        let message = message.map_err(|error| {
            Error::InvalidResponse(crate::ErrorDetail::failed("stream framing", error))
        })?;
        decode_converse_message(&message)
    });
    futures_util::stream::unfold(
        (Box::pin(events), false, false),
        |(mut events, seen, done)| async move {
            if done {
                return None;
            }
            loop {
                match events.next().await {
                    Some(Ok(None)) => continue,
                    Some(Ok(Some(event))) => return Some((Ok(event), (events, true, false))),
                    Some(Err(error)) => return Some((Err(error), (events, seen, true))),
                    None if seen => return None,
                    None => {
                        return Some((
                            Err(Error::Provider {
                                status: EMPTY_STREAM_STATUS,
                                body: "Bedrock answered the stream with HTTP 200 but its body decoded to no events".into(),
                            }),
                            (events, seen, true),
                        ));
                    }
                }
            }
        },
    )
    .boxed()
}

/// ConverseStream events to Chat Completions chunks, as Python's `AWSEventStreamDecoder`
/// `converse_chunk_parser` builds them.
///
/// DIVERGENCE: `json_mode` tool conversion is not done (the route never asks for it),
/// `redactedContent` is passed through without unwrapping, and the reasoning token estimate
/// Python makes from streamed thinking text is not made.
pub struct ConverseStreamIterator {
    shape: StreamShape,
    /// The tool argument fragments of the block in progress.
    content_blocks: Vec<String>,
    tool_calls_index: Option<i64>,
    thinking_ran: bool,
    thinking_block_index: i64,
    thinking_content_block_index: Option<u64>,
}

#[derive(Default)]
struct ConverseParsed {
    text: String,
    tool_use: Option<ChatCompletionToolCallChunk>,
    provider_specific_fields: Map<String, Value>,
    reasoning_content: Option<String>,
    thinking_blocks: Vec<ChatCompletionThinkingBlock>,
    finish_reason: Option<String>,
    usage: Option<ChatCompletionsUsage>,
}

impl ConverseStreamIterator {
    pub fn new(shape: StreamShape) -> Self {
        Self {
            shape,
            content_blocks: Vec::new(),
            tool_calls_index: None,
            thinking_ran: false,
            thinking_block_index: -1,
            thinking_content_block_index: None,
        }
    }

    fn tool_call(
        &self,
        id: Option<String>,
        name: Option<String>,
        arguments: String,
        index: u64,
    ) -> ChatCompletionToolCallChunk {
        ChatCompletionToolCallChunk {
            id,
            tool_type: "function".into(),
            function: ChatCompletionToolCallFunctionChunk {
                name,
                arguments,
                provider_specific_fields: None,
            },
            index: self
                .tool_calls_index
                .unwrap_or_else(|| i64::try_from(index).unwrap_or(i64::MAX)),
        }
    }

    fn thinking_blocks(
        &mut self,
        block: &Value,
        content_block_index: u64,
    ) -> Vec<ChatCompletionThinkingBlock> {
        if self.thinking_content_block_index != Some(content_block_index) {
            self.thinking_content_block_index = Some(content_block_index);
            self.thinking_block_index += 1;
        }
        let index = u64::try_from(self.thinking_block_index).ok();
        let text = |field: &str| block.get(field).and_then(Value::as_str).map(str::to_owned);
        let translated = if let Some(thinking) = text("text") {
            Some(ChatCompletionThinkingBlock::Thinking {
                thinking: Some(thinking),
                signature: None,
                cache_control: None,
                index,
            })
        } else if let Some(signature) = text("signature") {
            Some(ChatCompletionThinkingBlock::Thinking {
                thinking: Some(String::new()),
                signature: Some(signature),
                cache_control: None,
                index,
            })
        } else {
            text("redactedContent").map(|data| ChatCompletionThinkingBlock::RedactedThinking {
                data: Some(data),
                cache_control: None,
                index,
            })
        };
        translated.into_iter().collect()
    }

    fn start(&mut self, start: &Value, index: u64) -> ConverseParsed {
        let mut parsed = ConverseParsed::default();
        self.content_blocks.clear();
        if let Some(tool) = start.get("toolUse").filter(|tool| !tool.is_null()) {
            self.tool_calls_index = Some(self.tool_calls_index.map_or(0, |index| index + 1));
            let name = tool.get("name").and_then(Value::as_str).map(|name| {
                self.shape
                    .tool_name_reverse_map
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| name.to_owned())
            });
            let id = tool
                .get("toolUseId")
                .and_then(Value::as_str)
                .map(str::to_owned);
            parsed.tool_use = Some(self.tool_call(id, name, String::new(), index));
        } else if let Some(reasoning) = start.get("reasoningContent").filter(|r| !r.is_null()) {
            parsed.thinking_blocks = self.thinking_blocks(reasoning, index);
            parsed
                .provider_specific_fields
                .insert("reasoningContent".into(), reasoning.clone());
        }
        parsed
    }

    fn delta(&mut self, delta: &Value, index: u64) -> ConverseParsed {
        let mut parsed = ConverseParsed::default();
        if let Some(text) = delta.get("text").and_then(Value::as_str) {
            parsed.text = text.to_owned();
        } else if let Some(tool) = delta.get("toolUse") {
            let input = tool
                .get("input")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            self.content_blocks.push(input.clone());
            parsed.tool_use = Some(self.tool_call(None, None, input, index));
        } else if let Some(reasoning) = delta.get("reasoningContent") {
            parsed
                .provider_specific_fields
                .insert("reasoningContent".into(), reasoning.clone());
            parsed.reasoning_content = reasoning
                .get("text")
                .and_then(Value::as_str)
                .map(str::to_owned);
            parsed.thinking_blocks = self.thinking_blocks(reasoning, index);
            if !parsed.thinking_blocks.is_empty() {
                parsed.reasoning_content.get_or_insert_with(String::new);
            }
        } else if let Some(citations) = delta.get("citationsContent") {
            parsed
                .provider_specific_fields
                .insert("citationsContent".into(), citations.clone());
        }
        parsed
    }

    pub fn converse_chunk_parser(&mut self, data: &Value) -> ChatCompletionChunk {
        let index = data
            .get("contentBlockIndex")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let mut parsed = if let Some(start) = data.get("start") {
            self.start(start, index)
        } else if let Some(delta) = data.get("delta") {
            self.delta(delta, index)
        } else if data.get("contentBlockIndex").is_some() {
            // A block's stop: a tool that streamed only empty fragments closes with `{}`.
            let empty =
                !self.content_blocks.is_empty() && self.content_blocks.iter().all(String::is_empty);
            ConverseParsed {
                tool_use: empty.then(|| self.tool_call(None, None, "{}".into(), index)),
                ..ConverseParsed::default()
            }
        } else if let Some(reason) = data.get("stopReason") {
            ConverseParsed {
                finish_reason: Some(
                    finish_reason_for(reason.as_str().unwrap_or("stop")).to_owned(),
                ),
                ..ConverseParsed::default()
            }
        } else if let Some(usage) = data.get("usage") {
            ConverseParsed {
                usage: Some(converse_usage(usage, self.thinking_ran)),
                ..ConverseParsed::default()
            }
        } else {
            ConverseParsed::default()
        };
        if !parsed.thinking_blocks.is_empty() {
            self.thinking_ran = true;
        }
        let trace = data.get("trace").filter(|trace| match trace {
            Value::Null => false,
            Value::Object(fields) => !fields.is_empty(),
            Value::Array(items) => !items.is_empty(),
            Value::String(text) => !text.is_empty(),
            _ => true,
        });
        let carries_content = trace.is_some()
            || ["start", "delta", "contentBlockIndex", "stopReason"]
                .iter()
                .any(|key| data.get(*key).is_some());
        let thinking_blocks = std::mem::take(&mut parsed.thinking_blocks);
        let provider_specific_fields = std::mem::take(&mut parsed.provider_specific_fields);
        ChatCompletionChunk {
            choices: vec![ChatCompletionStreamingChoice {
                index: 0,
                delta: ChatCompletionDelta {
                    content: carries_content.then_some(parsed.text),
                    role: carries_content.then(|| "assistant".to_owned()),
                    tool_calls: parsed.tool_use.map(|tool| vec![tool]),
                    reasoning_content: parsed.reasoning_content,
                    thinking_blocks: (!thinking_blocks.is_empty()).then_some(thinking_blocks),
                    provider_specific_fields: (!provider_specific_fields.is_empty())
                        .then_some(provider_specific_fields),
                    ..ChatCompletionDelta::default()
                },
                finish_reason: parsed.finish_reason,
                logprobs: None,
            }],
            usage: parsed.usage,
            provider_specific_fields: Some(
                trace
                    .map(|trace| Map::from_iter([("trace".to_owned(), trace.clone())]))
                    .unwrap_or_default(),
            ),
            ..ChatCompletionChunk::default()
        }
    }
}

/// Python's `AmazonConverseConfig.transform_usage` for a stream's `metadata` usage.
pub fn converse_usage(usage: &Value, thinking_ran: bool) -> ChatCompletionsUsage {
    let count = |names: &[&str]| {
        names
            .iter()
            .find_map(|name| usage.get(*name).and_then(Value::as_u64))
            .unwrap_or(0)
    };
    let input = count(&["inputTokens"]);
    let output = count(&["outputTokens"]);
    let cache_read = count(&["cacheReadInputTokens", "cacheReadInputTokenCount"]);
    let cache_write = count(&["cacheWriteInputTokens", "cacheWriteInputTokenCount"]);
    let prompt = input + cache_read + cache_write;
    ChatCompletionsUsage {
        prompt_tokens: prompt,
        completion_tokens: output,
        total_tokens: usage
            .get("totalTokens")
            .and_then(Value::as_u64)
            .unwrap_or(prompt + output),
        prompt_tokens_details: PromptTokensDetails {
            cached_tokens: cache_read,
            cache_creation_tokens: cache_write,
            text_tokens: input,
            cache_write_tokens: Some(cache_write),
            cache_creation_token_details: None,
        },
        completion_tokens_details: Some(match thinking_ran {
            true => CompletionTokensDetails::default(),
            false => CompletionTokensDetails {
                reasoning_tokens: Some(0),
                text_tokens: Some(output),
            },
        }),
        server_tool_use: None,
        cache_creation_input_tokens: Some(cache_write),
        cache_read_input_tokens: Some(cache_read),
    }
}

impl StreamTransformer for ConverseStreamIterator {
    type Input = Value;
    type Output = ChatCompletionChunk;
    type Error = Error;

    /// Python hands only Converse events (those with a block index, a stop reason, metrics
    /// or a trace) to the chunk parser; `messageStart` becomes an empty generic chunk that
    /// never reaches the client.
    fn transform(&mut self, input: Value) -> Result<Vec<ChatCompletionChunk>, Error> {
        let converse = [
            "contentBlockIndex",
            "stopReason",
            "metrics",
            "trace",
            "usage",
        ]
        .iter()
        .any(|key| input.get(*key).is_some());
        Ok(match converse {
            true => vec![self.converse_chunk_parser(&input)],
            false => Vec::new(),
        })
    }

    fn finish(&mut self) -> Result<Vec<ChatCompletionChunk>, Error> {
        Ok(Vec::new())
    }
}

pub fn converse_chat_stream(shape: StreamShape) -> ChatStream {
    ChatStream::new(converse_event_stream, ConverseStreamIterator::new(shape))
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
    use crate::base_llm::messages::streaming;

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
        let from_sse = streaming::event_stream(in_pieces(sse.as_bytes()))
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

    fn converse_frame(kind: &str, name: &str, payload: serde_json::Value) -> Vec<u8> {
        let type_header = match kind {
            "exception" => ":exception-type",
            _ => ":event-type",
        };
        let message = Message::new(Bytes::from(serde_json::to_vec(&payload).unwrap()))
            .add_header(Header::new(
                ":message-type",
                HeaderValue::String(kind.to_owned().into()),
            ))
            .add_header(Header::new(
                type_header,
                HeaderValue::String(name.to_owned().into()),
            ));
        let mut wire = Vec::new();
        write_message_to(&message, &mut wire).unwrap();
        wire
    }

    async fn converse_events(wire: Vec<u8>) -> Vec<Result<serde_json::Value, Error>> {
        converse_event_stream(in_pieces(&wire)).collect().await
    }

    #[tokio::test]
    async fn converse_frames_decode_to_their_payloads_split_anywhere() {
        let wire = [
            converse_frame(
                "event",
                "messageStart",
                serde_json::json!({"role": "assistant"}),
            ),
            converse_frame(
                "event",
                "contentBlockDelta",
                serde_json::json!({"contentBlockIndex": 0, "delta": {"text": "hi"}}),
            ),
        ]
        .concat();
        let events = converse_events(wire).await;
        assert_eq!(
            events.into_iter().map(Result::unwrap).collect::<Vec<_>>(),
            vec![
                serde_json::json!({"role": "assistant"}),
                serde_json::json!({"contentBlockIndex": 0, "delta": {"text": "hi"}})
            ]
        );
    }

    #[rstest::rstest]
    #[case::throttling("throttlingException", 429)]
    #[case::validation("validationException", 400)]
    #[case::internal("internalServerException", 500)]
    #[case::unavailable("serviceUnavailableException", 503)]
    #[case::model_stream("modelStreamErrorException", 424)]
    #[case::unmodeled("somethingNewException", 400)]
    #[tokio::test]
    async fn an_exception_frame_is_the_providers_error_with_its_modeled_status(
        #[case] exception: &str,
        #[case] expected: u16,
    ) {
        let wire = [
            converse_frame(
                "event",
                "contentBlockDelta",
                serde_json::json!({"contentBlockIndex": 0, "delta": {"text": "hi"}}),
            ),
            converse_frame(
                "exception",
                exception,
                serde_json::json!({"message": "slow down"}),
            ),
        ]
        .concat();
        let mut events = converse_events(wire).await.into_iter();
        assert!(events.next().unwrap().is_ok());
        let Some(Err(Error::Provider { status, body })) = events.next() else {
            panic!("the exception ends the stream as the provider's error");
        };
        assert_eq!(status, expected);
        assert_eq!(body, format!("{exception} {{\"message\":\"slow down\"}}"));
        assert!(events.next().is_none());
    }

    #[tokio::test]
    async fn a_stream_without_events_is_a_bad_gateway() {
        let events = converse_events(Vec::new()).await;
        assert!(matches!(
            events.as_slice(),
            [Err(Error::Provider { status: 502, .. })]
        ));
    }

    fn converse_chunks(events: serde_json::Value) -> Vec<serde_json::Value> {
        let mut iterator = ConverseStreamIterator::new(StreamShape::default());
        let serde_json::Value::Array(events) = events else {
            panic!("events are a list");
        };
        events
            .into_iter()
            .flat_map(|event| iterator.transform(event).unwrap())
            .map(|chunk| serde_json::to_value(chunk).unwrap())
            .collect()
    }

    #[test]
    fn converse_text_and_stop_follow_python() {
        let chunks = converse_chunks(serde_json::json!([
            {"role": "assistant"},
            {"contentBlockIndex": 0, "delta": {"text": "Hello"}},
            {"contentBlockIndex": 0},
            {"stopReason": "end_turn"},
            {"usage": {"inputTokens": 10, "outputTokens": 12, "totalTokens": 48, "cacheReadInputTokens": 20, "cacheWriteInputTokens": 6}, "metrics": {"latencyMs": 1}}
        ]));
        assert_eq!(chunks.len(), 4, "messageStart is not a Converse chunk");
        assert_eq!(
            chunks[0],
            serde_json::json!({"choices": [{"index": 0, "delta": {"content": "Hello", "role": "assistant"}}], "provider_specific_fields": {}})
        );
        assert_eq!(
            chunks[1]["choices"][0]["delta"],
            serde_json::json!({"content": "", "role": "assistant"})
        );
        assert_eq!(chunks[2]["choices"][0]["finish_reason"], "stop");
        assert_eq!(chunks[3]["choices"][0]["delta"], serde_json::json!({}));
        assert_eq!(
            chunks[3]["usage"],
            serde_json::json!({
                "completion_tokens": 12, "prompt_tokens": 36, "total_tokens": 48,
                "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 12},
                "prompt_tokens_details": {"cached_tokens": 20, "text_tokens": 10, "cache_write_tokens": 6, "cache_creation_tokens": 6},
                "cache_creation_input_tokens": 6, "cache_read_input_tokens": 20
            })
        );
    }

    #[test]
    fn converse_tool_calls_stream_name_then_partial_arguments() {
        let chunks = converse_chunks(serde_json::json!([
            {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "t1", "name": "get_weather"}}},
            {"contentBlockIndex": 1, "delta": {"toolUse": {"input": "{\"city\": \"Pa"}}},
            {"contentBlockIndex": 1, "delta": {"toolUse": {"input": "ris\"}"}}},
            {"contentBlockIndex": 1},
            {"contentBlockIndex": 2, "start": {"toolUse": {"toolUseId": "t2", "name": "now"}}},
            {"contentBlockIndex": 2, "delta": {"toolUse": {"input": ""}}},
            {"contentBlockIndex": 2},
            {"stopReason": "tool_use"}
        ]));
        let tools = |chunk: &serde_json::Value| chunk["choices"][0]["delta"]["tool_calls"].clone();
        assert_eq!(
            tools(&chunks[0]),
            serde_json::json!([{"id": "t1", "type": "function", "function": {"name": "get_weather", "arguments": ""}, "index": 0}])
        );
        assert_eq!(
            tools(&chunks[1]),
            serde_json::json!([{"type": "function", "function": {"arguments": "{\"city\": \"Pa"}, "index": 0}])
        );
        assert_eq!(tools(&chunks[3]), serde_json::Value::Null);
        assert_eq!(tools(&chunks[4])[0]["index"], 1);
        assert_eq!(
            tools(&chunks[6]),
            serde_json::json!([{"type": "function", "function": {"arguments": "{}"}, "index": 1}])
        );
        assert_eq!(chunks[7]["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn converse_reasoning_streams_as_thinking_blocks() {
        let chunks = converse_chunks(serde_json::json!([
            {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "Let me think"}}},
            {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig"}}},
            {"contentBlockIndex": 1, "start": {"reasoningContent": {"redactedContent": "enc"}}},
            {"stopReason": "max_tokens"},
            {"usage": {"inputTokens": 1, "outputTokens": 5}, "metrics": {}}
        ]));
        assert_eq!(
            chunks[0]["choices"][0]["delta"],
            serde_json::json!({"content": "", "role": "assistant", "reasoning_content": "Let me think",
                "thinking_blocks": [{"type": "thinking", "thinking": "Let me think", "index": 0}],
                "provider_specific_fields": {"reasoningContent": {"text": "Let me think"}}})
        );
        assert_eq!(chunks[1]["choices"][0]["delta"]["reasoning_content"], "");
        assert_eq!(
            chunks[1]["choices"][0]["delta"]["thinking_blocks"],
            serde_json::json!([{"type": "thinking", "thinking": "", "signature": "sig", "index": 0}])
        );
        assert_eq!(
            chunks[2]["choices"][0]["delta"]["thinking_blocks"],
            serde_json::json!([{"type": "redacted_thinking", "data": "enc", "index": 1}])
        );
        assert_eq!(chunks[3]["choices"][0]["finish_reason"], "length");
        assert_eq!(
            chunks[4]["usage"]["completion_tokens_details"],
            serde_json::json!({})
        );
        assert_eq!(chunks[4]["usage"]["total_tokens"], 6);
    }

    #[test]
    fn a_guardrail_trace_rides_on_the_chunk() {
        let chunks = converse_chunks(serde_json::json!([
            {"stopReason": "guardrail_intervened", "trace": {"guardrail": {"action": "BLOCKED"}}}
        ]));
        assert_eq!(chunks[0]["choices"][0]["finish_reason"], "content_filter");
        assert_eq!(
            chunks[0]["provider_specific_fields"],
            serde_json::json!({"trace": {"guardrail": {"action": "BLOCKED"}}})
        );
    }
}
