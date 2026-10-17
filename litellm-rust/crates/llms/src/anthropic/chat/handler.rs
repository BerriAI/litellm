//! Anthropic Messages stream events to Chat Completions chunks, as Python's
//! `anthropic/chat/handler.py` `ModelResponseIterator.chunk_parser` builds them.
//!
//! Each event becomes one provider chunk, empty ones included; the host's stream wrapper
//! (Python's `CustomStreamWrapper`) decides which of them the client sees.
//!
//! DIVERGENCE: not built here, since the chat route never asks for what produces them:
//! `json_mode` / `response_format` tool conversion, `web_search_calls` and
//! `code_interpreter_results` summaries, and a tool call's `caller`. The reasoning token
//! estimate Python makes from streamed thinking text (with its tokenizer) is not made, so
//! such a stream reports `completion_tokens_details` without counts unless Anthropic sends
//! `output_tokens_details.thinking_tokens`.

use litellm_core_utils::core_helpers::finish_reason_for;
use litellm_llms_types::formats::{
    chat_completions::{
        ChatCompletionChunk, ChatCompletionDelta, ChatCompletionStreamingChoice,
        ChatCompletionThinkingBlock, ChatCompletionToolCallChunk,
        ChatCompletionToolCallFunctionChunk, ChatCompletionsUsage, CompletionTokensDetails,
        PromptTokensDetails,
    },
    messages::streaming::{
        MessagesContentBlock, MessagesContentBlockDelta, MessagesDelta, MessagesStreamEvent,
        MessagesStreamUsage,
    },
};
use serde_json::{Map, Value, json};

use crate::{
    Error,
    base_llm::{base_model_iterator::StreamTransformer, chat::streaming::StreamShape},
};

/// Anthropic sends no status with an `error` event; Python raises it as a 500.
const STREAM_ERROR_STATUS: u16 = 500;

const TOOL_BLOCKS: [&str; 2] = ["tool_use", "server_tool_use"];
const THINKING_BLOCKS: [&str; 2] = ["thinking", "redacted_thinking"];

pub struct ModelResponseIterator {
    shape: StreamShape,
    /// The deltas of the block in progress, other than text.
    content_blocks: Vec<MessagesContentBlockDelta>,
    tool_index: i64,
    thinking_block_index: i64,
    served_model: Option<String>,
    current_content_block_type: Option<String>,
    /// `message_start` usage overlaid with every later `message_delta` usage.
    cumulative_usage: Map<String, Value>,
    streamed_reasoning: bool,
    web_search_results: Vec<Value>,
    tool_results: Vec<Value>,
    compaction_blocks: Vec<Value>,
}

/// What one event contributes to its chunk.
#[derive(Default)]
struct Parsed {
    text: String,
    tool_use: Option<ChatCompletionToolCallChunk>,
    finish_reason: Option<String>,
    usage: Option<ChatCompletionsUsage>,
    provider_specific_fields: Map<String, Value>,
    reasoning_content: Option<String>,
    thinking_blocks: Vec<ChatCompletionThinkingBlock>,
}

impl ModelResponseIterator {
    pub fn new(shape: StreamShape) -> Self {
        Self {
            shape,
            content_blocks: Vec::new(),
            tool_index: -1,
            thinking_block_index: -1,
            served_model: None,
            current_content_block_type: None,
            cumulative_usage: Map::new(),
            streamed_reasoning: false,
            web_search_results: Vec::new(),
            tool_results: Vec::new(),
            compaction_blocks: Vec::new(),
        }
    }

    /// Whether the tool block so far streamed only empty argument fragments.
    pub fn check_empty_tool_call_args(&self) -> bool {
        match self.content_blocks.first() {
            None
            | Some(
                MessagesContentBlockDelta::TextDelta { .. }
                | MessagesContentBlockDelta::ThinkingDelta { .. },
            ) => false,
            Some(_) => self.content_blocks.iter().all(|delta| match delta {
                MessagesContentBlockDelta::InputJsonDelta { partial_json } => {
                    partial_json.is_empty()
                }
                _ => true,
            }),
        }
    }

    fn handle_usage(&mut self, usage: &MessagesStreamUsage) -> ChatCompletionsUsage {
        if let Ok(Value::Object(fields)) = serde_json::to_value(usage) {
            self.cumulative_usage.extend(fields);
        }
        anthropic_usage(&self.cumulative_usage, self.streamed_reasoning)
    }

    fn thinking_index(&self) -> Option<u64> {
        u64::try_from(self.thinking_block_index).ok()
    }

    fn tool_call(
        &self,
        id: Option<String>,
        name: Option<String>,
        arguments: String,
    ) -> ChatCompletionToolCallChunk {
        ChatCompletionToolCallChunk {
            id,
            tool_type: "function".into(),
            function: ChatCompletionToolCallFunctionChunk {
                name,
                arguments,
                provider_specific_fields: None,
            },
            index: self.tool_index,
        }
    }

    fn content_block_delta(&mut self, delta: MessagesContentBlockDelta) -> Parsed {
        let mut parsed = Parsed::default();
        if let MessagesContentBlockDelta::TextDelta { text } = delta {
            parsed.text = text;
            return parsed;
        }
        self.content_blocks.push(delta.clone());
        match delta {
            MessagesContentBlockDelta::TextDelta { .. } => {}
            MessagesContentBlockDelta::InputJsonDelta { partial_json } => {
                // A `web_search_tool_result` block streams JSON too, and is not a tool call.
                if self
                    .current_content_block_type
                    .as_deref()
                    .is_some_and(|block| TOOL_BLOCKS.contains(&block))
                {
                    parsed.tool_use = Some(self.tool_call(None, None, partial_json));
                }
            }
            MessagesContentBlockDelta::Citations { citation } => {
                parsed
                    .provider_specific_fields
                    .insert("citation".into(), citation);
            }
            MessagesContentBlockDelta::ThinkingDelta { thinking } => {
                if !thinking.is_empty() {
                    self.streamed_reasoning = true;
                    parsed.reasoning_content = Some(thinking.clone());
                    self.thinking_blocks(
                        &mut parsed,
                        ChatCompletionThinkingBlock::Thinking {
                            thinking: Some(thinking),
                            signature: None,
                            cache_control: None,
                            index: self.thinking_index(),
                        },
                    );
                }
            }
            MessagesContentBlockDelta::SignatureDelta { signature } => {
                if !signature.is_empty() {
                    self.thinking_blocks(
                        &mut parsed,
                        ChatCompletionThinkingBlock::Thinking {
                            thinking: Some(String::new()),
                            signature: Some(signature),
                            cache_control: None,
                            index: self.thinking_index(),
                        },
                    );
                    parsed.reasoning_content.get_or_insert_with(String::new);
                }
            }
            MessagesContentBlockDelta::CompactionDelta { content } => {
                parsed.provider_specific_fields.insert(
                    "compaction_delta".into(),
                    json!({"type": "compaction_delta", "content": content}),
                );
            }
        }
        parsed
    }

    /// A thinking block goes out both as `thinking_blocks` and under the same name in the
    /// delta's `provider_specific_fields`.
    fn thinking_blocks(&self, parsed: &mut Parsed, block: ChatCompletionThinkingBlock) {
        parsed.thinking_blocks = vec![block];
        if let Ok(blocks) = serde_json::to_value(&parsed.thinking_blocks) {
            parsed
                .provider_specific_fields
                .insert("thinking_blocks".into(), blocks);
        }
    }

    fn content_block_start(&mut self, block: MessagesContentBlock) -> Parsed {
        let mut parsed = Parsed::default();
        self.content_blocks.clear();
        let block_type = block.block_type.clone();
        self.current_content_block_type = Some(block_type.clone());
        if THINKING_BLOCKS.contains(&block_type.as_str()) {
            self.thinking_block_index += 1;
        }
        match block_type.as_str() {
            "text" => parsed.text = block.text.unwrap_or_default(),
            "tool_use" | "server_tool_use" => {
                self.tool_index += 1;
                let name = block.name.map(|name| {
                    self.shape
                        .tool_name_reverse_map
                        .get(&name)
                        .cloned()
                        .unwrap_or(name)
                });
                // The arguments arrive in the deltas; `input` here is always `{}`.
                parsed.tool_use = Some(self.tool_call(block.id, name, String::new()));
            }
            "redacted_thinking" => self.thinking_blocks(
                &mut parsed,
                ChatCompletionThinkingBlock::RedactedThinking {
                    data: block.data,
                    cache_control: None,
                    index: self.thinking_index(),
                },
            ),
            "compaction" => {
                let content = block.content.clone().unwrap_or_else(|| json!(""));
                self.compaction_blocks.push(block_value(&block));
                parsed.provider_specific_fields.insert(
                    "compaction_blocks".into(),
                    Value::Array(self.compaction_blocks.clone()),
                );
                parsed.provider_specific_fields.insert(
                    "compaction_start".into(),
                    json!({"type": "compaction", "content": content}),
                );
            }
            result if result.ends_with("_tool_result") => {
                if matches!(result, "web_search_tool_result" | "web_fetch_tool_result") {
                    self.web_search_results.push(block_value(&block));
                    parsed.provider_specific_fields.insert(
                        "web_search_results".into(),
                        Value::Array(self.web_search_results.clone()),
                    );
                } else {
                    self.tool_results.push(block_value(&block));
                    parsed.provider_specific_fields.insert(
                        "tool_results".into(),
                        Value::Array(self.tool_results.clone()),
                    );
                }
            }
            _ => {}
        }
        parsed
    }

    fn content_block_stop(&mut self) -> Parsed {
        let mut parsed = Parsed::default();
        let in_tool = self
            .current_content_block_type
            .as_deref()
            .is_some_and(|block| TOOL_BLOCKS.contains(&block));
        if in_tool && self.check_empty_tool_call_args() {
            parsed.tool_use = Some(self.tool_call(None, None, "{}".into()));
        }
        self.current_content_block_type = None;
        parsed
    }

    fn message_delta(
        &mut self,
        delta: MessagesDelta,
        usage: Option<MessagesStreamUsage>,
    ) -> Parsed {
        let mut parsed = Parsed {
            finish_reason: Some(
                finish_reason_for(delta.stop_reason.as_deref().unwrap_or("stop")).to_string(),
            ),
            usage: usage.map(|usage| self.handle_usage(&usage)),
            ..Parsed::default()
        };
        // Python's `refusal_provider_specific_fields`.
        if let Some(details) = delta.stop_details {
            parsed
                .provider_specific_fields
                .insert("stop_reason".into(), json!(delta.stop_reason));
            parsed
                .provider_specific_fields
                .insert("stop_details".into(), details);
        } else if delta.stop_reason.as_deref() == Some("refusal") {
            parsed
                .provider_specific_fields
                .insert("stop_reason".into(), json!("refusal"));
        }
        if let Some(container) = delta.container.filter(|container| !container.is_null()) {
            parsed
                .provider_specific_fields
                .insert("container".into(), container);
        }
        parsed
    }

    pub fn chunk_parser(
        &mut self,
        event: MessagesStreamEvent,
    ) -> Result<ChatCompletionChunk, Error> {
        let parsed = match event {
            MessagesStreamEvent::ContentBlockDelta { delta, .. } => self.content_block_delta(delta),
            MessagesStreamEvent::ContentBlockStart { content_block, .. } => {
                self.content_block_start(content_block)
            }
            MessagesStreamEvent::ContentBlockStop { .. } => self.content_block_stop(),
            MessagesStreamEvent::MessageDelta { delta, usage, .. } => {
                self.message_delta(delta, usage)
            }
            MessagesStreamEvent::MessageStart { message } => {
                self.served_model = Some(message.model);
                Parsed {
                    usage: Some(self.handle_usage(&message.usage)),
                    ..Parsed::default()
                }
            }
            MessagesStreamEvent::MessageStop { .. } | MessagesStreamEvent::Ping => {
                Parsed::default()
            }
            MessagesStreamEvent::Error { error } => {
                let body = json!({"type": "error", "error": error});
                return Err(Error::Provider {
                    status: STREAM_ERROR_STATUS,
                    body: body.to_string(),
                });
            }
        };
        Ok(self.chunk(parsed))
    }

    fn chunk(&self, parsed: Parsed) -> ChatCompletionChunk {
        let Parsed {
            text,
            tool_use,
            finish_reason,
            usage,
            provider_specific_fields,
            reasoning_content,
            thinking_blocks,
        } = parsed;
        ChatCompletionChunk {
            model: self.served_model.clone(),
            choices: vec![ChatCompletionStreamingChoice {
                index: 0,
                delta: ChatCompletionDelta {
                    content: Some(text),
                    tool_calls: tool_use.map(|tool| vec![tool]),
                    reasoning_content,
                    thinking_blocks: (!thinking_blocks.is_empty()).then_some(thinking_blocks),
                    provider_specific_fields: (!provider_specific_fields.is_empty())
                        .then_some(provider_specific_fields),
                    ..ChatCompletionDelta::default()
                },
                finish_reason,
                logprobs: None,
            }],
            usage,
            ..ChatCompletionChunk::default()
        }
    }
}

fn block_value(block: &MessagesContentBlock) -> Value {
    serde_json::to_value(block).unwrap_or(Value::Null)
}

fn count(usage: &Map<String, Value>, field: &str) -> u64 {
    usage.get(field).and_then(Value::as_u64).unwrap_or(0)
}

/// Python's `AnthropicConfig.calculate_usage` over a stream's cumulative usage.
pub fn anthropic_usage(
    usage: &Map<String, Value>,
    streamed_reasoning: bool,
) -> ChatCompletionsUsage {
    let iterations = usage
        .get("iterations")
        .and_then(Value::as_array)
        .filter(|iterations| !iterations.is_empty());
    let (input, completion, cache_creation, cache_read) = match iterations {
        Some(iterations) => {
            let sum = |field: &str| {
                iterations
                    .iter()
                    .filter_map(|iteration| iteration.get(field)?.as_u64())
                    .sum::<u64>()
            };
            (
                sum("input_tokens"),
                sum("output_tokens"),
                sum("cache_creation_input_tokens"),
                sum("cache_read_input_tokens"),
            )
        }
        None => (
            count(usage, "input_tokens"),
            count(usage, "output_tokens"),
            count(usage, "cache_creation_input_tokens"),
            count(usage, "cache_read_input_tokens"),
        ),
    };
    let prompt = input + cache_creation + cache_read;
    let thinking_tokens = usage
        .get("output_tokens_details")
        .and_then(|details| details.get("thinking_tokens"))
        .and_then(Value::as_u64);
    let completion_tokens_details = match thinking_tokens {
        Some(thinking) => {
            let reasoning = thinking.min(completion);
            CompletionTokensDetails {
                reasoning_tokens: Some(reasoning),
                text_tokens: Some(completion - reasoning),
            }
        }
        None if streamed_reasoning => CompletionTokensDetails::default(),
        None => CompletionTokensDetails {
            reasoning_tokens: Some(0),
            text_tokens: Some(completion),
        },
    };
    let cache_creation_token_details =
        usage
            .get("cache_creation")
            .and_then(Value::as_object)
            .map(|details| {
                Value::Object(
                    ["ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"]
                        .into_iter()
                        .filter_map(|field| Some((field.to_owned(), details.get(field)?.clone())))
                        .filter(|(_, value)| !value.is_null())
                        .collect(),
                )
            });
    let server_tool_use = usage
        .get("server_tool_use")
        .and_then(Value::as_object)
        .map(|tools| {
            ["web_search_requests", "tool_search_requests"]
                .into_iter()
                .filter_map(|field| Some((field.to_owned(), tools.get(field)?.clone())))
                .filter(|(_, value)| !value.is_null())
                .collect::<Map<String, Value>>()
        })
        .filter(|tools| !tools.is_empty())
        .map(Value::Object);
    ChatCompletionsUsage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
        prompt_tokens_details: PromptTokensDetails {
            cached_tokens: cache_read,
            cache_creation_tokens: cache_creation,
            text_tokens: input,
            cache_write_tokens: Some(cache_creation),
            cache_creation_token_details,
        },
        completion_tokens_details: Some(completion_tokens_details),
        server_tool_use,
        cache_creation_input_tokens: Some(cache_creation),
        cache_read_input_tokens: Some(cache_read),
    }
}

impl StreamTransformer for ModelResponseIterator {
    type Input = MessagesStreamEvent;
    type Output = ChatCompletionChunk;
    type Error = Error;

    fn transform(&mut self, input: Self::Input) -> Result<Vec<Self::Output>, Self::Error> {
        self.chunk_parser(input).map(|chunk| vec![chunk])
    }

    fn finish(&mut self) -> Result<Vec<Self::Output>, Self::Error> {
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn events(wire: Value) -> Vec<MessagesStreamEvent> {
        serde_json::from_value(wire).unwrap()
    }

    fn parse(wire: Value) -> Vec<Value> {
        let mut iterator = ModelResponseIterator::new(StreamShape::default());
        events(wire)
            .into_iter()
            .map(|event| serde_json::to_value(iterator.chunk_parser(event).unwrap()).unwrap())
            .collect()
    }

    fn delta(chunk: &Value) -> &Value {
        &chunk["choices"][0]["delta"]
    }

    #[rstest]
    fn text_deltas_become_content_and_message_start_names_the_model() {
        let chunks = parse(json!([
            {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x-1", "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 3, "output_tokens": 1}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
            {"type": "ping"}
        ]));
        assert_eq!(chunks[0]["model"], "claude-x-1");
        assert_eq!(chunks[0]["usage"]["prompt_tokens"], 3);
        assert_eq!(delta(&chunks[1]), &json!({"content": ""}));
        assert_eq!(delta(&chunks[2]), &json!({"content": "Hi"}));
        assert_eq!(chunks[2]["model"], "claude-x-1");
        assert_eq!(delta(&chunks[3]), &json!({"content": ""}));
    }

    #[rstest]
    fn tool_calls_stream_their_name_then_partial_arguments() {
        let chunks = parse(json!([
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"city\": \"Pa"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "ris\"}"}},
            {"type": "content_block_stop", "index": 1},
            {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_2", "name": "now", "input": {}}},
            {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": ""}},
            {"type": "content_block_stop", "index": 2},
            {"type": "content_block_start", "index": 3, "content_block": {"type": "tool_use", "id": "toolu_3", "name": "none", "input": {}}},
            {"type": "content_block_stop", "index": 3}
        ]));
        let tool = |chunk: &Value| delta(chunk)["tool_calls"].clone();
        assert_eq!(delta(&chunks[1]), &json!({"content": ""}));
        assert_eq!(
            tool(&chunks[2]),
            json!([{"id": "toolu_1", "type": "function", "function": {"name": "get_weather", "arguments": ""}, "index": 0}])
        );
        assert_eq!(
            tool(&chunks[3]),
            json!([{"type": "function", "function": {"arguments": "{\"city\": \"Pa"}, "index": 0}])
        );
        assert_eq!(tool(&chunks[4])[0]["function"]["arguments"], "ris\"}");
        assert_eq!(delta(&chunks[5]), &json!({"content": ""}));
        assert_eq!(tool(&chunks[6])[0]["index"], 1);
        // Only empty fragments streamed: the block closes with `{}`.
        assert_eq!(
            tool(&chunks[8]),
            json!([{"type": "function", "function": {"arguments": "{}"}, "index": 1}])
        );
        // No fragment at all: Python's check sees no deltas and adds nothing.
        assert_eq!(delta(&chunks[10]), &json!({"content": ""}));
    }

    #[rstest]
    fn a_reverse_mapped_tool_name_is_restored() {
        let mut shape = StreamShape::default();
        shape
            .tool_name_reverse_map
            .insert("ns_tool".into(), "ns/tool".into());
        let mut iterator = ModelResponseIterator::new(shape);
        let chunk = iterator
            .chunk_parser(events(json!([{"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t", "name": "ns_tool", "input": {}}}])).remove(0))
            .unwrap();
        let tool = &chunk.choices[0].delta.tool_calls.as_ref().unwrap()[0];
        assert_eq!(tool.function.name.as_deref(), Some("ns/tool"));
    }

    #[rstest]
    fn thinking_streams_as_reasoning_content_and_thinking_blocks() {
        let chunks = parse(json!([
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me think"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "redacted_thinking", "data": "enc"}}
        ]));
        let thinking = json!([{"type": "thinking", "thinking": "Let me think", "index": 0}]);
        assert_eq!(
            delta(&chunks[1]),
            &json!({"content": "", "reasoning_content": "Let me think", "thinking_blocks": thinking, "provider_specific_fields": {"thinking_blocks": thinking}})
        );
        let signature =
            json!([{"type": "thinking", "thinking": "", "signature": "sig", "index": 0}]);
        assert_eq!(
            delta(&chunks[2]),
            &json!({"content": "", "reasoning_content": "", "thinking_blocks": signature, "provider_specific_fields": {"thinking_blocks": signature}})
        );
        let redacted = json!([{"type": "redacted_thinking", "data": "enc", "index": 1}]);
        assert_eq!(
            delta(&chunks[4]),
            &json!({"content": "", "thinking_blocks": redacted, "provider_specific_fields": {"thinking_blocks": redacted}})
        );
    }

    #[rstest]
    #[case::end_turn("end_turn", "stop")]
    #[case::max_tokens("max_tokens", "length")]
    #[case::tool_use("tool_use", "tool_calls")]
    #[case::stop_sequence("stop_sequence", "stop")]
    #[case::refusal("refusal", "content_filter")]
    fn message_delta_maps_the_stop_reason(#[case] stop_reason: &str, #[case] expected: &str) {
        let chunks = parse(
            json!([{"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 2}}]),
        );
        assert_eq!(chunks[0]["choices"][0]["finish_reason"], expected);
    }

    #[rstest]
    fn a_refusal_keeps_its_stop_reason_in_the_provider_fields() {
        let chunks = parse(json!([{"type": "message_delta", "delta": {"stop_reason": "refusal"}}]));
        assert_eq!(
            delta(&chunks[0])["provider_specific_fields"],
            json!({"stop_reason": "refusal"})
        );
    }

    #[rstest]
    fn usage_accumulates_across_start_and_delta_with_cache_tokens() {
        let chunks = parse(json!([
            {"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant", "model": "x", "content": [], "stop_reason": null, "stop_sequence": null,
             "usage": {"input_tokens": 10, "cache_creation_input_tokens": 6, "cache_read_input_tokens": 20, "output_tokens": 1,
                       "cache_creation": {"ephemeral_5m_input_tokens": 2, "ephemeral_1h_input_tokens": 4}}}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 12, "server_tool_use": {"web_search_requests": 2}}}
        ]));
        assert_eq!(
            chunks[1]["usage"],
            json!({
                "completion_tokens": 12, "prompt_tokens": 36, "total_tokens": 48,
                "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 12},
                "prompt_tokens_details": {"cached_tokens": 20, "text_tokens": 10, "cache_write_tokens": 6, "cache_creation_tokens": 6,
                    "cache_creation_token_details": {"ephemeral_5m_input_tokens": 2, "ephemeral_1h_input_tokens": 4}},
                "server_tool_use": {"web_search_requests": 2},
                "cache_creation_input_tokens": 6, "cache_read_input_tokens": 20
            })
        );
    }

    #[rstest]
    fn reported_thinking_tokens_split_the_completion() {
        let usage = anthropic_usage(
            json!({"input_tokens": 1, "output_tokens": 10, "output_tokens_details": {"thinking_tokens": 4}})
                .as_object()
                .unwrap(),
            true,
        );
        assert_eq!(
            usage.completion_tokens_details,
            Some(CompletionTokensDetails {
                reasoning_tokens: Some(4),
                text_tokens: Some(6)
            })
        );
    }

    #[rstest]
    fn an_error_event_fails_with_the_providers_error() {
        let mut iterator = ModelResponseIterator::new(StreamShape::default());
        let error = iterator
            .chunk_parser(events(json!([{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}])).remove(0))
            .unwrap_err();
        let Error::Provider { status, body } = error else {
            panic!("an error event is the provider's error, got {error:?}");
        };
        assert_eq!(status, 500);
        assert_eq!(
            serde_json::from_str::<Value>(&body).unwrap(),
            json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
        );
    }

    #[rstest]
    fn citations_and_server_tool_results_go_to_provider_fields() {
        let chunks = parse(json!([
            {"type": "content_block_delta", "index": 0, "delta": {"type": "citations_delta", "citation": {"type": "char_location"}}},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "web_search_tool_result", "tool_use_id": "s", "content": []}}
        ]));
        assert_eq!(
            delta(&chunks[0])["provider_specific_fields"],
            json!({"citation": {"type": "char_location"}})
        );
        assert_eq!(
            delta(&chunks[1])["provider_specific_fields"]["web_search_results"][0]["tool_use_id"],
            "s"
        );
    }
}
