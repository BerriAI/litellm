//! Gemini `streamGenerateContent?alt=sse` events to Chat Completions chunks, as Python's
//! `ModelResponseIterator.chunk_parser` in `vertex_and_google_ai_studio_gemini.py` builds them.
//!
//! DIVERGENCE: grounding, URL context, safety and citation metadata on stream chunks are not
//! forwarded as hidden params, and inline images and audio in a chunk are dropped.

use litellm_core_utils::core_helpers::{finish_reason_for, unix_now};
use litellm_llms_types::formats::chat_completions::{
    ChatCompletionChunk, ChatCompletionDelta, ChatCompletionStreamingChoice,
};
use serde_json::{Map, Value};

use crate::{
    Error,
    base_llm::{base_model_iterator::StreamTransformer, chat::streaming::ChatStream, sse},
    vertex_ai::gemini::vertex_and_google_ai_studio_gemini::{
        calculate_usage, candidate_message, prompt_blocked, served_model_name,
    },
};

/// Gemini's `error` events carry their own HTTP code; Python falls back to a 500.
const STREAM_ERROR_STATUS: u16 = 500;

pub fn gemini_chat_stream() -> ChatStream {
    ChatStream::new(
        |bytes| sse::event_stream::<Value>(bytes, sse::is_done),
        GeminiStreamIterator::default(),
    )
}

#[derive(Default)]
pub struct GeminiStreamIterator {
    tool_index: i64,
    has_seen_tool_calls: bool,
}

fn stream_error(error: &Value) -> Error {
    let status = match error.get("code") {
        Some(Value::Number(code)) => code.as_u64().and_then(|code| u16::try_from(code).ok()),
        Some(Value::String(code)) => code.parse().ok(),
        _ => None,
    }
    .unwrap_or(STREAM_ERROR_STATUS);
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    Error::Provider {
        status,
        body: format!("{status} - {message}"),
    }
}

impl GeminiStreamIterator {
    fn choice(&mut self, position: usize, candidate: &Value) -> ChatCompletionStreamingChoice {
        let message = candidate_message(candidate, &mut self.tool_index);
        let has_tool_calls = !message.tool_calls.is_empty();
        self.has_seen_tool_calls |= has_tool_calls;
        let finish_reason = match (&message.finish_reason, has_tool_calls) {
            (_, true) => Some("tool_calls".to_owned()),
            (Some(reason), false) => Some(
                match (finish_reason_for(reason), self.has_seen_tool_calls) {
                    ("stop", true) => "tool_calls",
                    (mapped, _) => mapped,
                }
                .to_owned(),
            ),
            (None, false) => None,
        };
        let mut extra = Map::new();
        if let Some(annotations) = message.annotations {
            extra.insert("annotations".into(), Value::Array(annotations));
        }
        ChatCompletionStreamingChoice {
            index: candidate
                .get("index")
                .and_then(Value::as_u64)
                .unwrap_or(position as u64),
            delta: ChatCompletionDelta {
                content: message.content,
                role: None,
                tool_calls: has_tool_calls.then_some(message.tool_calls),
                reasoning_content: message.reasoning_content,
                thinking_blocks: message.thinking_blocks.filter(|blocks| !blocks.is_empty()),
                provider_specific_fields: message.provider_specific_fields,
                extra,
            },
            finish_reason,
            logprobs: message.logprobs,
        }
    }
}

impl StreamTransformer for GeminiStreamIterator {
    type Input = Value;
    type Output = ChatCompletionChunk;
    type Error = Error;

    fn transform(&mut self, event: Value) -> Result<Vec<ChatCompletionChunk>, Error> {
        if let Some(error) = event.get("error") {
            return Err(stream_error(error));
        }
        let choices = match prompt_blocked(&event) {
            true => vec![ChatCompletionStreamingChoice {
                index: 0,
                delta: ChatCompletionDelta {
                    role: Some("assistant".into()),
                    ..ChatCompletionDelta::default()
                },
                finish_reason: Some("content_filter".into()),
                logprobs: None,
            }],
            false => event
                .get("candidates")
                .and_then(Value::as_array)
                .map(|candidates| {
                    candidates
                        .iter()
                        .enumerate()
                        .map(|(position, candidate)| self.choice(position, candidate))
                        .collect()
                })
                .unwrap_or_default(),
        };
        let usage = match event.get("usageMetadata") {
            Some(_) => Some(calculate_usage(&event)?),
            None => None,
        };
        Ok(vec![ChatCompletionChunk {
            id: event
                .get("responseId")
                .and_then(Value::as_str)
                .map(str::to_owned),
            created: Some(unix_now()),
            model: served_model_name(&event),
            object: Some("chat.completion.chunk".into()),
            choices,
            usage,
            provider_specific_fields: None,
            extra: Map::new(),
        }])
    }

    fn finish(&mut self) -> Result<Vec<ChatCompletionChunk>, Error> {
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn first_choice(chunks: &[ChatCompletionChunk]) -> &ChatCompletionStreamingChoice {
        &chunks[0].choices[0]
    }

    #[test]
    fn text_streams_without_a_finish_until_gemini_sends_one() {
        let mut iterator = GeminiStreamIterator::default();
        let chunks = iterator
            .transform(json!({
                "candidates": [{"content": {"role": "model", "parts": [{"text": "Hel"}]}}],
                "modelVersion": "gemini-2.5-flash",
                "responseId": "r1"
            }))
            .unwrap();
        let choice = first_choice(&chunks);
        assert_eq!(choice.delta.content.as_deref(), Some("Hel"));
        assert_eq!(choice.finish_reason, None);
        assert_eq!(chunks[0].id.as_deref(), Some("r1"));
        assert_eq!(chunks[0].model.as_deref(), Some("gemini-2.5-flash"));
        assert!(chunks[0].usage.is_none());
    }

    #[test]
    fn the_last_event_carries_the_finish_and_usage() {
        let mut iterator = GeminiStreamIterator::default();
        let chunks = iterator
            .transform(json!({
                "candidates": [{"content": {"role": "model", "parts": [{"text": "lo"}]}, "finishReason": "MAX_TOKENS"}],
                "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2, "totalTokenCount": 6}
            }))
            .unwrap();
        assert_eq!(
            first_choice(&chunks).finish_reason.as_deref(),
            Some("length")
        );
        let usage = chunks[0].usage.as_ref().unwrap();
        assert_eq!((usage.prompt_tokens, usage.completion_tokens), (4, 2));
    }

    #[test]
    fn tool_calls_count_across_events_and_turn_stop_into_tool_calls() {
        let mut iterator = GeminiStreamIterator::default();
        let call = |name: &str| json!({"candidates": [{"content": {"parts": [{"functionCall": {"name": name, "args": {"a": 1}}}]}}]});
        let first = iterator.transform(call("one")).unwrap();
        let second = iterator.transform(call("two")).unwrap();
        let last = iterator
            .transform(json!({"candidates": [{"content": {"parts": [{"text": ""}]}, "finishReason": "STOP"}]}))
            .unwrap();
        let tool = |chunks: &[ChatCompletionChunk]| {
            first_choice(chunks).delta.tool_calls.as_ref().unwrap()[0].clone()
        };
        assert_eq!(tool(&first).index, 0);
        assert_eq!(tool(&second).index, 1);
        assert_eq!(tool(&second).function.arguments, "{\"a\": 1}");
        assert_eq!(
            first_choice(&first).finish_reason.as_deref(),
            Some("tool_calls")
        );
        assert_eq!(
            first_choice(&last).finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    #[rstest]
    #[case::numeric(json!({"error": {"code": 429, "message": "slow down", "status": "RESOURCE_EXHAUSTED"}}), 429, "429 - slow down")]
    #[case::text(json!({"error": {"code": "503", "message": "busy"}}), 503, "503 - busy")]
    #[case::bare(json!({"error": {"message": "oops"}}), 500, "500 - oops")]
    fn an_error_event_ends_the_stream_with_its_status(
        #[case] event: Value,
        #[case] status: u16,
        #[case] body: &str,
    ) {
        let error = GeminiStreamIterator::default()
            .transform(event)
            .unwrap_err();
        assert_eq!(
            error,
            Error::Provider {
                status,
                body: body.into()
            }
        );
    }

    #[test]
    fn a_blocked_prompt_is_a_content_filter() {
        let chunks = GeminiStreamIterator::default()
            .transform(json!({"promptFeedback": {"blockReason": "SAFETY"}}))
            .unwrap();
        let choice = first_choice(&chunks);
        assert_eq!(choice.finish_reason.as_deref(), Some("content_filter"));
        assert_eq!(choice.delta.role.as_deref(), Some("assistant"));
    }

    #[test]
    fn reasoning_and_signatures_stream_on_the_delta() {
        let chunks = GeminiStreamIterator::default()
            .transform(json!({"candidates": [{"content": {"parts": [
                {"text": "think", "thought": true, "thoughtSignature": "sig"}
            ]}}]}))
            .unwrap();
        let delta = &first_choice(&chunks).delta;
        assert_eq!(delta.reasoning_content.as_deref(), Some("think"));
        assert_eq!(delta.content, None);
        assert_eq!(
            delta.provider_specific_fields.as_ref().unwrap()["thought_signatures"],
            json!(["sig"])
        );
    }
}
