//! The answer that served a `/v1/messages` call, kept for the `response` of its logging payload:
//! a complete answer's body as the provider sent it, or what a stream's events add up to as they
//! pass. `litellm_standard_logging::messages_response` turns either into the chat
//! `ModelResponse` Python logs for the call.

use litellm_standard_logging::messages_response::{
    Error, LoggedResponse, ResponseContext, StreamedMessage, complete_response,
};
use serde_json::{Map, Value};

/// The served Messages answer, before the route renders it for the client.
#[derive(Clone, Debug, PartialEq)]
pub enum ServedMessage {
    /// A complete answer's body, which Python's `AnthropicConfig.transform_response` reads.
    Complete(Map<String, Value>),
    /// A stream's events so far, read as the Anthropic passthrough logging handler reads them.
    Streamed(Box<StreamedMessage>),
}

impl ServedMessage {
    /// Adds one stream event's `data` to the stream `served` keeps, starting it on the first.
    pub(crate) fn see(served: &mut Option<Self>, data: &str) {
        match served.get_or_insert_with(|| Self::Streamed(Box::default())) {
            Self::Streamed(stream) => stream.see(data),
            Self::Complete(_) => {}
        }
    }

    /// The `ModelResponse.model_dump()` Python logs for the answer. `streamed_model` is the model
    /// a stream's response is named by, the one Python priced the call with; a complete answer
    /// keeps the model its body names.
    pub fn logged_response(
        &self,
        streamed_model: &str,
        context: &ResponseContext<'_>,
    ) -> Result<LoggedResponse, Error> {
        match self {
            Self::Complete(body) => complete_response(body, context),
            Self::Streamed(stream) => stream.response(streamed_model, context),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    const START: &str = r#"{"type":"message_start","message":{"id":"msg_1","model":"claude","usage":{"input_tokens":3,"output_tokens":1}}}"#;
    const TEXT: &str =
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#;
    const DELTA: &str = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}"#;

    fn context() -> ResponseContext<'static> {
        ResponseContext {
            created: 1,
            fallback_id: "fallback",
            requested_speed: None,
            count_tokens: &|_| 0,
        }
    }

    fn logged(served: &ServedMessage) -> Value {
        json!(
            served
                .logged_response("claude-priced", &context())
                .expect("loggable")
        )
    }

    #[rstest]
    fn a_stream_is_kept_event_by_event() {
        let mut served = None;
        ServedMessage::see(&mut served, START);
        ServedMessage::see(&mut served, TEXT);
        ServedMessage::see(&mut served, DELTA);
        let logged = logged(&served.expect("kept"));
        assert_eq!(
            (
                &logged["id"],
                &logged["model"],
                &logged["choices"][0]["message"]["content"],
                &logged["usage"]["total_tokens"]
            ),
            (
                &json!("msg_1"),
                &json!("claude-priced"),
                &json!("Hi"),
                &json!(5)
            )
        );
    }

    #[rstest]
    fn a_complete_answer_keeps_the_model_its_body_names() {
        let body = json!({
            "id": "msg_2", "type": "message", "role": "assistant", "model": "claude-served",
            "content": [{"type": "text", "text": "Hi"}], "stop_reason": "end_turn",
            "stop_sequence": null, "usage": {"input_tokens": 3, "output_tokens": 2}
        });
        let served = ServedMessage::Complete(body.as_object().cloned().unwrap_or_default());
        assert_eq!(logged(&served)["model"], json!("claude-served"));
    }

    #[rstest]
    fn an_event_after_a_complete_answer_changes_nothing() {
        let mut served = Some(ServedMessage::Complete(Map::new()));
        ServedMessage::see(&mut served, START);
        assert_eq!(served, Some(ServedMessage::Complete(Map::new())));
    }
}
