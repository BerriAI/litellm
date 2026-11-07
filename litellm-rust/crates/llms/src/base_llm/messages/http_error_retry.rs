//! Python's `BaseAnthropicMessagesConfig` recovery from a rejected request: a 400 naming an
//! invalid replayed thinking block is answered by sending the body again without thinking
//! (`should_retry_anthropic_messages_on_http_error` and
//! `transform_anthropic_messages_request_on_http_error`).

use serde_json::Value;

/// Python's `max_retry_on_anthropic_messages_http_error`: the HTTP attempts a call may make.
pub const MAX_ATTEMPTS: usize = 2;

const THINKING_BLOCK_TYPES: [&str; 2] = ["thinking", "redacted_thinking"];

/// Python's `is_anthropic_invalid_thinking_block_error`.
pub fn is_invalid_thinking_block_error(error_text: &str) -> bool {
    if error_text.is_empty() {
        return false;
    }
    let lower = error_text.to_lowercase();
    if !lower.contains("thinking") {
        return false;
    }
    if lower.contains("signature") && (lower.contains("invalid") || lower.contains("valid string"))
    {
        return true;
    }
    lower.contains("must contain thinking")
}

/// Python's `is_anthropic_thinking_binding_mismatch_error`.
fn is_thinking_binding_mismatch_error(error_text: &str) -> bool {
    error_text
        .to_lowercase()
        .contains("bound to a different conversation")
}

/// Python's `anthropic_request_errors_on_thinking_binding_mismatch`: the caller pinned the
/// blocks with `prefix_mismatch_behavior: error`, so the retry would undo what it asked for.
fn errors_on_thinking_binding_mismatch(body: &Value) -> bool {
    body.get("thinking")
        .and_then(Value::as_object)
        .and_then(|thinking| thinking.get("block_binding"))
        .and_then(Value::as_object)
        .and_then(|binding| binding.get("prefix_mismatch_behavior"))
        .is_some_and(|behavior| behavior == "error")
}

/// Python's `should_retry_anthropic_messages_on_http_error` for the thinking case.
pub fn retries_without_thinking(status: u16, error_text: &str, body: &Value) -> bool {
    if status != 400 {
        return false;
    }
    let caller_asked_for_this_error =
        errors_on_thinking_binding_mismatch(body) && is_thinking_binding_mismatch_error(error_text);
    !caller_asked_for_this_error && is_invalid_thinking_block_error(error_text)
}

fn is_thinking_block(block: &Value) -> bool {
    block
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| THINKING_BLOCK_TYPES.contains(&kind))
}

/// Python's `strip_thinking_blocks_from_anthropic_messages`: a message left with an empty
/// content list is dropped, since Anthropic rejects one.
fn without_thinking_blocks(messages: Vec<Value>) -> Vec<Value> {
    messages
        .into_iter()
        .filter_map(|mut message| {
            if let Some(Value::Array(content)) = message.get_mut("content") {
                content.retain(|block| !is_thinking_block(block));
                if content.is_empty() {
                    return None;
                }
            }
            Some(message)
        })
        .collect()
}

/// Python's `strip_thinking_blocks_from_anthropic_messages_request_dict`: the body without its
/// replayed thinking blocks and without the `thinking` param.
pub fn without_thinking(body: Value) -> Value {
    let Value::Object(body) = body else {
        return body;
    };
    Value::Object(
        body.into_iter()
            .filter(|(key, _)| key != "thinking")
            .map(|(key, value)| match (key.as_str(), value) {
                ("messages", Value::Array(messages)) => {
                    (key, Value::Array(without_thinking_blocks(messages)))
                }
                (_, value) => (key, value),
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::invalid_signature(
        "messages.1.content.0: Invalid `signature` in `thinking` block",
        true
    )]
    #[case::not_a_string(
        "messages.2.content.0.thinking.signature.str: Input should be a valid string",
        true
    )]
    #[case::empty_thinking(
        "messages.1.content.0.thinking: each thinking block must contain thinking",
        true
    )]
    #[case::no_thinking("Invalid `signature` in block", false)]
    #[case::unrelated("max_tokens: too large", false)]
    #[case::empty("", false)]
    fn invalid_thinking_errors_are_recognized(#[case] text: &str, #[case] expected: bool) {
        assert_eq!(is_invalid_thinking_block_error(text), expected);
    }

    #[rstest]
    #[case::rejected_signature(400, "Invalid `signature` in `thinking` block", json!({}), true)]
    #[case::other_status(500, "Invalid `signature` in `thinking` block", json!({}), false)]
    #[case::pinned_binding(
        400,
        "messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a different conversation.",
        json!({"thinking": {"block_binding": {"prefix_mismatch_behavior": "error"}}}),
        false,
    )]
    #[case::pinned_binding_other_error(
        400,
        "Invalid `signature` in `thinking` block",
        json!({"thinking": {"block_binding": {"prefix_mismatch_behavior": "error"}}}),
        true,
    )]
    fn retry_follows_the_rejection(
        #[case] status: u16,
        #[case] text: &str,
        #[case] body: Value,
        #[case] expected: bool,
    ) {
        assert_eq!(retries_without_thinking(status, text, &body), expected);
    }

    #[test]
    fn thinking_is_stripped_from_the_body() {
        let body = json!({
            "model": "claude",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "x"}]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "t", "signature": "s"},
                    {"type": "text", "text": "Paris"}
                ]},
                "not a message"
            ],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "max_tokens": 64
        });
        assert_eq!(
            without_thinking(body),
            json!({
                "model": "claude",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [{"type": "text", "text": "Paris"}]},
                    "not a message"
                ],
                "max_tokens": 64
            })
        );
    }
}
