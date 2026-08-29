#![no_main]

use libfuzzer_sys::fuzz_target;
use serde_json::Value;

/// Validates that input validation logic doesn't panic on arbitrary JSON input.
/// This mirrors the validation in the chat completions service.
fn validate_chat_completions_body(body: &Value) -> Result<(), String> {
    let messages = body.get("messages").ok_or("missing messages")?;
    let messages_arr = messages.as_array().ok_or("messages not array")?;
    if messages_arr.is_empty() {
        return Err("empty messages".to_string());
    }
    for (i, msg) in messages_arr.iter().enumerate() {
        if msg.get("role").and_then(Value::as_str).is_none() {
            return Err(format!("messages[{i}] missing role"));
        }
    }
    if let Some(temp) = body.get("temperature").and_then(Value::as_f64) {
        if !(0.0..=2.0).contains(&temp) {
            return Err(format!("temperature {temp} out of range"));
        }
    }
    if let Some(top_p) = body.get("top_p").and_then(Value::as_f64) {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(format!("top_p {top_p} out of range"));
        }
    }
    if let Some(max_tokens) = body.get("max_tokens").and_then(Value::as_i64) {
        if max_tokens <= 0 {
            return Err(format!("max_tokens {max_tokens} not positive"));
        }
    }
    if let Some(stream) = body.get("stream") {
        if !stream.is_boolean() {
            return Err("stream not boolean".to_string());
        }
    }
    Ok(())
}

fuzz_target!(|data: &[u8]| {
    if let Ok(body) = serde_json::from_slice::<Value>(data) {
        // Validation should never panic, only return errors
        let _ = validate_chat_completions_body(&body);
    }
});
