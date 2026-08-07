use crate::constants::MESSAGES_ERROR_BODY_MAX_CHARS;

pub fn truncate_error_body(body: &str) -> String {
    if body.chars().count() <= MESSAGES_ERROR_BODY_MAX_CHARS {
        return body.to_string();
    }
    let truncated: String = body.chars().take(MESSAGES_ERROR_BODY_MAX_CHARS).collect();
    format!("{truncated}... (truncated)")
}
