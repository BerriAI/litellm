use litellm_core::error::{json_type_name, CoreError};
use litellm_core::messages::transformation::AnthropicMessagesProviderConfig;
use litellm_core::providers::azure_ai::messages::transformation::AZURE_ANTHROPIC_MESSAGES_CONFIG;
use litellm_core::CoreResult;
use serde_json::{Map, Value};

use crate::constants::MESSAGES_ERROR_BODY_MAX_CHARS;

pub(super) fn truncate_error_body(body: &str) -> String {
    if body.chars().count() <= MESSAGES_ERROR_BODY_MAX_CHARS {
        return body.to_string();
    }
    let truncated: String = body.chars().take(MESSAGES_ERROR_BODY_MAX_CHARS).collect();
    format!("{truncated}... (truncated)")
}

pub(super) fn messages_provider_config(
    provider: &str,
) -> Option<&'static dyn AnthropicMessagesProviderConfig> {
    match provider {
        "azure_ai" => Some(&AZURE_ANTHROPIC_MESSAGES_CONFIG),
        _ => None,
    }
}

pub(super) fn string_headers(
    extra_headers: Option<Map<String, Value>>,
) -> CoreResult<Vec<(String, String)>> {
    extra_headers
        .unwrap_or_default()
        .into_iter()
        .map(|(key, value)| {
            value
                .as_str()
                .map(|value| (key.clone(), value.to_string()))
                .ok_or_else(|| {
                    CoreError::InvalidRequest(format!(
                        "messages extra_headers.{key} must be a string, got {}",
                        json_type_name(&value)
                    ))
                })
        })
        .collect()
}

pub(super) fn has_header(headers: &[(String, String)], name: &str) -> bool {
    headers
        .iter()
        .any(|(key, _)| key.eq_ignore_ascii_case(name))
}
