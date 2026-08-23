use serde_json::{Map, Value};

use crate::error::CoreResult;
use crate::http_utils::string_headers as shared_string_headers;
use crate::providers::anthropic::messages::transformation::ANTHROPIC_MESSAGES_CONFIG;
use crate::providers::azure_ai::messages::transformation::AZURE_ANTHROPIC_MESSAGES_CONFIG;

use super::transformation::AnthropicMessagesProviderConfig;

pub(super) use crate::http_utils::{has_bearer_auth, has_header, truncate_error_body};

const HEADER_CONTEXT: &str = "messages";

pub(super) fn messages_provider_config(
    provider: &str,
) -> Option<&'static dyn AnthropicMessagesProviderConfig> {
    match provider {
        "anthropic" => Some(&ANTHROPIC_MESSAGES_CONFIG),
        "azure_ai" => Some(&AZURE_ANTHROPIC_MESSAGES_CONFIG),
        _ => None,
    }
}

pub(super) fn string_headers(
    extra_headers: Option<Map<String, Value>>,
) -> CoreResult<Vec<(String, String)>> {
    shared_string_headers(HEADER_CONTEXT, extra_headers)
}
