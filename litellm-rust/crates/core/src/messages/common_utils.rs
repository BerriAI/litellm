use crate::Error;
use crate::http_utils::string_headers as shared_string_headers;
use crate::providers::anthropic::messages::transformation::ANTHROPIC_MESSAGES_CONFIG;
use crate::providers::azure_ai::messages::transformation::AZURE_ANTHROPIC_MESSAGES_CONFIG;
use serde_json::{Map, Value};

use super::transformation::AnthropicMessagesProviderConfig;

pub(super) use crate::http_utils::{has_bearer_auth, has_header, truncate_error_body};

const HEADER_CONTEXT: &str = "messages";

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
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
) -> Result<Vec<(String, String)>, Error> {
    shared_string_headers(HEADER_CONTEXT, extra_headers)
}
