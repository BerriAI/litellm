use litellm_providers::{
    anthropic::experimental_pass_through::messages::transformation::ANTHROPIC_MESSAGES_CONFIG,
    azure_ai::anthropic::messages_transformation::AZURE_ANTHROPIC_MESSAGES_CONFIG,
    base_llm::anthropic_messages::transformation::BaseAnthropicMessagesConfig,
};
use serde_json::{Map, Value};

use super::Error;
use crate::http_utils::string_headers as shared_string_headers;
pub(super) use crate::http_utils::{has_bearer_auth, has_header, truncate_error_body};

const HEADER_CONTEXT: &str = "messages";

pub(super) fn messages_provider_config(
    provider: &str,
) -> Option<&'static dyn BaseAnthropicMessagesConfig> {
    match provider {
        "anthropic" => Some(&ANTHROPIC_MESSAGES_CONFIG),
        "azure_ai" => Some(&AZURE_ANTHROPIC_MESSAGES_CONFIG),
        _ => None,
    }
}

pub(super) fn string_headers(
    extra_headers: Option<Map<String, Value>>,
) -> Result<Vec<(String, String)>, Error> {
    shared_string_headers(HEADER_CONTEXT, extra_headers).map_err(Error::from)
}
