use serde_json::{Map, Value};

use crate::error::CoreResult;
use crate::http_utils::string_headers as shared_string_headers;
use crate::providers::anthropic::chat_completions::transformation::ANTHROPIC_CHAT_COMPLETIONS_CONFIG;

use super::transformation::ChatCompletionsProviderConfig;

const HEADER_CONTEXT: &str = "chat completions";

pub(super) fn chat_completions_provider_config(
    provider: &str,
) -> Option<&'static dyn ChatCompletionsProviderConfig> {
    match provider {
        "anthropic" => Some(&ANTHROPIC_CHAT_COMPLETIONS_CONFIG),
        #[cfg(feature = "bedrock-auth")]
        "bedrock" => Some(
            &crate::providers::bedrock::chat_completions::transformation::BEDROCK_CHAT_COMPLETIONS_CONFIG,
        ),
        _ => None,
    }
}

pub(super) fn string_headers(
    extra_headers: Option<Map<String, Value>>,
) -> CoreResult<Vec<(String, String)>> {
    shared_string_headers(HEADER_CONTEXT, extra_headers)
}
