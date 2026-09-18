use litellm_llms::{
    anthropic::chat::transformation::ANTHROPIC_CHAT_COMPLETIONS_CONFIG,
    base_llm::chat::transformation::BaseConfig,
};
use serde_json::{Map, Value};

use super::Error;
use crate::http_utils::string_headers as shared_string_headers;

const HEADER_CONTEXT: &str = "chat completions";

pub(super) fn chat_completions_provider_config(provider: &str) -> Option<&'static dyn BaseConfig> {
    match provider {
        "anthropic" => Some(&ANTHROPIC_CHAT_COMPLETIONS_CONFIG),
        "bedrock" => Some(
            &litellm_llms::bedrock::chat::converse_transformation::BEDROCK_CHAT_COMPLETIONS_CONFIG,
        ),
        _ => None,
    }
}

pub(super) fn string_headers(
    extra_headers: Option<Map<String, Value>>,
) -> Result<Vec<(String, String)>, Error> {
    shared_string_headers(HEADER_CONTEXT, extra_headers).map_err(Error::from)
}
