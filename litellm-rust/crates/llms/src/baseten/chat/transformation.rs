use litellm_llms_types::formats::chat_completions::{ChatCompletionsResponse, ChatMessage};
use serde_json::{Map, Value};

use crate::{
    Error,
    base_llm::chat::transformation::{
        BaseConfig, Headers, ProviderChatRequestData, ProviderChatResponseData,
        ValidatedEnvironment,
    },
    baseten::common_utils::{SECRET_NAMES, complete_url, validate_environment},
    openai_like::chat::transformation::OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG,
};

const SUPPORTED_PARAMS: &[(&str, &str)] = &[
    ("max_tokens", "max_tokens"),
    ("max_completion_tokens", "max_completion_tokens"),
    ("response_format", "response_format"),
    ("seed", "seed"),
    ("stop", "stop"),
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("user", "user"),
    ("presence_penalty", "presence_penalty"),
    ("frequency_penalty", "frequency_penalty"),
    ("n", "n"),
    ("stream_options", "stream_options"),
];

pub struct BasetenConfig;

pub const BASETEN_CHAT_COMPLETIONS_CONFIG: BasetenConfig = BasetenConfig;

impl BaseConfig for BasetenConfig {
    fn supported_openai_param_mappings(&self) -> &'static [(&'static str, &'static str)] {
        SUPPORTED_PARAMS
    }

    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        validate_environment(headers, api_key, env_lookup)
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(complete_url(api_base, "chat/completions", env_lookup))
    }

    fn transform_request(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        optional_params: Map<String, Value>,
    ) -> Result<ProviderChatRequestData, Error> {
        OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG.transform_request(model, messages, optional_params)
    }

    fn transform_response(
        &self,
        model: &str,
        response: ProviderChatResponseData,
    ) -> Result<ChatCompletionsResponse, Error> {
        OPENAI_LIKE_CHAT_COMPLETIONS_CONFIG.transform_response(model, response)
    }

    fn secret_names(&self) -> Vec<&'static str> {
        SECRET_NAMES.to_vec()
    }

    fn config_params(&self) -> &'static [&'static str] {
        &["extra_headers", "max_retries"]
    }
}
