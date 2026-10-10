use litellm_llms_types::{
    billing::BilledAmount,
    formats::messages::{MessagesRequest, MessagesResponse},
    providers::openrouter::OpenRouterUsage,
};
use litellm_router_types::LitellmParams;

use crate::{
    Error,
    anthropic::messages::transformation::{DEFAULT_HEADERS, update_headers_with_anthropic_beta},
    base_llm::messages::{
        context::MessagesTransformContext,
        transformation::{BaseMessagesConfig, Headers, ValidatedEnvironment},
    },
    openai_like::messages::transformation::{
        compatible_host_environment, compatible_host_url, without_billing_blocks,
    },
};

const API_KEY_ENV: &str = "OPENROUTER_API_KEY";
const API_BASE_ENV: &str = "OPENROUTER_API_BASE";
const DEFAULT_BASE: &str = "https://openrouter.ai/api/v1";

/// OpenRouter serves the Anthropic Messages API at `/api/v1/messages` behind its own bearer
/// key, passes extended cache hints through, and reports the amount it billed as `usage.cost`
/// on the response and on the final stream usage.
pub struct OpenRouterAnthropicMessagesConfig;
pub const OPENROUTER_MESSAGES_CONFIG: OpenRouterAnthropicMessagesConfig =
    OpenRouterAnthropicMessagesConfig;

impl BaseMessagesConfig for OpenRouterAnthropicMessagesConfig {
    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _params: &LitellmParams,
        _stream: bool,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(compatible_host_url(
            api_base,
            API_BASE_ENV,
            DEFAULT_BASE,
            env,
        ))
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        _context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        Ok(without_billing_blocks(request))
    }

    fn secret_names(&self) -> &'static [&'static str] {
        &[API_KEY_ENV, API_BASE_ENV]
    }

    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        _params: &LitellmParams,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        compatible_host_environment(headers, api_key, "OpenRouter", API_KEY_ENV, env)
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_HEADERS
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        update_headers_with_anthropic_beta(headers, request)
    }

    fn reported_cost(&self, response: &MessagesResponse) -> Option<serde_json::Number> {
        let usage: OpenRouterUsage = serde_json::from_value(response.usage.clone()?).ok()?;
        usage.cost.map(BilledAmount::into_number)
    }
}
