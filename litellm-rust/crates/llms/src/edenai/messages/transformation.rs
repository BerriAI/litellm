use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_layered::{Layered, non_blank, read};
use litellm_llms_types::formats::messages::{
    MessagesOptionalParams, MessagesRequest, MessagesResponse,
};
use litellm_llms_types::providers::anthropic::DEFAULT_HEADERS;
use litellm_router_types::LitellmParams;
use serde_json::Value;

use crate::{
    Error,
    anthropic::{
        common_utils::{filter_billing_headers_from_system, has_anthropic_credential},
        messages::transformation::update_headers_with_anthropic_beta,
    },
    base_llm::{
        auth::AuthScheme,
        messages::{
            context::MessagesTransformContext,
            transformation::{BaseMessagesConfig, Headers, ValidatedEnvironment},
        },
    },
    openai_like::messages::transformation::{complete_messages_url, portable_cache_control},
};

const API_KEY_ENV: &str = "EDENAI_API_KEY";
const API_BASE_ENV: &str = "EDENAI_API_BASE";
const DEFAULT_BASE: &str = "https://api.edenai.run/v3";

#[derive(Layered)]
#[expect(dead_code, reason = "only the derived partial is read")]
struct EdenAiKeys {
    #[layer(litellm_param = "api_key", env = "EDENAI_API_KEY", read = read::text, secret, setting = "API Key")]
    api_key: String,
    #[layer(litellm_param = "api_base", env = "EDENAI_API_BASE", read = read::text, setting = "API Base")]
    api_base: String,
}

impl EdenAiKeysPartial {
    fn env(env: &dyn Fn(&str) -> Option<String>) -> Self {
        Self::from_env(&env).unwrap_or_default()
    }
}

pub struct EdenAIAnthropicMessagesConfig;
pub const EDENAI_MESSAGES_CONFIG: EdenAIAnthropicMessagesConfig = EdenAIAnthropicMessagesConfig;

impl BaseMessagesConfig for EdenAIAnthropicMessagesConfig {
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        _optional_params: &MessagesOptionalParams,
        _params: &LitellmParams,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let key = non_blank(api_key)
            .or(EdenAiKeysPartial::env(env).api_key)
            .ok_or(Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "Eden AI",
                environment_variable: EdenAiKeys::API_KEY.env[0],
            }))?;
        if has_anthropic_credential(&headers) {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            });
        }
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                secret: SecretValue::new(key),
            },
        })
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _params: &LitellmParams,
        _stream: bool,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let base = non_blank(api_base)
            .or(EdenAiKeysPartial::env(env).api_base)
            .unwrap_or_else(|| DEFAULT_BASE.into());
        Ok(complete_messages_url(&base))
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        _context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        Ok(MessagesRequest {
            params: MessagesOptionalParams {
                system: request
                    .params
                    .system
                    .and_then(filter_billing_headers_from_system),
                ..request.params
            },
            ..request
        })
    }

    fn secret_names(&self) -> &'static [&'static str] {
        &[API_KEY_ENV, API_BASE_ENV]
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_HEADERS
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        update_headers_with_anthropic_beta(headers, request)
    }

    fn wire_body(&self, body: Value) -> Value {
        portable_cache_control(body)
    }

    fn reported_cost(&self, response: &MessagesResponse) -> Option<serde_json::Number> {
        let cost = match response.extra.get("cost")? {
            Value::Number(cost) => cost.as_f64()?,
            Value::String(cost) => cost.parse::<f64>().ok()?,
            _ => return None,
        };
        if cost < 0.0 {
            return None;
        }
        serde_json::Number::from_f64(cost)
    }
}
