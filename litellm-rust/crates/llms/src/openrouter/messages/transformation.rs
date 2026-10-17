use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_core_utils::settings::resolve_non_empty;
use litellm_http::request::{has_bearer_auth, without_headers};
use litellm_llms_types::{
    formats::messages::{MessagesOptionalParams, MessagesRequest},
    providers::anthropic::DEFAULT_HEADERS,
};
use litellm_router_types::LitellmParams;

use crate::{
    Error,
    anthropic::{
        common_utils::filter_billing_headers_from_system,
        messages::transformation::update_headers_with_anthropic_beta,
    },
    base_llm::{
        auth::{AuthScheme, Headers, ValidatedEnvironment},
        messages::{context::MessagesTransformContext, transformation::BaseMessagesConfig},
    },
};

const API_KEY_ENV: &str = "OPENROUTER_API_KEY";
const API_BASE_ENV: &str = "OPENROUTER_API_BASE";
const DEFAULT_BASE: &str = "https://openrouter.ai/api/v1";
const MESSAGES_PATH: &str = "/v1/messages";

pub struct OpenRouterAnthropicMessagesConfig;
pub const OPENROUTER_MESSAGES_CONFIG: OpenRouterAnthropicMessagesConfig =
    OpenRouterAnthropicMessagesConfig;

impl BaseMessagesConfig for OpenRouterAnthropicMessagesConfig {
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        _params: &LitellmParams,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let headers = without_headers(headers, &["x-api-key"]);
        if has_bearer_auth(&headers) {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            });
        }
        let key = resolve_non_empty(api_key, env, &[API_KEY_ENV]).ok_or(Error::Auth(
            litellm_auth::Error::MissingApiKey {
                provider: "OpenRouter",
                environment_variable: API_KEY_ENV,
            },
        ))?;
        Ok(ValidatedEnvironment {
            headers: without_headers(headers, &["authorization"]),
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
        params: &LitellmParams,
        _stream: bool,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let explicit = api_base
            .filter(|base| !base.trim().is_empty())
            .or(params.api_base.as_deref());
        let base = resolve_non_empty(explicit, env, &[API_BASE_ENV])
            .unwrap_or_else(|| DEFAULT_BASE.into());
        let trimmed = base.trim_end_matches('/');
        if trimmed.ends_with(MESSAGES_PATH) {
            return Ok(trimmed.into());
        }
        Ok(format!(
            "{}{MESSAGES_PATH}",
            trimmed.strip_suffix("/v1").unwrap_or(trimmed)
        ))
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serde_json::json;

    #[rstest]
    #[case::default(None, "https://openrouter.ai/api/v1/messages")]
    #[case::root(Some("https://proxy.test"), "https://proxy.test/v1/messages")]
    #[case::version(
        Some("https://proxy.test/api/v1/"),
        "https://proxy.test/api/v1/messages"
    )]
    #[case::complete(
        Some("https://proxy.test/api/v1/messages/"),
        "https://proxy.test/api/v1/messages"
    )]
    fn urls_append_messages_once(#[case] base: Option<&str>, #[case] expected: &str) {
        assert_eq!(
            OPENROUTER_MESSAGES_CONFIG
                .get_complete_url(
                    base,
                    "anthropic/test-model",
                    &LitellmParams::default(),
                    true,
                    &|_| None
                )
                .unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::call(
        Some("https://call.test/v1"),
        Some("https://deployment.test/v1"),
        "https://call.test/v1/messages"
    )]
    #[case::deployment(
        None,
        Some("https://deployment.test/v1"),
        "https://deployment.test/v1/messages"
    )]
    #[case::blank_call(
        Some(" "),
        Some("https://deployment.test/v1"),
        "https://deployment.test/v1/messages"
    )]
    #[case::environment(None, None, "https://secret.test/v1/messages")]
    fn base_precedence_is_call_deployment_then_secrets(
        #[case] base: Option<&str>,
        #[case] deployment: Option<&str>,
        #[case] expected: &str,
    ) {
        let params = LitellmParams {
            api_base: deployment.map(str::to_string),
            ..Default::default()
        };
        assert_eq!(
            OPENROUTER_MESSAGES_CONFIG
                .get_complete_url(base, "test-model", &params, false, &|name| (name
                    == API_BASE_ENV)
                    .then(|| "https://secret.test/v1".into()))
                .unwrap(),
            expected
        );
    }

    #[rstest]
    fn native_payload_is_preserved_except_first_party_billing_blocks() {
        let input = json!({"model": "anthropic/test-model", "max_tokens": 128, "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}], "system": [{"type": "text", "text": "x-anthropic-billing-header: private"}, {"type": "text", "text": "instructions"}], "metadata": {"user_id": "user", "custom": "keep"}, "provider": {"order": ["test-provider"]}, "thinking": {"type": "enabled", "budget_tokens": 64}});
        let request = serde_json::from_value(input.clone()).unwrap();
        let actual = OPENROUTER_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .unwrap();
        let expected = json!({"system": [{"type": "text", "text": "instructions"}]});
        let body = serde_json::to_value(actual).unwrap();
        assert_eq!(body["system"], expected["system"]);
        assert_eq!(body["messages"], input["messages"]);
        assert_eq!(body["metadata"], input["metadata"]);
        assert_eq!(body["provider"], input["provider"]);
        assert_eq!(body["thinking"], input["thinking"]);
    }
}
