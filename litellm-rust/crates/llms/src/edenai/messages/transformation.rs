use litellm_auth::CredentialPlacement;
use litellm_secret_value::SecretValue;

use litellm_core_utils::settings::resolve_non_empty;
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
        messages::{
            handler::shape_anthropic_messages_request,
            transformation::update_headers_with_anthropic_beta,
        },
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

pub struct EdenAIAnthropicMessagesConfig;
pub const EDENAI_MESSAGES_CONFIG: EdenAIAnthropicMessagesConfig = EdenAIAnthropicMessagesConfig;

impl BaseMessagesConfig for EdenAIAnthropicMessagesConfig {
    fn shape_request(
        &self,
        request: MessagesRequest,
        reasoning_auto_summary: bool,
    ) -> Result<MessagesRequest, Error> {
        shape_anthropic_messages_request(request, reasoning_auto_summary)
    }

    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        _optional_params: &MessagesOptionalParams,
        _params: &LitellmParams,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let key = resolve_non_empty(api_key, env, &[API_KEY_ENV]).ok_or(Error::Auth(
            litellm_auth::Error::MissingApiKey {
                provider: "Eden AI",
                environment_variable: API_KEY_ENV,
            },
        ))?;
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
        let base = resolve_non_empty(api_base, env, &[API_BASE_ENV])
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

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serde_json::json;

    #[rstest]
    #[case::bearer("Authorization", "Bearer caller")]
    #[case::api_key("X-Api-Key", "caller")]
    fn forwarded_credentials_still_require_an_edenai_key(#[case] name: &str, #[case] value: &str) {
        let error = EDENAI_MESSAGES_CONFIG
            .validate_environment(
                vec![(name.into(), value.into())],
                None,
                "test-model",
                &MessagesOptionalParams::default(),
                &LitellmParams::default(),
                &|_| None,
            )
            .unwrap_err();
        assert_eq!(
            error,
            Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "Eden AI",
                environment_variable: API_KEY_ENV,
            })
        );
    }

    #[rstest]
    #[case::explicit(Some("explicit"), "explicit")]
    #[case::missing(None, "environment")]
    #[case::blank(Some(" "), "environment")]
    fn configured_credentials_use_bearer_auth(
        #[case] api_key: Option<&str>,
        #[case] expected: &str,
    ) {
        let headers = vec![("x-routing".into(), "keep".into())];
        let environment = EDENAI_MESSAGES_CONFIG
            .validate_environment(
                headers.clone(),
                api_key,
                "test-model",
                &MessagesOptionalParams::default(),
                &LitellmParams::default(),
                &|name| (name == API_KEY_ENV).then(|| "environment".into()),
            )
            .unwrap();
        assert_eq!(environment.headers, headers);
        assert!(matches!(
            environment.auth,
            AuthScheme::Credential { placement: CredentialPlacement::Bearer, secret }
                if secret.expose() == expected
        ));
    }

    #[rstest]
    #[case::punctuation("functions.lookup:0", "functions_lookup_0")]
    #[case::thought_signature("lookup__thought__signature", "lookup")]
    fn shaping_normalizes_matching_tool_ids_and_preserves_native_parameters(
        #[case] source_id: &str,
        #[case] expected_id: &str,
    ) {
        let input = json!({"id": source_id, "cache_control": {"scope": "application"}});
        let thinking = json!({"type": "enabled", "budget_tokens": 512, "native_field": true});
        let body = json!({
            "model": "test-model", "max_tokens": 64, "thinking": thinking,
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "text", "text": " \n "},
                    {"type": "tool_use", "id": source_id, "name": "lookup", "input": input}
                ]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": source_id, "content": "ok"}]}
            ]
        });
        let shaped = EDENAI_MESSAGES_CONFIG
            .shape_request(serde_json::from_value(body).unwrap(), false)
            .unwrap();
        assert_eq!(
            serde_json::to_value(shaped).unwrap(),
            json!({
                "model": "test-model", "max_tokens": 64, "thinking": thinking,
                "messages": [
                    {"role": "assistant", "content": [{"type": "tool_use", "id": expected_id, "name": "lookup", "input": input}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": expected_id, "content": "ok"}]}
                ]
            })
        );
    }

    #[rstest]
    fn cache_sanitization_preserves_tool_application_data_and_extensions() {
        let payload = json!({"cache_control": {"type": "application", "ttl": "keep"}});
        let wire = EDENAI_MESSAGES_CONFIG.wire_body(json!({
            "model": "test-model",
            "messages": [{"role": "assistant", "content": [{
                "type": "tool_use", "id": "call-1", "name": "lookup", "input": payload,
                "cache_control": {"type": "ephemeral", "ttl": "1h", "scope": "global"}
            }]}],
            "future": null
        }));
        assert_eq!(
            wire,
            json!({
                "model": "test-model",
                "messages": [{"role": "assistant", "content": [{
                    "type": "tool_use", "id": "call-1", "name": "lookup", "input": payload,
                    "cache_control": {"type": "ephemeral"}
                }]}],
                "future": null
            })
        );
    }
}
