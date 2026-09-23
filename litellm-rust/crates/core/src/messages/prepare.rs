use litellm_core_utils::get_llm_provider_logic::{CustomLlmProvider, get_custom_llm_provider};
use litellm_llms::anthropic::common_utils::normalize_messages;
use litellm_llms::anthropic::experimental_pass_through::messages::transformation::{
    anthropic_beta_headers, inject_cache_control, normalize_context_management, normalize_reasoning,
};
use litellm_llms::base_llm::anthropic_messages::transformation::{
    BaseAnthropicMessagesConfig, MessagesAuthStrategy,
};
use litellm_types::llms::anthropic_messages::anthropic_request::AnthropicMessagesRequest;
use serde_json::{Map, Value};

use super::{
    Error,
    common_utils::{has_bearer_auth, has_header, messages_provider_config, string_headers},
};
use crate::messages::types::{MessagesRequest, ProviderMessagesRequest};

pub(super) fn prepare_provider_request(
    request: MessagesRequest<'_>,
) -> Result<ProviderMessagesRequest, Error> {
    let provider_info = get_custom_llm_provider(request.model, request.custom_llm_provider)
        .or_else(|| {
            request
                .custom_llm_provider
                .map(|provider| CustomLlmProvider {
                    model: request.model,
                    custom_llm_provider: provider,
                })
        })
        .ok_or_else(|| {
            Error::InvalidProvider(
                "unable to resolve custom_llm_provider for messages request".to_string(),
            )
        })?;
    let model = provider_info.model.to_string();
    let provider = provider_info.custom_llm_provider;

    let config = messages_provider_config(provider)
        .ok_or_else(|| Error::InvalidProvider(provider.to_string()))?;
    let env_lookup = |key: &str| std::env::var(key).ok();

    let body = if provider == "anthropic" {
        normalize_anthropic_body(request.body, request.messages_features)?
    } else {
        request.body
    };
    let headers = validate_environment(
        config,
        provider,
        request.extra_headers,
        request.api_key,
        &body,
        &env_lookup,
    )?;

    let typed_request: AnthropicMessagesRequest = serde_json::from_value(body).map_err(|err| {
        Error::InvalidRequest(format!("invalid Anthropic messages request: {err}"))
    })?;
    let transformed = config.transform_anthropic_messages_request(AnthropicMessagesRequest {
        model: model.clone(),
        ..typed_request
    })?;
    let body = serde_json::to_value(transformed).map_err(|err| {
        Error::InvalidRequest(format!(
            "failed to serialize Anthropic messages request: {err}"
        ))
    })?;

    let url = config.get_complete_url(request.api_base, &model, &env_lookup)?;

    Ok(ProviderMessagesRequest {
        provider: provider.to_string(),
        model,
        config,
        url,
        body,
        upstream_headers: headers,
        timeout: request.timeout,
    })
}

fn normalize_anthropic_body(
    body: Value,
    capabilities: &litellm_llms::anthropic::experimental_pass_through::messages::transformation::MessagesFeatures,
) -> Result<Value, Error> {
    let Value::Object(fields) = body else {
        return Ok(body);
    };
    if fields.get("max_tokens").is_none_or(Value::is_null) {
        return Err(Error::InvalidRequest(
            "max_tokens is required for Anthropic /v1/messages API".into(),
        ));
    }
    let tools = fields
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let body = Value::Object(
        fields
            .into_iter()
            .map(|(name, value)| -> Result<(String, Value), Error> {
                match (name.as_str(), value) {
                    ("messages", Value::Array(messages)) => {
                        Ok((name, Value::Array(normalize_messages(messages, &tools))))
                    }
                    ("context_management", value) => {
                        Ok((name, normalize_context_management(value)))
                    }
                    ("metadata", Value::Object(metadata)) => {
                        let user_id = metadata.get("user_id").filter(|value| !value.is_null());
                        if user_id.is_some_and(|value| !value.is_string()) {
                            return Err(Error::InvalidRequest(
                                "metadata.user_id must be a string".into(),
                            ));
                        }
                        Ok((
                            name,
                            Value::Object(
                                user_id
                                    .map(|value| ("user_id".into(), value.clone()))
                                    .into_iter()
                                    .collect(),
                            ),
                        ))
                    }
                    ("metadata", value) => Err(Error::InvalidRequest(format!(
                        "metadata must be an object, got {value}"
                    ))),
                    (_, value) => Ok((name, value)),
                }
            })
            .collect::<Result<Map<String, Value>, Error>>()?,
    );
    let body = inject_cache_control(body, capabilities);
    normalize_reasoning(body, capabilities).map_err(Error::InvalidRequest)
}

fn validate_environment(
    config: &dyn BaseAnthropicMessagesConfig,
    provider: &str,
    extra_headers: Option<Map<String, Value>>,
    api_key: Option<&str>,
    body: &Value,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<Vec<(String, String)>, Error> {
    let headers = string_headers(extra_headers)?;
    if provider == "anthropic" {
        return anthropic_environment(headers, api_key, body, env_lookup);
    }

    let mut headers = headers;

    let auth_strategy = config.auth_strategy();
    let already_authorized = has_header(&headers, auth_strategy.header_name())
        || (config.accepts_bearer_auth() && has_bearer_auth(&headers));
    if !already_authorized {
        let api_key = config.resolve_api_key(api_key, env_lookup)?;
        let auth_header = match auth_strategy {
            MessagesAuthStrategy::Bearer => {
                ("authorization".to_string(), format!("Bearer {api_key}"))
            }
            MessagesAuthStrategy::Header(name) => (name.to_string(), api_key),
        };
        headers.push(auth_header);
    }

    for (name, value) in config.default_headers() {
        if !has_header(&headers, name) {
            headers.push((name.to_string(), value.to_string()));
        }
    }

    Ok(headers)
}

fn anthropic_environment(
    headers: Vec<(String, String)>,
    api_key: Option<&str>,
    body: &Value,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<Vec<(String, String)>, Error> {
    let key = api_key
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .or_else(|| env_lookup("ANTHROPIC_API_KEY").filter(|value| !value.trim().is_empty()));
    let existing_authorization = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
        .map(|(_, value)| value.as_str());
    let oauth_header = existing_authorization
        .filter(|value| value.starts_with("Bearer sk-ant-oat"))
        .map(str::to_string)
        .or_else(|| {
            key.as_deref()
                .filter(|value| value.starts_with("sk-ant-oat"))
                .map(|value| format!("Bearer {value}"))
        });
    let headers = if let Some(authorization) = oauth_header {
        headers
            .into_iter()
            .filter(|(name, _)| {
                !name.eq_ignore_ascii_case("authorization")
                    && !name.eq_ignore_ascii_case("x-api-key")
            })
            .chain([
                ("authorization".into(), authorization),
                (
                    "anthropic-dangerous-direct-browser-access".into(),
                    "true".into(),
                ),
                ("anthropic-beta".into(), "oauth-2025-04-20".into()),
            ])
            .collect()
    } else if has_header(&headers, "x-api-key") || has_header(&headers, "authorization") {
        headers
    } else {
        let auth = key
            .map(|key| ("x-api-key".into(), key))
            .or_else(|| {
                env_lookup("ANTHROPIC_AUTH_TOKEN")
                    .filter(|value| !value.trim().is_empty())
                    .map(|token| ("authorization".into(), format!("Bearer {token}")))
            })
            .ok_or(litellm_auth::Error::MissingApiKey {
                provider: "Anthropic",
                environment_variable: "ANTHROPIC_API_KEY",
            })?;
        headers.into_iter().chain([auth]).collect()
    };
    let headers = [
        ("anthropic-version", "2023-06-01"),
        ("content-type", "application/json"),
    ]
    .into_iter()
    .fold(headers, |headers, (name, value)| {
        if has_header(&headers, name) {
            headers
        } else {
            headers
                .into_iter()
                .chain([(name.into(), value.into())])
                .collect()
        }
    });
    Ok(anthropic_beta_headers(headers, body))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn anthropic_messages_require_max_tokens_before_network() {
        let result = normalize_anthropic_body(
            json!({"model":"claude-sonnet-5","messages":[]}),
            &Default::default(),
        );
        assert!(
            matches!(result, Err(Error::InvalidRequest(message)) if message.contains("max_tokens"))
        );
    }

    #[test]
    fn anthropic_oauth_replaces_api_key_and_preserves_client_beta() {
        let headers = anthropic_environment(
            vec![("anthropic-beta".into(), "custom-beta".into())],
            Some("sk-ant-oat-token"),
            &json!({"speed":"fast"}),
            &|_| None,
        )
        .unwrap();

        assert!(
            headers
                .iter()
                .any(|(name, value)| name == "authorization" && value == "Bearer sk-ant-oat-token")
        );
        assert!(!headers.iter().any(|(name, _)| name == "x-api-key"));
        assert!(headers.iter().any(|(name, value)| {
            name == "anthropic-beta"
                && value.contains("custom-beta")
                && value.contains("oauth-2025-04-20")
        }));
    }

    #[test]
    fn anthropic_environment_uses_auth_token_when_api_key_absent() {
        let headers = anthropic_environment(vec![], None, &json!({}), &|name| {
            (name == "ANTHROPIC_AUTH_TOKEN").then(|| "token".into())
        })
        .unwrap();

        assert!(
            headers
                .iter()
                .any(|(name, value)| name == "authorization" && value == "Bearer token")
        );
        assert!(
            headers
                .iter()
                .any(|(name, value)| name == "anthropic-version" && value == "2023-06-01")
        );
    }

    #[test]
    fn anthropic_metadata_keeps_only_api_fields_and_rejects_invalid_user_id() {
        let body = normalize_anthropic_body(
            json!({"max_tokens":16,"metadata":{"user_id":"user","internal":"private"}}),
            &Default::default(),
        )
        .unwrap();
        assert_eq!(body["metadata"], json!({"user_id":"user"}));
        assert!(
            normalize_anthropic_body(
                json!({"max_tokens":16,"metadata":{"user_id":12}}),
                &Default::default()
            )
            .is_err()
        );
    }
}
