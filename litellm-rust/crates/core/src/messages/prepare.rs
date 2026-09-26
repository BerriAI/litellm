use litellm_core_utils::{
    dot_notation_indexing::delete_nested_value,
    get_llm_provider_logic::{CustomLlmProvider, get_custom_llm_provider},
    get_provider_specific_headers::get_provider_specific_headers,
    settings::Lookup,
};
use litellm_llms::{
    anthropic::messages::handler::shape_anthropic_messages_request,
    base_llm::anthropic_messages::transformation::{
        BaseAnthropicMessagesConfig, MessagesTransformContext,
    },
};
use litellm_types::llms::anthropic_messages::anthropic_request::AnthropicMessagesRequest;
use serde_json::{Map, Value};

use super::{
    Error,
    common_utils::{messages_provider_config, string_headers},
};
use crate::messages::types::{MessagesRequest, ProviderMessagesRequest};

pub(super) struct ResolvedProvider<'a> {
    pub(super) model: &'a str,
    pub(super) provider: &'a str,
    pub(super) config: &'static dyn BaseAnthropicMessagesConfig,
}

pub(super) fn resolve_provider<'a>(
    model: &'a str,
    custom_llm_provider: Option<&'a str>,
) -> Result<ResolvedProvider<'a>, Error> {
    let CustomLlmProvider {
        model,
        custom_llm_provider: provider,
    } = get_custom_llm_provider(model, custom_llm_provider)
        .or_else(|| {
            custom_llm_provider.map(|provider| CustomLlmProvider {
                model,
                custom_llm_provider: provider,
            })
        })
        .ok_or_else(|| {
            Error::InvalidProvider(
                "unable to resolve custom_llm_provider for messages request".to_string(),
            )
        })?;
    let config = messages_provider_config(provider)
        .ok_or_else(|| Error::InvalidProvider(provider.to_string()))?;
    Ok(ResolvedProvider {
        model,
        provider,
        config,
    })
}

pub(super) fn prepare_provider_request(
    request: MessagesRequest<'_>,
    resolved: ResolvedProvider<'_>,
    secrets: &dyn Lookup,
) -> Result<ProviderMessagesRequest, Error> {
    let ResolvedProvider {
        model,
        provider,
        config,
    } = resolved;
    let model = model.to_string();
    let env_lookup = |key: &str| secrets.get(key);

    let typed_request: AnthropicMessagesRequest =
        serde_json::from_value(request.body).map_err(invalid_request)?;
    let sanitized = shape_anthropic_messages_request(
        AnthropicMessagesRequest {
            model: model.clone(),
            ..typed_request
        },
        request.shaping.reasoning_auto_summary,
    )?;
    let trimmed =
        without_additional_drop_params(sanitized, &request.shaping.additional_drop_params)?;
    let transformed = config.transform_anthropic_messages_request(
        trimmed,
        &MessagesTransformContext::new(request.shaping.capabilities, request.shaping.drop_params),
    )?;

    let scoped = get_provider_specific_headers(request.provider_specific_header.as_ref(), provider);
    let forwarded = string_headers(Some(
        request
            .extra_headers
            .into_iter()
            .flatten()
            .chain(scoped)
            .collect(),
    ))?;
    let authenticated = config.authenticate(forwarded, request.api_key, &env_lookup)?;
    let headers = config.request_headers(
        with_default_headers(authenticated, config.default_headers()),
        &transformed,
    );

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

fn invalid_request(err: serde_json::Error) -> Error {
    Error::InvalidRequest(format!("invalid Anthropic messages request: {err}"))
}

fn without_additional_drop_params(
    request: AnthropicMessagesRequest,
    paths: &[String],
) -> Result<AnthropicMessagesRequest, Error> {
    if paths.is_empty() {
        return Ok(request);
    }
    let Value::Object(fields) = serde_json::to_value(request).map_err(invalid_request)? else {
        return Err(Error::InvalidRequest(
            "Anthropic messages request did not serialize to an object".to_string(),
        ));
    };
    let (required, optional): (Map<String, Value>, Map<String, Value>) = fields
        .into_iter()
        .partition(|(key, _)| matches!(key.as_str(), "model" | "messages"));
    let trimmed = paths.iter().fold(Value::Object(optional), |body, path| {
        delete_nested_value(body, path)
    });
    let merged: Map<String, Value> = required
        .into_iter()
        .chain(trimmed.as_object().cloned().unwrap_or_default())
        .collect();
    serde_json::from_value(Value::Object(merged)).map_err(invalid_request)
}

fn with_default_headers(
    headers: Vec<(String, String)>,
    defaults: &[(&str, &str)],
) -> Vec<(String, String)> {
    let missing: Vec<(String, String)> = defaults
        .iter()
        .filter(|(name, _)| {
            !headers
                .iter()
                .any(|(header, _)| header.eq_ignore_ascii_case(name))
        })
        .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
        .collect();
    headers.into_iter().chain(missing).collect()
}

#[cfg(test)]
mod tests {
    use litellm_types::utils::ProviderSpecificHeaders;
    use rstest::{fixture, rstest};
    use serde_json::json;

    use super::*;
    use crate::messages::types::MessagesShaping;

    #[fixture]
    fn shaping() -> MessagesShaping {
        MessagesShaping::default()
    }

    fn prepare(request: MessagesRequest<'_>) -> Result<ProviderMessagesRequest, Error> {
        prepare_with_secrets(request, &|_: &str| None)
    }

    fn prepare_with_secrets(
        request: MessagesRequest<'_>,
        secrets: &dyn Lookup,
    ) -> Result<ProviderMessagesRequest, Error> {
        let resolved = resolve_provider(request.model, request.custom_llm_provider)?;
        prepare_provider_request(request, resolved, secrets)
    }

    #[rstest]
    #[case::api_key(
        &[("ANTHROPIC_API_KEY", "sk-secret")],
        &[("x-api-key", "sk-secret")],
        "https://api.anthropic.com/v1/messages"
    )]
    #[case::auth_token(
        &[("ANTHROPIC_AUTH_TOKEN", "token")],
        &[("authorization", "Bearer token")],
        "https://api.anthropic.com/v1/messages"
    )]
    #[case::api_base(
        &[("ANTHROPIC_API_KEY", "sk-secret"), ("ANTHROPIC_API_BASE", "https://gateway.test")],
        &[("x-api-key", "sk-secret")],
        "https://gateway.test/v1/messages"
    )]
    #[case::sdk_base_url(
        &[("ANTHROPIC_API_KEY", "sk-secret"), ("ANTHROPIC_BASE_URL", "https://sdk.test")],
        &[("x-api-key", "sk-secret")],
        "https://sdk.test/v1/messages"
    )]
    fn credentials_and_base_come_from_the_resolved_secrets(
        shaping: MessagesShaping,
        #[case] secrets: &[(&str, &str)],
        #[case] expected_auth: &[(&str, &str)],
        #[case] expected_url: &str,
    ) {
        let lookup = |name: &str| {
            secrets
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        };
        let prepared = prepare_with_secrets(
            MessagesRequest {
                model: "claude-test",
                body: json!({"model": "claude-test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}),
                api_key: None,
                api_base: None,
                custom_llm_provider: Some("anthropic"),
                extra_headers: None,
                provider_specific_header: None,
                timeout: None,
                shaping,
            },
            &lookup,
        )
        .unwrap();
        let auth: Vec<(&str, &str)> = prepared
            .upstream_headers
            .iter()
            .filter(|(name, _)| matches!(name.as_str(), "x-api-key" | "authorization"))
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        assert_eq!(
            (auth.as_slice(), prepared.url.as_str()),
            (expected_auth, expected_url)
        );
    }

    fn prepared_body(body: Value, shaping: MessagesShaping) -> Result<Value, Error> {
        prepare(MessagesRequest {
            model: "anthropic/claude-test",
            body,
            api_key: Some("sk-test"),
            api_base: Some("https://anthropic.test"),
            custom_llm_provider: Some("anthropic"),
            extra_headers: None,
            provider_specific_header: None,
            timeout: None,
            shaping,
        })
        .map(|prepared| prepared.body)
    }

    #[rstest]
    #[case::nothing_forwarded(
        &[],
        &[("x-version", "1"), ("content-type", "application/json")],
        &[("x-version", "1"), ("content-type", "application/json")],
    )]
    #[case::forwarded_header_wins_in_any_case(
        &[("X-Version", "custom"), ("x-api-key", "k")],
        &[("x-version", "1"), ("content-type", "application/json")],
        &[("X-Version", "custom"), ("x-api-key", "k"), ("content-type", "application/json")],
    )]
    #[case::no_defaults(&[("x-api-key", "k")], &[], &[("x-api-key", "k")])]
    fn default_headers_fill_only_missing_names(
        #[case] forwarded: &[(&str, &str)],
        #[case] defaults: &[(&str, &str)],
        #[case] expected: &[(&str, &str)],
    ) {
        let owned = |headers: &[(&str, &str)]| -> Vec<(String, String)> {
            headers
                .iter()
                .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
                .collect()
        };
        assert_eq!(
            with_default_headers(owned(forwarded), defaults),
            owned(expected)
        );
    }

    #[rstest]
    #[case::top_level_and_nested_paths(
        json!({
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "context_management": {"edits": [{"type": "clear_thinking_20251015"}]},
            "metadata": {"user_id": "u1"},
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}, "input_examples": [{"q": "x"}]}]
        }),
        &["thinking", "context_management", "tools[*].input_examples"],
        json!({
            "max_tokens": 1024,
            "metadata": {"user_id": "u1"},
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}]
        }),
    )]
    #[case::no_paths(
        json!({"max_tokens": 16, "safeguards": [{"type": "dangerous_tool_use"}]}),
        &[],
        json!({"max_tokens": 16, "safeguards": [{"type": "dangerous_tool_use"}]}),
    )]
    #[case::model_and_messages_are_never_dropped(
        json!({"max_tokens": 16}),
        &["model", "messages", "messages[0].content"],
        json!({"max_tokens": 16}),
    )]
    fn prepared_body_drops_configured_paths(
        shaping: MessagesShaping,
        #[case] fields: Value,
        #[case] additional_drop_params: &[&str],
        #[case] expected_fields: Value,
    ) {
        let with_messages = |fields: Value| -> Value {
            let Value::Object(fields) = fields else {
                unreachable!()
            };
            Value::Object(
                [
                    ("model".to_string(), json!("claude-test")),
                    (
                        "messages".to_string(),
                        json!([{"role": "user", "content": "hi"}]),
                    ),
                ]
                .into_iter()
                .chain(fields)
                .collect(),
            )
        };
        let shaping = MessagesShaping {
            additional_drop_params: additional_drop_params
                .iter()
                .map(ToString::to_string)
                .collect(),
            ..shaping
        };
        assert_eq!(
            prepared_body(with_messages(fields), shaping),
            Ok(with_messages(expected_fields))
        );
    }

    #[rstest]
    #[case::model_prefix_picks_the_provider(
        "azure_ai/claude-test",
        None,
        &[("x-priority", "extra"), ("x-scoped", "azure_ai")]
    )]
    #[case::explicit_provider(
        "claude-test",
        Some("anthropic"),
        &[("x-priority", "scoped"), ("x-scoped", "anthropic")]
    )]
    #[case::provider_prefix_on_an_anthropic_model(
        "anthropic/claude-test",
        None,
        &[("x-priority", "scoped"), ("x-scoped", "anthropic")]
    )]
    fn provider_specific_headers_follow_the_resolved_provider(
        shaping: MessagesShaping,
        #[case] model: &str,
        #[case] custom_llm_provider: Option<&str>,
        #[case] expected: &[(&str, &str)],
    ) {
        let configured: ProviderSpecificHeaders = serde_json::from_value(json!([
            {"custom_llm_provider": "azure_ai", "extra_headers": {"x-scoped": "azure_ai"}},
            {"custom_llm_provider": "anthropic", "extra_headers": {"x-scoped": "anthropic", "x-priority": "scoped"}}
        ]))
        .unwrap();
        let prepared = prepare(MessagesRequest {
            model,
            body: json!({"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}),
            api_key: Some("sk-test"),
            api_base: Some("https://resource.services.ai.azure.com"),
            custom_llm_provider,
            extra_headers: Some(serde_json::from_value(json!({"x-priority": "extra"})).unwrap()),
            provider_specific_header: Some(configured),
            timeout: None,
            shaping,
        })
        .unwrap();
        let caller_headers: Vec<(&str, &str)> = prepared
            .upstream_headers
            .iter()
            .filter(|(name, _)| matches!(name.as_str(), "x-priority" | "x-scoped"))
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        assert_eq!(caller_headers, expected);
    }

    #[rstest]
    fn prepared_body_carries_the_provider_stripped_model(shaping: MessagesShaping) {
        assert_eq!(
            prepared_body(
                json!({
                    "model": "anthropic/claude-test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16
                }),
                shaping,
            ),
            Ok(json!({
                "model": "claude-test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16
            }))
        );
    }

    #[rstest]
    fn dropped_thinking_display_is_not_restored_by_auto_summary(shaping: MessagesShaping) {
        let shaping = MessagesShaping {
            reasoning_auto_summary: true,
            additional_drop_params: vec!["thinking.display".to_string()],
            ..shaping
        };
        assert_eq!(
            prepared_body(
                json!({
                    "model": "claude-test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 4096,
                    "thinking": {"type": "enabled", "budget_tokens": 2048}
                }),
                shaping,
            ),
            Ok(json!({
                "model": "claude-test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4096,
                "thinking": {"type": "enabled", "budget_tokens": 2048}
            }))
        );
    }

    #[rstest]
    fn dropping_an_invalid_metadata_user_id_does_not_skip_its_validation(shaping: MessagesShaping) {
        let shaping = MessagesShaping {
            additional_drop_params: vec!["metadata.user_id".to_string()],
            ..shaping
        };
        assert!(matches!(
            prepared_body(
                json!({
                    "model": "claude-test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                    "metadata": {"user_id": 123}
                }),
                shaping,
            ),
            Err(Error::InvalidRequest(_))
        ));
    }

    #[rstest]
    fn prepared_body_rejects_invalid_metadata_before_the_call(shaping: MessagesShaping) {
        assert_eq!(
            prepared_body(
                json!({
                    "model": "claude-test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                    "metadata": {"user_id": 123}
                }),
                shaping,
            ),
            Err(Error::InvalidRequest(
                "metadata.user_id must be a string, got 123".to_string()
            ))
        );
    }
}
