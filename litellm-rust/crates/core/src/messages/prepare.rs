use std::time::Duration;

use litellm_auth::SecretValue;
use litellm_core_utils::{
    dot_notation_indexing::delete_nested_value,
    get_llm_provider_logic::{CustomLlmProvider, get_custom_llm_provider},
    get_provider_specific_headers::get_provider_specific_headers,
    settings::Lookup,
};
use litellm_llms::{
    anthropic::messages::handler::shape_anthropic_messages_request,
    base_llm::{
        anthropic_messages::transformation::MessagesTransformContext,
        auth::{ValidatedEnvironment, with_default_headers},
    },
};
use litellm_secrets::source::SecretSource;
use litellm_types::llms::anthropic_messages::anthropic_request::AnthropicMessagesRequest;

use super::{
    Error, MessagesCall,
    common_utils::{MessagesProvider, string_headers},
};

pub(super) struct ResolvedProvider {
    model: String,
    pub(super) provider: MessagesProvider,
}

pub(super) struct ProviderMessagesRequest {
    pub(super) provider: MessagesProvider,
    pub(super) url: String,
    pub(super) body: AnthropicMessagesRequest,
    pub(super) environment: ValidatedEnvironment,
    pub(super) timeout: Option<Duration>,
    /// The caller's own credential, reported to the host beside the wire request.
    pub(super) api_key: Option<SecretValue>,
}

pub(super) async fn prepare(
    call: MessagesCall,
    secrets: &dyn SecretSource,
) -> Result<ProviderMessagesRequest, Error> {
    let resolved = resolve_provider(&call.body.model, call.custom_llm_provider.as_deref())?;
    let secrets = secrets
        .resolve(resolved.provider.config().secret_names())
        .await?;
    prepare_provider_request(call, resolved, secrets.as_ref())
}

pub(super) fn resolve_provider(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<ResolvedProvider, Error> {
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
    let provider = provider
        .parse()
        .map_err(|_| Error::InvalidProvider(provider.to_string()))?;
    Ok(ResolvedProvider {
        model: model.to_string(),
        provider,
    })
}

fn prepare_provider_request(
    call: MessagesCall,
    resolved: ResolvedProvider,
    secrets: &dyn Lookup,
) -> Result<ProviderMessagesRequest, Error> {
    let ResolvedProvider { model, provider } = resolved;
    let MessagesCall {
        body,
        api_key,
        api_base,
        extra_headers,
        provider_specific_header,
        timeout,
        shaping,
        ..
    } = call;
    let config = provider.config();
    let env_lookup = |key: &str| secrets.get(key);

    let sanitized = shape_anthropic_messages_request(
        AnthropicMessagesRequest { model, ..body },
        shaping.reasoning_auto_summary,
    )?;
    let trimmed = without_additional_drop_params(sanitized, &shaping.additional_drop_params)?;
    let transformed = config.transform_anthropic_messages_request(
        trimmed,
        &MessagesTransformContext::new(shaping.capabilities, shaping.drop_params),
    )?;

    let scoped =
        get_provider_specific_headers(provider_specific_header.as_ref(), provider.as_str());
    let forwarded = string_headers(Some(
        extra_headers.into_iter().flatten().chain(scoped).collect(),
    ))?;
    let validated = config.validate_environment(
        forwarded,
        api_key.as_deref(),
        &transformed.model,
        &env_lookup,
    )?;
    let environment = ValidatedEnvironment {
        headers: config.request_headers(
            with_default_headers(validated.headers, config.default_headers()),
            &transformed,
        ),
        auth: validated.auth,
    };

    let url = if transformed.params.stream == Some(true) {
        config.complete_stream_url(api_base.as_deref(), &transformed.model, &env_lookup)?
    } else {
        config.get_complete_url(api_base.as_deref(), &transformed.model, &env_lookup)?
    };

    Ok(ProviderMessagesRequest {
        provider,
        url,
        body: transformed,
        environment,
        timeout,
        api_key: api_key.map(SecretValue::new),
    })
}



fn without_additional_drop_params(
    request: AnthropicMessagesRequest,
    paths: &[String],
) -> Result<AnthropicMessagesRequest, Error> {
    if paths.is_empty() {
        return Ok(request);
    }
    let params = serde_json::to_value(request.params).map_err(|e| Error::RequestEncoding(e.into()))?;
    let trimmed = paths
        .iter()
        .fold(params, |params, path| delete_nested_value(params, path));
    Ok(AnthropicMessagesRequest {
        params: serde_json::from_value(trimmed).map_err(|e| Error::RequestDecoding(e.into()))?,
        ..request
    })
}

#[cfg(test)]
mod tests {
    use litellm_llms::base_llm::auth::resolve_auth;
    use litellm_types::utils::ProviderSpecificHeaders;
    use rstest::{fixture, rstest};
    use serde_json::{Map, Value, json};

    use super::*;
    use crate::messages::MessagesShaping;

    #[fixture]
    fn shaping() -> MessagesShaping {
        MessagesShaping::default()
    }

    fn body(value: Value) -> AnthropicMessagesRequest {
        serde_json::from_value(value).unwrap()
    }

    fn prepare(call: MessagesCall) -> Result<ProviderMessagesRequest, Error> {
        prepare_with_secrets(call, &|_: &str| None)
    }

    fn prepare_with_secrets(
        call: MessagesCall,
        secrets: &dyn Lookup,
    ) -> Result<ProviderMessagesRequest, Error> {
        let resolved = resolve_provider(&call.body.model, call.custom_llm_provider.as_deref())?;
        prepare_provider_request(call, resolved, secrets)
    }

    /// The headers as they go on the wire, credential applied.
    fn wire_headers(prepared: &ProviderMessagesRequest) -> Vec<(String, String)> {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(resolve_auth(
                &litellm_auth::AuthServices::default(),
                prepared.environment.clone(),
                &|_| None,
            ))
            .unwrap()
            .headers
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
            MessagesCall {
                body: body(
                    json!({"model": "claude-test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}),
                ),
                api_key: None,
                api_base: None,
                custom_llm_provider: Some("anthropic".into()),
                extra_headers: None,
                provider_specific_header: None,
                timeout: None,
                shaping,
            },
            &lookup,
        )
        .unwrap();
        let headers = wire_headers(&prepared);
        let auth: Vec<(&str, &str)> = headers
            .iter()
            .filter(|(name, _)| matches!(name.as_str(), "x-api-key" | "authorization"))
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        assert_eq!(
            (auth.as_slice(), prepared.url.as_str()),
            (expected_auth, expected_url)
        );
    }

    fn prepared_body(fields: Value, shaping: MessagesShaping) -> Result<Value, Error> {
        prepare(MessagesCall {
            body: body(fields),
            api_key: Some("sk-test".into()),
            api_base: Some("https://anthropic.test".into()),
            custom_llm_provider: Some("anthropic".into()),
            extra_headers: None,
            provider_specific_header: None,
            timeout: None,
            shaping,
        })
        .map(|prepared| serde_json::to_value(prepared.body).unwrap())
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
        let prepared = prepare(MessagesCall {
            body: body(
                json!({"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}),
            ),
            api_key: Some("sk-test".into()),
            api_base: Some("https://resource.services.ai.azure.com".into()),
            custom_llm_provider: custom_llm_provider.map(Into::into),
            extra_headers: Some(Map::from_iter([("x-priority".into(), json!("extra"))])),
            provider_specific_header: Some(configured),
            timeout: None,
            shaping,
        })
        .unwrap();
        let caller_headers: Vec<(&str, &str)> = prepared
            .environment
            .headers
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

    #[rstest]
    #[case::streaming(json!({"stream": true}), true)]
    #[case::explicitly_off(json!({"stream": false}), false)]
    #[case::unset(json!({}), false)]
    fn prepared_request_reports_whether_it_streams(
        shaping: MessagesShaping,
        #[case] extra: Value,
        #[case] expected: bool,
    ) {
        let body = [
            ("model", json!("claude-test")),
            ("messages", json!([{"role": "user", "content": "hi"}])),
            ("max_tokens", json!(16)),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .chain(extra.as_object().cloned().unwrap_or_default())
        .collect();
        let prepared = prepare(MessagesCall {
            body: self::body(Value::Object(body)),
            api_key: Some("sk-test".into()),
            api_base: Some("https://anthropic.test".into()),
            custom_llm_provider: Some("anthropic".into()),
            extra_headers: None,
            provider_specific_header: None,
            timeout: None,
            shaping,
        })
        .unwrap();
        assert_eq!(prepared.body.params.stream.unwrap_or(false), expected);
    }

    #[rstest]
    fn untyped_messages_fail_as_request_decoding(shaping: MessagesShaping) {
        assert!(matches!(
            prepared_body(
                json!({"model": "claude-test", "messages": "nope", "max_tokens": 16}),
                shaping,
            ),
            Err(Error::RequestDecoding(_))
        ));
    }
}
