use std::time::Duration;

use litellm_auth::SecretValue;
use litellm_core_utils::{
    dot_notation_indexing::delete_nested_value,
    get_provider_specific_headers::get_provider_specific_headers, settings::Lookup,
};
use litellm_http::request::with_default_headers;
use litellm_llms::anthropic::messages::transformation::is_claude_model;
use litellm_llms::base_llm::{
    auth::ValidatedEnvironment, messages::context::MessagesTransformContext,
};
use litellm_llms_types::formats::messages::MessagesRequest;
use litellm_secrets::source::SecretSource;

use super::{
    Error, MessagesCall,
    common_utils::{MessagesProvider, messages_provider, string_headers},
    types::invalid_request,
};
use litellm_inference::provider::resolve_llm_provider;

struct ResolvedProvider {
    model: String,
    provider: MessagesProvider,
}

pub(super) struct ProviderMessagesRequest {
    pub(super) provider: MessagesProvider,
    pub(super) url: String,
    pub(super) body: MessagesRequest,
    pub(super) environment: ValidatedEnvironment,
    pub(super) timeout: Option<Duration>,
    /// The caller's own credential, reported to the host beside the wire request.
    pub(super) api_key: Option<SecretValue>,
}

#[tracing::instrument(name = "litellm.prepare", level = "debug", skip_all)]
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

fn resolve_provider(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<ResolvedProvider, Error> {
    let resolved = resolve_llm_provider(model, custom_llm_provider, "messages")?;
    let provider = messages_provider(resolved.provider, resolved.model)
        .ok_or_else(|| Error::InvalidProvider(<&str>::from(resolved.provider).to_string()))?;
    Ok(ResolvedProvider {
        model: resolved.model.to_string(),
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
        litellm_params,
        extra_headers,
        provider_specific_header,
        timeout,
        shaping,
        ..
    } = call;
    let connection_params = litellm_router_types::LitellmParams {
        api_base: api_base.clone().or(litellm_params.api_base.clone()),
        ..litellm_params
    };
    let config = provider.config();
    let env_lookup = |key: &str| secrets.get(key);

    let sanitized = config.shape_request(
        MessagesRequest { model, ..body },
        shaping.settings.reasoning_auto_summary,
    )?;
    let trimmed =
        without_additional_drop_params(sanitized, &shaping.settings.additional_drop_params)?;
    let transformed = config.transform_anthropic_messages_request(
        trimmed,
        &MessagesTransformContext {
            claude_base_model: shaping
                .settings
                .base_model
                .as_deref()
                .is_some_and(is_claude_model),
            ..MessagesTransformContext::new(shaping.capabilities, shaping.settings.drop_params)
        },
    )?;

    let scoped =
        get_provider_specific_headers(provider_specific_header.as_ref(), provider.as_str());
    let forwarded = string_headers(Some(
        extra_headers.into_iter().flatten().chain(scoped).collect(),
    ))?;
    let requested = with_default_headers(forwarded, config.default_headers());
    let forwarded = config.request_headers(requested.clone(), &transformed);
    let environment = config.validate_environment(
        forwarded,
        api_key.as_deref(),
        &transformed.model,
        &connection_params,
        &env_lookup,
    )?;

    let url = config.get_complete_url(
        api_base.as_deref(),
        &transformed.model,
        &connection_params,
        transformed.params.stream == Some(true),
        &env_lookup,
    )?;

    let body = config.request_body(transformed, &url, &requested, shaping.settings.drop_params)?;

    Ok(ProviderMessagesRequest {
        provider,
        url,
        body,
        environment,
        timeout: timeout.or_else(|| shaping.settings.request_timeout()),
        api_key: api_key.map(SecretValue::new),
    })
}

fn without_additional_drop_params(
    request: MessagesRequest,
    paths: &[String],
) -> Result<MessagesRequest, Error> {
    if paths.is_empty() {
        return Ok(request);
    }
    let params = serde_json::to_value(request.params).map_err(invalid_request)?;
    let trimmed = paths
        .iter()
        .fold(params, |params, path| delete_nested_value(params, path));
    Ok(MessagesRequest {
        params: serde_json::from_value(trimmed).map_err(invalid_request)?,
        ..request
    })
}

#[cfg(test)]
mod tests {
    use litellm_llms::base_llm::auth::resolve_auth;
    use litellm_llms_types::headers::ProviderSpecificHeaders;
    use rstest::{fixture, rstest};
    use serde_json::{Map, Value, json};

    use super::*;
    use crate::{MessagesSettings, MessagesShaping};

    #[fixture]
    fn shaping() -> MessagesShaping {
        MessagesShaping::default()
    }

    fn body(value: Value) -> MessagesRequest {
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
                litellm_params: Default::default(),
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

    #[rstest]
    #[case::nothing_configured(None, None, None)]
    #[case::configured_setting(None, Some(1800.0), Some(Duration::from_secs(1800)))]
    #[case::call_timeout_wins(
        Some(Duration::from_secs(30)),
        Some(1800.0),
        Some(Duration::from_secs(30))
    )]
    #[case::call_timeout_only(Some(Duration::from_secs(30)), None, Some(Duration::from_secs(30)))]
    #[case::unusable_setting_ignored(None, Some(-1.0), None)]
    #[case::zero_setting_ignored(None, Some(0.0), None)]
    fn timeout_falls_back_to_the_configured_request_timeout(
        shaping: MessagesShaping,
        #[case] call_timeout: Option<Duration>,
        #[case] request_timeout: Option<f64>,
        #[case] expected: Option<Duration>,
    ) {
        let prepared = prepare(MessagesCall {
            body: body(json!({"model": "claude-test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16})),
            api_key: Some("sk-test".into()),
            api_base: Some("https://anthropic.test".into()),
            custom_llm_provider: Some("anthropic".into()),
            litellm_params: Default::default(),
            extra_headers: None,
            provider_specific_header: None,
            timeout: call_timeout,
            shaping: MessagesShaping {
                settings: MessagesSettings { request_timeout, ..shaping.settings },
                ..shaping
            },
        })
        .unwrap();
        assert_eq!(prepared.timeout, expected);
    }

    fn prepared_body(fields: Value, shaping: MessagesShaping) -> Result<Value, Error> {
        prepare(MessagesCall {
            body: body(fields),
            api_key: Some("sk-test".into()),
            api_base: Some("https://anthropic.test".into()),
            custom_llm_provider: Some("anthropic".into()),
            litellm_params: Default::default(),
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
            settings: MessagesSettings {
                additional_drop_params: additional_drop_params
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
                ..shaping.settings
            },
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
            litellm_params: Default::default(),
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
            settings: MessagesSettings {
                reasoning_auto_summary: true,
                additional_drop_params: vec!["thinking.display".to_string()],
                ..shaping.settings
            },
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
            settings: MessagesSettings {
                additional_drop_params: vec!["metadata.user_id".to_string()],
                ..shaping.settings
            },
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
                litellm_llms::ErrorDetail::InvalidValue {
                    field: "metadata.user_id",
                    expected: "a string",
                    actual: json!(123),
                }
            ))
        );
    }
    #[rstest]
    #[case::mantle(
        "bedrock_mantle/claude-test",
        "https://bedrock-mantle.us-west-2.api.aws",
        "/anthropic/v1/messages",
        "bedrock"
    )]
    #[case::platform(
        "bedrock/claude_platform/claude-test",
        "https://aws-external-anthropic.us-west-2.api.aws",
        "/v1/messages",
        "aws-external-anthropic"
    )]
    fn explicit_endpoint_region_is_used_by_the_signer(
        shaping: MessagesShaping,
        #[case] model: &str,
        #[case] base: &str,
        #[case] path: &str,
        #[case] service: &str,
    ) {
        let prepared = prepare_with_secrets(MessagesCall {
            body: body(json!({"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]})),
            api_key: None,
            api_base: Some(base.into()),
            custom_llm_provider: None,
            litellm_params: Default::default(),
            extra_headers: None,
            provider_specific_header: None,
            timeout: None,
            shaping,
        }, &|name: &str| match name {
            "AWS_REGION_NAME" => Some("us-east-1".into()),
            "ANTHROPIC_AWS_WORKSPACE_ID" => Some("workspace".into()),
            _ => None,
        }).unwrap();
        assert_eq!(prepared.url, format!("{base}{path}"));
        assert!(
            matches!(prepared.environment.auth, litellm_llms::base_llm::auth::AuthScheme::AwsSigV4 { region, service: actual, .. } if region == "us-west-2" && actual == service)
        );
    }

    type Upstream<'a> = (Option<&'a str>, &'a [(&'a str, &'a str)]);

    const ANTHROPIC_BY_DEFAULT: Upstream<'static> = (None, &[]);
    const ANTHROPIC_API_BASE: Upstream<'static> = (Some("https://api.anthropic.com"), &[]);
    const ANTHROPIC_BASE_URL_ENV: Upstream<'static> = (
        None,
        &[("ANTHROPIC_BASE_URL", "https://api.anthropic.com/")],
    );

    fn versioned_body(
        version: Option<&str>,
        (api_base, env): Upstream<'_>,
        drop_params: bool,
        shaping: MessagesShaping,
    ) -> Result<Value, Error> {
        let lookup = |name: &str| {
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        };
        prepare_with_secrets(
            MessagesCall {
                body: body(json!({
                    "model": "claude-test",
                    "max_tokens": 16,
                    "temperature": 0.2,
                    "messages": [{"role": "user", "content": "hi"}]
                })),
                api_key: Some("sk-test".into()),
                api_base: api_base.map(Into::into),
                custom_llm_provider: Some("anthropic".into()),
                litellm_params: Default::default(),
                extra_headers: version
                    .map(|version| Map::from_iter([("anthropic-version".into(), json!(version))])),
                provider_specific_header: None,
                timeout: None,
                shaping: MessagesShaping {
                    settings: MessagesSettings {
                        drop_params,
                        ..shaping.settings
                    },
                    ..shaping
                },
            },
            &lookup,
        )
        .map(|prepared| serde_json::to_value(prepared.body).unwrap())
    }

    #[rstest]
    #[case::caller_on_2023_06_01(Some("2023-06-01"), false, Some(json!(0.2)))]
    #[case::default_version(None, false, Some(json!(0.2)))]
    #[case::caller_on_2026_10_21_under_drop_params(Some("2026-10-21"), true, None)]
    fn temperature_reaches_anthropic_only_under_an_api_version_that_has_it(
        shaping: MessagesShaping,
        #[case] version: Option<&str>,
        #[case] drop_params: bool,
        #[case] expected: Option<Value>,
        #[values(ANTHROPIC_BY_DEFAULT, ANTHROPIC_API_BASE, ANTHROPIC_BASE_URL_ENV)]
        upstream: Upstream<'_>,
    ) {
        let prepared = versioned_body(version, upstream, drop_params, shaping).unwrap();
        assert_eq!(prepared.get("temperature").cloned(), expected);
    }

    #[rstest]
    fn an_api_version_without_sampling_params_refuses_them_before_the_anthropic_call(
        shaping: MessagesShaping,
        #[values(ANTHROPIC_BY_DEFAULT, ANTHROPIC_API_BASE, ANTHROPIC_BASE_URL_ENV)]
        upstream: Upstream<'_>,
    ) {
        let Err(Error::InvalidRequest(detail)) =
            versioned_body(Some("2026-10-21"), upstream, false, shaping)
        else {
            panic!("expected an invalid request");
        };
        assert_eq!(
            detail.to_string(),
            "claude-test does not support temperature=0.2. anthropic-version 2026-10-21 removed temperature, top_p and top_k. To drop unsupported params, set `litellm.drop_params = True`."
        );
    }

    #[rstest]
    #[case::deepseek_api_base(Some("https://api.deepseek.com/anthropic"), &[])]
    #[case::minimax_api_base(Some("https://api.minimax.io/anthropic"), &[])]
    #[case::api_base_env(None, &[("ANTHROPIC_API_BASE", "https://api.deepseek.com/anthropic")])]
    #[case::sdk_base_url_env(None, &[("ANTHROPIC_BASE_URL", "https://api.minimax.io/anthropic")])]
    fn a_third_party_messages_api_gets_sampling_params_under_anthropic_version_2026_10_21(
        shaping: MessagesShaping,
        #[case] api_base: Option<&str>,
        #[case] env: &[(&str, &str)],
        #[values(false, true)] drop_params: bool,
    ) {
        let prepared =
            versioned_body(Some("2026-10-21"), (api_base, env), drop_params, shaping).unwrap();
        assert_eq!(prepared.get("temperature"), Some(&json!(0.2)));
    }

    fn aws_route_body(
        (model, base): (&str, &str),
        version: &str,
        drop_params: bool,
        shaping: MessagesShaping,
    ) -> Result<Value, Error> {
        prepare_with_secrets(
            MessagesCall {
                body: body(json!({
                    "model": model,
                    "max_tokens": 16,
                    "temperature": 0.2,
                    "messages": [{"role": "user", "content": "hi"}]
                })),
                api_key: None,
                api_base: Some(base.into()),
                custom_llm_provider: None,
                litellm_params: Default::default(),
                extra_headers: Some(Map::from_iter([(
                    "anthropic-version".into(),
                    json!(version),
                )])),
                provider_specific_header: None,
                timeout: None,
                shaping: MessagesShaping {
                    settings: MessagesSettings {
                        drop_params,
                        ..shaping.settings
                    },
                    ..shaping
                },
            },
            &|name: &str| match name {
                "AWS_REGION_NAME" => Some("us-west-2".into()),
                "ANTHROPIC_AWS_WORKSPACE_ID" => Some("workspace".into()),
                _ => None,
            },
        )
        .map(|prepared| serde_json::to_value(prepared.body).unwrap())
    }

    const MANTLE: (&str, &str) = (
        "bedrock_mantle/claude-test",
        "https://bedrock-mantle.us-west-2.api.aws",
    );
    const CLAUDE_PLATFORM: (&str, &str) = (
        "bedrock/claude_platform/claude-test",
        "https://aws-external-anthropic.us-west-2.api.aws",
    );

    #[rstest]
    #[case::caller_on_2023_06_01("2023-06-01", false, Some(json!(0.2)))]
    #[case::caller_on_2026_10_21_under_drop_params("2026-10-21", true, None)]
    fn temperature_reaches_anthropic_on_aws_only_under_an_api_version_that_has_it(
        shaping: MessagesShaping,
        #[case] version: &str,
        #[case] drop_params: bool,
        #[case] expected: Option<Value>,
        #[values(MANTLE, CLAUDE_PLATFORM)] route: (&str, &str),
    ) {
        let prepared = aws_route_body(route, version, drop_params, shaping).unwrap();
        assert_eq!(prepared.get("temperature").cloned(), expected);
    }

    #[rstest]
    fn anthropic_on_aws_refuses_sampling_params_under_anthropic_version_2026_10_21(
        shaping: MessagesShaping,
        #[values(MANTLE, CLAUDE_PLATFORM)] route: (&str, &str),
    ) {
        let Err(Error::InvalidRequest(detail)) =
            aws_route_body(route, "2026-10-21", false, shaping)
        else {
            panic!("expected an invalid request");
        };
        assert_eq!(
            detail.to_string(),
            "claude-test does not support temperature=0.2. anthropic-version 2026-10-21 removed temperature, top_p and top_k. To drop unsupported params, set `litellm.drop_params = True`."
        );
    }

    fn vertex_call_requesting(
        shaping: MessagesShaping,
        requested: &str,
        drop_params: bool,
    ) -> MessagesCall {
        MessagesCall {
            body: body(
                json!({"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}),
            ),
            api_key: None,
            api_base: Some("https://vertex.test".into()),
            custom_llm_provider: Some("vertex_ai".into()),
            litellm_params: Default::default(),
            extra_headers: Some(Map::from_iter([
                ("authorization".to_string(), json!("Bearer vertex-token")),
                ("anthropic-version".to_string(), json!(requested)),
            ])),
            provider_specific_header: None,
            timeout: None,
            shaping: MessagesShaping {
                settings: MessagesSettings {
                    drop_params,
                    ..shaping.settings
                },
                ..shaping
            },
        }
    }

    fn vertex_project(name: &str) -> Option<String> {
        (name == "VERTEXAI_PROJECT").then(|| "version-project".to_string())
    }

    #[rstest]
    #[case::sdk_2("2026-10-21", false, "2026-10-21")]
    #[case::sdk_1("2023-06-01", false, "vertex-2023-10-16")]
    #[case::unknown_under_drop_params("latest", true, "vertex-2023-10-16")]
    fn vertex_gets_the_requested_version_in_the_body_alone(
        shaping: MessagesShaping,
        #[case] requested: &str,
        #[case] drop_params: bool,
        #[case] body_version: &str,
    ) {
        let prepared = prepare_with_secrets(
            vertex_call_requesting(shaping, requested, drop_params),
            &vertex_project,
        )
        .unwrap();
        let version_headers: Vec<String> = wire_headers(&prepared)
            .into_iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-version"))
            .map(|(_, value)| value)
            .collect();
        let body = serde_json::to_value(prepared.body).unwrap();
        assert_eq!(
            (version_headers, &body["anthropic_version"]),
            (Vec::<String>::new(), &json!(body_version))
        );
    }

    #[rstest]
    fn vertex_refuses_an_unsupported_version_without_drop_params(shaping: MessagesShaping) {
        assert!(matches!(
            prepare_with_secrets(
                vertex_call_requesting(shaping, "latest", false),
                &vertex_project
            ),
            Err(Error::InvalidRequest(
                litellm_llms::ErrorDetail::UnsupportedParameter { .. }
            ))
        ));
    }
}
