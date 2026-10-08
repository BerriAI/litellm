use std::time::Duration;

use litellm_auth::SecretValue;
use litellm_core_utils::{
    dot_notation_indexing::delete_nested_value,
    get_provider_specific_headers::get_provider_specific_headers, settings::Lookup,
};
use litellm_http::request::with_default_headers;
use litellm_llms::{
    base_llm::{
        auth::{Headers, ValidatedEnvironment},
        messages::{context::MessagesTransformContext, transformation::BaseMessagesConfig},
    },
    bedrock::messages::invoke_transformations::anthropic_claude3_transformation::BEDROCK_ANTHROPIC_MESSAGES_CONFIG,
};
use litellm_llms_types::formats::messages::MessagesRequest;
use litellm_secrets::source::SecretSource;

use super::{
    Error, MessagesCall,
    common_utils::{MessagesProvider, messages_provider, string_headers},
    types::{MessagesShaping, invalid_request},
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
    let provider = messages_provider(resolved.provider)
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
        extra_headers,
        provider_specific_header,
        timeout,
        shaping,
        ..
    } = call;
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
        &MessagesTransformContext::new(shaping.capabilities, shaping.settings.drop_params),
    )?;

    let scoped =
        get_provider_specific_headers(provider_specific_header.as_ref(), provider.as_str());
    let forwarded = string_headers(Some(
        extra_headers.into_iter().flatten().chain(scoped).collect(),
    ))?;
    let stream = transformed.params.stream == Some(true);
    let (validated, url) = provider_environment(
        provider,
        config,
        &shaping,
        forwarded,
        api_key.as_deref(),
        api_base.as_deref(),
        &transformed.model,
        stream,
        &env_lookup,
    )?;
    let environment = ValidatedEnvironment {
        headers: config.request_headers(
            with_default_headers(validated.headers, config.default_headers()),
            &transformed,
        ),
        auth: validated.auth,
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

#[allow(clippy::too_many_arguments)]
fn provider_environment(
    provider: MessagesProvider,
    config: &dyn BaseMessagesConfig,
    shaping: &MessagesShaping,
    forwarded: Headers,
    api_key: Option<&str>,
    api_base: Option<&str>,
    model: &str,
    stream: bool,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<(ValidatedEnvironment, String), Error> {
    match provider {
        MessagesProvider::Bedrock => {
            let connection = shaping.bedrock_connection.clone().unwrap_or_default();
            Ok((
                BEDROCK_ANTHROPIC_MESSAGES_CONFIG.connection_environment(
                    forwarded,
                    api_key,
                    model,
                    &connection,
                    shaping.bedrock_request_metadata.as_ref(),
                    env_lookup,
                )?,
                BEDROCK_ANTHROPIC_MESSAGES_CONFIG.connection_url(
                    api_base,
                    model,
                    stream,
                    &connection,
                    env_lookup,
                )?,
            ))
        }
        _ => Ok((
            config.validate_environment(forwarded, api_key, model, env_lookup)?,
            if stream {
                config.complete_stream_url(api_base, model, env_lookup)?
            } else {
                config.get_complete_url(api_base, model, env_lookup)?
            },
        )),
    }
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
    use litellm_llms::{
        base_llm::auth::{AuthScheme, resolve_auth},
        bedrock::{
            messages::connection::BedrockMessagesConnection,
            request_metadata::{BedrockMetadataSource, BedrockRequestMetadataInput},
        },
    };
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

    fn environment_for(
        provider: MessagesProvider,
        shaping: &MessagesShaping,
        forwarded: Vec<(String, String)>,
        api_key: Option<&str>,
        api_base: Option<&str>,
        model: &str,
        stream: bool,
    ) -> Result<(ValidatedEnvironment, String), Error> {
        provider_environment(
            provider,
            provider.config(),
            shaping,
            forwarded,
            api_key,
            api_base,
            model,
            stream,
            &|_: &str| None,
        )
    }

    fn trusted_metadata_input() -> BedrockRequestMetadataInput {
        BedrockRequestMetadataInput {
            allowed_fields: vec!["user_api_key_alias".to_string()],
            sources: vec![BedrockMetadataSource {
                identity: vec![("user_api_key_alias".into(), "prod-key".into())],
                spend_logs: vec![],
            }],
        }
    }

    #[rstest]
    #[case::bedrock(
        MessagesProvider::Bedrock,
        "anthropic.claude-3",
        None,
        None,
        "{\"user_api_key_alias\":\"prod-key\"}"
    )]
    #[case::anthropic(
        MessagesProvider::Anthropic,
        "claude-test",
        Some("sk-test"),
        None,
        "caller-set"
    )]
    #[case::azure_ai(
        MessagesProvider::AzureAi,
        "claude-test",
        Some("sk-test"),
        Some("https://resource.services.ai.azure.com"),
        "caller-set"
    )]
    fn trusted_metadata_headers_are_applied_only_to_bedrock(
        shaping: MessagesShaping,
        #[case] provider: MessagesProvider,
        #[case] model: &str,
        #[case] api_key: Option<&str>,
        #[case] api_base: Option<&str>,
        #[case] expected: &str,
    ) {
        let shaping = MessagesShaping {
            bedrock_request_metadata: Some(trusted_metadata_input()),
            ..shaping
        };
        let (validated, _) = environment_for(
            provider,
            &shaping,
            vec![(
                "x-AMZN-bedrock-Request-METADATA".into(),
                "caller-set".into(),
            )],
            api_key,
            api_base,
            model,
            false,
        )
        .unwrap();
        let metadata_headers: Vec<&str> = validated
            .headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("X-Amzn-Bedrock-Request-Metadata"))
            .map(|(_, value)| value.as_str())
            .collect();
        assert_eq!(metadata_headers, vec![expected]);
    }

    #[rstest]
    #[case::region_only(
        None,
        None,
        "anthropic.claude-3",
        false,
        "https://bedrock-runtime.us-east-2.amazonaws.com/model/override%2Fmodel/invoke"
    )]
    #[case::region_only_stream(
        None,
        None,
        "anthropic.claude-3",
        true,
        "https://bedrock-runtime.us-east-2.amazonaws.com/model/override%2Fmodel/invoke-with-response-stream"
    )]
    #[case::connection_api_base(
        Some("https://projected.test"),
        None,
        "anthropic.claude-3",
        false,
        "https://projected.test/model/override%2Fmodel/invoke"
    )]
    #[case::connection_api_base_stream(
        Some("https://projected.test"),
        None,
        "anthropic.claude-3",
        true,
        "https://projected.test/model/override%2Fmodel/invoke-with-response-stream"
    )]
    #[case::call_api_base_wins(
        Some("https://projected.test"),
        Some("https://caller.test"),
        "anthropic.claude-3",
        false,
        "https://caller.test/model/override%2Fmodel/invoke"
    )]
    #[case::connection_region_beats_model_region(
        None,
        None,
        "us-west-2/anthropic.claude-3",
        false,
        "https://bedrock-runtime.us-east-2.amazonaws.com/model/override%2Fmodel/invoke"
    )]
    fn bedrock_connection_controls_url_workspace_and_signing_scope(
        mut shaping: MessagesShaping,
        #[case] connection_api_base: Option<&str>,
        #[case] call_api_base: Option<&str>,
        #[case] model: &str,
        #[case] stream: bool,
        #[case] expected_url: &str,
    ) {
        shaping.bedrock_connection = Some(BedrockMessagesConnection {
            api_base: connection_api_base.map(str::to_string),
            region: Some("us-east-2".into()),
            model_id: Some("override/model".into()),
            workspace_id: Some("trusted-project".into()),
        });
        let (validated, url) = environment_for(
            MessagesProvider::Bedrock,
            &shaping,
            vec![("Anthropic-Workspace-Id".into(), "caller-project".into())],
            None,
            call_api_base,
            model,
            stream,
        )
        .unwrap();
        assert_eq!(url, expected_url);
        let workspace: Vec<&str> = validated
            .headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-workspace-id"))
            .map(|(_, value)| value.as_str())
            .collect();
        assert_eq!(workspace, vec!["trusted-project"]);
        match &validated.auth {
            AuthScheme::AwsSigV4 { region, .. } => {
                assert_eq!(region, "us-east-2");
            }
            other => panic!("expected SigV4 auth, got {other:?}"),
        }
    }

    #[rstest]
    #[case::non_stream(
        false,
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3/invoke"
    )]
    #[case::stream(
        true,
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3/invoke-with-response-stream"
    )]
    fn default_bedrock_inputs_keep_the_model_derived_connection(
        shaping: MessagesShaping,
        #[case] stream: bool,
        #[case] expected_url: &str,
    ) {
        let (validated, url) = environment_for(
            MessagesProvider::Bedrock,
            &shaping,
            vec![("Anthropic-Workspace-Id".into(), "caller-project".into())],
            None,
            None,
            "us-west-2/anthropic.claude-3",
            stream,
        )
        .unwrap();
        assert_eq!(url, expected_url);
        let workspace: Vec<&str> = validated
            .headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-workspace-id"))
            .map(|(_, value)| value.as_str())
            .collect();
        assert_eq!(workspace, vec!["caller-project"]);
        match &validated.auth {
            AuthScheme::AwsSigV4 { region, .. } => {
                assert_eq!(region, "us-west-2");
            }
            other => panic!("expected SigV4 auth, got {other:?}"),
        }
    }

    #[rstest]
    fn bedrock_prepare_still_fails_at_the_unimplemented_shaping_gate(shaping: MessagesShaping) {
        assert!(matches!(
            prepare(MessagesCall {
                body: body(
                    json!({"model": "bedrock/anthropic.claude-3", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}),
                ),
                api_key: None,
                api_base: None,
                custom_llm_provider: Some("bedrock".into()),
                extra_headers: None,
                provider_specific_header: None,
                timeout: None,
                shaping,
            }),
            Err(Error::Unsupported(_))
        ));
    }
}
