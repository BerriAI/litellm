use litellm_auth::SecretValue;
use litellm_core_utils::get_llm_provider_logic::{CustomLlmProvider, get_custom_llm_provider};
use litellm_llms::base_llm::{
    auth::{ValidatedEnvironment, with_default_headers},
    chat::transformation::BaseConfig,
};
use litellm_types::llms::openai::ChatMessage;
use serde_json::Value;

use super::{
    Error,
    common_utils::{chat_completions_provider_config, string_headers},
};
use crate::chat_completions::types::{
    ChatCompletionsRequest, ProviderChatCompletionsRequest, ResolvedChatCompletionsRequest,
};

pub(super) struct ResolvedProvider {
    pub(super) model: String,
    pub(super) custom_llm_provider: String,
    pub(super) config: &'static dyn BaseConfig,
}

pub(super) fn resolve_provider_config<'a>(
    model: &'a str,
    custom_llm_provider: Option<&'a str>,
) -> Result<ResolvedProvider, Error> {
    let provider_info = get_custom_llm_provider(model, custom_llm_provider)
        .or_else(|| {
            custom_llm_provider.map(|provider| CustomLlmProvider {
                model,
                custom_llm_provider: provider,
            })
        })
        .ok_or_else(|| {
            Error::InvalidProvider(
                "unable to resolve custom_llm_provider for chat completions request".to_string(),
            )
        })?;
    let config = chat_completions_provider_config(provider_info.custom_llm_provider)
        .ok_or_else(|| Error::InvalidProvider(provider_info.custom_llm_provider.to_string()))?;
    Ok(ResolvedProvider {
        model: provider_info.model.to_string(),
        custom_llm_provider: provider_info.custom_llm_provider.to_string(),
        config,
    })
}

pub(super) fn parse_messages(messages: Value) -> Result<Vec<ChatMessage>, Error> {
    serde_json::from_value(messages)
        .map_err(|err| Error::InvalidRequest(format!("invalid chat completions messages: {err}")))
}

pub(super) fn resolve_request(
    request: ChatCompletionsRequest<'_>,
) -> Result<ResolvedChatCompletionsRequest<'_>, Error> {
    let ResolvedProvider {
        model,
        custom_llm_provider,
        config,
    } = resolve_provider_config(request.model, request.custom_llm_provider)?;
    let messages = parse_messages(request.messages)?;
    if messages.is_empty() {
        return Err(Error::InvalidRequest(
            "chat completions requires at least one message".to_string(),
        ));
    }
    if let Some(reason) = config.unsupported_reason(&messages, &request.optional_params) {
        return Err(Error::Unsupported(reason.0));
    }
    Ok(ResolvedChatCompletionsRequest {
        model,
        custom_llm_provider,
        config,
        messages,
        optional_params: request.optional_params,
        api_key: request.api_key,
        api_base: request.api_base,
        extra_headers: request.extra_headers,
        timeout: request.timeout,
    })
}

fn validate_environment(
    request: &ResolvedChatCompletionsRequest<'_>,
    model: &str,
    config: &dyn BaseConfig,
) -> Result<ValidatedEnvironment, Error> {
    let env_lookup = |key: &str| std::env::var(key).ok();
    let forwarded = string_headers(request.extra_headers.clone())?;
    let validated = config.validate_environment(
        forwarded,
        request.api_key,
        model,
        &request.optional_params,
        &env_lookup,
    )?;
    Ok(ValidatedEnvironment {
        headers: with_default_headers(validated.headers, config.default_headers()),
        auth: validated.auth,
    })
}

pub(super) fn prepare_provider_request(
    request: ResolvedChatCompletionsRequest<'_>,
) -> Result<ProviderChatCompletionsRequest, Error> {
    let environment = validate_environment(&request, &request.model, request.config)?;
    let model = request.model;
    let config = request.config;
    let env_lookup = |key: &str| std::env::var(key).ok();
    let url = config.get_complete_url(
        request.api_base,
        &model,
        &request.optional_params,
        &env_lookup,
    )?;
    let transformed =
        config.transform_request(&model, request.messages, request.optional_params.clone())?;

    Ok(ProviderChatCompletionsRequest {
        model,
        custom_llm_provider: request.custom_llm_provider,
        config,
        url,
        body: transformed.body,
        optional_params: request.optional_params,
        environment,
        timeout: request.timeout,
        api_key: request.api_key.map(|key| SecretValue::new(key.to_string())),
    })
}

#[cfg(test)]
mod tests {
    use litellm_auth::CredentialPlacement;
    use litellm_llms::base_llm::auth::{AuthScheme, resolve_auth};
    use serde_json::{Map, Value, json};

    use super::{prepare_provider_request, resolve_request};
    use crate::chat_completions::{
        Error,
        types::{ChatCompletionsRequest, ProviderChatCompletionsRequest},
    };

    fn prepare_chat_completions_call(
        request: ChatCompletionsRequest<'_>,
    ) -> Result<ProviderChatCompletionsRequest, Error> {
        prepare_provider_request(resolve_request(request)?)
    }

    /// The headers as they go on the wire, credential applied.
    fn wire_headers(prepared: &ProviderChatCompletionsRequest) -> Vec<(String, String)> {
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

    fn request<'a>(
        model: &'a str,
        provider: Option<&'a str>,
        messages: Value,
        optional_params: Value,
    ) -> ChatCompletionsRequest<'a> {
        ChatCompletionsRequest {
            model,
            messages,
            optional_params: match optional_params {
                Value::Object(map) => map,
                other => panic!("params must be an object, got {other}"),
            },
            api_key: Some("sk-test"),
            api_base: None,
            custom_llm_provider: provider,
            extra_headers: None,
            timeout: None,
        }
    }

    /// `ProviderChatCompletionsRequest` deliberately has no `Debug` (its headers
    /// carry resolved credentials), so unwrap the failure case by hand.
    fn decline(request: ChatCompletionsRequest<'_>) -> Error {
        match prepare_chat_completions_call(request) {
            Err(error) => error,
            Ok(prepared) => panic!("expected a decline, prepared a call to {}", prepared.url),
        }
    }

    #[test]
    fn resolves_the_provider_from_the_model_prefix() {
        let prepared = prepare_chat_completions_call(request(
            "anthropic/claude-sonnet-4-5",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({"max_tokens": 16}),
        ))
        .expect("prepares");
        assert_eq!(prepared.model, "claude-sonnet-4-5");
        assert_eq!(prepared.url, "https://api.anthropic.com/v1/messages");
        assert_eq!(prepared.body["model"], json!("claude-sonnet-4-5"));
    }

    #[test]
    fn strips_an_explicit_provider_prefix_from_the_model() {
        let prepared = prepare_chat_completions_call(request(
            "anthropic/claude-sonnet-4-5",
            Some("anthropic"),
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        ))
        .expect("prepares");
        assert_eq!(prepared.model, "claude-sonnet-4-5");
    }

    #[test]
    fn adds_the_auth_and_default_headers() {
        let prepared = prepare_chat_completions_call(request(
            "claude-sonnet-4-5",
            Some("anthropic"),
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        ))
        .expect("prepares");
        assert!(
            wire_headers(&prepared).contains(&("x-api-key".to_string(), "sk-test".to_string()))
        );
        assert!(
            wire_headers(&prepared)
                .contains(&("anthropic-version".to_string(), "2023-06-01".to_string()))
        );
        assert!(matches!(
            prepared.environment.auth,
            AuthScheme::Credential {
                placement: CredentialPlacement::Header("x-api-key"),
                ..
            }
        ));
    }

    #[test]
    fn the_deployment_credential_replaces_a_caller_supplied_auth_header() {
        // Python builds `{**headers, **anthropic_headers}`, so the deployment's key
        // overwrites a forwarded one. Honouring the caller's would let whoever sends
        // the request choose the Anthropic principal it bills to.
        let mut call = request(
            "claude-sonnet-4-5",
            Some("anthropic"),
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        );
        call.extra_headers = Some(Map::from_iter([(
            "X-Api-Key".to_string(),
            json!("sk-caller"),
        )]));
        let prepared = prepare_chat_completions_call(call).expect("prepares");
        let headers = wire_headers(&prepared);
        let keys: Vec<_> = headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("x-api-key"))
            .collect();
        assert_eq!(keys.len(), 1, "got {:?}", headers);
        assert_eq!(keys[0].1, "sk-test");
    }

    #[test]
    fn a_forwarded_authorization_header_suppresses_the_resolved_api_key_header() {
        // Anthropic's `validate_environment` pops `x-api-key` and sets `authorization`
        // for an OAuth token, so re-adding the key here would put the credential into
        // a header the host removed on purpose.
        let mut call = request(
            "claude-sonnet-4-5",
            Some("anthropic"),
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        );
        call.extra_headers = Some(Map::from_iter([
            (
                "Authorization".to_string(),
                json!("Bearer sk-ant-oat01-token"),
            ),
            ("X-Api-Key".to_string(), json!("sk-caller")),
        ]));
        let prepared = prepare_chat_completions_call(call).expect("prepares");
        assert!(
            !wire_headers(&prepared)
                .iter()
                .any(|(name, value)| name.eq_ignore_ascii_case("x-api-key") && value == "sk-test"),
            "the resolved key must not be applied over an OAuth bearer, got {:?}",
            wire_headers(&prepared)
        );
        assert!(
            wire_headers(&prepared)
                .iter()
                .any(|(name, value)| name.eq_ignore_ascii_case("authorization")
                    && value == "Bearer sk-ant-oat01-token")
        );
    }

    #[test]
    fn an_unrelated_forwarded_authorization_does_not_defer_the_resolved_key() {
        // Only an OAuth bearer replaces the credential. Python sends the deployment's
        // `x-api-key` alongside any other forwarded `authorization`, so deferring on
        // the mere presence of that header would drop the deployment's auth.
        let mut call = request(
            "claude-sonnet-4-5",
            Some("anthropic"),
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        );
        call.extra_headers = Some(Map::from_iter([
            ("Authorization".to_string(), json!("Bearer unrelated")),
            ("X-Api-Key".to_string(), json!("sk-caller")),
        ]));
        let prepared = prepare_chat_completions_call(call).expect("prepares");
        let headers = wire_headers(&prepared);
        let keys: Vec<_> = headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("x-api-key"))
            .collect();
        assert_eq!(keys.len(), 1, "got {:?}", headers);
        assert_eq!(keys[0].1, "sk-test");
        assert!(
            wire_headers(&prepared)
                .iter()
                .any(|(name, value)| name.eq_ignore_ascii_case("authorization")
                    && value == "Bearer unrelated"),
            "the unrelated authorization must survive, got {:?}",
            wire_headers(&prepared)
        );
    }

    #[test]
    fn declines_an_unsupported_request_before_resolving_credentials() {
        let mut call = request(
            "claude-sonnet-4-5",
            Some("anthropic"),
            json!([{"role": "user", "content": "hi"}]),
            json!({"stream": true}),
        );
        call.api_key = None;
        // No api_key is set and no env is consulted: the gate must run first, so the
        // error is the decline rather than a missing-credential error.
        assert_eq!(decline(call), Error::Unsupported("streaming"));
    }

    #[test]
    fn rejects_an_unknown_provider() {
        assert_eq!(
            decline(request(
                "openai/gpt-4o",
                None,
                json!([{"role": "user", "content": "hi"}]),
                json!({}),
            )),
            Error::InvalidProvider("openai".to_string())
        );
    }

    #[test]
    fn rejects_a_model_with_no_resolvable_provider() {
        assert!(matches!(
            decline(request(
                "claude-sonnet-4-5",
                None,
                json!([{"role": "user", "content": "hi"}]),
                json!({}),
            )),
            Error::InvalidProvider(_)
        ));
    }

    #[test]
    fn rejects_an_empty_or_malformed_message_list() {
        assert_eq!(
            decline(request(
                "anthropic/claude-sonnet-4-5",
                None,
                json!([]),
                json!({}),
            )),
            Error::InvalidRequest("chat completions requires at least one message".to_string())
        );
        assert!(matches!(
            decline(request(
                "anthropic/claude-sonnet-4-5",
                None,
                json!("not a list"),
                json!({}),
            )),
            Error::InvalidRequest(_)
        ));
    }

    #[test]
    fn rejects_non_string_extra_headers() {
        let mut call = request(
            "anthropic/claude-sonnet-4-5",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        );
        call.extra_headers = Some(Map::from_iter([("x-trace".to_string(), json!(7))]));
        assert_eq!(
            decline(call),
            Error::Headers(litellm_http::request::HeaderError {
                context: "chat completions",
                name: "x-trace".to_string(),
                actual: "number",
            })
        );
    }

    #[test]
    fn prepares_a_bedrock_call_without_resolving_credentials() {
        let mut call = request(
            "bedrock/us-east-1/anthropic.claude-v2",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({"maxTokens": 16}),
        );
        call.api_key = None;
        let prepared = prepare_chat_completions_call(call).expect("prepares");
        assert_eq!(
            prepared.url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/converse"
        );
        assert!(matches!(
            &prepared.environment.auth,
            AuthScheme::AwsSigV4 { region, service: "bedrock", .. } if region == "us-east-1"
        ));
        // SigV4 signs the serialized body, so prepare must not have added an
        // Authorization header; the signer does it over the bytes sent.
        assert!(
            !prepared
                .environment
                .headers
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case("authorization"))
        );
        assert_eq!(prepared.body["inferenceConfig"], json!({"maxTokens": 16}));
    }

    #[tokio::test]
    async fn a_forwarded_client_header_does_not_enter_the_bedrock_signature() {
        // Python signs only the AWS header set and reattaches the rest, so a header
        // the caller forwarded rides along without joining the canonical request.
        // Signing it makes Converse 403 on a deployment that works on Python.
        let mut call = request(
            "bedrock/us-east-1/anthropic.claude-v2",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({
                "maxTokens": 16,
                "aws_access_key_id": "AKIDEXAMPLE",
                "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
            }),
        );
        // A key would resolve to a bearer token and never reach the signer.
        call.api_key = None;
        call.extra_headers = Some(Map::from_iter([(
            "x-request-id".to_string(),
            json!("abc-123"),
        )]));
        let prepared = prepare_chat_completions_call(call).expect("prepares");
        let authenticated = resolve_auth(
            &litellm_auth::AuthServices::default(),
            prepared.environment,
            &|_| None,
        )
        .await
        .expect("resolves");
        let signed = crate::chat_completions::handler::outbound_request(
            authenticated,
            prepared.url,
            &prepared.body,
            prepared.timeout,
        )
        .expect("signs");

        let authorization = signed
            .header("authorization")
            .expect("carries an authorization header")
            .to_string();
        assert!(
            authorization.starts_with("AWS4-HMAC-SHA256"),
            "expected a SigV4 signature, got {authorization}"
        );
        assert!(
            !authorization.contains("x-request-id"),
            "forwarded header reached SignedHeaders: {authorization}"
        );
        // It still goes on the wire, it is just not part of the signature.
        assert!(
            signed
                .headers()
                .iter()
                .any(|(name, value)| name == "x-request-id" && value == "abc-123"),
            "forwarded header was dropped instead of reattached"
        );
    }

    #[tokio::test]
    async fn a_forwarded_header_the_signer_computes_declines_to_python() {
        // Reattaching the caller's copy next to the computed one puts the name on
        // the wire twice and Bedrock rejects the pair, so a request carrying one
        // has to go to Python instead of being signed here.
        for forwarded in [
            "Authorization",
            "x-amz-date",
            "x-amz-security-token",
            "Date",
        ] {
            let mut call = request(
                "bedrock/us-east-1/anthropic.claude-v2",
                None,
                json!([{"role": "user", "content": "hi"}]),
                json!({
                    "maxTokens": 16,
                    "aws_access_key_id": "AKIDEXAMPLE",
                    "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
                }),
            );
            call.api_key = None;
            call.extra_headers = Some(Map::from_iter([(forwarded.to_string(), json!("forged"))]));
            let prepared = prepare_chat_completions_call(call).expect("prepares");
            let authenticated = resolve_auth(
                &litellm_auth::AuthServices::default(),
                prepared.environment,
                &|_| None,
            )
            .await
            .expect("resolves");
            let error = crate::chat_completions::handler::outbound_request(
                authenticated,
                prepared.url,
                &prepared.body,
                prepared.timeout,
            )
            .expect_err("{forwarded} should decline instead of being signed");
            assert!(
                matches!(error, Error::Unsupported(_)),
                "{forwarded} declined as {error:?}, which the host would not fall back on"
            );
        }
    }

    #[test]
    fn a_bedrock_deployment_bearer_outranks_a_forwarded_authorization() {
        // `get_request_headers` assigns `headers["Authorization"]` unconditionally
        // once a bearer token resolves, so the deployment's identity wins on
        // Python. Keeping the caller's would authorize and bill the call as a
        // different principal, and only when the deployment carries `rust: true`.
        let mut call = request(
            "bedrock/us-east-1/anthropic.claude-v2",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({"maxTokens": 16}),
        );
        call.extra_headers = Some(Map::from_iter([(
            "Authorization".to_string(),
            json!("Bearer caller-supplied"),
        )]));
        let prepared = prepare_chat_completions_call(call).expect("prepares");
        let headers = wire_headers(&prepared);
        let authorizations: Vec<_> = headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("authorization"))
            .map(|(_, value)| value.as_str())
            .collect();
        assert_eq!(
            authorizations,
            vec!["Bearer sk-test"],
            "the deployment token must be the only authorization on the wire"
        );
    }

    #[test]
    fn an_anthropic_forwarded_oauth_bearer_still_outranks_the_resolved_key() {
        // The opposite precedence, and deliberate: Anthropic's own transform
        // honours a forwarded OAuth bearer, so the Bedrock fix above must not be
        // generalized into a rule that the configured key always wins.
        //
        // An OAuth bearer is the whole of that exception. This forwarded a plain
        // `x-api-key` until round 17, which read as the same claim and was not:
        // Python overwrites a forwarded `x-api-key` with the deployment's.
        let mut call = request(
            "claude-sonnet-4-5",
            Some("anthropic"),
            json!([{"role": "user", "content": "hi"}]),
            json!({}),
        );
        call.extra_headers = Some(Map::from_iter([(
            "authorization".to_string(),
            json!("Bearer sk-ant-oat01-forwarded"),
        )]));
        let prepared = prepare_chat_completions_call(call).expect("prepares");
        let headers = wire_headers(&prepared);
        let keys: Vec<_> = headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("x-api-key"))
            .map(|(_, value)| value.as_str())
            .collect();
        assert!(keys.is_empty(), "got {:?}", headers);
        assert!(
            wire_headers(&prepared)
                .iter()
                .any(|(name, value)| name.eq_ignore_ascii_case("authorization")
                    && value == "Bearer sk-ant-oat01-forwarded")
        );
    }

    #[test]
    fn a_bedrock_api_key_is_sent_as_a_bearer_token_instead_of_being_signed() {
        // The configured bearer identity has its own account and quota boundary,
        // so a request carrying one must not be signed as whatever principal the
        // host's AWS credentials resolve to.
        let prepared = prepare_chat_completions_call(request(
            "bedrock/us-east-1/anthropic.claude-v2",
            None,
            json!([{"role": "user", "content": "hi"}]),
            json!({"maxTokens": 16}),
        ))
        .expect("prepares");
        assert!(matches!(
            &prepared.environment.auth,
            AuthScheme::Credential { placement: CredentialPlacement::Bearer, secret }
                if secret.expose() == "sk-test"
        ));
        assert!(
            wire_headers(&prepared)
                .iter()
                .any(|(name, value)| name.eq_ignore_ascii_case("authorization")
                    && value == "Bearer sk-test"),
            "prepare did not carry the bearer token"
        );
    }

    fn decline_reason(
        model: &str,
        provider: Option<&str>,
        messages: Value,
        params: Value,
    ) -> Option<&'static str> {
        let params = match params {
            Value::Object(map) => map,
            other => panic!("params must be an object, got {other}"),
        };
        crate::chat_completions::chat_completions_decline_reason(model, provider, messages, &params)
    }

    #[test]
    fn the_gate_accepts_what_prepare_accepts() {
        assert_eq!(
            decline_reason(
                "anthropic/claude-sonnet-4-5",
                None,
                json!([{"role": "user", "content": "hi"}]),
                json!({"max_tokens": 16}),
            ),
            None
        );
    }

    #[test]
    fn the_gate_declines_without_resolving_credentials_or_calling_out() {
        assert_eq!(
            decline_reason(
                "anthropic/claude-sonnet-4-5",
                None,
                json!([{"role": "user", "content": "hi"}]),
                json!({"stream": true}),
            ),
            Some("streaming")
        );
        assert_eq!(
            decline_reason(
                "openai/gpt-4o",
                None,
                json!([{"role": "user", "content": "hi"}]),
                json!({}),
            ),
            Some("provider is not on the rust chat completions path")
        );
        assert_eq!(
            decline_reason(
                "claude-sonnet-4-5",
                None,
                json!([{"role": "user", "content": "hi"}]),
                json!({}),
            ),
            Some("provider is not on the rust chat completions path")
        );
        assert_eq!(
            decline_reason(
                "anthropic/claude-sonnet-4-5",
                None,
                json!("nope"),
                json!({})
            ),
            Some("unreadable message list")
        );
        assert_eq!(
            decline_reason("anthropic/claude-sonnet-4-5", None, json!([]), json!({})),
            Some("empty message list")
        );
    }

    #[test]
    fn the_gate_agrees_with_prepare_on_every_case_it_accepts() {
        // A gate that accepts what prepare then declines would make the host emit
        // its pre-call logging on a path that falls back, so pin the agreement.
        for (messages, params) in [
            (
                json!([{"role": "user", "content": "hi"}]),
                json!({"max_tokens": 8}),
            ),
            (
                json!([{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]),
                json!({"temperature": 0.1}),
            ),
            (
                json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]),
                json!({}),
            ),
        ] {
            assert_eq!(
                decline_reason(
                    "anthropic/claude-sonnet-4-5",
                    None,
                    messages.clone(),
                    params.clone()
                ),
                None,
                "gate declined {messages}"
            );
            prepare_chat_completions_call(request(
                "anthropic/claude-sonnet-4-5",
                None,
                messages.clone(),
                params,
            ))
            .unwrap_or_else(|error| panic!("prepare declined {messages}: {error}"));
        }
    }
}
