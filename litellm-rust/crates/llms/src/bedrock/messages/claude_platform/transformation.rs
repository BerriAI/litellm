use litellm_auth::{
    CredentialPlacement, CredentialPlanKind, CredentialRule, ExistingHeaderBehavior,
    ProviderAuthPolicy, SecretValue,
};
use litellm_auth_aws::{
    AwsCredentialSource,
    constants::{
        AWS_ACCESS_KEY_ID, AWS_DEFAULT_REGION, AWS_EXTERNAL_ID, AWS_PROFILE_NAME, AWS_REGION,
        AWS_REGION_NAME, AWS_ROLE_NAME, AWS_SECRET_ACCESS_KEY, AWS_SESSION_NAME, AWS_SESSION_TOKEN,
        AWS_STS_ENDPOINT, AWS_WEB_IDENTITY_TOKEN,
    },
};
use litellm_http::request::with_header;
use litellm_llms_types::formats::messages::{MessagesOptionalParams, MessagesRequest};
use serde_json::{Map, Value};

use super::common_utils::{
    ANTHROPIC_AWS_API_BASE_ENV, ANTHROPIC_AWS_API_KEY_ENV, ANTHROPIC_AWS_BASE_URL_ENV,
    ANTHROPIC_AWS_WORKSPACE_ID_ENV, ANTHROPIC_WORKSPACE_ID_ENV, CLAUDE_PLATFORM_SERVICE_NAME,
    ClaudePlatformSettings, WORKSPACE_HEADER, complete_claude_platform_url, is_non_request_param,
    resolve_api_key, resolve_region, resolve_workspace_id, strip_claude_platform_route,
};
use crate::{
    Error, ErrorDetail,
    anthropic::{
        beta_headers::BetaPolicy,
        common_utils::DEFAULT_ANTHROPIC_HEADERS,
        messages::{
            handler::shape_anthropic_messages_request,
            transformation::{
                FIRST_PARTY_REQUEST_POLICY, transform_messages_request_with,
                update_headers_with_anthropic_beta,
            },
        },
    },
    base_llm::{
        auth::{AuthScheme, Headers, ValidatedEnvironment},
        messages::{context::MessagesTransformContext, transformation::BaseMessagesConfig},
    },
};

const AUTH_POLICY: ProviderAuthPolicy = ProviderAuthPolicy {
    rules: &[CredentialRule {
        kind: CredentialPlanKind::Static,
        placement: CredentialPlacement::Header("x-api-key"),
    }],
    accepted_existing_headers: &["x-api-key"],
    existing_header_behavior: ExistingHeaderBehavior::Preserve,
    scope: None,
    audience: None,
};

const SECRET_NAMES: &[&str] = &[
    ANTHROPIC_AWS_API_KEY_ENV,
    ANTHROPIC_AWS_WORKSPACE_ID_ENV,
    ANTHROPIC_WORKSPACE_ID_ENV,
    ANTHROPIC_AWS_BASE_URL_ENV,
    ANTHROPIC_AWS_API_BASE_ENV,
    AWS_REGION_NAME,
    AWS_REGION,
    AWS_DEFAULT_REGION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_SESSION_TOKEN,
    AWS_SESSION_NAME,
    AWS_PROFILE_NAME,
    AWS_ROLE_NAME,
    AWS_WEB_IDENTITY_TOKEN,
    AWS_STS_ENDPOINT,
    AWS_EXTERNAL_ID,
];

pub struct ClaudePlatformMessagesConfig {
    settings: ClaudePlatformSettings,
}

pub const CLAUDE_PLATFORM_MESSAGES_CONFIG: ClaudePlatformMessagesConfig =
    ClaudePlatformMessagesConfig::new(ClaudePlatformSettings::EMPTY);

impl ClaudePlatformMessagesConfig {
    pub const fn new(settings: ClaudePlatformSettings) -> Self {
        Self { settings }
    }

    fn request_params(
        &self,
        params: MessagesOptionalParams,
    ) -> Result<MessagesOptionalParams, Error> {
        let Value::Object(fields) = serde_json::to_value(params).map_err(invalid_params)? else {
            return Err(Error::InvalidRequest(ErrorDetail::Message(
                "Messages params did not serialize to an object".into(),
            )));
        };
        let kept: Map<String, Value> = fields
            .into_iter()
            .filter(|(key, _)| {
                !is_non_request_param(key) && !self.settings.unsupported_params.rejects(key)
            })
            .collect();
        serde_json::from_value(Value::Object(kept)).map_err(invalid_params)
    }
}

fn invalid_params(error: serde_json::Error) -> Error {
    Error::InvalidRequest(ErrorDetail::failed(
        "filtering Claude Platform params",
        error,
    ))
}

impl BaseMessagesConfig for ClaudePlatformMessagesConfig {
    fn shape_request(
        &self,
        request: MessagesRequest,
        reasoning_auto_summary: bool,
    ) -> Result<MessagesRequest, Error> {
        shape_anthropic_messages_request(request, reasoning_auto_summary)
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        complete_claude_platform_url(api_base, &self.settings, env_lookup)
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        let filtered = MessagesRequest {
            model: strip_claude_platform_route(&request.model).to_string(),
            params: self.request_params(request.params)?,
            messages: request.messages,
        };
        transform_messages_request_with(filtered, context, FIRST_PARTY_REQUEST_POLICY)
    }

    fn secret_names(&self) -> &'static [&'static str] {
        SECRET_NAMES
    }

    /// A key, the caller's or `ANTHROPIC_AWS_API_KEY`, goes in `x-api-key` unsigned, and a
    /// forwarded `x-api-key` is kept. With neither, the request is SigV4 signed.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let workspace_id = resolve_workspace_id(&self.settings, env_lookup)?;
        let headers = with_header(headers, WORKSPACE_HEADER, workspace_id);
        let api_key = resolve_api_key(api_key, env_lookup);
        if api_key.is_some() || AUTH_POLICY.has_existing_credential(&headers) {
            return Ok(ValidatedEnvironment::with_api_key(
                &AUTH_POLICY,
                headers,
                api_key.map(SecretValue::new),
                litellm_auth::Error::MissingApiKey {
                    provider: "Claude Platform on AWS",
                    environment_variable: ANTHROPIC_AWS_API_KEY_ENV,
                },
            )?);
        }
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::AwsSigV4 {
                region: resolve_region(&self.settings, env_lookup)?,
                service: CLAUDE_PLATFORM_SERVICE_NAME,
                credentials: Box::new(AwsCredentialSource::from_params(
                    &self.settings.aws_param_map(),
                    env_lookup,
                )),
            },
        })
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_ANTHROPIC_HEADERS
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        BetaPolicy::Forward.apply(update_headers_with_anthropic_beta(headers, request))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use litellm_llms_types::formats::messages::MessagesResponse;
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::bedrock::messages::claude_platform::common_utils::UnsupportedParams;

    fn request(value: Value) -> MessagesRequest {
        serde_json::from_value(value).unwrap()
    }

    fn headers(pairs: &[(&str, &str)]) -> Headers {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    fn config(unsupported_params: UnsupportedParams) -> ClaudePlatformMessagesConfig {
        ClaudePlatformMessagesConfig::new(ClaudePlatformSettings {
            workspace_id: Some("wrkspc_1".into()),
            unsupported_params,
            ..ClaudePlatformSettings::EMPTY
        })
    }

    fn body_with(fields: Value) -> Value {
        let Value::Object(fields) = fields else {
            panic!("case fields are an object")
        };
        Value::Object(
            [
                ("model".to_string(), json!("claude_platform/claude-x")),
                ("max_tokens".to_string(), json!(16)),
                (
                    "messages".to_string(),
                    json!([{"role": "user", "content": "hi"}]),
                ),
            ]
            .into_iter()
            .chain(fields)
            .collect(),
        )
    }

    fn transformed(config: &ClaudePlatformMessagesConfig, fields: Value) -> MessagesRequest {
        config
            .transform_anthropic_messages_request(
                request(body_with(fields)),
                &MessagesTransformContext::default(),
            )
            .unwrap()
    }

    #[test]
    fn the_wire_body_drops_workspace_override_aws_and_rejected_keys_and_the_route_prefix() {
        let body = serde_json::to_value(transformed(
            &config(UnsupportedParams::Default),
            json!({
                "workspace_id": "w",
                "aws_workspace_id": "w",
                "anthropic-workspace-id": "w",
                "anthropic_workspace_id": "w",
                "claude_platform_unsupported_params": ["speed"],
                "aws_region_name": "us-east-1",
                "aws_secret_access_key": "s",
                "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
                "temperature": 0.5,
                "custom_field": 1
            }),
        ))
        .unwrap();
        assert_eq!(
            body,
            json!({
                "model": "claude-x",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.5,
                "custom_field": 1
            })
        );
    }

    #[rstest]
    #[case::replacement_keeps_context_management(
        UnsupportedParams::Replaced(BTreeSet::from(["speed".to_string()])),
        json!({"speed": "fast", "context_management": {"edits": []}}),
        json!({"context_management": {"edits": []}})
    )]
    #[case::empty_replacement_rejects_nothing(
        UnsupportedParams::Replaced(BTreeSet::new()),
        json!({"context_management": {"edits": []}}),
        json!({"context_management": {"edits": []}})
    )]
    #[case::ignored_override_keeps_the_default(
        UnsupportedParams::Ignored { found: "string" },
        json!({"context_management": {"edits": []}, "speed": "fast"}),
        json!({"speed": "fast"})
    )]
    fn the_configured_rejected_set_replaces_the_default(
        #[case] unsupported: UnsupportedParams,
        #[case] fields: Value,
        #[case] expected_extra: Value,
    ) {
        let config = config(unsupported);
        let context = MessagesTransformContext::with_lookup(
            crate::base_llm::messages::context::MessagesModelCapabilities {
                supports_speed: true,
                ..Default::default()
            },
            false,
            &|_: &str| None,
        );
        assert_eq!(
            config
                .transform_anthropic_messages_request(request(body_with(fields)), &context)
                .unwrap(),
            MessagesRequest {
                model: "claude-x".into(),
                ..request(body_with(expected_extra))
            }
        );
    }

    #[rstest]
    #[case::dropped_context_management_pulls_no_beta(
        UnsupportedParams::Default,
        json!({"context_management": {"edits": [{"type": "compact_20260112"}]}}),
        &[("anthropic-beta", "user-beta")],
        &[("anthropic-beta", "user-beta")]
    )]
    #[case::kept_context_management_adds_its_beta_and_forwards_user_betas(
        UnsupportedParams::Replaced(BTreeSet::new()),
        json!({"context_management": {"edits": [{"type": "compact_20260112"}]}}),
        &[("anthropic-beta", "user-beta")],
        &[("anthropic-beta", "compact-2026-01-12,user-beta")]
    )]
    fn automatic_betas_follow_the_filtered_body_and_user_betas_pass_unfiltered(
        #[case] unsupported: UnsupportedParams,
        #[case] fields: Value,
        #[case] forwarded: &[(&str, &str)],
        #[case] expected: &[(&str, &str)],
    ) {
        let config = config(unsupported);
        let body = transformed(&config, fields);
        assert_eq!(
            config.request_headers(headers(forwarded), &body),
            headers(expected)
        );
    }

    #[test]
    fn a_missing_workspace_is_an_auth_error_before_any_credential_is_chosen() {
        let error = CLAUDE_PLATFORM_MESSAGES_CONFIG
            .validate_environment(Vec::new(), Some("sk"), "claude-x", &env(&[]))
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Auth(litellm_auth::Error::ProviderAuthentication(ref message))
                if message.contains("workspace")
        ));
    }

    #[test]
    fn the_workspace_header_replaces_a_forwarded_one() {
        let validated = config(UnsupportedParams::Default)
            .validate_environment(
                headers(&[("Anthropic-Workspace-Id", "caller")]),
                Some("sk"),
                "claude-x",
                &env(&[]),
            )
            .unwrap();
        assert_eq!(
            validated.headers,
            headers(&[("anthropic-workspace-id", "wrkspc_1")])
        );
    }

    #[test]
    fn the_env_workspace_is_used_without_a_configured_one() {
        let validated = CLAUDE_PLATFORM_MESSAGES_CONFIG
            .validate_environment(
                Vec::new(),
                Some("sk"),
                "claude-x",
                &env(&[(ANTHROPIC_AWS_WORKSPACE_ID_ENV, "wrkspc_env")]),
            )
            .unwrap();
        assert_eq!(
            validated.headers,
            headers(&[("anthropic-workspace-id", "wrkspc_env")])
        );
    }

    #[rstest]
    #[case::caller_key(Some("sk-call"), &[], &[], Some("sk-call"))]
    #[case::env_key(None, &[(ANTHROPIC_AWS_API_KEY_ENV, "sk-env")], &[], Some("sk-env"))]
    #[case::blank_caller_key_uses_env(Some(" "), &[(ANTHROPIC_AWS_API_KEY_ENV, "sk-env")], &[], Some("sk-env"))]
    fn an_api_key_goes_in_x_api_key_unsigned(
        #[case] api_key: Option<&str>,
        #[case] pairs: &'static [(&'static str, &'static str)],
        #[case] forwarded: &[(&str, &str)],
        #[case] expected: Option<&str>,
    ) {
        let validated = config(UnsupportedParams::Default)
            .validate_environment(headers(forwarded), api_key, "claude-x", &env(pairs))
            .unwrap();
        assert!(matches!(
            validated.auth,
            AuthScheme::Credential {
                placement: CredentialPlacement::Header("x-api-key"),
                ref secret,
            } if Some(secret.expose()) == expected
        ));
    }

    #[test]
    fn a_forwarded_x_api_key_is_sent_unsigned() {
        let validated = config(UnsupportedParams::Default)
            .validate_environment(
                headers(&[("x-api-key", "caller")]),
                None,
                "claude-x",
                &env(&[]),
            )
            .unwrap();
        assert!(matches!(validated.auth, AuthScheme::Forwarded));
    }

    #[test]
    fn without_a_key_the_request_is_sigv4_signed_for_the_gateway_service() {
        let config = ClaudePlatformMessagesConfig::new(ClaudePlatformSettings {
            workspace_id: Some("wrkspc_1".into()),
            aws_params: BTreeMap::from([
                ("aws_region_name".to_string(), "us-east-1".to_string()),
                ("aws_access_key_id".to_string(), "AKID".to_string()),
                ("aws_secret_access_key".to_string(), "secret".to_string()),
            ]),
            ..ClaudePlatformSettings::EMPTY
        });
        let validated = config
            .validate_environment(Vec::new(), None, "claude-x", &env(&[]))
            .unwrap();
        match validated.auth {
            AuthScheme::AwsSigV4 {
                region,
                service,
                credentials,
            } => {
                assert_eq!(
                    (region.as_str(), service),
                    ("us-east-1", CLAUDE_PLATFORM_SERVICE_NAME)
                );
                assert!(matches!(*credentials, AwsCredentialSource::HostSupplied(_)));
            }
            other => panic!("expected SigV4, got {other:?}"),
        }
    }

    #[test]
    fn signing_without_a_region_is_an_auth_error() {
        let error = config(UnsupportedParams::Default)
            .validate_environment(Vec::new(), None, "claude-x", &env(&[]))
            .unwrap_err();
        assert!(error.to_string().contains("Missing AWS region"), "{error}");
    }

    #[test]
    fn responses_pass_through_unchanged() {
        let response: MessagesResponse = serde_json::from_value(json!({
            "id": "msg_1", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "hello"}], "model": "claude-x",
            "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 2}
        }))
        .unwrap();
        assert_eq!(
            CLAUDE_PLATFORM_MESSAGES_CONFIG
                .transform_anthropic_messages_response("claude-x", response.clone())
                .unwrap(),
            response
        );
        assert!(CLAUDE_PLATFORM_MESSAGES_CONFIG.stream_decoder().is_none());
    }

    #[rstest]
    #[case::api_key(Some("sk"))]
    #[case::sigv4(None)]
    fn secret_names_cover_every_lookup(#[case] api_key: Option<&str>) {
        let requested = std::cell::RefCell::new(Vec::<String>::new());
        let record = |name: &str| -> Option<String> {
            requested.borrow_mut().push(name.to_string());
            (name == AWS_REGION_NAME).then(|| "us-east-1".to_string())
        };
        let config = config(UnsupportedParams::Default);
        let _ = config.validate_environment(Vec::new(), api_key, "claude-x", &record);
        let _ = CLAUDE_PLATFORM_MESSAGES_CONFIG.validate_environment(
            Vec::new(),
            api_key,
            "claude-x",
            &record,
        );
        let _ = config.get_complete_url(None, "claude-x", &record);
        let requested = requested.into_inner();
        assert!(!requested.is_empty());
        let undeclared: Vec<&String> = requested
            .iter()
            .filter(|name| !SECRET_NAMES.contains(&name.as_str()))
            .collect();
        assert_eq!(undeclared, Vec::<&String>::new());
    }
}
