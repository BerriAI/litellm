use std::sync::Arc;

use litellm_auth::{ResolvedCredential, TokenFuture, TokenProvider, TokenProviderHandle};
use litellm_http::request::{with_default_headers, with_header};
use litellm_llms_types::formats::messages::MessagesRequest;

use crate::{
    Error,
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
        messages::{
            context::MessagesTransformContext,
            transformation::{BaseMessagesConfig, complete_messages_url},
        },
    },
    github_copilot::common_utils::{
        COPILOT_DEFAULT_HEADERS, CopilotTokenCache, resolve_copilot_api_base,
    },
};

const MESSAGES_PROXY: &str = "messages-proxy";
const MESSAGES_PROXY_API_VERSION: &str = "2026-06-01";

/// Copilot routes on these, so a caller value would send the request down another path.
const FORCED_HEADERS: &[(&str, &str)] = &[
    ("openai-intent", MESSAGES_PROXY),
    ("x-interaction-type", MESSAGES_PROXY),
    ("x-github-api-version", MESSAGES_PROXY_API_VERSION),
];

pub struct GithubCopilotMessagesConfig {
    tokens: &'static dyn CopilotTokenCache,
}

impl GithubCopilotMessagesConfig {
    pub const fn new(tokens: &'static dyn CopilotTokenCache) -> Self {
        Self { tokens }
    }
}

impl BaseMessagesConfig for GithubCopilotMessagesConfig {
    fn shape_request(
        &self,
        request: MessagesRequest,
        reasoning_auto_summary: bool,
    ) -> Result<MessagesRequest, Error> {
        shape_anthropic_messages_request(request, reasoning_auto_summary)
    }

    /// The caller's `api_base` is ignored so the Copilot bearer never reaches a host the
    /// caller picked.
    fn get_complete_url(
        &self,
        _api_base: Option<&str>,
        _model: &str,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(complete_messages_url(&resolve_copilot_api_base(
            self.tokens,
        )))
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        transform_messages_request_with(request, context, FIRST_PARTY_REQUEST_POLICY)
    }

    fn secret_names(&self) -> &'static [&'static str] {
        &[]
    }

    /// The caller's `api_key` is ignored: the Copilot token is acquired at send time and
    /// replaces any forwarded `authorization`.
    fn validate_environment(
        &self,
        headers: Headers,
        _api_key: Option<&str>,
        _model: &str,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let headers = FORCED_HEADERS.iter().fold(
            with_default_headers(headers, DEFAULT_ANTHROPIC_HEADERS),
            |headers, (name, value)| with_header(headers, name, (*value).to_string()),
        );
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::Token {
                provider: TokenProviderHandle::new(Arc::new(CopilotBearer(self.tokens))),
            },
        })
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        COPILOT_DEFAULT_HEADERS
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        BetaPolicy::Forward.apply(update_headers_with_anthropic_beta(headers, request))
    }
}

#[derive(Debug)]
struct CopilotBearer(&'static dyn CopilotTokenCache);

impl TokenProvider for CopilotBearer {
    fn acquire(&self) -> TokenFuture<'_> {
        Box::pin(async move {
            let token = self.0.acquire().await?;
            Ok(ResolvedCredential::AccessToken {
                token: token.token,
                expires_on: Some(token.expires_at),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell,
        time::{Duration, SystemTime},
    };

    use litellm_auth::{AuthServices, ErrorDetail, SecretValue};
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::{
        base_llm::{auth::resolve_auth, messages::context::MessagesModelCapabilities},
        github_copilot::common_utils::{
            CopilotToken, CopilotTokenFuture, DEFAULT_GITHUB_COPILOT_API_BASE,
        },
    };

    #[derive(Debug)]
    struct FakeTokens {
        cached: Option<CopilotToken>,
        acquired: Result<CopilotToken, litellm_auth::Error>,
    }

    impl CopilotTokenCache for FakeTokens {
        fn cached(&self) -> Option<CopilotToken> {
            self.cached.clone()
        }

        fn acquire(&self) -> CopilotTokenFuture<'_> {
            Box::pin(async move { self.acquired.clone() })
        }
    }

    fn expiry() -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(4_000_000_000)
    }

    fn token(secret: &str, api_endpoint: Option<&str>) -> CopilotToken {
        CopilotToken {
            token: SecretValue::new(secret),
            expires_at: expiry(),
            api_endpoint: api_endpoint.map(str::to_string),
        }
    }

    fn config(
        cached: Option<CopilotToken>,
        acquired: Result<CopilotToken, litellm_auth::Error>,
    ) -> GithubCopilotMessagesConfig {
        GithubCopilotMessagesConfig::new(Box::leak(Box::new(FakeTokens { cached, acquired })))
    }

    fn fresh_config() -> GithubCopilotMessagesConfig {
        let fresh = token(
            "copilot-fresh",
            Some("https://api.individual.githubcopilot.com"),
        );
        config(Some(fresh.clone()), Ok(fresh))
    }

    fn headers(pairs: &[(&str, &str)]) -> Headers {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    fn request_from(value: serde_json::Value) -> MessagesRequest {
        serde_json::from_value(value).expect("valid request")
    }

    fn validated(
        config: &GithubCopilotMessagesConfig,
        forwarded: &[(&str, &str)],
    ) -> ValidatedEnvironment {
        config
            .validate_environment(
                headers(forwarded),
                Some("sk-caller"),
                "claude-sonnet-4-5",
                &|_| None,
            )
            .expect("environment validates")
    }

    #[rstest]
    #[case::bare_endpoint(
        "https://api.business.githubcopilot.com",
        "https://api.business.githubcopilot.com/v1/messages"
    )]
    #[case::trailing_slash(
        "https://api.business.githubcopilot.com/",
        "https://api.business.githubcopilot.com/v1/messages"
    )]
    #[case::complete_endpoint(
        "https://api.business.githubcopilot.com/v1/messages",
        "https://api.business.githubcopilot.com/v1/messages"
    )]
    fn the_cached_token_endpoint_decides_the_url_over_the_caller_base(
        #[case] endpoint: &str,
        #[case] expected: &str,
    ) {
        let config = config(
            Some(token("expired", Some(endpoint))),
            Err(litellm_auth::Error::ProviderAuthentication("unused".into())),
        );
        assert_eq!(
            config
                .get_complete_url(
                    Some("https://attacker.example"),
                    "claude-sonnet-4-5",
                    &|_| None
                )
                .expect("url builds"),
            expected
        );
    }

    #[rstest]
    #[case::no_cached_token(None)]
    #[case::token_without_endpoint(Some(token("t", None)))]
    #[case::blank_endpoint(Some(token("t", Some("  "))))]
    fn without_a_cached_endpoint_the_default_copilot_host_is_used(
        #[case] cached: Option<CopilotToken>,
    ) {
        let config = config(
            cached,
            Err(litellm_auth::Error::ProviderAuthentication("unused".into())),
        );
        assert_eq!(
            config
                .get_complete_url(
                    Some("https://attacker.example"),
                    "claude-sonnet-4-5",
                    &|_| None
                )
                .expect("url builds"),
            complete_messages_url(DEFAULT_GITHUB_COPILOT_API_BASE)
        );
    }

    #[tokio::test]
    async fn the_copilot_token_is_the_bearer_whatever_the_caller_sent() {
        let config = fresh_config();
        let authenticated = resolve_auth(
            &AuthServices::default(),
            validated(
                &config,
                &[("Authorization", "Bearer caller"), ("x-trace", "1")],
            ),
            &|_: &str| None,
        )
        .await
        .expect("auth resolves");
        assert_eq!(
            authenticated
                .headers
                .iter()
                .filter(|(name, _)| name.eq_ignore_ascii_case("authorization"))
                .collect::<Vec<_>>(),
            vec![&(
                "authorization".to_string(),
                "Bearer copilot-fresh".to_string()
            )]
        );
    }

    #[tokio::test]
    async fn the_bearer_carries_the_token_expiry() {
        let AuthScheme::Token { provider } = validated(&fresh_config(), &[]).auth else {
            panic!("copilot authenticates with a token")
        };
        assert_eq!(
            provider.acquire().await.expect("token acquired"),
            ResolvedCredential::AccessToken {
                token: SecretValue::new("copilot-fresh"),
                expires_on: Some(expiry()),
            }
        );
    }

    #[tokio::test]
    async fn a_token_failure_is_an_authentication_error() {
        let failure =
            litellm_auth::Error::CredentialAcquisition(ErrorDetail::from("device flow timed out"));
        let config = config(None, Err(failure.clone()));
        let error = resolve_auth(
            &AuthServices::default(),
            validated(&config, &[]),
            &|_: &str| None,
        )
        .await
        .expect_err("token failure surfaces");
        assert_eq!(error, failure);
    }

    #[test]
    fn routing_headers_are_forced_and_copilot_defaults_only_fill_gaps() {
        let config = fresh_config();
        let caller = [
            ("OpenAI-Intent", "conversation-panel"),
            ("X-Interaction-Type", "chat"),
            ("X-GitHub-Api-Version", "2020-01-01"),
            ("Editor-Version", "vscode/9.9.9"),
            ("Anthropic-Version", "2099-01-01"),
        ];
        let sent = with_default_headers(
            validated(&config, &caller).headers,
            config.default_headers(),
        );
        let value = |name: &str| -> Vec<&str> {
            sent.iter()
                .filter(|(key, _)| key.eq_ignore_ascii_case(name))
                .map(|(_, value)| value.as_str())
                .collect()
        };
        for (name, forced) in FORCED_HEADERS {
            assert_eq!(value(name), vec![*forced], "{name}");
        }
        assert_eq!(value("editor-version"), vec!["vscode/9.9.9"]);
        assert_eq!(value("anthropic-version"), vec!["2099-01-01"]);
        for (name, default) in COPILOT_DEFAULT_HEADERS
            .iter()
            .filter(|(name, _)| *name != "editor-version")
        {
            assert_eq!(value(name), vec![*default], "{name}");
        }
    }

    #[test]
    fn anthropic_version_is_filled_when_the_caller_sent_none() {
        let sent = validated(&fresh_config(), &[]).headers;
        assert!(sent.contains(&(
            "anthropic-version".to_string(),
            DEFAULT_ANTHROPIC_HEADERS[0].1.to_string()
        )));
    }

    #[test]
    fn betas_are_forwarded_unfiltered_alongside_feature_betas() {
        let request = request_from(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "context_management": {"edits": [{"type": "compact_20260112"}]}
        }));
        assert_eq!(
            fresh_config().request_headers(
                headers(&[("anthropic-beta", "example-beta-2099-01-01")]),
                &request
            ),
            headers(&[(
                "anthropic-beta",
                "compact-2026-01-12,example-beta-2099-01-01"
            )])
        );
    }

    #[test]
    fn the_request_keeps_claude_thinking_rules_and_billing_metadata() {
        let request = request_from(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 256,
            "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=1"}],
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let context = MessagesTransformContext::with_lookup(
            MessagesModelCapabilities {
                supports_reasoning: true,
                thinking_always_on: true,
                ..Default::default()
            },
            false,
            &|_: &str| None,
        );
        let transformed = serde_json::to_value(
            fresh_config()
                .transform_anthropic_messages_request(request, &context)
                .expect("request transforms"),
        )
        .expect("serializable");
        assert_eq!(transformed.get("thinking"), None);
        assert_eq!(
            transformed["system"],
            json!([{"type": "text", "text": "x-anthropic-billing-header: cc_version=1"}])
        );
    }

    #[test]
    fn secret_names_cover_every_lookup() {
        let requested = RefCell::new(Vec::<String>::new());
        let record = |name: &str| -> Option<String> {
            requested.borrow_mut().push(name.to_string());
            None
        };
        let config = fresh_config();
        let _ = config.validate_environment(Vec::new(), None, "claude", &record);
        let _ = config.get_complete_url(None, "claude", &record);
        let undeclared: Vec<String> = requested
            .into_inner()
            .into_iter()
            .filter(|name| !config.secret_names().contains(&name.as_str()))
            .collect();
        assert_eq!(undeclared, Vec::<String>::new());
    }
    #[rstest]
    #[case::structured_output(Some(serde_json::json!({"type": "json_object"})), None, "structured-outputs-2025-11-13")]
    #[case::context_management(None, Some(litellm_llms_types::formats::messages::ContextManagement {
        edits: Some(vec![litellm_llms_types::recognized::Recognized::Known(litellm_llms_types::formats::messages::ContextEdit::ClearToolUses { extra: Default::default() })]),
        ..Default::default()
    }), "context-management-2025-06-27")]
    fn native_feature_betas_keep_messages_proxy_routing(
        #[case] output_format: Option<serde_json::Value>,
        #[case] context_management: Option<
            litellm_llms_types::formats::messages::ContextManagement,
        >,
        #[case] expected_beta: &str,
    ) {
        use litellm_llms_types::{
            formats::messages::{Message, MessageContent, MessageRole, MessagesOptionalParams},
            recognized::Recognized,
        };
        let request = MessagesRequest {
            model: "native-claude-model".into(),
            messages: vec![Message {
                role: MessageRole::User,
                content: MessageContent::Text("Hello".into()),
                extra: Default::default(),
            }],
            params: MessagesOptionalParams {
                max_tokens: Some(16),
                output_format: output_format.map(Recognized::Unrecognized),
                context_management: context_management.map(Recognized::Known),
                ..Default::default()
            },
        };
        let config = fresh_config();
        let sent = config.request_headers(validated(&config, &[]).headers, &request);
        for (name, expected) in [
            ("openai-intent", "messages-proxy"),
            ("x-interaction-type", "messages-proxy"),
            ("x-github-api-version", "2026-06-01"),
            ("anthropic-beta", expected_beta),
        ] {
            assert_eq!(
                sent.iter()
                    .filter(|(key, _)| key.eq_ignore_ascii_case(name))
                    .map(|(_, value)| value.as_str())
                    .collect::<Vec<_>>(),
                vec![expected]
            );
        }
    }
}
