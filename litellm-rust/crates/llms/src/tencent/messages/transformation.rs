use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_layered::{Layered, non_blank, read};
use litellm_llms_types::{
    formats::messages::{MessagesOptionalParams, MessagesRequest},
    providers::anthropic::{DEFAULT_HEADERS, MESSAGES_PATH},
};
use litellm_router_types::LitellmParams;

use crate::{
    Error,
    anthropic::{
        common_utils::{filter_billing_headers_from_system, has_anthropic_credential},
        messages::{
            handler::shape_anthropic_messages_request,
            transformation::{transform_messages_request, update_headers_with_anthropic_beta},
        },
    },
    base_llm::{
        auth::{AuthScheme, Headers, ValidatedEnvironment},
        messages::{context::MessagesTransformContext, transformation::BaseMessagesConfig},
    },
};

const API_KEY_ENV: &str = "TENCENT_API_KEY";
const API_BASE_ENV: &str = "TENCENT_API_BASE";
const ANTHROPIC_API_BASE_ENV: &str = "TENCENT_ANTHROPIC_API_BASE";
const DEFAULT_BASE: &str = "https://tokenhub-intl.tencentcloudmaas.com";

#[derive(Layered)]
#[expect(dead_code, reason = "only the derived partial is read")]
struct TencentKeys {
    #[layer(litellm_param = "api_key", env = "TENCENT_API_KEY", read = read::text, secret, setting = "API Key")]
    api_key: String,
    #[layer(litellm_param = "api_base", env = ["TENCENT_ANTHROPIC_API_BASE", "TENCENT_API_BASE"], read = read::text, setting = "API Base")]
    api_base: String,
}

impl TencentKeysPartial {
    fn env(env: &dyn Fn(&str) -> Option<String>) -> Self {
        Self::from_env(&env).unwrap_or_default()
    }
}

pub struct TencentAnthropicMessagesConfig;
pub const TENCENT_MESSAGES_CONFIG: TencentAnthropicMessagesConfig = TencentAnthropicMessagesConfig;

impl BaseMessagesConfig for TencentAnthropicMessagesConfig {
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
        if has_anthropic_credential(&headers) {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            });
        }
        let key = non_blank(api_key)
            .or(TencentKeysPartial::env(env).api_key)
            .ok_or(Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "Tencent",
                environment_variable: TencentKeys::API_KEY.env[0],
            }))?;
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::Credential {
                placement: CredentialPlacement::Header("x-api-key"),
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
        let base = non_blank(api_base)
            .or(TencentKeysPartial::env(env).api_base)
            .unwrap_or_else(|| DEFAULT_BASE.into());
        let base = base.trim_end_matches('/');
        if base.ends_with(MESSAGES_PATH) {
            return Ok(base.into());
        }
        let base = base
            .strip_suffix("/v1/chat/completions")
            .or_else(|| base.strip_suffix("/v1"))
            .unwrap_or(base);
        Ok(format!("{base}{MESSAGES_PATH}"))
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        let request = transform_messages_request(request, context)?;
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
        &[API_KEY_ENV, ANTHROPIC_API_BASE_ENV, API_BASE_ENV]
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
    use rstest::rstest;

    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[rstest]
    #[case::root("https://proxy.test", "https://proxy.test/v1/messages")]
    #[case::root_slash("https://proxy.test/", "https://proxy.test/v1/messages")]
    #[case::full_endpoint("https://proxy.test/v1/messages", "https://proxy.test/v1/messages")]
    #[case::full_endpoint_slash(
        "https://proxy.test/v1/messages/",
        "https://proxy.test/v1/messages"
    )]
    #[case::custom_path(
        "https://proxy.test/anthropic",
        "https://proxy.test/anthropic/v1/messages"
    )]
    #[case::versioned("https://proxy.test/v1", "https://proxy.test/v1/messages")]
    #[case::chat_endpoint(
        "https://proxy.test/v1/chat/completions/",
        "https://proxy.test/v1/messages"
    )]
    fn endpoint_is_appended_once(#[case] base: &str, #[case] expected: &str) {
        assert_eq!(
            TENCENT_MESSAGES_CONFIG.get_complete_url(
                Some(base),
                "test-model",
                &Default::default(),
                false,
                &no_env,
            ),
            Ok(expected.into())
        );
    }

    #[rstest]
    #[case::default(None, None, None, &format!("{DEFAULT_BASE}{MESSAGES_PATH}"))]
    #[case::explicit(
        Some("https://explicit.test"),
        Some("https://anthropic.test"),
        Some("https://env.test"),
        "https://explicit.test/v1/messages"
    )]
    #[case::anthropic_base(
        None,
        Some("https://anthropic.test"),
        Some("https://env.test"),
        "https://anthropic.test/v1/messages"
    )]
    #[case::generic_base(
        None,
        None,
        Some("https://env.test/v1"),
        "https://env.test/v1/messages"
    )]
    #[case::blank_anthropic_base(
        None,
        Some(" "),
        Some("https://env.test"),
        "https://env.test/v1/messages"
    )]
    #[case::blank_explicit(
        Some(" "),
        Some("https://anthropic.test/v1/messages/"),
        None,
        "https://anthropic.test/v1/messages"
    )]
    fn base_precedence(
        #[case] base: Option<&str>,
        #[case] anthropic_base: Option<&str>,
        #[case] generic_base: Option<&str>,
        #[case] expected: &str,
    ) {
        let env = |name: &str| match name {
            ANTHROPIC_API_BASE_ENV => anthropic_base.map(str::to_owned),
            API_BASE_ENV => generic_base.map(str::to_owned),
            _ => None,
        };
        assert_eq!(
            TENCENT_MESSAGES_CONFIG.get_complete_url(
                base,
                "test-model",
                &Default::default(),
                false,
                &env,
            ),
            Ok(expected.into())
        );
    }

    #[rstest]
    #[case::explicit(Some("explicit"), Some("environment"), "explicit")]
    #[case::environment(None, Some("environment"), "environment")]
    #[case::blank_explicit(Some(" "), Some("environment"), "environment")]
    fn key_precedence(
        #[case] key: Option<&str>,
        #[case] env_key: Option<&str>,
        #[case] expected: &str,
    ) {
        let env = |name: &str| {
            (name == API_KEY_ENV)
                .then(|| env_key.map(str::to_owned))
                .flatten()
        };
        let environment = TENCENT_MESSAGES_CONFIG
            .validate_environment(
                vec![],
                key,
                "test-model",
                &MessagesOptionalParams::default(),
                &Default::default(),
                &env,
            )
            .unwrap();
        assert!(matches!(
            environment.auth,
            AuthScheme::Credential {
                placement: CredentialPlacement::Header("x-api-key"), ref secret
            } if secret.expose() == expected
        ));
    }
}
