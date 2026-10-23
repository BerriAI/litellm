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

const API_KEY_ENV: &str = "MINIMAX_API_KEY";
const API_BASE_ENV: &str = "MINIMAX_API_BASE";
const DEFAULT_BASE: &str = "https://api.minimax.io/anthropic";

#[derive(Layered)]
#[expect(dead_code, reason = "only the derived partial is read")]
struct MinimaxKeys {
    #[layer(litellm_param = "api_key", env = "MINIMAX_API_KEY", read = read::text, secret, setting = "API Key")]
    api_key: String,
    #[layer(litellm_param = "api_base", env = "MINIMAX_API_BASE", read = read::text, setting = "API Base")]
    api_base: String,
}

impl MinimaxKeysPartial {
    fn env(env: &dyn Fn(&str) -> Option<String>) -> Self {
        Self::from_env(&env).unwrap_or_default()
    }
}

pub struct MinimaxMessagesConfig;
pub const MINIMAX_MESSAGES_CONFIG: MinimaxMessagesConfig = MinimaxMessagesConfig;

impl BaseMessagesConfig for MinimaxMessagesConfig {
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
            .or(MinimaxKeysPartial::env(env).api_key)
            .ok_or(Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "MiniMax",
                environment_variable: MinimaxKeys::API_KEY.env[0],
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
            .or(MinimaxKeysPartial::env(env).api_base)
            .unwrap_or_else(|| DEFAULT_BASE.into());
        let base = base.trim_end_matches('/');
        if base.ends_with(MESSAGES_PATH) {
            return Ok(base.into());
        }
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
        &[API_KEY_ENV, API_BASE_ENV]
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
    fn endpoint_is_appended_once(#[case] base: &str, #[case] expected: &str) {
        assert_eq!(
            MINIMAX_MESSAGES_CONFIG.get_complete_url(
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
    #[case::explicit(Some("https://explicit.test"), "https://explicit.test/v1/messages")]
    #[case::environment(None, "https://env.test/v1/messages")]
    #[case::blank_explicit(Some(" "), "https://env.test/v1/messages")]
    fn base_precedence(#[case] base: Option<&str>, #[case] expected: &str) {
        let env =
            |name: &str| (name == API_BASE_ENV).then(|| "https://env.test/v1/messages/".into());
        assert_eq!(
            MINIMAX_MESSAGES_CONFIG.get_complete_url(
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
        let environment = MINIMAX_MESSAGES_CONFIG
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
