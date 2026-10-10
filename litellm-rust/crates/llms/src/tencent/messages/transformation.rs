use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_core_utils::settings::resolve_non_empty;
use litellm_llms_types::formats::messages::MessagesRequest;
use litellm_router_types::LitellmParams;

use crate::{
    Error,
    anthropic::{
        common_utils::has_anthropic_credential,
        messages::transformation::{DEFAULT_HEADERS, update_headers_with_anthropic_beta},
    },
    base_llm::{
        auth::AuthScheme,
        messages::{
            context::MessagesTransformContext,
            transformation::{BaseMessagesConfig, Headers, ValidatedEnvironment},
        },
    },
    openai_like::messages::transformation::without_billing_blocks,
};

const API_KEY_ENV: &str = "TENCENT_API_KEY";
const API_BASE_ENV: &str = "TENCENT_API_BASE";
const ANTHROPIC_API_BASE_ENV: &str = "TENCENT_ANTHROPIC_API_BASE";
const DEFAULT_BASE: &str = "https://tokenhub-intl.tencentcloudmaas.com";

pub struct TencentAnthropicMessagesConfig;
pub const TENCENT_MESSAGES_CONFIG: TencentAnthropicMessagesConfig = TencentAnthropicMessagesConfig;

impl BaseMessagesConfig for TencentAnthropicMessagesConfig {
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        _params: &LitellmParams,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        if has_anthropic_credential(&headers) {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            });
        }
        let key = resolve_non_empty(api_key, env, &[API_KEY_ENV]).ok_or(Error::Auth(
            litellm_auth::Error::MissingApiKey {
                provider: "Tencent",
                environment_variable: API_KEY_ENV,
            },
        ))?;
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
        let base = resolve_non_empty(api_base, env, &[ANTHROPIC_API_BASE_ENV, API_BASE_ENV])
            .unwrap_or_else(|| DEFAULT_BASE.into());
        let base = base.trim_end_matches('/');
        let base = base.strip_suffix("/v1/chat/completions").unwrap_or(base);
        Ok(crate::openai_like::messages::transformation::complete_messages_url(base))
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        _context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        Ok(without_billing_blocks(request))
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
