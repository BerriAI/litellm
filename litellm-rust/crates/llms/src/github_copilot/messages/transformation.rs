use litellm_auth::CredentialPlacement;
use litellm_http::request::{with_default_headers, with_header};
use litellm_llms_types::formats::messages::MessagesRequest;
use litellm_llms_types::providers::github_copilot::{DEFAULT_HEADERS, MESSAGES_PATH};
use litellm_router_types::{GithubCopilotSession, LitellmParams};

use crate::{
    Error,
    anthropic::messages::transformation::{
        ANTHROPIC_MESSAGES_CONFIG, update_headers_with_anthropic_beta,
    },
    base_llm::{
        auth::AuthScheme,
        messages::{
            context::MessagesTransformContext,
            transformation::{BaseMessagesConfig, Headers, ValidatedEnvironment},
        },
    },
};

pub struct GithubCopilotAnthropicMessagesConfig;
pub const COPILOT_MESSAGES_CONFIG: GithubCopilotAnthropicMessagesConfig =
    GithubCopilotAnthropicMessagesConfig;

impl BaseMessagesConfig for GithubCopilotAnthropicMessagesConfig {
    fn shape_request(
        &self,
        request: MessagesRequest,
        reasoning_auto_summary: bool,
    ) -> Result<MessagesRequest, Error> {
        ANTHROPIC_MESSAGES_CONFIG.shape_request(request, reasoning_auto_summary)
    }

    fn validate_environment(
        &self,
        headers: Headers,
        _api_key: Option<&str>,
        _model: &str,
        params: &LitellmParams,
        _env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                secret: host_session(params)?.token.clone(),
            },
        })
    }

    fn get_complete_url(
        &self,
        _api_base: Option<&str>,
        _model: &str,
        params: &LitellmParams,
        _stream: bool,
        _env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let api_base = host_session(params)?.api_base.trim_end_matches('/');
        Ok(if api_base.ends_with(MESSAGES_PATH) {
            api_base.to_string()
        } else {
            format!("{api_base}{MESSAGES_PATH}")
        })
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        ANTHROPIC_MESSAGES_CONFIG.transform_anthropic_messages_request(request, context)
    }

    fn secret_names(&self) -> &'static [&'static str] {
        &[]
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_HEADERS
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        let headers = update_headers_with_anthropic_beta(headers, request);
        let headers = with_default_headers(
            headers,
            &[("x-request-id", &uuid::Uuid::new_v4().to_string())],
        );
        let headers = with_header(headers, "openai-intent", "messages-proxy".into());
        let headers = with_header(headers, "x-interaction-type", "messages-proxy".into());
        with_header(headers, "x-github-api-version", "2026-06-01".into())
    }
}

fn host_session(params: &LitellmParams) -> Result<&GithubCopilotSession, Error> {
    params.github_copilot_session.as_ref().ok_or_else(|| {
        litellm_auth::Error::ProviderAuthentication(
            "GitHub Copilot session was not resolved by the host".into(),
        )
        .into()
    })
}
