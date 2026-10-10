use litellm_http::request::{with_default_headers, with_header};
use litellm_llms_types::formats::messages::MessagesRequest;
use litellm_router_types::LitellmParams;

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

const DEFAULT_HEADERS: &[(&str, &str)] = &[
    ("content-type", "application/json"),
    ("copilot-integration-id", "vscode-chat"),
    ("editor-version", "vscode/1.95.0"),
    ("editor-plugin-version", "copilot-chat/0.26.7"),
    ("user-agent", "GitHubCopilotChat/0.26.7"),
    ("x-vscode-user-agent-library-version", "electron-fetch"),
    ("anthropic-version", "2023-06-01"),
];

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
        if params.extra.contains_key("github_copilot_user_session") {
            return Err(Error::Unsupported(
                "native Copilot per-user session projection",
            ));
        }
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::CopilotSession {
                path: "/v1/messages",
            },
        })
    }

    fn get_complete_url(
        &self,
        _api_base: Option<&str>,
        _model: &str,
        _params: &LitellmParams,
        _stream: bool,
        _env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(format!(
            "{}/v1/messages",
            litellm_auth_copilot::DEFAULT_API_BASE
        ))
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
