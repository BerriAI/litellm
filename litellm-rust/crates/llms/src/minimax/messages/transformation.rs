use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_core_utils::settings::resolve_non_empty;
use litellm_http::request::{has_bearer_auth, has_header};
use litellm_types::llms::anthropic_messages::anthropic_request::{
    AnthropicMessagesOptionalParams, AnthropicMessagesRequest, ContentBlockType, SystemPrompt,
};

use crate::{
    Error,
    anthropic::messages::{
        handler::shape_anthropic_messages_request,
        transformation::{
            DEFAULT_HEADERS, transform_messages_request, update_headers_with_anthropic_beta,
        },
    },
    base_llm::{
        auth::{AuthScheme, Headers, ValidatedEnvironment},
        messages::{
            context::MessagesTransformContext,
            normalization::fold_system_role_messages,
            transformation::{BaseAnthropicMessagesConfig, MESSAGES_PATH_SUFFIX},
        },
    },
};

const MINIMAX_API_KEY_ENV: &str = "MINIMAX_API_KEY";
const MINIMAX_API_BASE_ENV: &str = "MINIMAX_API_BASE";
const DEFAULT_API_BASE: &str = "https://api.minimax.io/anthropic/v1/messages";
const BILLING_HEADER_PREFIX: &str = "x-anthropic-billing-header:";

pub struct MinimaxMessagesConfig;

pub const MINIMAX_MESSAGES_CONFIG: MinimaxMessagesConfig = MinimaxMessagesConfig;

impl BaseAnthropicMessagesConfig for MinimaxMessagesConfig {
    fn shape_request(
        &self,
        request: AnthropicMessagesRequest,
        reasoning_auto_summary: bool,
    ) -> Result<AnthropicMessagesRequest, Error> {
        shape_anthropic_messages_request(request, reasoning_auto_summary)
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let base = resolve_non_empty(api_base, env_lookup, &[MINIMAX_API_BASE_ENV])
            .unwrap_or_else(|| DEFAULT_API_BASE.to_string());
        let base = base.trim_end_matches('/');
        if base.ends_with(MESSAGES_PATH_SUFFIX) {
            return Ok(base.to_string());
        }
        Ok(format!("{base}{MESSAGES_PATH_SUFFIX}"))
    }

    fn transform_anthropic_messages_request(
        &self,
        request: AnthropicMessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<AnthropicMessagesRequest, Error> {
        let request = fold_system_role_messages(request);
        transform_messages_request(
            AnthropicMessagesRequest {
                params: AnthropicMessagesOptionalParams {
                    system: request.params.system.and_then(strip_billing_metadata),
                    ..request.params
                },
                ..request
            },
            context,
        )
    }

    fn secret_names(&self) -> &'static [&'static str] {
        &[MINIMAX_API_KEY_ENV, MINIMAX_API_BASE_ENV]
    }

    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        if has_header(&headers, "x-api-key") || has_bearer_auth(&headers) {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            });
        }
        let key = resolve_non_empty(api_key, env_lookup, &[MINIMAX_API_KEY_ENV]).ok_or(
            litellm_auth::Error::MissingApiKey {
                provider: "MiniMax",
                environment_variable: MINIMAX_API_KEY_ENV,
            },
        )?;
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::Credential {
                placement: CredentialPlacement::Header("x-api-key"),
                secret: SecretValue::new(key),
            },
        })
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_HEADERS
    }

    fn request_headers(&self, headers: Headers, request: &AnthropicMessagesRequest) -> Headers {
        update_headers_with_anthropic_beta(headers, request)
    }
}

fn strip_billing_metadata(system: SystemPrompt) -> Option<SystemPrompt> {
    match system {
        SystemPrompt::Text(text) => {
            (!text.starts_with(BILLING_HEADER_PREFIX)).then_some(SystemPrompt::Text(text))
        }
        SystemPrompt::Blocks(blocks) => {
            let filtered: Vec<_> = blocks
                .into_iter()
                .filter(|block| {
                    !(block.is_type(ContentBlockType::Text)
                        && block
                            .text
                            .as_deref()
                            .is_some_and(|text| text.starts_with(BILLING_HEADER_PREFIX)))
                })
                .collect();
            (!filtered.is_empty()).then_some(SystemPrompt::Blocks(filtered))
        }
    }
}
