use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_core_utils::settings::resolve_non_empty;
use litellm_http::request::header_value;
use litellm_llms_types::{
    formats::messages::{MessagesOptionalParams, MessagesRequest, MessagesTool},
    recognized::Recognized,
};
use serde_json::Value;

use crate::{
    Error,
    base_llm::{
        auth::AuthScheme,
        messages::{
            context::MessagesTransformContext,
            normalization::fold_system_role_messages,
            transformation::{BaseMessagesConfig, Headers, ValidatedEnvironment},
        },
    },
};

const API_BASE_ENV_NAMES: &[&str] = &["DEEPSEEK_ANTHROPIC_API_BASE", "DEEPSEEK_API_BASE"];
const DEFAULT_API_BASE: &str = "https://api.deepseek.com/anthropic";

pub struct DeepSeekAnthropicMessagesConfig;

pub const DEEPSEEK_MESSAGES_CONFIG: DeepSeekAnthropicMessagesConfig =
    DeepSeekAnthropicMessagesConfig;

impl BaseMessagesConfig for DeepSeekAnthropicMessagesConfig {
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        if ["x-api-key", "authorization"]
            .iter()
            .any(|name| header_value(&headers, name).is_some_and(|value| !value.trim().is_empty()))
        {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            });
        }
        let key = resolve_non_empty(api_key, env_lookup, &["DEEPSEEK_API_KEY"]).ok_or(
            litellm_auth::Error::MissingApiKey {
                provider: "DeepSeek",
                environment_variable: "DEEPSEEK_API_KEY",
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

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let resolved = resolve_non_empty(api_base, env_lookup, API_BASE_ENV_NAMES)
            .unwrap_or_else(|| DEFAULT_API_BASE.to_string());
        let base = resolved.trim_end_matches('/');
        if base.ends_with("/v1/messages") && base.contains("/anthropic/") {
            return Ok(base.to_string());
        }
        let unversioned = base
            .trim_end_matches("/v1/messages")
            .trim_end_matches("/v1")
            .trim_end_matches("/beta");
        let anthropic_base =
            if unversioned.ends_with("/anthropic") || unversioned.contains("/anthropic/") {
                unversioned.to_string()
            } else {
                format!("{unversioned}/anthropic")
            };
        Ok(format!("{anthropic_base}/v1/messages"))
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        _context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        Ok(fold_system_role_messages(MessagesRequest {
            params: MessagesOptionalParams {
                tools: request
                    .params
                    .tools
                    .map(|tools| tools.into_iter().map(sanitize_tool).collect()),
                ..request.params
            },
            ..request
        }))
    }

    fn secret_names(&self) -> &'static [&'static str] {
        &[
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_ANTHROPIC_API_BASE",
            "DEEPSEEK_API_BASE",
        ]
    }
}

fn sanitize_tool(tool: Recognized<MessagesTool>) -> Recognized<MessagesTool> {
    match tool {
        Recognized::Unrecognized(Value::Object(fields))
            if fields.get("type").and_then(Value::as_str) == Some("custom") =>
        {
            Recognized::Unrecognized(Value::Object(
                fields
                    .into_iter()
                    .filter(|(name, _)| name != "type")
                    .collect(),
            ))
        }
        other => other,
    }
}
