use litellm_core_utils::settings::{Lookup, ProcessEnvironment};
use litellm_types::llms::anthropic_messages::anthropic_request::AnthropicMessagesRequest;
use serde_json::{Map, Value, json};

use super::{
    headers::{Headers, authenticate, with_feature_betas},
    thinking::{ThinkingBudgets, ThinkingContext, translate_thinking},
};
use crate::{
    anthropic::common_utils::{
        AnthropicModelCapabilities, has_advisor_tool, strip_advisor_blocks,
        strip_encrypted_reasoning_blocks,
    },
    base_llm::{
        anthropic_messages::transformation::{
            BaseAnthropicMessagesConfig, MessagesTransformContext,
        },
        chat::transformation::Error,
    },
};

const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
const ANTHROPIC_API_BASE_ENV: &str = "ANTHROPIC_API_BASE";
const ANTHROPIC_BASE_URL_ENV: &str = "ANTHROPIC_BASE_URL";
const DEFAULT_ANTHROPIC_API_BASE: &str = "https://api.anthropic.com";
const MESSAGES_PATH_SUFFIX: &str = "/v1/messages";

pub struct AnthropicMessagesConfig;

pub const ANTHROPIC_MESSAGES_CONFIG: AnthropicMessagesConfig = AnthropicMessagesConfig;

impl MessagesTransformContext {
    /// The context for a model whose capability flags the host resolved, with the thinking
    /// budgets read from the process environment.
    pub fn new(capabilities: AnthropicModelCapabilities, drop_params: bool) -> Self {
        Self::with_lookup(capabilities, drop_params, &ProcessEnvironment)
    }

    pub fn with_lookup(
        capabilities: AnthropicModelCapabilities,
        drop_params: bool,
        env: &impl Lookup,
    ) -> Self {
        Self {
            thinking: ThinkingContext {
                capabilities,
                budgets: ThinkingBudgets::from_lookup(env),
            },
            drop_params,
        }
    }
}

impl BaseAnthropicMessagesConfig for AnthropicMessagesConfig {
    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(complete_anthropic_url(api_base, env_lookup))
    }

    fn transform_anthropic_messages_request(
        &self,
        request: AnthropicMessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<AnthropicMessagesRequest, Error> {
        if request.max_tokens.is_none() {
            return Err(Error::InvalidRequest(
                "max_tokens is required for Anthropic /v1/messages API".to_string(),
            ));
        }
        let request = drop_unsupported_params(request, context)?;
        let request = translate_thinking(request, &context.thinking)?;
        let context_management = request
            .context_management
            .as_ref()
            .and_then(map_openai_context_management_to_anthropic)
            .or_else(|| request.context_management.clone());
        let messages = if has_advisor_tool(request.tools.as_deref()) {
            request.messages
        } else {
            strip_advisor_blocks(request.messages)
        };
        Ok(AnthropicMessagesRequest {
            messages: strip_encrypted_reasoning_blocks(messages),
            context_management,
            ..request
        })
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        resolve_anthropic_api_key(api_key, env_lookup).map_err(Error::from)
    }

    fn authenticate(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Headers, Error> {
        authenticate(headers, api_key, env_lookup).map_err(Error::from)
    }

    fn request_headers(&self, headers: Headers, request: &AnthropicMessagesRequest) -> Headers {
        with_feature_betas(headers, request)
    }
}

fn unsupported_param(model: &str, param: &str, value: &Value, hint: &str) -> Error {
    Error::InvalidRequest(format!(
        "{model} does not support {param}={value}. {hint}To drop unsupported params, set `litellm.drop_params = True`."
    ))
}

/// Python's `_maybe_drop_speed_param` and `_apply_sampling_param` gates: a parameter the
/// model removed is dropped under `drop_params`, else rejected before the call.
fn drop_unsupported_params(
    request: AnthropicMessagesRequest,
    context: &MessagesTransformContext,
) -> Result<AnthropicMessagesRequest, Error> {
    let capabilities = &context.thinking.capabilities;
    let model = request.model.clone();
    let reject = |param: &str, value: Value, hint: &str| -> Result<(), Error> {
        if context.drop_params {
            return Ok(());
        }
        Err(unsupported_param(&model, param, &value, hint))
    };
    let speed = match request.speed.as_deref() {
        Some(speed) if !capabilities.supports_speed => {
            reject("speed", json!(speed), "")?;
            None
        }
        _ => request.speed.clone(),
    };
    if capabilities.supports_sampling_params {
        return Ok(AnthropicMessagesRequest { speed, ..request });
    }
    let temperature = match request.temperature {
        Some(temperature) if temperature != 1.0 => {
            reject(
                "temperature",
                json!(temperature),
                "Only temperature=1 is supported. ",
            )?;
            None
        }
        temperature => temperature,
    };
    if let Some(top_p) = request.top_p {
        reject("top_p", json!(top_p), "")?;
    }
    if let Some(top_k) = request.top_k {
        reject("top_k", json!(top_k), "")?;
    }
    Ok(AnthropicMessagesRequest {
        speed,
        temperature,
        top_p: None,
        top_k: None,
        ..request
    })
}

/// OpenAI's `[{"type": "compaction", "compact_threshold": N}]` list becomes Anthropic's
/// `{"edits": [{"type": "compact_20260112", "trigger": {...}}]}`. An Anthropic-shaped value
/// is returned as is; anything else yields `None`.
pub fn map_openai_context_management_to_anthropic(context_management: &Value) -> Option<Value> {
    match context_management {
        Value::Object(edits) if edits.contains_key("edits") => Some(context_management.clone()),
        Value::Array(entries) => {
            let edits: Vec<Value> = entries
                .iter()
                .filter_map(Value::as_object)
                .filter(|entry| entry.get("type").and_then(Value::as_str) == Some("compaction"))
                .map(|entry| {
                    let trigger = entry.get("compact_threshold").and_then(Value::as_f64).map(
                        |threshold| json!({"type": "input_tokens", "value": threshold as i64}),
                    );
                    let passthrough = entry
                        .iter()
                        .filter(|(key, _)| !matches!(key.as_str(), "type" | "compact_threshold"))
                        .map(|(key, value)| (key.clone(), value.clone()));
                    Value::Object(
                        [("type".to_string(), json!("compact_20260112"))]
                            .into_iter()
                            .chain(trigger.map(|trigger| ("trigger".to_string(), trigger)))
                            .chain(passthrough)
                            .collect::<Map<String, Value>>(),
                    )
                })
                .collect();
            (!edits.is_empty()).then(|| json!({"edits": edits}))
        }
        _ => None,
    }
}

pub fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

pub fn resolve_anthropic_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, litellm_auth::Error> {
    non_empty(api_key)
        .map(str::to_string)
        .or_else(|| env_lookup(ANTHROPIC_API_KEY_ENV).filter(|value| !value.trim().is_empty()))
        .ok_or(litellm_auth::Error::MissingApiKey {
            provider: "Anthropic",
            environment_variable: ANTHROPIC_API_KEY_ENV,
        })
}

pub fn complete_anthropic_url(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    let api_base = resolve_anthropic_api_base(api_base, env_lookup);

    let api_base = api_base.trim_end_matches('/');
    if api_base.ends_with(MESSAGES_PATH_SUFFIX) {
        return api_base.to_string();
    }
    format!("{api_base}{MESSAGES_PATH_SUFFIX}")
}

/// `api_base`, else `ANTHROPIC_API_BASE`, else the SDK's `ANTHROPIC_BASE_URL`, else the
/// public endpoint.
pub fn resolve_anthropic_api_base(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    let env = |name: &str| env_lookup(name).filter(|value| !value.trim().is_empty());
    non_empty(api_base)
        .map(str::to_string)
        .or_else(|| env(ANTHROPIC_API_BASE_ENV))
        .or_else(|| env(ANTHROPIC_BASE_URL_ENV))
        .unwrap_or_else(|| DEFAULT_ANTHROPIC_API_BASE.to_string())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn request(fields: Value) -> AnthropicMessagesRequest {
        let mut body = json!({
            "model": "claude",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}]
        });
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        serde_json::from_value(body).unwrap()
    }

    fn context(
        capabilities: AnthropicModelCapabilities,
        drop_params: bool,
    ) -> MessagesTransformContext {
        MessagesTransformContext::with_lookup(capabilities, drop_params, &|_: &str| None)
    }

    fn transform(
        fields: Value,
        capabilities: AnthropicModelCapabilities,
        drop_params: bool,
    ) -> Result<Value, Error> {
        ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(
                request(fields),
                &context(capabilities, drop_params),
            )
            .map(|transformed| serde_json::to_value(transformed).unwrap())
    }

    #[test]
    fn url_defaults_to_public_anthropic_endpoint() {
        assert_eq!(
            complete_anthropic_url(None, &|_| None),
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[test]
    fn url_appends_messages_suffix_to_custom_base() {
        assert_eq!(
            complete_anthropic_url(Some("https://proxy.internal"), &|_| None),
            "https://proxy.internal/v1/messages"
        );
    }

    #[test]
    fn url_leaves_complete_messages_endpoint_untouched() {
        assert_eq!(
            complete_anthropic_url(Some("https://proxy.internal/v1/messages"), &|_| None),
            "https://proxy.internal/v1/messages"
        );
    }

    #[rstest]
    #[case(ANTHROPIC_API_BASE_ENV)]
    #[case(ANTHROPIC_BASE_URL_ENV)]
    fn url_falls_back_to_either_env_base(#[case] variable: &str) {
        let with_env = |key: &str| (key == variable).then(|| "https://env.anthropic".to_string());
        assert_eq!(
            complete_anthropic_url(Some("  "), &with_env),
            "https://env.anthropic/v1/messages"
        );
    }

    #[test]
    fn api_key_prefers_param_then_env_then_errors() {
        assert_eq!(
            resolve_anthropic_api_key(Some("sk-param"), &|_| None).unwrap(),
            "sk-param"
        );
        let with_env = |key: &str| (key == ANTHROPIC_API_KEY_ENV).then(|| "sk-env".to_string());
        assert_eq!(
            resolve_anthropic_api_key(Some("  "), &with_env).unwrap(),
            "sk-env"
        );
        assert_eq!(
            resolve_anthropic_api_key(None, &|_| None)
                .expect_err("missing key")
                .to_string(),
            "Missing Anthropic API Key - Set `api_key` or the ANTHROPIC_API_KEY environment variable"
        );
    }

    #[test]
    fn auth_strategy_and_default_headers_match_anthropic() {
        assert_eq!(
            ANTHROPIC_MESSAGES_CONFIG.auth_strategy().header_name(),
            "x-api-key"
        );
        assert_eq!(
            ANTHROPIC_MESSAGES_CONFIG.default_headers(),
            &[
                ("anthropic-version", "2023-06-01"),
                ("content-type", "application/json"),
            ]
        );
    }

    #[test]
    fn max_tokens_is_required() {
        let mut body = request(json!({}));
        body.max_tokens = None;
        let error = ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(
                body,
                &context(AnthropicModelCapabilities::default(), false),
            )
            .expect_err("rejected");
        assert!(
            matches!(error, Error::InvalidRequest(message) if message.contains("max_tokens is required"))
        );
    }

    #[test]
    fn advisor_history_is_stripped_only_when_the_advisor_tool_is_absent() {
        let history = json!([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "adv", "name": "advisor", "input": {}},
                {"type": "advisor_tool_result", "tool_use_id": "adv", "content": "x"},
                {"type": "text", "text": "a"}
            ]}
        ]);
        let stripped = transform(
            json!({"messages": history}),
            AnthropicModelCapabilities::default(),
            false,
        )
        .unwrap();
        assert_eq!(
            stripped["messages"][1]["content"],
            json!([{"type": "text", "text": "a"}])
        );
        let kept = transform(
            json!({"messages": history, "tools": [{"type": "advisor_20260301", "name": "advisor"}]}),
            AnthropicModelCapabilities::default(),
            false,
        )
        .unwrap();
        assert_eq!(kept["messages"][1]["content"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn encrypted_reasoning_blocks_are_stripped_from_the_wire() {
        let sent = transform(
            json!({"messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "t", "signature": "litellm_encrypted_reasoning:abc"},
                    {"type": "text", "text": "a"}
                ]}
            ]}),
            AnthropicModelCapabilities::default(),
            false,
        )
        .unwrap();
        assert_eq!(
            sent["messages"][1]["content"],
            json!([{"type": "text", "text": "a"}])
        );
    }

    #[test]
    fn openai_context_management_list_becomes_anthropic_edits() {
        let sent = transform(
            json!({"context_management": [
                {"type": "compaction", "compact_threshold": 150000, "note": "keep"},
                {"type": "other"}
            ]}),
            AnthropicModelCapabilities::default(),
            false,
        )
        .unwrap();
        assert_eq!(
            sent["context_management"],
            json!({"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150000}, "note": "keep"}]})
        );
        let native = json!({"edits": [{"type": "clear_tool_uses_20250919"}]});
        let untouched = transform(
            json!({"context_management": native}),
            AnthropicModelCapabilities::default(),
            false,
        )
        .unwrap();
        assert_eq!(untouched["context_management"], native);
        assert_eq!(
            map_openai_context_management_to_anthropic(&json!([{"type": "other"}])),
            None
        );
    }

    fn no_sampling() -> AnthropicModelCapabilities {
        AnthropicModelCapabilities {
            supports_sampling_params: false,
            ..Default::default()
        }
    }

    #[test]
    fn removed_sampling_params_are_dropped_under_drop_params_and_rejected_otherwise() {
        let dropped = transform(
            json!({"temperature": 0.2, "top_p": 0.9, "top_k": 5}),
            no_sampling(),
            true,
        )
        .unwrap();
        for param in ["temperature", "top_p", "top_k"] {
            assert!(dropped.get(param).is_none(), "{param} should be dropped");
        }
        let kept_unit_temperature =
            transform(json!({"temperature": 1.0}), no_sampling(), false).unwrap();
        assert_eq!(kept_unit_temperature["temperature"], json!(1.0));
        let rejected = transform(json!({"top_k": 5}), no_sampling(), false).expect_err("rejected");
        assert!(
            matches!(rejected, Error::InvalidRequest(message) if message.contains("does not support top_k=5"))
        );
        let supported = transform(
            json!({"temperature": 0.2, "top_p": 0.9, "top_k": 5}),
            AnthropicModelCapabilities::default(),
            false,
        )
        .unwrap();
        assert_eq!(supported["top_k"], json!(5));
    }

    #[test]
    fn speed_is_gated_on_the_model_flag() {
        let dropped = transform(
            json!({"speed": "fast"}),
            AnthropicModelCapabilities::default(),
            true,
        )
        .unwrap();
        assert!(dropped.get("speed").is_none());
        let rejected = transform(
            json!({"speed": "fast"}),
            AnthropicModelCapabilities::default(),
            false,
        )
        .expect_err("rejected");
        assert!(
            matches!(rejected, Error::InvalidRequest(message) if message.contains("does not support speed="))
        );
        let kept = transform(
            json!({"speed": "fast"}),
            AnthropicModelCapabilities {
                supports_speed: true,
                ..Default::default()
            },
            false,
        )
        .unwrap();
        assert_eq!(kept["speed"], json!("fast"));
    }
}
