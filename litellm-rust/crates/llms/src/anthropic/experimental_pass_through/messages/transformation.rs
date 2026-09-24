use litellm_core_utils::settings::{Lookup, ProcessEnvironment};
use litellm_types::llms::anthropic_messages::anthropic_request::AnthropicMessagesRequest;
use serde_json::{Map, Value, json};

use super::{
    headers::{authenticate, with_feature_betas},
    thinking::{ThinkingBudgets, ThinkingContext, translate_thinking},
};
use crate::{
    anthropic::common_utils::{
        AnthropicModelCapabilities, has_advisor_tool, strip_advisor_blocks,
        strip_encrypted_reasoning_blocks,
    },
    base_llm::{
        anthropic_messages::transformation::{
            BaseAnthropicMessagesConfig, Headers, MessagesTransformContext,
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

fn unsupported_param(model: &str, param: &str, value: &str, hint: &str) -> Error {
    Error::InvalidRequest(format!(
        "{model} does not support {param}={value}. {hint}To drop unsupported params, set `litellm.drop_params = True`."
    ))
}

fn drop_unsupported_params(
    request: AnthropicMessagesRequest,
    context: &MessagesTransformContext,
) -> Result<AnthropicMessagesRequest, Error> {
    let capabilities = &context.thinking.capabilities;
    let model = request.model.clone();
    let reject = |param: &str, value: String, hint: &str| -> Result<(), Error> {
        if context.drop_params {
            return Ok(());
        }
        Err(unsupported_param(&model, param, &value, hint))
    };
    let speed = match request.speed.as_deref() {
        Some(speed) if !capabilities.supports_speed => {
            reject("speed", format!("'{speed}'"), "")?;
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
                json!(temperature).to_string(),
                "Only temperature=1 is supported. ",
            )?;
            None
        }
        temperature => temperature,
    };
    if let Some(top_p) = request.top_p {
        reject("top_p", json!(top_p).to_string(), "")?;
    }
    if let Some(top_k) = request.top_k {
        reject("top_k", json!(top_k).to_string(), "")?;
    }
    Ok(AnthropicMessagesRequest {
        speed,
        temperature,
        top_p: None,
        top_k: None,
        ..request
    })
}

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
    use std::process::Command;

    use rstest::{fixture, rstest};

    use super::*;
    use crate::anthropic::common_utils::{ENCRYPTED_REASONING_SIGNATURE_PREFIX, beta};

    type Env = &'static [(&'static str, &'static str)];

    const BOTH_BASE_ENVS: Env = &[
        (ANTHROPIC_API_BASE_ENV, "https://api-base.example.com"),
        (ANTHROPIC_BASE_URL_ENV, "https://base-url.example.com"),
    ];
    const API_KEY_ENV: Env = &[(ANTHROPIC_API_KEY_ENV, "sk-env")];
    const MISSING_API_KEY: &str =
        "Missing Anthropic API Key - Set `api_key` or the ANTHROPIC_API_KEY environment variable";
    const LOW_BUDGET_ENV: &str = "DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET";
    const PROCESS_ENV_PROBE: &str = "LITELLM_MESSAGES_TRANSFORM_CONTEXT_PROBE";

    fn merged(base: Value, fields: Value) -> Value {
        Value::Object(
            base.as_object()
                .unwrap()
                .clone()
                .into_iter()
                .chain(fields.as_object().unwrap().clone())
                .collect(),
        )
    }

    fn body(fields: Value) -> Value {
        merged(
            json!({
                "model": "claude",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }),
            fields,
        )
    }

    fn request(fields: Value) -> AnthropicMessagesRequest {
        serde_json::from_value(body(fields)).unwrap()
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn env(vars: Env) -> impl Fn(&str) -> Option<String> {
        move |name| {
            vars.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    fn headers(pairs: &[(&str, &str)]) -> Headers {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    fn transform(
        fields: Value,
        capabilities: AnthropicModelCapabilities,
        drop_params: bool,
    ) -> Result<Value, Error> {
        ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(
                request(fields),
                &MessagesTransformContext::with_lookup(capabilities, drop_params, &no_env),
            )
            .map(|transformed| serde_json::to_value(transformed).unwrap())
    }

    fn invalid(message: &str) -> Result<Value, Error> {
        Err(Error::InvalidRequest(message.to_string()))
    }

    fn advisor_history() -> Value {
        json!([
            {"role": "user", "content": "Build a worker pool."},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me consult the advisor."},
                {"type": "server_tool_use", "id": "srvtoolu_abc123", "name": "advisor", "input": {}},
                {"type": "advisor_tool_result", "tool_use_id": "srvtoolu_abc123", "content": {"type": "advisor_result", "text": "Use channels."}},
                {"type": "text", "text": "Here is the implementation."}
            ]}
        ])
    }

    #[fixture]
    fn unmapped() -> AnthropicModelCapabilities {
        AnthropicModelCapabilities::default()
    }

    #[fixture]
    fn sampling_removed() -> AnthropicModelCapabilities {
        AnthropicModelCapabilities {
            supports_sampling_params: false,
            ..Default::default()
        }
    }

    #[fixture]
    fn fast_mode() -> AnthropicModelCapabilities {
        AnthropicModelCapabilities {
            supports_speed: true,
            ..Default::default()
        }
    }

    #[rstest]
    #[case::alone(json!({"max_tokens": null}))]
    #[case::ahead_of_the_param_gate(json!({"max_tokens": null, "speed": "fast"}))]
    fn missing_max_tokens_is_rejected(#[case] fields: Value, unmapped: AnthropicModelCapabilities) {
        assert_eq!(
            transform(fields, unmapped, false),
            invalid("max_tokens is required for Anthropic /v1/messages API")
        );
    }

    #[rstest]
    #[case::sampling_params_on_a_sampling_model(
        unmapped(),
        false,
        json!({"temperature": 0.3, "top_p": 0.9, "top_k": 40})
    )]
    #[case::sampling_params_on_a_sampling_model_under_drop_params(
        unmapped(),
        true,
        json!({"temperature": 0.3, "top_p": 0.9, "top_k": 40})
    )]
    #[case::unit_temperature_on_a_sampling_removed_model(
        sampling_removed(),
        false,
        json!({"temperature": 1.0})
    )]
    #[case::unit_temperature_on_a_sampling_removed_model_under_drop_params(
        sampling_removed(),
        true,
        json!({"temperature": 1.0})
    )]
    #[case::speed_on_a_fast_mode_model(fast_mode(), false, json!({"speed": "fast"}))]
    #[case::speed_on_a_fast_mode_model_under_drop_params(fast_mode(), true, json!({"speed": "fast"}))]
    #[case::native_context_management_edits(unmapped(), false, json!({"context_management": {"edits": [{
        "type": "clear_tool_uses_20250919",
        "trigger": {"type": "input_tokens", "value": 30000},
        "keep": {"type": "tool_uses", "value": 3},
        "clear_at_least": {"type": "input_tokens", "value": 5000},
        "exclude_tools": ["web_search"],
        "clear_tool_inputs": false
    }]}}))]
    #[case::first_party_billing_header_system_block(unmapped(), false, json!({"system": [
        {"type": "text", "text": "x-anthropic-billing-header: cc_version=1"},
        {"type": "text", "text": "real system prompt"}
    ]}))]
    #[case::anthropic_signed_reasoning_history(unmapped(), false, json!({"messages": [
        {"role": "user", "content": "Solve it."},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "plan", "signature": "EqQBCkYIAxgCIkA_anthropic_signed"},
            {"type": "redacted_thinking", "data": "EmwKAhgBEgy_anthropic_minted"},
            {"type": "text", "text": "The answer."}
        ]}
    ]}))]
    #[case::advisor_history_alongside_the_advisor_tool(unmapped(), false, json!({
        "messages": advisor_history(),
        "tools": [{"type": "advisor_20260301", "name": "advisor"}]
    }))]
    fn request_is_forwarded_unchanged(
        #[case] capabilities: AnthropicModelCapabilities,
        #[case] drop_params: bool,
        #[case] fields: Value,
    ) {
        assert_eq!(
            transform(fields.clone(), capabilities, drop_params),
            Ok(body(fields))
        );
    }

    #[rstest]
    #[case::temperature(sampling_removed(), json!({"temperature": 0.3}), json!({}))]
    #[case::top_p(sampling_removed(), json!({"top_p": 0.9}), json!({}))]
    #[case::top_k(sampling_removed(), json!({"top_k": 40}), json!({}))]
    #[case::every_sampling_param_keeping_the_rest(
        sampling_removed(),
        json!({"temperature": 0.3, "top_p": 0.9, "top_k": 40, "stream": true}),
        json!({"stream": true})
    )]
    #[case::speed_on_a_sampling_model(
        unmapped(),
        json!({"speed": "fast", "temperature": 0.5}),
        json!({"temperature": 0.5})
    )]
    #[case::speed_on_a_sampling_removed_model(
        sampling_removed(),
        json!({"speed": "fast", "temperature": 1.0}),
        json!({"temperature": 1.0})
    )]
    fn removed_params_are_dropped_under_drop_params(
        #[case] capabilities: AnthropicModelCapabilities,
        #[case] fields: Value,
        #[case] expected: Value,
    ) {
        assert_eq!(transform(fields, capabilities, true), Ok(body(expected)));
    }

    #[rstest]
    #[case::temperature(
        sampling_removed(),
        json!({"temperature": 0.3}),
        "claude does not support temperature=0.3. Only temperature=1 is supported. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::temperature_just_below_one(
        sampling_removed(),
        json!({"temperature": 0.99}),
        "claude does not support temperature=0.99. Only temperature=1 is supported. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::whole_number_temperature_keeps_its_decimal(
        sampling_removed(),
        json!({"temperature": 2.0}),
        "claude does not support temperature=2.0. Only temperature=1 is supported. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::top_p(
        sampling_removed(),
        json!({"top_p": 0.9}),
        "claude does not support top_p=0.9. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::top_k(
        sampling_removed(),
        json!({"top_k": 5}),
        "claude does not support top_k=5. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::top_k_next_to_unit_temperature(
        sampling_removed(),
        json!({"temperature": 1.0, "top_k": 5}),
        "claude does not support top_k=5. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::temperature_ahead_of_top_k(
        sampling_removed(),
        json!({"temperature": 0.5, "top_k": 5}),
        "claude does not support temperature=0.5. Only temperature=1 is supported. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::top_p_ahead_of_top_k(
        sampling_removed(),
        json!({"top_p": 0.9, "top_k": 5}),
        "claude does not support top_p=0.9. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::speed(
        unmapped(),
        json!({"speed": "fast"}),
        "claude does not support speed='fast'. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::speed_ahead_of_sampling_params(
        sampling_removed(),
        json!({"speed": "fast", "temperature": 0.5}),
        "claude does not support speed='fast'. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    fn removed_params_are_rejected_without_drop_params(
        #[case] capabilities: AnthropicModelCapabilities,
        #[case] fields: Value,
        #[case] message: &str,
    ) {
        assert_eq!(transform(fields, capabilities, false), invalid(message));
    }

    #[rstest]
    #[case::compaction_threshold(
        json!([{"type": "compaction", "compact_threshold": 200000}]),
        Some(json!({"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 200000}}]}))
    )]
    #[case::other_keys_pass_through(
        json!([{"type": "compaction", "compact_threshold": 150000, "instructions": "Focus on preserving code snippets"}]),
        Some(json!({"edits": [{
            "type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": 150000},
            "instructions": "Focus on preserving code snippets"
        }]}))
    )]
    #[case::float_threshold_is_truncated(
        json!([{"type": "compaction", "compact_threshold": 150000.9}]),
        Some(json!({"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150000}}]}))
    )]
    #[case::compaction_without_threshold(
        json!([{"type": "compaction"}]),
        Some(json!({"edits": [{"type": "compact_20260112"}]}))
    )]
    #[case::non_numeric_threshold_is_dropped(
        json!([{"type": "compaction", "compact_threshold": "150000"}]),
        Some(json!({"edits": [{"type": "compact_20260112"}]}))
    )]
    #[case::non_object_entries_are_skipped(
        json!([42, "compaction", null, [], {"type": "compaction", "compact_threshold": 1000}]),
        Some(json!({"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 1000}}]}))
    )]
    #[case::only_compaction_entries_are_mapped_in_order(
        json!([
            {"type": "compaction", "compact_threshold": 1000},
            {"type": "other", "compact_threshold": 5},
            {"type": "compaction", "instructions": "second"}
        ]),
        Some(json!({"edits": [
            {"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 1000}},
            {"type": "compact_20260112", "instructions": "second"}
        ]}))
    )]
    #[case::list_without_compaction(json!([{"type": "other"}]), None)]
    #[case::empty_list(json!([]), None)]
    #[case::anthropic_edits_pass_through(
        json!({"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150000}}]}),
        Some(json!({"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150000}}]}))
    )]
    #[case::object_without_edits(json!({"type": "compaction"}), None)]
    #[case::scalar(json!("compaction"), None)]
    fn openai_context_management_maps_to_anthropic_edits(
        #[case] context_management: Value,
        #[case] expected: Option<Value>,
    ) {
        assert_eq!(
            map_openai_context_management_to_anthropic(&context_management),
            expected
        );
    }

    #[rstest]
    #[case::openai_list_is_mapped(
        json!([{"type": "compaction", "compact_threshold": 200000}]),
        json!({"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 200000}}]})
    )]
    #[case::unmappable_list_is_kept(json!([{"type": "other"}]), json!([{"type": "other"}]))]
    #[case::unmappable_object_is_kept(json!({"type": "other"}), json!({"type": "other"}))]
    fn context_management_reaches_the_wire(
        #[case] context_management: Value,
        #[case] expected: Value,
        unmapped: AnthropicModelCapabilities,
    ) {
        assert_eq!(
            transform(
                json!({"context_management": context_management}),
                unmapped,
                false
            ),
            Ok(body(json!({"context_management": expected})))
        );
    }

    #[rstest]
    #[case::without_tools(json!({}))]
    #[case::with_only_other_tools(json!({"tools": [{"name": "get_weather", "input_schema": {"type": "object"}}]}))]
    fn advisor_history_is_stripped_without_the_advisor_tool(
        #[case] tools: Value,
        unmapped: AnthropicModelCapabilities,
    ) {
        let stripped = json!([
            {"role": "user", "content": "Build a worker pool."},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me consult the advisor."},
                {"type": "text", "text": "Here is the implementation."}
            ]}
        ]);
        assert_eq!(
            transform(
                merged(tools.clone(), json!({"messages": advisor_history()})),
                unmapped,
                false
            ),
            Ok(body(merged(tools, json!({"messages": stripped}))))
        );
    }

    #[rstest]
    fn bridge_minted_reasoning_is_stripped_from_the_wire(unmapped: AnthropicModelCapabilities) {
        let messages = json!([
            {"role": "user", "content": "Solve it."},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "plan", "signature": format!("{ENCRYPTED_REASONING_SIGNATURE_PREFIX}gAAAA_1")},
                {"type": "redacted_thinking", "data": format!("{ENCRYPTED_REASONING_SIGNATURE_PREFIX}gAAAA_2")},
                {"type": "text", "text": "The answer."}
            ]},
            {"role": "user", "content": "And the next one?"}
        ]);
        assert_eq!(
            transform(json!({"messages": messages}), unmapped, false),
            Ok(body(json!({"messages": [
                {"role": "user", "content": "Solve it."},
                {"role": "assistant", "content": [{"type": "text", "text": "The answer."}]},
                {"role": "user", "content": "And the next one?"}
            ]})))
        );
    }

    #[test]
    fn thinking_is_translated_with_the_context_budgets() {
        let context = MessagesTransformContext::with_lookup(
            AnthropicModelCapabilities {
                supports_reasoning: true,
                ..Default::default()
            },
            false,
            &env(&[(LOW_BUDGET_ENV, "2000")]),
        );
        let transformed = ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(
                request(json!({"max_tokens": 4096, "reasoning_effort": "low"})),
                &context,
            )
            .map(|transformed| serde_json::to_value(transformed).unwrap());
        assert_eq!(
            transformed,
            Ok(body(json!({
                "max_tokens": 4096,
                "thinking": {"type": "enabled", "budget_tokens": 2000}
            })))
        );
    }

    #[test]
    fn new_reads_thinking_budgets_from_the_process_environment() {
        if std::env::var_os(PROCESS_ENV_PROBE).is_some() {
            assert_eq!(
                MessagesTransformContext::new(sampling_removed(), true),
                MessagesTransformContext {
                    thinking: ThinkingContext {
                        capabilities: sampling_removed(),
                        budgets: ThinkingBudgets {
                            low: 2000,
                            ..ThinkingBudgets::default()
                        },
                    },
                    drop_params: true,
                }
            );
            return;
        }
        let (_, test_path) = concat!(
            module_path!(),
            "::new_reads_thinking_budgets_from_the_process_environment"
        )
        .split_once("::")
        .unwrap();
        let other_tiers = ["MINIMAL", "MEDIUM", "HIGH", "XHIGH", "MAX"]
            .map(|tier| format!("DEFAULT_REASONING_EFFORT_{tier}_THINKING_BUDGET"));
        let output = other_tiers
            .iter()
            .fold(
                Command::new(std::env::current_exe().unwrap()),
                |mut command, name| {
                    command.env_remove(name);
                    command
                },
            )
            .args([test_path, "--exact"])
            .env(PROCESS_ENV_PROBE, "1")
            .env(LOW_BUDGET_ENV, "2000")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "{stdout}{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[rstest]
    #[case::public_endpoint_by_default(None, &[], "https://api.anthropic.com")]
    #[case::explicit_api_base_beats_env(
        Some("https://explicit.example.com"),
        BOTH_BASE_ENVS,
        "https://explicit.example.com"
    )]
    #[case::explicit_api_base_is_trimmed(
        Some("  https://explicit.example.com  "),
        &[],
        "https://explicit.example.com"
    )]
    #[case::blank_api_base_falls_back_to_env(
        Some("  "),
        BOTH_BASE_ENVS,
        "https://api-base.example.com"
    )]
    #[case::api_base_env_beats_base_url_env(None, BOTH_BASE_ENVS, "https://api-base.example.com")]
    #[case::base_url_env_without_api_base_env(
        None,
        &[(ANTHROPIC_BASE_URL_ENV, "https://base-url.example.com")],
        "https://base-url.example.com"
    )]
    #[case::blank_api_base_env_falls_back_to_base_url_env(
        None,
        &[(ANTHROPIC_API_BASE_ENV, " \t "), (ANTHROPIC_BASE_URL_ENV, "https://base-url.example.com")],
        "https://base-url.example.com"
    )]
    #[case::blank_envs_fall_back_to_public_endpoint(
        None,
        &[(ANTHROPIC_API_BASE_ENV, ""), (ANTHROPIC_BASE_URL_ENV, "  ")],
        "https://api.anthropic.com"
    )]
    fn api_base_resolution(
        #[case] api_base: Option<&str>,
        #[case] vars: Env,
        #[case] expected: &str,
    ) {
        assert_eq!(resolve_anthropic_api_base(api_base, &env(vars)), expected);
    }

    #[rstest]
    #[case::public_endpoint(None, &[], "https://api.anthropic.com/v1/messages")]
    #[case::base_url_env(
        None,
        &[(ANTHROPIC_BASE_URL_ENV, "https://custom.example.com")],
        "https://custom.example.com/v1/messages"
    )]
    #[case::custom_base(Some("https://proxy.internal"), &[], "https://proxy.internal/v1/messages")]
    #[case::trailing_slash(Some("https://proxy.internal/"), &[], "https://proxy.internal/v1/messages")]
    #[case::complete_endpoint(
        Some("https://proxy.internal/v1/messages"),
        &[],
        "https://proxy.internal/v1/messages"
    )]
    #[case::complete_endpoint_with_trailing_slash(
        Some("https://proxy.internal/v1/messages/"),
        &[],
        "https://proxy.internal/v1/messages"
    )]
    fn complete_url_ends_in_the_messages_path(
        #[case] api_base: Option<&str>,
        #[case] vars: Env,
        #[case] expected: &str,
    ) {
        assert_eq!(
            ANTHROPIC_MESSAGES_CONFIG.get_complete_url(api_base, "claude", &env(vars)),
            Ok(expected.to_string())
        );
    }

    #[rstest]
    #[case::param_beats_env(Some("sk-param"), API_KEY_ENV, Ok("sk-param"))]
    #[case::param_is_trimmed(Some(" sk-param "), &[], Ok("sk-param"))]
    #[case::blank_param_falls_back_to_env(Some("  "), API_KEY_ENV, Ok("sk-env"))]
    #[case::env_without_param(None, API_KEY_ENV, Ok("sk-env"))]
    #[case::blank_env_is_missing(None, &[(ANTHROPIC_API_KEY_ENV, " ")], Err(MISSING_API_KEY))]
    #[case::nothing_is_missing(None, &[], Err(MISSING_API_KEY))]
    fn api_key_resolution(
        #[case] api_key: Option<&str>,
        #[case] vars: Env,
        #[case] expected: Result<&str, &str>,
    ) {
        assert_eq!(
            resolve_anthropic_api_key(api_key, &env(vars)).map_err(|error| error.to_string()),
            expected.map(str::to_string).map_err(str::to_string)
        );
    }

    #[test]
    fn config_reports_a_missing_key_as_an_auth_error() {
        assert_eq!(
            ANTHROPIC_MESSAGES_CONFIG.resolve_api_key(None, &no_env),
            Err(Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "Anthropic",
                environment_variable: ANTHROPIC_API_KEY_ENV,
            }))
        );
    }

    #[test]
    fn config_authenticates_with_the_anthropic_auth_token() {
        assert_eq!(
            ANTHROPIC_MESSAGES_CONFIG.authenticate(
                vec![],
                None,
                &env(&[("ANTHROPIC_AUTH_TOKEN", "auth-token")])
            ),
            Ok(headers(&[("authorization", "Bearer auth-token")]))
        );
    }

    #[test]
    fn config_requests_the_betas_the_request_features_need() {
        assert_eq!(
            ANTHROPIC_MESSAGES_CONFIG.request_headers(
                headers(&[("x-api-key", "sk")]),
                &request(json!({"speed": "fast"}))
            ),
            headers(&[
                ("x-api-key", "sk"),
                ("anthropic-beta", beta::FAST_MODE_2026_02_01)
            ])
        );
    }

    #[rstest]
    #[case::absent(None, None)]
    #[case::blank(Some(" \t "), None)]
    #[case::padded(Some("  value "), Some("value"))]
    fn non_empty_trims_and_drops_blank_values(
        #[case] value: Option<&str>,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(non_empty(value), expected);
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
}
