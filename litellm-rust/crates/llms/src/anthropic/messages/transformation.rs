use litellm_auth::CredentialPlacement;
use litellm_core_utils::settings::{Lookup, ProcessEnvironment};
use litellm_types::{
    llms::{
        anthropic::{AnthropicBeta, BetaSet},
        anthropic_messages::anthropic_request::{
            AnthropicMessage, AnthropicMessagesOptionalParams, AnthropicMessagesRequest,
            ContextEdit, ContextManagement, Speed,
        },
    },
    recognized::Recognized,
};
use serde_json::{Map, Value, json};

use super::thinking::{ThinkingBudgets, ThinkingContext, translate_thinking};
use crate::{
    Error,
    anthropic::common_utils::{
        ANTHROPIC_API_BASE_ENV, ANTHROPIC_API_KEY_ENV, ANTHROPIC_AUTH_TOKEN_ENV,
        ANTHROPIC_BASE_URL_ENV, AnthropicModelCapabilities, OauthHandling, complete_anthropic_url,
        get_auth_header, has_advisor_tool, has_anthropic_credential, is_tool_search_used,
        merge_beta_headers, optionally_handle_anthropic_oauth, requires_native_compaction_beta,
        strip_advisor_blocks, strip_encrypted_reasoning_blocks,
    },
    base_llm::{
        anthropic_messages::transformation::{
            BaseAnthropicMessagesConfig, Headers, MessagesTransformContext, ValidatedEnvironment,
        },
        auth::AuthScheme,
    },
};

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
        if request.params.max_tokens.is_none() {
            return Err(Error::MissingField("max_tokens"));
        }
        let request = drop_unsupported_params(request, context)?;
        let request = translate_thinking(request, &context.thinking)?;
        let context_management = request
            .params
            .context_management
            .clone()
            .map(map_openai_context_management_to_anthropic);
        let messages = if has_advisor_tool(request.params.tools.as_deref()) {
            request.messages
        } else {
            strip_advisor_blocks(request.messages)
        };
        Ok(AnthropicMessagesRequest {
            messages: strip_encrypted_reasoning_blocks(messages),
            params: AnthropicMessagesOptionalParams {
                context_management,
                ..request.params
            },
            ..request
        })
    }

    fn secret_names(&self) -> &'static [&'static str] {
        &[
            ANTHROPIC_API_KEY_ENV,
            ANTHROPIC_AUTH_TOKEN_ENV,
            ANTHROPIC_API_BASE_ENV,
            ANTHROPIC_BASE_URL_ENV,
        ]
    }

    /// Python's `validate_anthropic_messages_environment` up to the beta merge, which
    /// `request_headers` does once the request is final.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let headers = match optionally_handle_anthropic_oauth(headers, api_key) {
            OauthHandling::Bearer { headers, token } => {
                return Ok(ValidatedEnvironment {
                    headers,
                    auth: AuthScheme::Credential {
                        placement: CredentialPlacement::Bearer,
                        secret: token,
                    },
                });
            }
            OauthHandling::Untouched(headers) => headers,
        };
        if has_anthropic_credential(&headers) {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            });
        }
        let auth = get_auth_header(api_key, env_lookup).ok_or(Error::Auth(
            litellm_auth::Error::MissingApiKey {
                provider: "Anthropic",
                environment_variable: ANTHROPIC_API_KEY_ENV,
            },
        ))?;
        Ok(ValidatedEnvironment { headers, auth })
    }

    fn request_headers(&self, headers: Headers, request: &AnthropicMessagesRequest) -> Headers {
        update_headers_with_anthropic_beta(headers, request)
    }
}

fn update_headers_with_anthropic_beta(
    headers: Headers,
    request: &AnthropicMessagesRequest,
) -> Headers {
    merge_beta_headers(headers, feature_betas(request))
}

fn feature_betas(request: &AnthropicMessagesRequest) -> BetaSet {
    let params = &request.params;
    let tools = params.tools.as_deref();
    [
        requires_native_compaction_beta(params.compaction.as_ref(), &request.messages)
            .then_some(AnthropicBeta::Compact20260904),
        uses_structured_output(params).then_some(AnthropicBeta::StructuredOutputs20251113),
        (params.speed == Some(Recognized::Known(Speed::Fast)))
            .then_some(AnthropicBeta::FastMode20260201),
        messages_carry_output_config(&request.messages)
            .then_some(AnthropicBeta::PerTurnControl20260701),
        has_advisor_tool(tools).then_some(AnthropicBeta::AdvisorTool20260301),
        is_tool_search_used(tools).then_some(AnthropicBeta::AdvancedToolUse20251120),
    ]
    .into_iter()
    .flatten()
    .chain(context_management_betas(params.context_management.as_ref()))
    .collect()
}

fn is_compact_edit(edit: &Recognized<ContextEdit>) -> bool {
    matches!(edit, Recognized::Known(ContextEdit::Compact { .. }))
}

fn context_management_betas(
    context_management: Option<&Recognized<ContextManagement>>,
) -> impl Iterator<Item = AnthropicBeta> {
    let edits = context_management
        .and_then(Recognized::known)
        .and_then(|context_management| context_management.edits.as_deref())
        .unwrap_or_default();
    let compact = edits.iter().any(is_compact_edit);
    let other = edits.iter().any(|edit| !is_compact_edit(edit));
    compact
        .then_some(AnthropicBeta::Compact20260112)
        .into_iter()
        .chain(other.then_some(AnthropicBeta::ContextManagement20250627))
}

fn uses_structured_output(params: &AnthropicMessagesOptionalParams) -> bool {
    params.output_format.is_some()
        || params
            .output_config
            .as_ref()
            .and_then(Recognized::known)
            .is_some_and(|config| config.format.is_some())
}

fn messages_carry_output_config(messages: &[AnthropicMessage]) -> bool {
    messages
        .iter()
        .any(|message| message.extra.contains_key("output_config"))
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
    let params = request.params;
    let speed = match &params.speed {
        Some(speed) if !capabilities.supports_speed => {
            reject("speed", format!("'{}'", speed_text(speed)), "")?;
            None
        }
        _ => params.speed.clone(),
    };
    if capabilities.supports_sampling_params {
        return Ok(AnthropicMessagesRequest {
            params: AnthropicMessagesOptionalParams { speed, ..params },
            ..request
        });
    }
    let temperature = match params.temperature {
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
    if let Some(top_p) = params.top_p {
        reject("top_p", json!(top_p).to_string(), "")?;
    }
    if let Some(top_k) = params.top_k {
        reject("top_k", json!(top_k).to_string(), "")?;
    }
    Ok(AnthropicMessagesRequest {
        params: AnthropicMessagesOptionalParams {
            speed,
            temperature,
            top_p: None,
            top_k: None,
            ..params
        },
        ..request
    })
}

fn speed_text(speed: &Recognized<Speed>) -> String {
    match speed {
        Recognized::Known(speed) => speed.as_str().to_string(),
        Recognized::Unrecognized(Value::String(text)) => text.clone(),
        Recognized::Unrecognized(other) => other.to_string(),
    }
}

fn compact_edit_from_openai(entry: &Map<String, Value>) -> Option<ContextEdit> {
    if entry.get("type").and_then(Value::as_str) != Some("compaction") {
        return None;
    }
    let trigger = entry
        .get("compact_threshold")
        .and_then(Value::as_f64)
        .map(|threshold| json!({"type": "input_tokens", "value": threshold as i64}));
    let passthrough = entry
        .iter()
        .filter(|(key, _)| !matches!(key.as_str(), "type" | "compact_threshold"))
        .map(|(key, value)| (key.clone(), value.clone()));
    Some(ContextEdit::Compact {
        extra: trigger
            .map(|trigger| ("trigger".to_string(), trigger))
            .into_iter()
            .chain(passthrough)
            .collect(),
    })
}

/// An OpenAI-style `context_management` list becomes Anthropic `edits` when it holds
/// compaction entries. Anything else, native edits included, is sent as it came.
pub fn map_openai_context_management_to_anthropic(
    context_management: Recognized<ContextManagement>,
) -> Recognized<ContextManagement> {
    let Recognized::Unrecognized(Value::Array(entries)) = &context_management else {
        return context_management;
    };
    let edits: Vec<Recognized<ContextEdit>> = entries
        .iter()
        .filter_map(Value::as_object)
        .filter_map(compact_edit_from_openai)
        .map(Recognized::Known)
        .collect();
    if edits.is_empty() {
        return context_management;
    }
    Recognized::Known(ContextManagement {
        edits: Some(edits),
        extra: Map::new(),
    })
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use rstest::{fixture, rstest};

    use super::*;
    use crate::anthropic::common_utils::ENCRYPTED_REASONING_SIGNATURE_PREFIX;

    type Env = &'static [(&'static str, &'static str)];

    const OAUTH_TOKEN: &str = "sk-ant-oat01-token";
    const OAUTH_BEARER: &str = "Bearer sk-ant-oat01-token";
    const OAUTH_BETA: &str = "oauth-2025-04-20";
    const BROWSER_ACCESS: (&str, &str) = ("anthropic-dangerous-direct-browser-access", "true");
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
            Err(Error::MissingField("max_tokens"))
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
        None
    )]
    #[case::object_without_edits(json!({"type": "compaction"}), None)]
    #[case::scalar(json!("compaction"), None)]
    fn openai_context_management_maps_to_anthropic_edits(
        #[case] context_management: Value,
        #[case] mapped: Option<Value>,
    ) {
        let parsed: Recognized<ContextManagement> =
            serde_json::from_value(context_management.clone()).unwrap();
        assert_eq!(
            serde_json::to_value(map_openai_context_management_to_anthropic(parsed)).unwrap(),
            mapped.unwrap_or(context_management)
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

    fn betas(values: &[&str]) -> BetaSet {
        values.join(",").parse().unwrap()
    }

    fn validated(
        forwarded: &[(&str, &str)],
        api_key: Option<&str>,
        vars: Env,
    ) -> Result<ValidatedEnvironment, Error> {
        ANTHROPIC_MESSAGES_CONFIG.validate_environment(
            headers(forwarded),
            api_key,
            "claude",
            &env(vars),
        )
    }

    fn credential(auth: &AuthScheme) -> Option<(&'static str, &str)> {
        match auth {
            AuthScheme::Credential { placement, secret } => {
                Some((placement.header_name(), secret.expose()))
            }
            AuthScheme::Forwarded => None,
            other => panic!("unexpected auth scheme {other:?}"),
        }
    }

    #[rstest]
    #[case::forwarded_oauth_bearer(
        &[("anthropic-version", "2023-06-01"), ("X-Api-Key", "sk-caller"), ("Authorization", OAUTH_BEARER)],
        Some("sk-deployment"),
        &[("ANTHROPIC_API_KEY", "sk-env")],
        &[("anthropic-version", "2023-06-01"), ("anthropic-beta", OAUTH_BETA), BROWSER_ACCESS],
        Some(("Authorization", OAUTH_TOKEN)),
    )]
    #[case::oauth_api_key(
        &[("x-api-key", OAUTH_TOKEN), ("anthropic-beta", "web-search-2025-03-05")],
        Some(OAUTH_TOKEN),
        &[],
        &[("anthropic-beta", "oauth-2025-04-20,web-search-2025-03-05"), BROWSER_ACCESS],
        Some(("Authorization", OAUTH_TOKEN)),
    )]
    #[case::forwarded_x_api_key_is_kept_over_the_deployment_key(
        &[("X-API-KEY", "caller-key")],
        Some("sk-other"),
        &[("ANTHROPIC_API_KEY", "sk-env")],
        &[("X-API-KEY", "caller-key")],
        None,
    )]
    #[case::forwarded_non_oauth_bearer_is_kept(
        &[("Authorization", "Bearer some-proxy-token")],
        Some("sk-ant-api03-regular"),
        &[],
        &[("Authorization", "Bearer some-proxy-token")],
        None,
    )]
    #[case::oauth_token_without_the_bearer_scheme_is_kept(
        &[("authorization", OAUTH_TOKEN)],
        None,
        &[],
        &[("authorization", OAUTH_TOKEN)],
        None,
    )]
    #[case::api_key_param(
        &[("anthropic-beta", "web-search-2025-03-05")],
        Some("sk-param"),
        &[("ANTHROPIC_API_KEY", "sk-env"), ("ANTHROPIC_AUTH_TOKEN", "env-token")],
        &[("anthropic-beta", "web-search-2025-03-05")],
        Some(("x-api-key", "sk-param")),
    )]
    #[case::env_key_when_the_param_is_blank(
        &[],
        Some("  "),
        &[("ANTHROPIC_API_KEY", "sk-env"), ("ANTHROPIC_AUTH_TOKEN", "env-token")],
        &[],
        Some(("x-api-key", "sk-env")),
    )]
    #[case::auth_token_when_no_key_is_set(
        &[],
        None,
        &[("ANTHROPIC_API_KEY", " \t"), ("ANTHROPIC_AUTH_TOKEN", "env-token")],
        &[],
        Some(("Authorization", "env-token")),
    )]
    #[case::oauth_env_key_as_a_bearer(
        &[],
        None,
        &[("ANTHROPIC_API_KEY", "sk-ant-oat01-env")],
        &[],
        Some(("Authorization", "sk-ant-oat01-env")),
    )]
    fn validate_environment_shapes_the_headers_and_names_the_credential(
        #[case] forwarded: &[(&str, &str)],
        #[case] api_key: Option<&str>,
        #[case] vars: Env,
        #[case] expected_headers: &[(&str, &str)],
        #[case] expected_credential: Option<(&str, &str)>,
    ) {
        let environment = validated(forwarded, api_key, vars).unwrap();
        assert_eq!(environment.headers, headers(expected_headers));
        assert_eq!(credential(&environment.auth), expected_credential);
    }

    #[rstest]
    #[case::no_credentials(&[], None, &[])]
    #[case::empty_api_key(&[], Some(""), &[])]
    #[case::whitespace_only_env_values(&[], None, &[("ANTHROPIC_API_KEY", "  "), ("ANTHROPIC_AUTH_TOKEN", " \t")])]
    #[case::unrelated_forwarded_headers(&[("anthropic-beta", "web-search-2025-03-05")], None, &[])]
    fn missing_credentials_are_an_auth_error(
        #[case] forwarded: &[(&str, &str)],
        #[case] api_key: Option<&str>,
        #[case] vars: Env,
    ) {
        assert!(matches!(
            validated(forwarded, api_key, vars),
            Err(Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "Anthropic",
                environment_variable: "ANTHROPIC_API_KEY",
            }))
        ));
    }

    #[rstest]
    #[case::no_features(json!({}), &[])]
    #[case::output_format(json!({"output_format": {"type": "json_schema"}}), &["structured-outputs-2025-11-13"])]
    #[case::null_output_format(json!({"output_format": null}), &[])]
    #[case::output_config_format(
        json!({"output_config": {"format": {"type": "json_schema"}, "effort": "xhigh"}}),
        &["structured-outputs-2025-11-13"]
    )]
    #[case::null_output_config_format(json!({"output_config": {"format": null}}), &[])]
    #[case::top_level_output_config_without_format(json!({"output_config": {"effort": "high"}}), &[])]
    #[case::fast_speed(json!({"speed": "fast"}), &["fast-mode-2026-02-01"])]
    #[case::standard_speed(json!({"speed": "standard"}), &[])]
    #[case::unknown_speed(json!({"speed": "turbo"}), &[])]
    #[case::compaction_param(json!({"compaction": {"enabled": true}}), &["compact-2026-09-04"])]
    #[case::empty_compaction_param(json!({"compaction": {}}), &["compact-2026-09-04"])]
    #[case::signed_compaction_block_in_history(
        json!({"messages": [
            {"role": "assistant", "content": [{"type": "compaction", "content": "summary", "signature": "sig"}]},
            {"role": "user", "content": "Continue"},
        ]}),
        &["compact-2026-09-04"]
    )]
    #[case::unsigned_compaction_block_in_history(
        json!({"messages": [
            {"role": "assistant", "content": [{"type": "compaction", "content": "summary", "signature": ""}]},
            {"role": "user", "content": "Continue"},
        ]}),
        &[]
    )]
    #[case::advisor_tool(
        json!({"tools": [{"type": "advisor_20260301", "name": "advisor", "model": "claude-opus-4-6"}]}),
        &["advisor-tool-2026-03-01"]
    )]
    #[case::no_tools(json!({"tools": []}), &[])]
    #[case::regex_tool_search(
        json!({"tools": [{"type": "tool_search_tool_regex_20251119"}]}),
        &["advanced-tool-use-2025-11-20"]
    )]
    #[case::bm25_tool_search(
        json!({"tools": [{"type": "tool_search_tool_bm25_20251119"}]}),
        &["advanced-tool-use-2025-11-20"]
    )]
    #[case::unrelated_server_tool(json!({"tools": [{"type": "web_search_20250305", "name": "web_search"}]}), &[])]
    #[case::only_compact_edits(
        json!({"context_management": {"edits": [{"type": "compact_20260112"}]}}),
        &["compact-2026-01-12"]
    )]
    #[case::only_other_edits(
        json!({"context_management": {"edits": [{"type": "clear_tool_uses_20250919", "keep": {"type": "tool_uses", "value": 3}}]}}),
        &["context-management-2025-06-27"]
    )]
    #[case::compact_and_other_edits(
        json!({"context_management": {"edits": [{"type": "compact_20260112"}, {"type": "clear_tool_uses_20250919"}]}}),
        &["compact-2026-01-12", "context-management-2025-06-27"]
    )]
    #[case::edit_without_a_type(json!({"context_management": {"edits": [{}]}}), &["context-management-2025-06-27"])]
    #[case::unknown_edit_type(json!({"context_management": {"edits": [{"type": "future"}]}}), &["context-management-2025-06-27"])]
    #[case::empty_edits(json!({"context_management": {"edits": []}}), &[])]
    #[case::context_management_without_edits(json!({"context_management": {}}), &[])]
    #[case::unmapped_openai_context_management(json!({"context_management": [{"type": "other"}]}), &[])]
    #[case::per_message_output_config(
        json!({"messages": [{"role": "user", "content": "hi", "output_config": {"effort": "low"}}]}),
        &["per-turn-control-2026-07-01"]
    )]
    #[case::per_message_null_output_config(
        json!({"messages": [{"role": "user", "content": "hi", "output_config": null}]}),
        &["per-turn-control-2026-07-01"]
    )]
    fn feature_betas_follow_the_request(#[case] fields: Value, #[case] expected: &[&str]) {
        assert_eq!(feature_betas(&request(fields)), betas(expected));
    }

    #[rstest]
    #[case::no_betas(&[("x-api-key", "k"), ("anthropic-version", "2023-06-01")], json!({}), &[("x-api-key", "k"), ("anthropic-version", "2023-06-01")])]
    #[case::blank_beta_header(&[("Anthropic-Beta", " , "), ("x-api-key", "k")], json!({}), &[("Anthropic-Beta", " , "), ("x-api-key", "k")])]
    #[case::feature_beta_is_appended(
        &[("x-api-key", "k")],
        json!({"speed": "fast"}),
        &[("x-api-key", "k"), ("anthropic-beta", "fast-mode-2026-02-01")],
    )]
    #[case::existing_betas_are_normalized_without_features(
        &[("Anthropic-Beta", "web-search-2025-03-05, interleaved-thinking-2025-05-14 ,web-search-2025-03-05"), ("x-api-key", "k")],
        json!({}),
        &[("x-api-key", "k"), ("anthropic-beta", "interleaved-thinking-2025-05-14,web-search-2025-03-05")],
    )]
    #[case::existing_advisor_beta_is_kept_without_an_advisor_tool(
        &[("anthropic-beta", "advisor-tool-2026-03-01")],
        json!({"tools": []}),
        &[("anthropic-beta", "advisor-tool-2026-03-01")],
    )]
    #[case::feature_already_sent_is_not_duplicated(
        &[("anthropic-beta", "fast-mode-2026-02-01")],
        json!({"speed": "fast"}),
        &[("anthropic-beta", "fast-mode-2026-02-01")],
    )]
    #[case::differently_cased_beta_header_is_replaced_by_one_sorted_header(
        &[("Anthropic-Beta", "interleaved-thinking-2025-05-14")],
        json!({"messages": [{"role": "system", "content": "env", "output_config": {"effort": "low"}}]}),
        &[("anthropic-beta", "interleaved-thinking-2025-05-14,per-turn-control-2026-07-01")],
    )]
    #[case::every_beta_header_casing_is_unioned_into_one_header(
        &[("anthropic-beta", "interleaved-thinking-2025-05-14"), ("Anthropic-Beta", "web-search-2025-03-05")],
        json!({"speed": "fast"}),
        &[("anthropic-beta", "fast-mode-2026-02-01,interleaved-thinking-2025-05-14,web-search-2025-03-05")],
    )]
    #[case::unknown_client_betas_survive_alongside_the_added_one(
        &[("anthropic-beta", "claude-code-20250219,interleaved-thinking-2025-05-14,context-management-2025-06-27,per-turn-control-2026-07-01,effort-2025-11-24")],
        json!({"messages": [{"role": "user", "content": "hi", "output_config": {"effort": "low"}}]}),
        &[("anthropic-beta", "claude-code-20250219,context-management-2025-06-27,effort-2025-11-24,interleaved-thinking-2025-05-14,per-turn-control-2026-07-01")],
    )]
    fn request_headers_merge_the_feature_betas(
        #[case] input: &[(&str, &str)],
        #[case] fields: Value,
        #[case] expected: &[(&str, &str)],
    ) {
        assert_eq!(
            ANTHROPIC_MESSAGES_CONFIG.request_headers(headers(input), &request(fields)),
            headers(expected)
        );
    }

    #[test]
    fn every_feature_merges_with_the_oauth_beta_sorted() {
        let environment = validated(&[], Some(OAUTH_TOKEN), &[]).unwrap();
        let all_features = request(json!({
            "compaction": {"enabled": true},
            "output_format": {"type": "json_schema"},
            "speed": "fast",
            "tools": [{"type": "advisor_20260301"}, {"type": "tool_search_tool_bm25_20251119"}],
            "context_management": {"edits": [{"type": "compact_20260112"}, {"type": "clear_thinking_20251015"}]},
            "messages": [{"role": "user", "content": "hi", "output_config": {"effort": "low"}}],
        }));
        assert_eq!(
            ANTHROPIC_MESSAGES_CONFIG.request_headers(environment.headers, &all_features),
            headers(&[
                BROWSER_ACCESS,
                (
                    "anthropic-beta",
                    "advanced-tool-use-2025-11-20,advisor-tool-2026-03-01,compact-2026-01-12,compact-2026-09-04,context-management-2025-06-27,fast-mode-2026-02-01,oauth-2025-04-20,per-turn-control-2026-07-01,structured-outputs-2025-11-13"
                ),
            ])
        );
        assert_eq!(
            credential(&environment.auth),
            Some(("Authorization", OAUTH_TOKEN))
        );
    }

    #[test]
    fn default_headers_match_anthropic() {
        assert_eq!(
            ANTHROPIC_MESSAGES_CONFIG.default_headers(),
            &[
                ("anthropic-version", "2023-06-01"),
                ("content-type", "application/json"),
            ]
        );
    }

    #[test]
    fn secret_names_cover_every_credential_and_base_lookup() {
        let requested = std::cell::RefCell::new(Vec::<String>::new());
        let record = |name: &str| -> Option<String> {
            requested.borrow_mut().push(name.to_string());
            None
        };
        let _ = ANTHROPIC_MESSAGES_CONFIG.validate_environment(Vec::new(), None, "claude", &record);
        let _ = ANTHROPIC_MESSAGES_CONFIG.get_complete_url(None, "claude", &record);
        let requested = requested.into_inner();
        assert!(!requested.is_empty());
        let undeclared: Vec<&String> = requested
            .iter()
            .filter(|name| {
                !ANTHROPIC_MESSAGES_CONFIG
                    .secret_names()
                    .contains(&name.as_str())
            })
            .collect();
        assert_eq!(undeclared, Vec::<&String>::new());
    }
}
