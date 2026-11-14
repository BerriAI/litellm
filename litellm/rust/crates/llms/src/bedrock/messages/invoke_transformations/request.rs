use litellm_llms_types::{
    formats::messages::{Message, MessagesRequest},
    recognized::{Recognized, map_known_runs},
};
use serde_json::{Map, Value, json};

use crate::{
    Error, ErrorDetail,
    anthropic::{
        common_utils::supports_effort_param, messages::transformation::transform_messages_request,
    },
    base_llm::{
        auth::Headers,
        messages::{
            context::{MessagesModelCapabilities, MessagesTransformContext, ThinkingBudgets},
            mid_conversation_system::convert_mid_conversation_system_turns,
            normalization::normalize_system_role_messages,
        },
    },
};

use super::{
    betas::{
        BETA_GATE_EMPTY, bedrock_invoke_betas, strip_beta_gated_request,
        strip_beta_gated_system_messages,
    },
    structured_output::apply_structured_output,
};

pub(super) type Body = Map<String, Value>;

pub const DEFAULT_BEDROCK_ANTHROPIC_API_VERSION: &str = "bedrock-2023-05-31";

const ALLOWED_TOP_LEVEL_FIELDS: [&str; 17] = [
    "anthropic_version",
    "max_tokens",
    "messages",
    "anthropic_beta",
    "system",
    "stop_sequences",
    "temperature",
    "top_p",
    "top_k",
    "tools",
    "tool_choice",
    "thinking",
    "metadata",
    "output_config",
    "safeguards",
    "context_management",
    "compaction",
];

const ADAPTIVE_THINKING_KEYS: [&str; 3] = ["type", "display", "block_binding"];

const EXTENDED_THINKING_PATTERNS: [&str; 12] = [
    "opus-4.5", "opus_4.5", "opus-4-5", "opus_4_5", "opus-4.1", "opus_4.1", "opus-4-1", "opus_4_1",
    "opus-4", "opus_4", "sonnet-4", "sonnet_4",
];

const CLEAR_THINKING_EDIT: &str = "clear_thinking_20251015";

const UNNAMED_TOOL_PREFIX: &str = "litellm_unnamed_tool_";

pub(super) fn effort_rank(effort: &str) -> Option<u8> {
    match effort {
        "low" => Some(0),
        "medium" => Some(1),
        "high" => Some(2),
        "max" => Some(3),
        "xhigh" => Some(4),
        _ => None,
    }
}

pub(super) fn is_int(value: &Value) -> bool {
    value.is_i64() || value.is_u64()
}

pub(super) fn python_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(fields) => !fields.is_empty(),
    }
}

fn to_body(request: &MessagesRequest) -> Result<Body, Error> {
    match serde_json::to_value(request) {
        Ok(Value::Object(body)) => Ok(body),
        Ok(_) => Err(Error::InvalidRequest(ErrorDetail::Expected {
            subject: "Bedrock invoke request",
            expected: "an object",
        })),
        Err(error) => Err(Error::InvalidRequest(ErrorDetail::failed(
            "Bedrock invoke request serialization",
            error,
        ))),
    }
}

fn from_body(body: Body) -> Result<MessagesRequest, Error> {
    serde_json::from_value(Value::Object(body)).map_err(|error| {
        Error::InvalidRequest(ErrorDetail::invalid("Bedrock invoke request", error))
    })
}

fn clamp_effort(effort: &str, ceiling: Option<&str>) -> Option<String> {
    let ceiling = ceiling?;
    (effort_rank(effort)? > effort_rank(ceiling)?).then(|| ceiling.to_string())
}

fn ceiling(capabilities: &MessagesModelCapabilities) -> Option<String> {
    capabilities
        .bedrock
        .output_config_effort_ceiling
        .map(|level| <&str>::from(level).to_string())
}

fn clamped_reasoning_effort(
    request: MessagesRequest,
    capabilities: &MessagesModelCapabilities,
) -> Result<MessagesRequest, Error> {
    if !capabilities.supports_adaptive_thinking {
        return Ok(request);
    }
    let mut body = to_body(&request)?;
    let Some(clamped) = body
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .and_then(|effort| clamp_effort(effort, ceiling(capabilities).as_deref()))
    else {
        return Ok(request);
    };
    body.insert("reasoning_effort".into(), Value::String(clamped));
    from_body(body)
}

pub(super) fn normalize_output_config_effort(body: &mut Body, ceiling: Option<&str>) {
    let Some(Value::Object(output_config)) = body.get_mut("output_config") else {
        return;
    };
    if let Some(clamped) = output_config
        .get("effort")
        .and_then(Value::as_str)
        .and_then(|effort| clamp_effort(effort, ceiling))
    {
        output_config.insert("effort".into(), Value::String(clamped));
    }
}

pub(super) fn is_display_updates(thinking: Option<&Value>) -> bool {
    let Some(Value::Object(thinking)) = thinking else {
        return false;
    };
    matches!(
        thinking.get("type").and_then(Value::as_str),
        Some("adaptive" | "enabled")
    ) && thinking.get("display").and_then(Value::as_str) == Some("updates")
}

fn carried_thinking_keys(thinking: &Body, rebuilt: &[&str], drops_unknown: bool) -> Body {
    thinking
        .iter()
        .filter(|(key, _)| !rebuilt.contains(&key.as_str()))
        .filter(|(key, _)| ADAPTIVE_THINKING_KEYS.contains(&key.as_str()) || !drops_unknown)
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn readd_display_updates(body: &mut Body, drops_unknown: bool) {
    let Some(Value::Object(thinking)) = body.get("thinking") else {
        return;
    };
    if thinking.get("type").and_then(Value::as_str) != Some("adaptive") {
        return;
    }
    let mut rebuilt = carried_thinking_keys(thinking, &["display"], drops_unknown);
    rebuilt.insert("display".into(), Value::String("updates".into()));
    body.insert("thinking".into(), Value::Object(rebuilt));
}

fn system_as_blocks(body: &mut Body) {
    match body.get("system") {
        Some(Value::String(text)) => {
            let blocks = json!([{"type": "text", "text": text}]);
            body.insert("system".into(), blocks);
        }
        Some(Value::Null) => {
            body.shift_remove("system");
        }
        Some(Value::Array(blocks)) if blocks.is_empty() => {
            body.shift_remove("system");
        }
        _ => {}
    }
}

fn converted_mid_conversation_turns(messages: Value) -> Result<Value, Error> {
    let messages: Vec<Recognized<Message>> = serde_json::from_value(messages).map_err(|error| {
        Error::InvalidRequest(ErrorDetail::invalid("Bedrock invoke messages", error))
    })?;
    serde_json::to_value(map_known_runs(
        messages,
        convert_mid_conversation_system_turns,
    ))
    .map_err(|error| {
        Error::InvalidRequest(ErrorDetail::failed(
            "Bedrock invoke messages serialization",
            error,
        ))
    })
}

fn normalize_mid_conversation_system(
    body: &mut Body,
    supports_mid_conversation_system: bool,
    drops_unsupported_params: bool,
) -> Result<(), Error> {
    if supports_mid_conversation_system {
        return Ok(());
    }
    let Some(slot @ Value::Array(_)) = body.get_mut("messages") else {
        return Ok(());
    };
    let Value::Array(messages) = std::mem::take(slot) else {
        return Ok(());
    };
    let messages = if drops_unsupported_params {
        strip_beta_gated_system_messages(messages, &BETA_GATE_EMPTY)
    } else {
        messages
    };
    *slot = converted_mid_conversation_turns(Value::Array(messages))?;
    Ok(())
}

fn reject_web_search_tools(body: &Body, model: &str) -> Result<(), Error> {
    let Some(Value::Array(tools)) = body.get("tools") else {
        return Ok(());
    };
    let web_search = tools.iter().filter_map(Value::as_object).find_map(|tool| {
        tool.get("type")
            .and_then(Value::as_str)
            .filter(|tool_type| tool_type.starts_with("web_search_") && *tool_type != "function")
    });
    match web_search {
        Some(tool_type) => Err(Error::InvalidRequest(
            ErrorDetail::UnsupportedBedrockWebSearchTool {
                tool_type: tool_type.to_string(),
                model: model.to_string(),
            },
        )),
        None => Ok(()),
    }
}

pub(super) fn supports_extended_thinking(
    model: &str,
    capabilities: &MessagesModelCapabilities,
) -> bool {
    let lowered = model.to_lowercase();
    capabilities.supports_adaptive_thinking
        || EXTENDED_THINKING_PATTERNS
            .iter()
            .any(|pattern| lowered.contains(pattern))
}

fn effort_from_thinking_budget(budget: i128, budgets: &ThinkingBudgets) -> &'static str {
    if budget >= i128::from(budgets.xhigh) {
        "xhigh"
    } else if budget >= i128::from(budgets.high) {
        "high"
    } else if budget >= i128::from(budgets.medium) {
        "medium"
    } else {
        "low"
    }
}

fn clear_thinking_budget(value: Option<&Value>, minimum: u64) -> Result<i128, Error> {
    match value {
        None | Some(Value::Null) => Ok(i128::from(minimum)),
        Some(Value::Bool(flag)) => Ok(i128::from(*flag)),
        Some(Value::Number(number)) => number
            .as_i64()
            .map(i128::from)
            .or_else(|| number.as_u64().map(i128::from))
            .or_else(|| {
                number
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .map(|value| value.trunc() as i128)
            })
            .ok_or_else(|| {
                Error::InvalidRequest(ErrorDetail::Expected {
                    subject: "thinking.budget_tokens",
                    expected: "a finite number",
                })
            }),
        Some(_) => Err(Error::InvalidRequest(ErrorDetail::Expected {
            subject: "thinking.budget_tokens",
            expected: "an integer",
        })),
    }
}

fn inject_adaptive_thinking(
    body: &mut Body,
    budget: i128,
    budgets: &ThinkingBudgets,
    drops_unknown: bool,
) {
    let mut output_config = match body.get("output_config") {
        Some(Value::Object(output_config)) => output_config.clone(),
        _ => Map::new(),
    };
    output_config
        .entry("effort")
        .or_insert_with(|| Value::String(effort_from_thinking_budget(budget, budgets).into()));
    let thinking = match body.get("thinking") {
        Some(Value::Object(thinking)) => thinking.clone(),
        _ => Map::new(),
    };
    let mut rebuilt = carried_thinking_keys(&thinking, &["type", "budget_tokens"], drops_unknown);
    rebuilt.insert("type".into(), Value::String("adaptive".into()));
    body.insert("output_config".into(), Value::Object(output_config));
    body.insert("thinking".into(), Value::Object(rebuilt));
}

fn ensure_thinking_for_clear_thinking(
    body: &mut Body,
    model: &str,
    capabilities: &MessagesModelCapabilities,
    budgets: &ThinkingBudgets,
    drops_unknown: bool,
) -> Result<bool, Error> {
    let Some(Value::Object(context_management)) = body.get("context_management") else {
        return Ok(false);
    };
    let Some(Value::Array(edits)) = context_management.get("edits") else {
        return Ok(false);
    };
    let needs_thinking = edits
        .iter()
        .any(|edit| edit.get("type").and_then(Value::as_str) == Some(CLEAR_THINKING_EDIT));
    if !needs_thinking || !supports_extended_thinking(model, capabilities) {
        return Ok(false);
    }
    let adaptive = capabilities.supports_adaptive_thinking;
    if let Some(Value::Object(thinking)) = body.get("thinking") {
        match thinking.get("type").and_then(Value::as_str) {
            Some("adaptive") => return Ok(false),
            Some("enabled") if !adaptive => return Ok(false),
            Some("enabled") => {
                let budget =
                    clear_thinking_budget(thinking.get("budget_tokens"), budgets.bedrock_min)?;
                inject_adaptive_thinking(body, budget, budgets, drops_unknown);
                return Ok(true);
            }
            _ => {}
        }
    }
    let minimum = budgets.bedrock_min;
    let too_small_max_tokens = body
        .get("max_tokens")
        .filter(|value| is_int(value))
        .and_then(|value| {
            value
                .as_i64()
                .map(i128::from)
                .or_else(|| value.as_u64().map(i128::from))
        })
        .filter(|max_tokens| *max_tokens <= i128::from(minimum));
    if let Some(max_tokens) = too_small_max_tokens {
        litellm_tracing::warn!(
            "Bedrock clear_thinking_20251015: max_tokens={max_tokens} is not greater than minimum thinking budget ({minimum}); cannot inject thinking safely"
        );
        return Ok(false);
    }
    if adaptive {
        inject_adaptive_thinking(body, i128::from(minimum), budgets, drops_unknown);
    } else {
        body.insert(
            "thinking".into(),
            json!({"type": "enabled", "budget_tokens": minimum}),
        );
    }
    Ok(true)
}

fn sanitize_cache_control(cache_control: &mut Value, supports_ttl: bool) {
    let Value::Object(cache_control) = cache_control else {
        return;
    };
    cache_control.shift_remove("scope");
    let keeps_ttl = supports_ttl
        && matches!(
            cache_control.get("ttl").and_then(Value::as_str),
            Some("5m" | "1h")
        );
    if !keeps_ttl {
        cache_control.shift_remove("ttl");
    }
}

fn sanitize_block_list(blocks: &mut [Value], supports_ttl: bool) {
    blocks
        .iter_mut()
        .filter_map(|block| block.get_mut("cache_control"))
        .for_each(|cache_control| sanitize_cache_control(cache_control, supports_ttl));
}

fn sanitize_cache_controls(body: &mut Body, supports_ttl: bool) {
    if let Some(Value::Array(tools)) = body.get_mut("tools") {
        sanitize_block_list(tools, supports_ttl);
    }
    if let Some(Value::Array(system)) = body.get_mut("system") {
        sanitize_block_list(system, supports_ttl);
    }
    if let Some(Value::Array(messages)) = body.get_mut("messages") {
        messages
            .iter_mut()
            .filter_map(|message| message.get_mut("content"))
            .filter_map(Value::as_array_mut)
            .for_each(|content| sanitize_block_list(content, supports_ttl));
    }
}

fn strip_unsupported_output_config_keys(body: &mut Body, model: &str, accepts_output_config: bool) {
    let Some(Value::Object(output_config)) = body.get("output_config") else {
        return;
    };
    if output_config.keys().all(|key| key == "format") || accepts_output_config {
        return;
    }
    litellm_tracing::warn!(
        "Bedrock Invoke: stripping unsupported `output_config` keys for model={model}: neither `supports_output_config` nor any `supports_*_reasoning_effort` flag is set in model_prices_and_context_window.json. Add the capability flag to the model JSON entry if this model accepts `output_config`."
    );
    match output_config
        .get("format")
        .filter(|format| !format.is_null())
        .cloned()
    {
        Some(format) => {
            body.insert("output_config".into(), json!({"format": format}));
        }
        None => {
            body.shift_remove("output_config");
        }
    }
}

fn normalize_custom_types_to_object(schema: &mut Value) {
    let Value::Object(node) = schema else {
        return;
    };
    if node.get("type").and_then(Value::as_str) == Some("custom") {
        node.insert("type".into(), Value::String("object".into()));
    }
    for key in ["items", "additionalProperties"] {
        if let Some(child) = node.get_mut(key) {
            normalize_custom_types_to_object(child);
        }
    }
    if let Some(Value::Object(properties)) = node.get_mut("properties") {
        properties
            .values_mut()
            .for_each(normalize_custom_types_to_object);
    }
    for combiner in ["allOf", "anyOf", "oneOf"] {
        if let Some(Value::Array(branches)) = node.get_mut(combiner) {
            branches
                .iter_mut()
                .for_each(normalize_custom_types_to_object);
        }
    }
}

fn normalize_tool(index: usize, tool: &mut Value) {
    let Value::Object(tool) = tool else {
        return;
    };
    if let Some(Value::Object(custom)) = tool.shift_remove("custom")
        && !tool.contains_key("defer_loading")
        && let Some(deferred @ Value::Bool(_)) = custom.get("defer_loading")
    {
        tool.insert("defer_loading".into(), deferred.clone());
    }
    if let Some(schema @ Value::Object(_)) = tool.get_mut("input_schema") {
        normalize_custom_types_to_object(schema);
    }
    let unnamed = match tool.get("name") {
        None | Some(Value::Null) => true,
        Some(Value::String(name)) => name.trim().is_empty(),
        Some(_) => false,
    };
    if unnamed {
        tool.insert(
            "name".into(),
            Value::String(format!("{UNNAMED_TOOL_PREFIX}{index}")),
        );
    }
}

fn normalize_tools(body: &mut Body) {
    if let Some(Value::Array(tools)) = body.get_mut("tools") {
        tools
            .iter_mut()
            .enumerate()
            .for_each(|(index, tool)| normalize_tool(index, tool));
    }
}

fn tools_without_eager_input_streaming(body: &mut Body) {
    let Some(Value::Array(tools)) = body.get_mut("tools") else {
        return;
    };
    if !tools.iter().all(Value::is_object) {
        return;
    }
    tools
        .iter_mut()
        .filter_map(Value::as_object_mut)
        .for_each(|tool| {
            tool.shift_remove("eager_input_streaming");
        });
}

fn drop_compaction_without_beta(body: &mut Body, betas: &[String]) {
    if body.contains_key("compaction")
        && !(body.get("compaction").is_some_and(Value::is_object)
            && betas.iter().any(|beta| beta == "compact-2026-09-04"))
    {
        litellm_tracing::debug!(
            "Bedrock Invoke: dropping `compaction`; compact-2026-09-04 is not served on this route"
        );
        body.shift_remove("compaction");
    }
}

fn pop_output_config_under_global_drop(
    body: &mut Body,
    global_drop_params: bool,
    supports_effort: bool,
) {
    let has_non_format_keys = matches!(
        body.get("output_config"),
        Some(Value::Object(output_config)) if output_config.keys().any(|key| key != "format")
    );
    if global_drop_params && has_non_format_keys && !supports_effort {
        litellm_tracing::warn!(
            "Bedrock Invoke: dropping `output_config` the model does not accept"
        );
        body.shift_remove("output_config");
    }
}

fn allowlisted(body: Body) -> Body {
    let stripped: Vec<&String> = body
        .keys()
        .filter(|key| !ALLOWED_TOP_LEVEL_FIELDS.contains(&key.as_str()))
        .collect();
    if !stripped.is_empty() {
        litellm_tracing::debug!(
            "Bedrock Invoke: stripping unsupported top-level request fields: {stripped:?}"
        );
    }
    body.into_iter()
        .filter(|(key, _)| ALLOWED_TOP_LEVEL_FIELDS.contains(&key.as_str()))
        .collect()
}

pub(super) fn transform_bedrock_invoke_request(
    request: MessagesRequest,
    context: &MessagesTransformContext,
    headers: Headers,
) -> Result<(MessagesRequest, Headers), Error> {
    let model = request.model.clone();
    let stream = request.params.stream.clone();
    let capabilities = &context.thinking.capabilities;
    let drops_unknown_thinking = context.drop_params || context.global_drop_params;
    let requested_display_updates =
        is_display_updates(serde_json::to_value(&request.params.thinking).ok().as_ref());
    let request = clamped_reasoning_effort(request, capabilities)?;
    let request = transform_messages_request(request, context)?;
    let request = normalize_system_role_messages(request, true);
    let mut body = to_body(&request)?;
    let shared_params = body.clone();
    if requested_display_updates {
        readd_display_updates(&mut body, drops_unknown_thinking);
    }
    system_as_blocks(&mut body);
    normalize_mid_conversation_system(
        &mut body,
        capabilities.supports_mid_conversation_system,
        context.drops_unsupported_params,
    )?;
    reject_web_search_tools(&body, &model)?;
    body.entry("anthropic_version")
        .or_insert_with(|| Value::String(DEFAULT_BEDROCK_ANTHROPIC_API_VERSION.into()));
    body.shift_remove("stream");
    body.shift_remove("model");
    let injected_thinking = ensure_thinking_for_clear_thinking(
        &mut body,
        &model,
        capabilities,
        &context.thinking.budgets,
        drops_unknown_thinking,
    )?;
    sanitize_cache_controls(&mut body, capabilities.bedrock.supports_prompt_cache_ttl);
    apply_structured_output(&mut body, &model, capabilities.bedrock.native_output)?;
    let ceiling = ceiling(capabilities);
    normalize_output_config_effort(&mut body, ceiling.as_deref());
    let supports_effort = supports_effort_param(capabilities);
    strip_unsupported_output_config_keys(
        &mut body,
        &model,
        capabilities.bedrock.supports_output_config || supports_effort,
    );
    normalize_tools(&mut body);
    let (betas, headers) = bedrock_invoke_betas(
        &mut body,
        &shared_params,
        headers,
        &model,
        capabilities,
        injected_thinking,
    )?;
    if !betas.is_empty() {
        body.insert(
            "anthropic_beta".into(),
            Value::Array(betas.iter().cloned().map(Value::String).collect()),
        );
    }
    drop_compaction_without_beta(&mut body, &betas);
    if context.drops_unsupported_params {
        strip_beta_gated_request(&mut body, &betas);
    }
    tools_without_eager_input_streaming(&mut body);
    pop_output_config_under_global_drop(&mut body, context.global_drop_params, supports_effort);
    let mut body = allowlisted(body);
    body.insert("model".into(), Value::String(model));
    if let Some(stream) = stream {
        body.insert(
            "stream".into(),
            serde_json::to_value(stream).unwrap_or(Value::Null),
        );
    }
    Ok((from_body(body)?, headers))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;
    use crate::{
        base_llm::messages::{
            context::{BedrockCapabilities, NativeOutputSupport, SupportedEffortTiers},
            transformation::BaseMessagesConfig,
        },
        bedrock::messages::invoke_transformations::anthropic_claude_transformation::AmazonAnthropicClaudeMessagesConfig,
    };
    use litellm_llms_types::formats::messages::EffortLevel;

    fn body(value: Value) -> Body {
        value.as_object().unwrap().clone()
    }

    fn adaptive() -> MessagesModelCapabilities {
        MessagesModelCapabilities {
            supports_reasoning: true,
            supports_adaptive_thinking: true,
            supports_output_config: true,
            effort_tiers: SupportedEffortTiers {
                low: true,
                medium: true,
                high: true,
                max: true,
                xhigh: true,
                ..SupportedEffortTiers::default()
            },
            ..MessagesModelCapabilities::default()
        }
    }

    fn with_bedrock(
        capabilities: MessagesModelCapabilities,
        bedrock: BedrockCapabilities,
    ) -> MessagesModelCapabilities {
        MessagesModelCapabilities {
            bedrock,
            ..capabilities
        }
    }

    fn context(
        capabilities: MessagesModelCapabilities,
        drop_params: bool,
    ) -> MessagesTransformContext {
        MessagesTransformContext {
            drops_unsupported_params: drop_params,
            ..MessagesTransformContext::new(capabilities, drop_params)
        }
    }

    fn transformed_with(
        request: Value,
        context: &MessagesTransformContext,
        headers: Headers,
    ) -> Result<(Value, Headers), Error> {
        let request: MessagesRequest = serde_json::from_value(request).unwrap();
        let (request, headers) = AmazonAnthropicClaudeMessagesConfig
            .transform_request_with_headers(request, context, headers)?;
        Ok((
            AmazonAnthropicClaudeMessagesConfig.wire_body(serde_json::to_value(&request).unwrap()),
            headers,
        ))
    }

    fn rejection(error: Error) -> String {
        match error {
            Error::InvalidRequest(detail) => detail.to_string(),
            other => panic!("expected a request rejection, got {other:?}"),
        }
    }

    fn transformed(request: Value, capabilities: MessagesModelCapabilities) -> Value {
        transformed_with(request, &context(capabilities, false), Vec::new())
            .unwrap()
            .0
    }

    fn hello(model: &str, params: Value) -> Value {
        let mut request = json!({
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
            "max_tokens": 4096,
        });
        request
            .as_object_mut()
            .unwrap()
            .extend(params.as_object().unwrap().clone());
        request
    }

    #[test]
    fn allowlist_filters_anthropic_only_fields() {
        let wire = transformed(
            hello(
                "anthropic.claude-opus-4-7",
                json!({
                    "temperature": 0.5,
                    "mcp_servers": [{"type": "url", "url": "https://example.com"}],
                    "container": {"skills": []},
                    "inference_geo": "us",
                    "output_config": {"effort": "low"},
                    "context_management": {"edits": []},
                    "stream": true,
                }),
            ),
            MessagesModelCapabilities {
                supports_sampling_params: true,
                ..adaptive()
            },
        );
        let fields = wire.as_object().unwrap();
        assert!(
            fields
                .keys()
                .all(|key| ALLOWED_TOP_LEVEL_FIELDS.contains(&key.as_str())),
            "{wire}"
        );
        assert_eq!(wire["output_config"], json!({"effort": "low"}));
        assert_eq!(wire["max_tokens"], json!(4096));
        assert_eq!(wire["temperature"], json!(0.5));
        assert_eq!(
            wire["anthropic_version"],
            json!(DEFAULT_BEDROCK_ANTHROPIC_API_VERSION)
        );
    }

    #[test]
    fn a_callers_anthropic_version_wins() {
        let wire = transformed(
            hello("anthropic.claude-3", json!({"anthropic_version": "custom"})),
            MessagesModelCapabilities::default(),
        );
        assert_eq!(wire["anthropic_version"], json!("custom"));
    }

    #[rstest]
    #[case::ttl_dropped_without_support(false, json!({"type": "ephemeral", "ttl": "1h", "scope": "global"}), json!({"type": "ephemeral"}))]
    #[case::ttl_kept_with_support(true, json!({"type": "ephemeral", "ttl": "1h", "scope": "global"}), json!({"type": "ephemeral", "ttl": "1h"}))]
    #[case::unknown_ttl_dropped_with_support(true, json!({"type": "ephemeral", "ttl": "2h"}), json!({"type": "ephemeral"}))]
    fn cache_control_keeps_only_supported_ttl_on_tools_system_and_messages(
        #[case] supports_ttl: bool,
        #[case] cache_control: Value,
        #[case] expected: Value,
    ) {
        let mut request = body(json!({
            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}, "cache_control": cache_control.clone()}],
            "system": [{"type": "text", "text": "You are helpful.", "cache_control": cache_control.clone()}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello", "cache_control": cache_control}]}],
        }));
        sanitize_cache_controls(&mut request, supports_ttl);
        assert_eq!(
            (
                &request["tools"][0]["cache_control"],
                &request["system"][0]["cache_control"],
                &request["messages"][0]["content"][0]["cache_control"],
            ),
            (&expected, &expected, &expected)
        );
    }

    #[rstest]
    #[case::hoists_bool_defer_loading(
        json!([{"name": "Read", "input_schema": {"type": "object"}, "custom": {"defer_loading": true}}, {"name": "Write", "input_schema": {"type": "object"}}]),
        json!([{"name": "Read", "input_schema": {"type": "object"}, "defer_loading": true}, {"name": "Write", "input_schema": {"type": "object"}}]),
    )]
    #[case::keeps_an_existing_defer_loading(
        json!([{"name": "Read", "defer_loading": false, "custom": {"defer_loading": true}}]),
        json!([{"name": "Read", "defer_loading": false}]),
    )]
    #[case::drops_a_non_bool_defer_loading(
        json!([{"name": "Read", "custom": {"defer_loading": "yes", "other": 1}}]),
        json!([{"name": "Read"}]),
    )]
    #[case::custom_schema_types_become_object(
        json!([{"name": "t", "input_schema": {"type": "custom", "properties": {"a": {"type": "custom", "items": {"type": "custom"}}}, "anyOf": [{"type": "custom"}]}}]),
        json!([{"name": "t", "input_schema": {"type": "object", "properties": {"a": {"type": "object", "items": {"type": "object"}}}, "anyOf": [{"type": "object"}]}}]),
    )]
    #[case::unnamed_tools_get_their_index(
        json!([{"name": "first"}, {"input_schema": {"type": "object"}}, {"name": "  "}, {"name": null}]),
        json!([{"name": "first"}, {"input_schema": {"type": "object"}, "name": "litellm_unnamed_tool_1"}, {"name": "litellm_unnamed_tool_2"}, {"name": "litellm_unnamed_tool_3"}]),
    )]
    fn tools_are_normalized_for_bedrock(#[case] tools: Value, #[case] expected: Value) {
        let mut request = body(json!({"tools": tools}));
        normalize_tools(&mut request);
        assert_eq!(request["tools"], expected);
    }

    #[test]
    fn eager_input_streaming_adds_its_beta_and_is_stripped_from_tools() {
        let wire = transformed(
            hello(
                "anthropic.claude-sonnet-4-5",
                json!({"tools": [{"name": "t", "input_schema": {"type": "object"}, "eager_input_streaming": true}]}),
            ),
            MessagesModelCapabilities::default(),
        );
        assert_eq!(
            wire["anthropic_beta"],
            json!(["fine-grained-tool-streaming-2025-05-14"])
        );
        assert_eq!(
            wire["tools"],
            json!([{"name": "t", "input_schema": {"type": "object"}}])
        );
    }

    #[test]
    fn a_non_bool_eager_input_streaming_is_refused() {
        let error = transformed_with(
            hello(
                "anthropic.claude-sonnet-4-5",
                json!({"tools": [{"name": "t", "input_schema": {"type": "object"}, "eager_input_streaming": "yes"}]}),
            ),
            &context(MessagesModelCapabilities::default(), false),
            Vec::new(),
        )
        .unwrap_err();
        assert_eq!(rejection(error), "eager_input_streaming must be a boolean");
    }

    #[rstest]
    #[case::stripped_without_support(
        MessagesModelCapabilities::default(),
        json!({"effort": "high"}),
        None,
    )]
    #[case::kept_with_output_config_support(
        adaptive(),
        json!({"effort": "high"}),
        Some(json!({"effort": "high"})),
    )]
    #[case::bedrock_flag_keeps_it(
        with_bedrock(MessagesModelCapabilities::default(), BedrockCapabilities { supports_output_config: true, ..BedrockCapabilities::default() }),
        json!({"effort": "high"}),
        Some(json!({"effort": "high"})),
    )]
    fn output_config_keys_follow_model_support(
        #[case] capabilities: MessagesModelCapabilities,
        #[case] output_config: Value,
        #[case] expected: Option<Value>,
    ) {
        let mut request = body(json!({"output_config": output_config}));
        strip_unsupported_output_config_keys(
            &mut request,
            "model",
            capabilities.bedrock.supports_output_config || supports_effort_param(&capabilities),
        );
        assert_eq!(request.get("output_config").cloned(), expected);
    }

    #[test]
    fn output_config_format_is_inlined_without_native_output() {
        let schema = json!({"type": "object", "properties": {"answer": {"type": "string"}}});
        let wire = transformed(
            hello(
                "anthropic.claude-opus-4-7",
                json!({"output_config": {"effort": "xhigh", "format": {"type": "json_schema", "schema": schema}}}),
            ),
            adaptive(),
        );
        assert_eq!(wire["output_config"], json!({"effort": "xhigh"}));
        let content = wire["messages"][0]["content"].as_array().unwrap();
        assert_eq!(
            content.last().unwrap()["text"],
            json!(r#"{"type": "object", "properties": {"answer": {"type": "string"}}}"#)
        );
    }

    #[rstest]
    #[case::opus_4_5_ceiling_high(Some(EffortLevel::High), "high")]
    #[case::opus_4_6_ceiling_max(Some(EffortLevel::Max), "max")]
    #[case::opus_4_7_no_ceiling(None, "xhigh")]
    fn output_config_effort_is_clamped_to_the_bedrock_ceiling(
        #[case] ceiling: Option<EffortLevel>,
        #[case] expected: &str,
    ) {
        let capabilities = with_bedrock(
            MessagesModelCapabilities {
                supports_output_config: true,
                ..MessagesModelCapabilities::default()
            },
            BedrockCapabilities {
                output_config_effort_ceiling: ceiling,
                ..BedrockCapabilities::default()
            },
        );
        let wire = transformed(
            hello(
                "anthropic.claude-opus-4-5",
                json!({"output_config": {"effort": "xhigh"}}),
            ),
            capabilities,
        );
        assert_eq!(wire["output_config"], json!({"effort": expected}));
    }

    #[test]
    fn global_drop_params_pops_output_config_the_model_does_not_take() {
        let context = MessagesTransformContext {
            global_drop_params: true,
            ..context(MessagesModelCapabilities::default(), false)
        };
        let mut request =
            body(json!({"output_config": {"effort": "high", "format": {"type": "json_schema"}}}));
        pop_output_config_under_global_drop(&mut request, context.global_drop_params, false);
        assert_eq!(request.get("output_config"), None);
    }

    #[rstest]
    #[case::adaptive_without_thinking(adaptive(), json!({"max_tokens": 32000}), true, Some(json!({"type": "adaptive"})), Some("low"))]
    #[case::adaptive_already_adaptive(adaptive(), json!({"max_tokens": 32000, "thinking": {"type": "adaptive"}}), false, Some(json!({"type": "adaptive"})), None)]
    #[case::adaptive_legacy_budget(adaptive(), json!({"max_tokens": 32000, "thinking": {"type": "enabled", "budget_tokens": 4096}}), true, Some(json!({"type": "adaptive"})), Some("high"))]
    #[case::adaptive_explicit_zero_budget(adaptive(), json!({"max_tokens": 32000, "thinking": {"type": "enabled", "budget_tokens": 0}}), true, Some(json!({"type": "adaptive"})), Some("low"))]
    #[case::non_adaptive_keeps_enabled(MessagesModelCapabilities::default(), json!({"max_tokens": 32000, "thinking": {"type": "enabled", "budget_tokens": 2048}}), false, Some(json!({"type": "enabled", "budget_tokens": 2048})), None)]
    #[case::non_adaptive_injects_enabled(MessagesModelCapabilities::default(), json!({"max_tokens": 32000}), true, Some(json!({"type": "enabled", "budget_tokens": 1024})), None)]
    #[case::max_tokens_too_small(MessagesModelCapabilities::default(), json!({"max_tokens": 1024}), false, None, None)]
    #[case::display_updates_preserved(adaptive(), json!({"max_tokens": 32000, "thinking": {"type": "enabled", "budget_tokens": 8192, "display": "updates"}}), true, Some(json!({"display": "updates", "type": "adaptive"})), Some("xhigh"))]
    fn clear_thinking_injects_the_thinking_bedrock_requires(
        #[case] capabilities: MessagesModelCapabilities,
        #[case] request: Value,
        #[case] injected: bool,
        #[case] thinking: Option<Value>,
        #[case] effort: Option<&str>,
    ) {
        let mut request = body(request);
        request.insert(
            "context_management".into(),
            json!({"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}),
        );
        let changed = ensure_thinking_for_clear_thinking(
            &mut request,
            "us.anthropic.claude-opus-4-5",
            &capabilities,
            &ThinkingBudgets::default(),
            false,
        )
        .unwrap();
        assert_eq!(
            (
                changed,
                request.get("thinking").cloned(),
                request
                    .get("output_config")
                    .and_then(|config| config.get("effort"))
                    .and_then(Value::as_str)
            ),
            (injected, thinking, effort)
        );
    }

    #[rstest]
    #[case::xhigh(8192, "xhigh")]
    #[case::high(4096, "high")]
    #[case::medium(2048, "medium")]
    #[case::low(2047, "low")]
    #[case::zero(0, "low")]
    fn effort_follows_thinking_budget_tiers(#[case] budget: i128, #[case] expected: &str) {
        assert_eq!(
            effort_from_thinking_budget(budget, &ThinkingBudgets::default()),
            expected
        );
    }

    #[rstest]
    #[case::sonnet_4("anthropic.claude-sonnet-4-20250514-v1:0", false, true)]
    #[case::opus_4_1("anthropic.claude-opus-4-1", false, true)]
    #[case::haiku_3("anthropic.claude-3-haiku", false, false)]
    #[case::adaptive_flag("anthropic.claude-fable-5", true, true)]
    fn extended_thinking_follows_adaptive_flag_then_names(
        #[case] model: &str,
        #[case] supports_adaptive_thinking: bool,
        #[case] expected: bool,
    ) {
        let capabilities = MessagesModelCapabilities {
            supports_adaptive_thinking,
            ..MessagesModelCapabilities::default()
        };
        assert_eq!(supports_extended_thinking(model, &capabilities), expected);
    }

    #[test]
    fn clear_thinking_edit_is_never_forwarded() {
        let wire = transformed(
            hello(
                "anthropic.claude-sonnet-4-5",
                json!({"max_tokens": 32000, "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}, {"type": "clear_tool_uses_20250919"}]}}),
            ),
            MessagesModelCapabilities::default(),
        );
        assert_eq!(
            wire["context_management"],
            json!({"edits": [{"type": "clear_tool_uses_20250919"}]})
        );
        assert_eq!(
            wire["thinking"],
            json!({"type": "enabled", "budget_tokens": 1024})
        );
        assert_eq!(
            wire["anthropic_beta"],
            json!(["context-management-2025-06-27"])
        );
    }

    #[rstest]
    #[case::non_list_edits_dropped(json!({"edits": "x"}), None, json!(null))]
    #[case::unsupported_edits_dropped(json!({"edits": [{"type": "clear_thinking_20251015"}]}), None, json!(null))]
    #[case::compact_kept(json!({"edits": [{"type": "compact_20260112"}]}), Some(json!({"edits": [{"type": "compact_20260112"}]})), json!(["compact-2026-01-12"]))]
    fn context_management_keeps_bedrock_edits_with_their_betas(
        #[case] context_management: Value,
        #[case] expected: Option<Value>,
        #[case] betas: Value,
    ) {
        let wire = transformed(
            hello(
                "anthropic.claude-3",
                json!({"context_management": context_management}),
            ),
            MessagesModelCapabilities::default(),
        );
        assert_eq!(wire.get("context_management").cloned(), expected);
        assert_eq!(
            wire.get("anthropic_beta").cloned().unwrap_or(Value::Null),
            betas
        );
    }

    #[rstest]
    #[case::object_compaction_kept(json!({"type": "auto"}), Some(json!({"type": "auto"})))]
    #[case::string_compaction_dropped(json!("auto"), None)]
    #[case::list_compaction_dropped(json!([]), None)]
    fn compaction_needs_an_object_and_its_beta(
        #[case] compaction: Value,
        #[case] expected: Option<Value>,
    ) {
        let wire = transformed(
            hello("anthropic.claude-3", json!({"compaction": compaction})),
            MessagesModelCapabilities::default(),
        );
        assert_eq!(wire.get("compaction").cloned(), expected);
    }

    #[test]
    fn leading_system_turns_are_hoisted_and_string_system_becomes_blocks() {
        let wire = transformed(
            json!({
                "model": "anthropic.claude-3",
                "max_tokens": 10,
                "system": "base",
                "messages": [
                    {"role": "system", "content": "first"},
                    {"role": "user", "content": "hi"},
                ],
            }),
            MessagesModelCapabilities::default(),
        );
        assert_eq!(
            wire["system"],
            json!([{"type": "text", "text": "base"}, {"type": "text", "text": "first"}])
        );
        assert_eq!(wire["messages"], json!([{"role": "user", "content": "hi"}]));
    }

    #[rstest]
    #[case::kept_in_place(true, "system")]
    #[case::converted_to_user(false, "user")]
    fn mid_conversation_system_turns_follow_model_support(
        #[case] supports_mid_conversation_system: bool,
        #[case] role: &str,
    ) {
        let wire = transformed(
            json!({
                "model": "anthropic.claude-3",
                "max_tokens": 10,
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "system", "content": "reminder"},
                ],
            }),
            MessagesModelCapabilities {
                supports_mid_conversation_system,
                ..MessagesModelCapabilities::default()
            },
        );
        assert_eq!(wire["messages"][2]["role"], json!(role));
    }

    #[rstest]
    #[case::web_search_20250305("web_search_20250305")]
    #[case::web_search_20260209("web_search_20260209")]
    fn server_web_search_tools_are_rejected(#[case] tool_type: &str) {
        let error = transformed_with(
            hello(
                "anthropic.claude-sonnet-4-5",
                json!({"tools": [{"type": tool_type, "name": "web_search"}]}),
            ),
            &context(MessagesModelCapabilities::default(), false),
            Vec::new(),
        )
        .unwrap_err();
        assert_eq!(
            rejection(error),
            format!(
                "Bedrock does not support Anthropic's server-side web search tool (tool type '{tool_type}', model 'anthropic.claude-sonnet-4-5'). To use web search with this model, enable LiteLLM's web search interception so the proxy executes the search instead: https://docs.litellm.ai/docs/integrations/websearch_interception. Alternatively, remove the web_search tool from the request."
            )
        );
    }

    #[test]
    fn a_converted_websearch_function_tool_is_allowed() {
        let wire = transformed(
            hello(
                "anthropic.claude-sonnet-4-5",
                json!({"tools": [{"name": "litellm_web_search", "input_schema": {"type": "object"}}]}),
            ),
            MessagesModelCapabilities::default(),
        );
        assert_eq!(wire["tools"][0]["name"], json!("litellm_web_search"));
    }

    #[test]
    fn reasoning_effort_is_clamped_before_the_effort_gate_on_adaptive_models() {
        let capabilities = with_bedrock(
            MessagesModelCapabilities {
                effort_tiers: SupportedEffortTiers {
                    low: true,
                    medium: true,
                    high: true,
                    max: true,
                    ..SupportedEffortTiers::default()
                },
                ..adaptive()
            },
            BedrockCapabilities {
                output_config_effort_ceiling: Some(EffortLevel::Max),
                ..BedrockCapabilities::default()
            },
        );
        let wire = transformed(
            hello(
                "anthropic.claude-opus-4-6",
                json!({"reasoning_effort": "xhigh"}),
            ),
            capabilities,
        );
        assert_eq!(wire["output_config"]["effort"], json!("max"));
        assert_eq!(wire.get("reasoning_effort"), None);
    }

    #[test]
    fn native_output_forwards_a_cleaned_schema() {
        let capabilities = with_bedrock(
            adaptive(),
            BedrockCapabilities {
                native_output: NativeOutputSupport {
                    enabled: true,
                    mapped: true,
                    ..NativeOutputSupport::default()
                },
                ..BedrockCapabilities::default()
            },
        );
        let wire = transformed(
            hello(
                "anthropic.claude-opus-4-7",
                json!({"output_config": {"format": {"type": "json_schema", "schema": {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}, "required": ["n"]}}}}),
            ),
            capabilities,
        );
        assert_eq!(
            wire["output_config"],
            json!({"format": {"type": "json_schema", "schema": {"type": "object", "properties": {"n": {"description": "Note: minimum value: 1.", "type": "integer"}}, "required": ["n"], "additionalProperties": false}}})
        );
        assert_eq!(wire["messages"][0]["content"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn model_and_stream_leave_the_wire_body() {
        let wire = transformed(
            hello("anthropic.claude-3", json!({"stream": true})),
            MessagesModelCapabilities::default(),
        );
        assert_eq!((wire.get("model"), wire.get("stream")), (None, None));
    }
}
