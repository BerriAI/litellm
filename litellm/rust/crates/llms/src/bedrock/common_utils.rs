pub use litellm_model_catalog::{
    bedrock_base_model, split_bedrock_region_path, strip_bedrock_routing_prefix,
};
use serde_json::{Map, Value, json};

use crate::{Error, ErrorDetail};

pub mod structured_output;

pub fn is_bedrock_application_inference_profile_arn(model: &str) -> bool {
    model.contains(":application-inference-profile/")
}

const UNNAMED_TOOL_PREFIX: &str = "litellm_unnamed_tool_";

pub(crate) fn is_int(value: &Value) -> bool {
    value.is_i64() || value.is_u64()
}

fn effort_rank(effort: &str) -> Option<u8> {
    match effort {
        "low" => Some(0),
        "medium" => Some(1),
        "high" => Some(2),
        "max" => Some(3),
        "xhigh" => Some(4),
        _ => None,
    }
}

pub(crate) fn clamp_effort(effort: &str, ceiling: Option<&str>) -> Option<String> {
    let ceiling = ceiling?;
    (effort_rank(effort)? > effort_rank(ceiling)?).then(|| ceiling.to_string())
}

/// Port of Python's `normalize_bedrock_opus_output_config_effort`.
pub(crate) fn normalize_bedrock_opus_output_config_effort(
    output_config: Option<&mut Value>,
    ceiling: Option<&str>,
) {
    let Some(Value::Object(output_config)) = output_config else {
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

/// Port of Python's `strip_unsupported_bedrock_invoke_output_config_keys`.
pub(crate) fn strip_unsupported_bedrock_invoke_output_config_keys(
    model: &str,
    request_body: &mut Map<String, Value>,
    accepts_output_config: bool,
) {
    let Some(Value::Object(output_config)) = request_body.get("output_config") else {
        return;
    };
    if output_config.keys().all(|key| key == "format") || accepts_output_config {
        return;
    }
    tracing::warn!(
        "Bedrock Invoke: stripping unsupported `output_config` keys for model={model}: neither `supports_output_config` nor any `supports_*_reasoning_effort` flag is set in model_prices_and_context_window.json. Add the capability flag to the model JSON entry if this model accepts `output_config`."
    );
    match output_config
        .get("format")
        .filter(|format| !format.is_null())
        .cloned()
    {
        Some(format) => {
            request_body.insert("output_config".into(), json!({"format": format}));
        }
        None => {
            request_body.shift_remove("output_config");
        }
    }
}

fn tools_mut(request_body: &mut Map<String, Value>) -> Option<&mut Vec<Value>> {
    match request_body.get_mut("tools") {
        Some(Value::Array(tools)) => Some(tools),
        _ => None,
    }
}

/// Port of Python's `_normalize_custom_field_on_tool`.
fn normalize_custom_field_on_tool(tool: &mut Value) {
    let Value::Object(tool) = tool else {
        return;
    };
    if let Some(Value::Object(custom)) = tool.shift_remove("custom")
        && !tool.contains_key("defer_loading")
        && let Some(deferred @ Value::Bool(_)) = custom.get("defer_loading")
    {
        tool.insert("defer_loading".into(), deferred.clone());
    }
}

/// Port of Python's `normalize_custom_field_on_tools`.
pub(crate) fn normalize_custom_field_on_tools(request_body: &mut Map<String, Value>) {
    if let Some(tools) = tools_mut(request_body) {
        tools.iter_mut().for_each(normalize_custom_field_on_tool);
    }
}

/// Port of Python's `normalize_json_schema_custom_types_to_object`.
fn normalize_json_schema_custom_types_to_object(schema: &mut Value) {
    let Value::Object(node) = schema else {
        return;
    };
    if node.get("type").and_then(Value::as_str) == Some("custom") {
        node.insert("type".into(), Value::String("object".into()));
    }
    for key in ["items", "additionalProperties"] {
        if let Some(child) = node.get_mut(key) {
            normalize_json_schema_custom_types_to_object(child);
        }
    }
    if let Some(Value::Object(properties)) = node.get_mut("properties") {
        properties
            .values_mut()
            .for_each(normalize_json_schema_custom_types_to_object);
    }
    for combiner in ["allOf", "anyOf", "oneOf"] {
        if let Some(Value::Array(branches)) = node.get_mut(combiner) {
            branches
                .iter_mut()
                .for_each(normalize_json_schema_custom_types_to_object);
        }
    }
}

/// Port of Python's `normalize_tool_input_schema_types_for_bedrock_invoke`.
pub(crate) fn normalize_tool_input_schema_types_for_bedrock_invoke(
    request_body: &mut Map<String, Value>,
) {
    let Some(tools) = tools_mut(request_body) else {
        return;
    };
    tools
        .iter_mut()
        .filter_map(|tool| tool.get_mut("input_schema"))
        .filter(|input_schema| input_schema.is_object())
        .for_each(normalize_json_schema_custom_types_to_object);
}

/// Port of Python's `ensure_bedrock_anthropic_messages_tool_names`.
pub(crate) fn ensure_bedrock_anthropic_messages_tool_names(request_body: &mut Map<String, Value>) {
    let Some(tools) = tools_mut(request_body) else {
        return;
    };
    for (index, tool) in tools.iter_mut().enumerate() {
        let Value::Object(tool) = tool else {
            continue;
        };
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
}

/// Port of Python's `tools_without_eager_input_streaming`.
pub(crate) fn tools_without_eager_input_streaming(request_body: &mut Map<String, Value>) {
    let Some(tools) = tools_mut(request_body) else {
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

pub(crate) fn is_valid_region(region: &str) -> bool {
    !region.is_empty()
        && region
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

pub(crate) fn validated_region(region: String) -> Result<String, Error> {
    if is_valid_region(&region) {
        return Ok(region);
    }
    Err(Error::InvalidRequest(ErrorDetail::InvalidValue {
        field: "aws_region_name",
        expected: "lowercase letters, digits, and hyphens",
        actual: region.into(),
    }))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;
    use crate::{
        anthropic::common_utils::supports_effort_param,
        base_llm::messages::context::{
            BedrockCapabilities, MessagesModelCapabilities, SupportedEffortTiers,
        },
    };

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
        let mut request = json!({"tools": tools}).as_object().unwrap().clone();
        normalize_custom_field_on_tools(&mut request);
        normalize_tool_input_schema_types_for_bedrock_invoke(&mut request);
        ensure_bedrock_anthropic_messages_tool_names(&mut request);
        assert_eq!(request["tools"], expected);
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
        let mut request = json!({"output_config": output_config})
            .as_object()
            .unwrap()
            .clone();
        strip_unsupported_bedrock_invoke_output_config_keys(
            "model",
            &mut request,
            capabilities.bedrock.supports_output_config || supports_effort_param(&capabilities),
        );
        assert_eq!(request.get("output_config").cloned(), expected);
    }
}
