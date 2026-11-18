use litellm_auth_aws::is_bedrock_region;
use serde_json::{Map, Value, json};

use crate::{Error, ErrorDetail};

pub mod structured_output;

const ROUTE_PREFIXES: [&str; 3] = ["bedrock/converse/", "bedrock/", "converse/"];
const ROUTING_PREFIXES: [&str; 8] = [
    "bedrock/",
    "chat_completions/",
    "converse/",
    "invoke/",
    "openai/",
    "mantle/",
    "nova-2/",
    "nova/",
];
const CROSS_REGION_INFERENCE_PREFIXES: [&str; 8] =
    ["global", "us", "eu", "apac", "jp", "au", "ca", "us-gov"];

pub fn bedrock_base_model(model: &str) -> &str {
    let unrouted = ROUTE_PREFIXES
        .into_iter()
        .find_map(|prefix| model.strip_prefix(prefix))
        .unwrap_or(model);
    if unrouted.starts_with("nova-2/") {
        return "amazon.nova-2-custom";
    }
    if unrouted.starts_with("nova/") {
        return "amazon.nova-custom";
    }
    let named = without_throughput_suffix(model_name_from_arn(strip_bedrock_routing_prefix(model)));
    if let Some((region, rest)) = named.split_once('.')
        && CROSS_REGION_INFERENCE_PREFIXES.contains(&region)
    {
        return rest;
    }
    match named.split_once('/') {
        Some((region, rest)) if is_bedrock_region(region) => rest,
        _ => named,
    }
}

pub fn is_bedrock_application_inference_profile_arn(model: &str) -> bool {
    model.contains(":application-inference-profile/")
}

/// Port of Python's `strip_bedrock_routing_prefix`.
pub fn strip_bedrock_routing_prefix(model: &str) -> &str {
    ROUTING_PREFIXES.into_iter().fold(model, |model, prefix| {
        model.strip_prefix(prefix).unwrap_or(model)
    })
}

pub fn split_bedrock_region_path(model: &str) -> (Option<&str>, &str) {
    let stripped = strip_bedrock_routing_prefix(model);
    match stripped.split_once('/') {
        Some((region, model_id)) if is_bedrock_region(region) => (Some(region), model_id),
        _ => (None, stripped),
    }
}

fn model_name_from_arn(model: &str) -> &str {
    if !model.to_lowercase().contains("arn") {
        return model;
    }
    model.rsplit('/').next().unwrap_or(model)
}

fn without_throughput_suffix(model: &str) -> &str {
    let unthrottled = model
        .strip_suffix('k')
        .and_then(|rest| rest.rsplit_once(':'))
        .filter(|(head, units)| {
            is_digits(units)
                && head
                    .rsplit_once(':')
                    .is_some_and(|(_, minor)| is_digits(minor))
        })
        .map_or(model, |(head, _)| head);
    unthrottled
        .strip_suffix(']')
        .and_then(|rest| rest.rsplit_once('['))
        .filter(|(_, tag)| {
            !tag.is_empty()
                && tag
                    .chars()
                    .all(|char| char.is_alphanumeric() || char == '_')
        })
        .map_or(unthrottled, |(head, _)| head)
}

fn is_digits(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
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
    litellm_tracing::warn!(
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

pub(crate) fn validated_region(region: String) -> Result<String, Error> {
    if !region.is_empty()
        && region
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
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
    #[case::cross_region("us.anthropic.claude-x-v1:0", "anthropic.claude-x-v1:0")]
    #[case::canada_cross_region("ca.anthropic.claude-x-v1:0", "anthropic.claude-x-v1:0")]
    #[case::converse_route("bedrock/converse/anthropic.claude-x", "anthropic.claude-x")]
    #[case::mantle_route(
        "bedrock/mantle/anthropic.claude-sonnet-5",
        "anthropic.claude-sonnet-5"
    )]
    #[case::invoke_route_cross_region("invoke/global.anthropic.claude-x", "anthropic.claude-x")]
    #[case::throughput_suffix("anthropic.claude-x-v2:0:51k", "anthropic.claude-x-v2:0")]
    #[case::version_pair_without_throughput("anthropic.claude-x:1:2", "anthropic.claude-x:1:2")]
    #[case::context_tag("anthropic.claude-x[1m]", "anthropic.claude-x")]
    #[case::region_path("us-east-1/anthropic.claude-x", "anthropic.claude-x")]
    #[case::region_path_keeps_cross_region(
        "bedrock/us-east-1/us.anthropic.claude-x",
        "us.anthropic.claude-x"
    )]
    #[case::profile_arn(
        "arn:aws:bedrock:us-east-1:123:application-inference-profile/us.anthropic.claude-x",
        "anthropic.claude-x"
    )]
    #[case::nova_2_custom(
        "bedrock/nova-2/arn:aws:bedrock:us-east-1:1:custom-model/x",
        "amazon.nova-2-custom"
    )]
    #[case::nova_custom("bedrock/nova/arn:aws:x", "amazon.nova-custom")]
    fn base_model_matches_the_python_resolution(#[case] model: &str, #[case] expected: &str) {
        assert_eq!(bedrock_base_model(model), expected);
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
