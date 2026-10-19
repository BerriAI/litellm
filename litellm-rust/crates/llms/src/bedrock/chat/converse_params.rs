//! Python's `AmazonConverseConfig.get_supported_openai_params` and `map_openai_params`, and
//! the Bedrock model-id reads behind them.

use serde_json::{Map, Value, json};

use crate::{
    Error, ErrorDetail,
    anthropic::chat::params::{
        ModelCapabilities, apply_sampling_param, cap_thinking_budget, map_context_management,
        map_reasoning_effort, output_config_effort, python_str, translate_legacy_thinking,
        unmapped_reasoning_effort,
    },
    base_llm::chat::{
        params::{
            ChatRequestContext, DEFAULT_MAX_TOKENS, DropPolicy, RESPONSE_FORMAT_TOOL_NAME,
            capability, is_thinking_enabled, model_field, ordered, python_dumps, unsupported_value,
        },
        response_format::ResponseFormatSchema,
    },
};

/// `get_bedrock_cross_region_inference_regions`.
const CROSS_REGION_PREFIXES: [&str; 7] = ["global", "us", "eu", "apac", "jp", "au", "us-gov"];

/// `BEDROCK_MIN_THINKING_BUDGET_TOKENS`.
const MIN_THINKING_BUDGET_TOKENS: i64 = 1024;

/// `BEDROCK_OPENAI_COMPAT_MIN_MAX_TOKENS`.
const OPENAI_COMPAT_MIN_MAX_TOKENS: i64 = 16;

/// `_BEDROCK_OUTPUT_CONFIG_EFFORT_ORDER`.
const OUTPUT_CONFIG_EFFORT_ORDER: [&str; 5] = ["low", "medium", "high", "xhigh", "max"];

/// `strip_bedrock_routing_prefix`: each prefix in turn, as Python applies them.
pub fn strip_bedrock_routing_prefix(model: &str) -> String {
    let mut model = model.to_owned();
    for prefix in [
        "bedrock/",
        "chat_completions/",
        "converse/",
        "invoke/",
        "openai/",
        "mantle/",
        "nova-2/",
        "nova/",
    ] {
        if model.starts_with(prefix) {
            model = model
                .split_once('/')
                .map(|(_, rest)| rest.to_owned())
                .unwrap_or_default();
        }
    }
    model
}

/// `strip_bedrock_throughput_suffix`: `:<n>:<n>k` loses its throughput and `[...]` is cut.
fn strip_throughput_suffix(model: &str) -> String {
    let mut model = model.to_owned();
    if let Some(head) = model.strip_suffix('k')
        && let Some((rest, digits)) = head.rsplit_once(':')
        && !digits.is_empty()
        && digits.chars().all(|c| c.is_ascii_digit())
        && let Some((_, minor)) = rest.rsplit_once(':')
        && !minor.is_empty()
        && minor.chars().all(|c| c.is_ascii_digit())
    {
        model = rest.to_owned();
    }
    if model.ends_with(']')
        && let Some(open) = model.rfind('[')
        && model[open + 1..model.len() - 1]
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_')
        && open + 1 < model.len() - 1
    {
        model.truncate(open);
    }
    model
}

/// Whether `segment` reads as an AWS region, such as `us-east-1` or `us-gov-west-1`.
fn is_aws_region(segment: &str) -> bool {
    let parts: Vec<&str> = segment.split('-').collect();
    parts.len() >= 3
        && parts[0].len() == 2
        && parts[0].chars().all(|c| c.is_ascii_lowercase())
        && parts
            .last()
            .is_some_and(|last| !last.is_empty() && last.chars().all(|c| c.is_ascii_digit()))
        && parts[1..parts.len() - 1]
            .iter()
            .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_lowercase()))
}

/// `get_bedrock_base_model`.
pub fn bedrock_base_model(model: &str) -> String {
    let mut stripped = model;
    for prefix in ["bedrock/converse/", "bedrock/", "converse/"] {
        if let Some(rest) = stripped.strip_prefix(prefix) {
            stripped = rest;
            break;
        }
    }
    if stripped.starts_with("nova-2/") {
        return "amazon.nova-2-custom".into();
    }
    if stripped.starts_with("nova/") {
        return "amazon.nova-custom".into();
    }
    let mut model = strip_bedrock_routing_prefix(model);
    if model.to_lowercase().contains("arn") {
        model = model.rsplit('/').next().unwrap_or_default().to_owned();
    }
    let model = strip_throughput_suffix(&model);
    if let Some((region, rest)) = model.split_once('.')
        && CROSS_REGION_PREFIXES.contains(&region)
    {
        return rest.to_owned();
    }
    if let Some((region, rest)) = model.split_once('/')
        && is_aws_region(region)
    {
        return rest.to_owned();
    }
    model
}

/// `_is_nova_2_model`.
fn is_nova_2_model(model: &str) -> bool {
    let mut stripped = model;
    for prefix in ["bedrock/converse/", "bedrock/", "converse/"] {
        if let Some(rest) = stripped.strip_prefix(prefix) {
            stripped = rest;
            break;
        }
    }
    for prefix in ["us.", "eu.", "apac."] {
        if let Some(rest) = stripped.strip_prefix(prefix) {
            stripped = rest;
            break;
        }
    }
    stripped.starts_with("amazon.nova-2-") || stripped.starts_with("nova-2/")
}

/// `openai.gpt-<digit>` anywhere in the id.
fn is_openai_gpt_reasoning_model(model: &str) -> bool {
    model.match_indices("openai.gpt-").any(|(at, _)| {
        model[at + "openai.gpt-".len()..]
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_digit())
    })
}

/// `xai.grok-` after a start, `.` or `/`.
fn is_xai_grok_model(model: &str) -> bool {
    model
        .match_indices("xai.grok-")
        .any(|(at, _)| at == 0 || matches!(model.as_bytes()[at - 1], b'.' | b'/'))
}

/// `openai.gpt-<major>[.<minor>]` after a start, `.` or `/`.
fn openai_gpt_version(model: &str) -> Option<(u32, u32)> {
    model.match_indices("openai.").find_map(|(at, _)| {
        if at != 0 && !matches!(model.as_bytes()[at - 1], b'.' | b'/') {
            return None;
        }
        let rest = model[at..].strip_prefix("openai.gpt-")?;
        let major: String = rest.chars().take_while(char::is_ascii_digit).collect();
        if major.is_empty() || major.len() > 3 {
            return None;
        }
        let after = &rest[major.len()..];
        let minor = after
            .strip_prefix('.')
            .map(|tail| {
                tail.chars()
                    .take_while(char::is_ascii_digit)
                    .collect::<String>()
            })
            .filter(|minor| !minor.is_empty() && minor.len() <= 3)
            .unwrap_or_default();
        Some((major.parse().ok()?, minor.parse().unwrap_or(0)))
    })
}

/// `bedrock_rejects_stop_sequences`.
fn rejects_stop_sequences(model: &str) -> bool {
    is_xai_grok_model(model) || openai_gpt_version(model).is_some_and(|version| version >= (5, 6))
}

/// The model a Converse request is for, and the cost-map reads its rules make.
pub struct BedrockModel<'a> {
    pub model: String,
    pub base: String,
    pub context: ChatRequestContext<'a>,
    pub caps: ModelCapabilities<'a>,
}

impl<'a> BedrockModel<'a> {
    pub fn new(model: &str, context: &ChatRequestContext<'a>) -> Self {
        Self {
            model: model.to_owned(),
            base: bedrock_base_model(model),
            context: *context,
            caps: ModelCapabilities::new(model, context),
        }
    }

    pub fn is_anthropic(&self) -> bool {
        self.base.starts_with("anthropic")
    }

    /// The keys Python's provider-aware lookups resolve `model` through.
    pub fn candidates(&self) -> Vec<String> {
        let stripped = strip_bedrock_routing_prefix(&self.model);
        let mut candidates = vec![
            self.model.clone(),
            format!("bedrock/{}", self.model),
            format!("bedrock_converse/{}", self.model),
            stripped.clone(),
            self.base.clone(),
            format!("bedrock/{}", self.base),
        ];
        candidates.dedup();
        candidates
    }

    pub fn flag(&self, key: &str) -> Result<Option<bool>, Error> {
        capability(&self.context, &self.candidates(), key)
    }

    pub fn supports(&self, key: &str) -> Result<bool, Error> {
        Ok(self.flag(key)? == Some(true))
    }

    /// `_supports_sampling_params`.
    pub fn supports_sampling(&self) -> Result<bool, Error> {
        if self.is_anthropic() {
            return Ok(true);
        }
        let mut keys = vec![self.model.clone()];
        keys.extend(["global.", "us.", "eu."].map(|prefix| format!("{prefix}{}", self.base)));
        Ok(capability(&self.context, &keys, "supports_sampling_params")?.unwrap_or(true))
    }

    fn is_deepseek(&self) -> bool {
        self.model.contains("deepseek") || self.base.contains("deepseek")
    }

    fn is_deepseek_r1(&self) -> bool {
        self.model.contains("deepseek.r1") || self.base.contains("deepseek.r1")
    }

    /// `_model_accepts_anthropic_thinking_param`.
    fn accepts_anthropic_thinking(&self) -> Result<bool, Error> {
        if self.is_deepseek() {
            return Ok(false);
        }
        Ok(self.model.contains("claude-3-7")
            || self.model.contains("claude-sonnet-4")
            || self.model.contains("claude-opus-4")
            || self.supports("supports_reasoning")?)
    }

    /// `bedrock_converse_supports_parallel_tool_use_config`.
    pub fn supports_parallel_tool_use_config(&self) -> Result<bool, Error> {
        capability(
            &self.context,
            &[self.model.clone(), self.base.clone()],
            "supports_parallel_tool_use_config",
        )
        .map(|flag| flag == Some(true))
    }

    /// `bedrock_model_accepts_cache_points`.
    pub fn accepts_cache_points(&self) -> bool {
        let openai_family = self
            .model
            .match_indices("openai.")
            .any(|(at, _)| at == 0 || matches!(self.model.as_bytes()[at - 1], b'.' | b'/'));
        if openai_family {
            return false;
        }
        let Some(info) = self.context.model_info else {
            return true;
        };
        let map_keys = [self.model.clone(), self.base.clone()];
        let registered = ["", "converse/", "converse_like/"]
            .map(|route| format!("bedrock/{route}{}", self.model));
        let explicit = registered
            .iter()
            .chain(map_keys.iter())
            .filter_map(|key| info.entry(key))
            .find_map(|entry| {
                entry
                    .get("supports_prompt_cache_breakpoint")
                    .filter(|flag| !flag.is_null())
                    .map(|flag| flag == &Value::Bool(true))
            });
        if let Some(explicit) = explicit {
            return explicit;
        }
        let entries: Vec<_> = map_keys.iter().filter_map(|key| info.entry(key)).collect();
        entries.is_empty()
            || entries
                .iter()
                .any(|entry| entry.get("supports_prompt_caching") == Some(&Value::Bool(true)))
    }

    /// `supports_bedrock_prompt_cache_ttl`.
    pub fn cache_ttl_supported(&self) -> bool {
        model_field(
            &self.context,
            &[self.model.clone(), self.base.clone()],
            "cache_creation_input_token_cost_above_1hr",
        )
        .is_some()
    }

    /// `bedrock_converse_supports_strict_tools`.
    pub fn supports_strict_tools(&self) -> bool {
        if !self.is_anthropic() {
            return false;
        }
        self.context
            .model_info
            .and_then(|info| {
                [self.base.clone(), format!("bedrock/{}", self.base)]
                    .iter()
                    .filter_map(|key| info.entry(key))
                    .find_map(|entry| {
                        entry
                            .get("bedrock_converse_supports_strict_tools")
                            .and_then(Value::as_bool)
                    })
            })
            .unwrap_or(true)
    }

    /// `bedrock_converse_supports_tool_result_images`.
    pub fn supports_tool_result_images(&self) -> bool {
        let Some(info) = self.context.model_info else {
            return true;
        };
        let keys = [
            self.model.clone(),
            strip_bedrock_routing_prefix(&self.model),
        ];
        keys.iter().filter_map(|key| info.entry(key)).all(|entry| {
            entry.get("supports_bedrock_converse_tool_result_images") != Some(&Value::Bool(false))
        })
    }

    /// `bedrock_converse_accepts_redacted_thinking` for a model id that names its model.
    pub fn accepts_redacted_thinking(&self) -> bool {
        self.base.starts_with("anthropic") || self.base.starts_with("claude")
    }

    /// `normalize_bedrock_opus_output_config_effort`: an effort above the model's ceiling
    /// comes down to it.
    pub fn normalize_output_config_effort(&self, output_config: &mut Map<String, Value>) {
        let Some(effort) = output_config.get("effort").and_then(Value::as_str) else {
            return;
        };
        let Some(rank) = OUTPUT_CONFIG_EFFORT_ORDER.iter().position(|e| *e == effort) else {
            return;
        };
        let ceiling = model_field(
            &self.context,
            &self.candidates(),
            "bedrock_output_config_effort_ceiling",
        );
        let Some(ceiling) = ceiling.as_ref().and_then(Value::as_str) else {
            return;
        };
        if let Some(ceiling_rank) = OUTPUT_CONFIG_EFFORT_ORDER
            .iter()
            .position(|e| *e == ceiling)
            && rank > ceiling_rank
        {
            output_config.insert("effort".into(), json!(ceiling));
        }
    }

    /// `_validate_anthropic_adaptive_effort`.
    pub fn validate_adaptive_effort(&self, effort: &Value) -> Result<(), Error> {
        let text = effort
            .as_str()
            .filter(|effort| OUTPUT_CONFIG_EFFORT_ORDER.contains(effort));
        let Some(text) = text else {
            let shown = match effort {
                Value::String(text) => format!("'{text}'"),
                other => python_dumps(other),
            };
            return Err(unsupported_value(format!(
                "Invalid reasoning_effort/output_config.effort value: {shown}. Must be one of: 'low', 'medium', 'high', 'xhigh', or 'max'."
            )));
        };
        let refused = match text {
            "max"
                if !(self.caps.is_adaptive()?
                    || self.caps.supports("supports_max_reasoning_effort")?) =>
            {
                Some("max")
            }
            "xhigh" if !self.caps.supports("supports_xhigh_reasoning_effort")? => Some("xhigh"),
            _ => None,
        };
        match refused {
            Some(value) => Err(Error::InvalidRequest(ErrorDetail::UnsupportedValue {
                field: "effort",
                value,
                model: self.model.clone(),
            })),
            None => Ok(()),
        }
    }

    /// `AnthropicConfig.model_supports_effort_param` under `bedrock`.
    pub fn supports_effort_param(&self) -> Result<bool, Error> {
        if self.caps.supports("supports_output_config")? {
            return Ok(true);
        }
        for level in ["low", "minimal", "medium", "high", "xhigh", "max"] {
            if self
                .caps
                .supports(&format!("supports_{level}_reasoning_effort"))?
            {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// `get_supported_openai_params`.
pub fn supported_params(model: &BedrockModel<'_>) -> Result<Vec<&'static str>, Error> {
    let mut params = vec![
        "max_tokens",
        "max_completion_tokens",
        "stream",
        "stream_options",
        "stop",
    ];
    if model.supports_sampling()? {
        params.extend(["temperature", "top_p"]);
    }
    params.extend([
        "extra_headers",
        "response_format",
        "requestMetadata",
        "service_tier",
        "parallel_tool_calls",
    ]);
    if model.model.contains("arn") {
        params.extend([
            "tools",
            "tool_choice",
            "thinking",
            "reasoning_effort",
            "output_config",
        ]);
        if model.model.to_lowercase().contains("nova") {
            params.push("web_search_options");
        }
        return Ok(params);
    }
    let base = model.base.as_str();
    if [
        "anthropic",
        "mistral",
        "cohere",
        "meta.llama3-1",
        "meta.llama3-2",
        "meta.llama3-3",
        "meta.llama4",
        "amazon.nova",
    ]
    .iter()
    .any(|family| base.starts_with(family))
        || model.supports("supports_function_calling")?
    {
        params.push("tools");
    }
    if base.starts_with("amazon.nova") {
        params.push("web_search_options");
    }
    if model.supports("supports_tool_choice")? {
        params.push("tool_choice");
    }
    if model.model.contains("gpt-oss")
        || is_openai_gpt_reasoning_model(&model.model)
        || is_openai_gpt_reasoning_model(base)
    {
        params.push("reasoning_effort");
    } else if model.is_deepseek() {
        if !model.is_deepseek_r1() {
            params.push("reasoning_effort");
        }
    } else if is_nova_2_model(&model.model) {
        params.push("reasoning_effort");
    } else if model.accepts_anthropic_thinking()? {
        params.extend(["thinking", "reasoning_effort", "output_config"]);
    }
    if model.is_anthropic() {
        params.push("context_management");
    }
    Ok(params)
}

/// `map_tool_choice_values`.
pub fn map_tool_choice_values(
    model: &BedrockModel<'_>,
    tool_choice: &Value,
    drop: DropPolicy,
) -> Result<Option<Value>, Error> {
    // `forced_tool_use_downgraded`: refused without `drop_params`, `auto` with it.
    let downgraded = || -> Result<bool, Error> {
        if !model.caps.forced_tool_use_unsupported()? {
            return Ok(false);
        }
        drop.unsupported(|| forced_tool_use_refusal(&model.model))?;
        Ok(true)
    };
    match tool_choice {
        Value::String(choice) if choice == "none" => {
            drop.unsupported(|| {
                unsupported_value(
                    "Bedrock doesn't support tool_choice=none. To drop it from the call, set `litellm.drop_params = True.",
                )
            })?;
            Ok(None)
        }
        Value::String(choice) if choice == "required" => Ok(Some(match downgraded()? {
            true => json!({"auto": {}}),
            false => json!({"any": {}}),
        })),
        Value::String(choice) if choice == "auto" => Ok(Some(json!({"auto": {}}))),
        Value::Object(choice) => {
            if downgraded()? {
                return Ok(Some(json!({"auto": {}})));
            }
            let name = choice
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            Ok(Some(
                json!({"tool": {"name": make_valid_bedrock_tool_name(name)}}),
            ))
        }
        other => Err(unsupported_value(format!(
            "Bedrock doesn't support tool_choice={}. Supported tool_choice values=['auto', 'required', json object]. To drop it from the call, set `litellm.drop_params = True.",
            python_str(other)
        ))),
    }
}

fn forced_tool_use_refusal(model: &str) -> Error {
    unsupported_value(format!(
        "{model} does not support forced tool use (tool_choice='required' or a named tool). Use tool_choice='auto' and tell the model in the prompt when to call the tool, or set `litellm.drop_params = True` to downgrade to 'auto' automatically."
    ))
}

/// `make_valid_bedrock_tool_name`: `[a-zA-Z][a-zA-Z0-9_-]*`.
pub fn make_valid_bedrock_tool_name(name: &str) -> String {
    let Some(first) = name.chars().next() else {
        return String::new();
    };
    let prefixed = match first.is_alphabetic() {
        true => name.to_owned(),
        false => format!("a{name}"),
    };
    prefixed
        .chars()
        .map(|c| match c.is_alphanumeric() || c == '_' || c == '-' {
            true => c,
            false => '_',
        })
        .collect()
}

/// `_add_additional_properties_to_schema`.
fn with_closed_objects(schema: &Value) -> Value {
    let Value::Object(object) = schema else {
        return schema.clone();
    };
    let mut result = object.clone();
    if result.get("type").and_then(Value::as_str) == Some("object")
        && !result.contains_key("additionalProperties")
    {
        result.insert("additionalProperties".into(), json!(false));
    }
    for key in ["properties", "$defs", "definitions"] {
        if let Some(Value::Object(children)) = result.get(key).cloned() {
            let closed: Map<String, Value> = children
                .iter()
                .map(|(name, child)| (name.clone(), with_closed_objects(child)))
                .collect();
            result.insert(key.into(), Value::Object(closed));
        }
    }
    if let Some(items @ Value::Object(_)) = result.get("items").cloned() {
        result.insert("items".into(), with_closed_objects(&items));
    }
    for key in ["anyOf", "allOf", "oneOf"] {
        if let Some(Value::Array(items)) = result.get(key).cloned() {
            result.insert(
                key.into(),
                Value::Array(items.iter().map(with_closed_objects).collect()),
            );
        }
    }
    Value::Object(result)
}

/// `_create_output_config_for_response_format`.
pub fn output_config_for_response_format(
    schema: Option<&Value>,
    name: Option<&Value>,
    description: Option<&Value>,
) -> Value {
    let schema_text = match schema {
        Some(schema) => python_dumps(&with_closed_objects(schema)),
        None => "{}".to_owned(),
    };
    let mut json_schema = Map::from_iter([("schema".to_owned(), json!(schema_text))]);
    if let Some(name) = name.filter(|v| !v.is_null()) {
        json_schema.insert("name".into(), name.clone());
    }
    if let Some(description) = description.filter(|v| !v.is_null()) {
        json_schema.insert("description".into(), description.clone());
    }
    json!({"textFormat": {"type": "json_schema", "structure": {"jsonSchema": json_schema}}})
}

/// What `map_openai_params` hands to the request build.
#[derive(Debug, Default)]
pub struct MappedParams {
    pub params: Map<String, Value>,
    pub json_mode: bool,
    /// The schema of the `json_tool_call` tool a `response_format` added.
    pub response_format_schema: Option<ResponseFormatSchema>,
}

fn add_tools(out: &mut Map<String, Value>, tools: Vec<Value>) {
    match out.get_mut("tools") {
        Some(Value::Array(existing)) => existing.extend(tools),
        _ => {
            out.insert("tools".into(), Value::Array(tools));
        }
    }
}

/// `_handle_reasoning_effort_parameter`.
fn map_reasoning_effort_param(
    out: &mut Map<String, Value>,
    model: &BedrockModel<'_>,
    effort: &str,
) -> Result<(), Error> {
    if model.model.contains("gpt-oss") || model.model.contains("deepseek") {
        out.insert("reasoning_effort".into(), json!(effort));
    } else if is_openai_gpt_reasoning_model(&model.model) {
        out.insert("reasoning".into(), json!({"effort": effort}));
    } else if is_nova_2_model(&model.model) {
        if !matches!(effort, "low" | "medium" | "high") {
            return Err(unsupported_value(format!(
                "Invalid reasoning_effort value '{effort}' for Nova 2 models. Supported values: ['low', 'medium', 'high']"
            )));
        }
        out.insert(
            "reasoningConfig".into(),
            json!({"type": "enabled", "maxReasoningEffort": effort}),
        );
    } else {
        match map_reasoning_effort(effort, &model.caps)? {
            None => {
                out.remove("thinking");
                out.remove("output_config");
            }
            Some(thinking) => {
                out.insert("thinking".into(), thinking);
                if model.caps.is_adaptive()? {
                    let mapped = output_config_effort(effort)
                        .ok_or_else(|| unmapped_reasoning_effort(effort))?;
                    let mut output_config = match out.get("output_config") {
                        Some(Value::Object(existing)) => existing.clone(),
                        _ => Map::new(),
                    };
                    output_config
                        .entry("effort")
                        .or_insert_with(|| json!(mapped));
                    model.normalize_output_config_effort(&mut output_config);
                    if let Some(effort) = output_config.get("effort") {
                        model.validate_adaptive_effort(effort)?;
                    }
                    out.insert("output_config".into(), Value::Object(output_config));
                    out.insert("_output_config_normalized".into(), json!(true));
                }
            }
        }
    }
    Ok(())
}

/// `AmazonConverseConfig.map_openai_params`.
pub fn map_openai_params(
    openai: &Map<String, Value>,
    model: &BedrockModel<'_>,
    drop: DropPolicy,
) -> Result<MappedParams, Error> {
    let thinking_requested = is_thinking_enabled(openai);
    let mut out = Map::new();
    let mut json_mode = false;
    let mut response_format_schema = None;
    for (name, value) in ordered(openai) {
        match name {
            "response_format" => {
                let Some(format) = value.as_object() else {
                    continue;
                };
                if format.get("type").and_then(Value::as_str) == Some("text") {
                    continue;
                }
                let (schema, schema_name, description) = match format.get("response_schema") {
                    Some(schema) => (Some(schema.clone()), None, None),
                    None => match format.get("json_schema") {
                        Some(json_schema) => (
                            json_schema.get("schema").cloned(),
                            json_schema.get("name").cloned(),
                            json_schema.get("description").cloned(),
                        ),
                        None => (None, None, None),
                    },
                };
                let schema = schema.filter(|schema| !schema.is_null());
                if schema.is_some() && model.supports("supports_native_structured_output")? {
                    out.insert(
                        "outputConfig".into(),
                        output_config_for_response_format(
                            schema.as_ref(),
                            schema_name.as_ref(),
                            description.as_ref(),
                        ),
                    );
                } else if let Some(schema) = schema {
                    let mut function = Map::from_iter([
                        ("name".to_owned(), json!(RESPONSE_FORMAT_TOOL_NAME)),
                        ("parameters".to_owned(), schema.clone()),
                    ]);
                    if let Some(description) =
                        description.filter(|d| d.as_str().is_some_and(|d| !d.is_empty()))
                    {
                        function.insert("description".into(), description);
                    }
                    add_tools(
                        &mut out,
                        vec![json!({"type": "function", "function": function})],
                    );
                    response_format_schema = ResponseFormatSchema::from_schema(&schema);
                    if model.supports("supports_tool_choice")?
                        && !thinking_requested
                        && !model.caps.forced_tool_use_unsupported()?
                    {
                        out.insert(
                            "tool_choice".into(),
                            json!({"tool": {"name": RESPONSE_FORMAT_TOOL_NAME}}),
                        );
                    }
                }
                json_mode = true;
            }
            "max_tokens" | "max_completion_tokens" => {
                let requires_min = is_openai_gpt_reasoning_model(&model.model)
                    || model.model.contains("xai.grok-");
                let value = match value.as_i64() {
                    Some(tokens) if requires_min && value.is_i64() => {
                        json!(tokens.max(OPENAI_COMPAT_MIN_MAX_TOKENS))
                    }
                    _ => value.clone(),
                };
                out.insert("maxTokens".into(), value);
            }
            "stream" => {
                out.insert("stream".into(), value.clone());
            }
            "stop" if !rejects_stop_sequences(&model.model) => {
                let stop = match value {
                    Value::String(text) if text.is_empty() => continue,
                    Value::String(text) => json!([text]),
                    other => other.clone(),
                };
                out.insert("stopSequences".into(), stop);
            }
            "temperature" | "top_p" => {
                let output_key = match name {
                    "top_p" => "topP",
                    _ => name,
                };
                if model.is_anthropic() {
                    apply_sampling_param(&mut out, &model.caps, drop, name, value, output_key)?;
                } else if !model.supports_sampling()? {
                    drop.unsupported(|| {
                        unsupported_value(format!(
                            "{} does not support {name}={}. To drop unsupported params, set `litellm.drop_params = True`.",
                            model.model,
                            python_str(value)
                        ))
                    })?;
                } else {
                    out.insert(output_key.into(), value.clone());
                }
            }
            "tools" => {
                let Some(tools) = value.as_array() else {
                    continue;
                };
                if tools
                    .iter()
                    .any(|tool| tool.get("type").and_then(Value::as_str) == Some("tool_search"))
                {
                    return Err(Error::Unsupported("the OpenAI tool_search tool on Bedrock"));
                }
                add_tools(&mut out, tools.clone());
            }
            "tool_choice" => {
                if let Some(choice) = map_tool_choice_values(model, value, drop)? {
                    out.insert("tool_choice".into(), choice);
                }
            }
            "parallel_tool_calls" => {
                out.insert(
                    "_parallel_tool_use_config".into(),
                    json!({"tool_choice": {"type": "auto", "disable_parallel_tool_use": !crate::base_llm::chat::params::truthy(value)}}),
                );
            }
            "thinking" if model.is_deepseek() => {}
            "thinking" if !is_openai_gpt_reasoning_model(&model.model) => {
                let adaptive_request =
                    value.get("type").and_then(Value::as_str) == Some("adaptive");
                if adaptive_request && !model.caps.is_adaptive()? {
                    let max_tokens = openai
                        .get("max_completion_tokens")
                        .filter(|v| crate::base_llm::chat::params::truthy(v))
                        .or_else(|| openai.get("max_tokens"))
                        .filter(|v| !v.is_null());
                    let legacy = map_reasoning_effort("medium", &model.caps)?;
                    if let Some(capped) =
                        legacy.and_then(|legacy| cap_thinking_budget(legacy, max_tokens))
                    {
                        out.insert("thinking".into(), capped);
                    }
                } else {
                    out.insert("thinking".into(), value.clone());
                    translate_legacy_thinking(&mut out, &model.caps, drop)?;
                }
            }
            "reasoning_effort" => {
                let Some(effort) = value.as_str() else {
                    continue;
                };
                if model.is_deepseek_r1() {
                    continue;
                }
                if is_openai_gpt_reasoning_model(&model.model) {
                    return Err(Error::Unsupported(
                        "reasoning_effort for an OpenAI GPT model on Bedrock Converse",
                    ));
                }
                map_reasoning_effort_param(&mut out, model, effort)?;
            }
            "context_management" => {
                if let Some(mapped) = match value {
                    Value::Array(_) => map_context_management(value),
                    Value::Object(_) => Some(value.clone()),
                    _ => None,
                } {
                    out.insert("context_management".into(), mapped);
                }
            }
            "service_tier" => {
                let Some(tier) = value.as_str() else { continue };
                let tier = match tier {
                    "auto" => "default",
                    other => other,
                };
                if matches!(tier, "default" | "flex" | "priority") {
                    out.insert("serviceTier".into(), json!({"type": tier}));
                }
            }
            "web_search_options"
                if value.is_object() && model.model.to_lowercase().contains("nova") =>
            {
                add_tools(
                    &mut out,
                    vec![json!({"systemTool": {"name": "nova_grounding"}})],
                );
            }
            _ => {}
        }
    }
    if !model.model.contains("gpt-oss") && !is_nova_2_model(&model.model) {
        if let Some(Value::Object(thinking)) = out.get_mut("thinking")
            && let Some(budget) = thinking.get("budget_tokens").and_then(Value::as_i64)
            && thinking.get("budget_tokens").is_some_and(Value::is_i64)
            && budget < MIN_THINKING_BUDGET_TOKENS
        {
            thinking.insert("budget_tokens".into(), json!(MIN_THINKING_BUDGET_TOKENS));
        }
        let max_tokens_in_request =
            openai.contains_key("max_tokens") || openai.contains_key("max_completion_tokens");
        if is_thinking_enabled(&out)
            && !max_tokens_in_request
            && let Some(budget) = out
                .get("thinking")
                .and_then(|thinking| thinking.get("budget_tokens"))
                .and_then(Value::as_i64)
        {
            out.insert(
                "maxTokens".into(),
                json!(budget + DEFAULT_MAX_TOKENS as i64),
            );
        }
    }
    if is_thinking_enabled(&out)
        && out
            .get("tool_choice")
            .and_then(Value::as_object)
            .is_some_and(|choice| choice.contains_key("any") || choice.contains_key("tool"))
    {
        out.insert("tool_choice".into(), json!({"auto": {}}));
    }
    Ok(MappedParams {
        params: out,
        json_mode,
        response_format_schema,
    })
}

/// `_validate_request_metadata`.
pub fn validate_request_metadata(metadata: &Value) -> Result<(), Error> {
    let Some(metadata) = metadata.as_object() else {
        return Err(unsupported_value("requestMetadata must be a dictionary"));
    };
    if metadata.len() > 16 {
        return Err(unsupported_value(
            "requestMetadata can contain a maximum of 16 items",
        ));
    }
    let allowed =
        |c: char| c.is_ascii_alphanumeric() || c.is_whitespace() || ":_@$#=/+,.-".contains(c);
    for (key, value) in metadata {
        let Some(value) = value.as_str() else {
            return Err(unsupported_value("requestMetadata values must be strings"));
        };
        if key.is_empty() || key.chars().count() > 256 {
            return Err(unsupported_value(
                "requestMetadata key length must be 1-256 characters",
            ));
        }
        if value.chars().count() > 256 {
            return Err(unsupported_value(
                "requestMetadata value length must be 0-256 characters",
            ));
        }
        if !key.chars().all(allowed) {
            return Err(unsupported_value(format!(
                "requestMetadata key '{key}' contains invalid characters. Allowed: [a-zA-Z0-9\\s:_@$#=/+,.-]"
            )));
        }
        if !value.chars().all(allowed) {
            return Err(unsupported_value(format!(
                "requestMetadata value '{value}' contains invalid characters. Allowed: [a-zA-Z0-9\\s:_@$#=/+,.-]"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(
        "us.meta.llama3-2-11b-instruct-v1:0",
        "meta.llama3-2-11b-instruct-v1:0"
    )]
    #[case(
        "bedrock/converse/anthropic.claude-3-5-sonnet-20240620-v1:0",
        "anthropic.claude-3-5-sonnet-20240620-v1:0"
    )]
    #[case(
        "anthropic.claude-3-5-sonnet-20241022-v2:0:51k",
        "anthropic.claude-3-5-sonnet-20241022-v2:0"
    )]
    #[case("bedrock/nova-2/arn:aws:bedrock:x", "amazon.nova-2-custom")]
    #[case(
        "arn:aws:bedrock:us-east-1:1:inference-profile/us.anthropic.claude-x",
        "anthropic.claude-x"
    )]
    #[case("us-west-2/mistral.mistral-large", "mistral.mistral-large")]
    #[case("global.anthropic.claude-sonnet-4-5", "anthropic.claude-sonnet-4-5")]
    fn base_models_match_python(#[case] model: &str, #[case] expected: &str) {
        assert_eq!(bedrock_base_model(model), expected);
    }

    #[rstest]
    #[case("get weather", "get_weather")]
    #[case("1tool", "a1tool")]
    #[case("", "")]
    #[case("ok-name_1", "ok-name_1")]
    fn tool_names_are_made_valid(#[case] name: &str, #[case] expected: &str) {
        assert_eq!(make_valid_bedrock_tool_name(name), expected);
    }

    #[rstest]
    fn gpt_versions_parse() {
        assert_eq!(openai_gpt_version("openai.gpt-5.6"), Some((5, 6)));
        assert!(rejects_stop_sequences("us.xai.grok-4"));
        assert!(!rejects_stop_sequences("openai.gpt-oss-20b-1:0"));
    }

    #[rstest]
    fn request_metadata_is_validated() {
        assert!(validate_request_metadata(&json!({"a": "b"})).is_ok());
        assert_eq!(
            validate_request_metadata(&json!({"a": 1}))
                .unwrap_err()
                .to_string(),
            "invalid request: requestMetadata values must be strings"
        );
    }
}
