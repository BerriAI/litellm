//! Python's `AnthropicConfig.map_openai_params` and the model-map reads behind it: the
//! OpenAI params a chat request carries, as the Messages API takes them.

use serde_json::{Map, Value, json};

use crate::{
    Error, ErrorDetail,
    base_llm::chat::{
        params::{
            ChatRequestContext, DEFAULT_MAX_TOKENS, DropPolicy, RESPONSE_FORMAT_TOOL_NAME,
            capability, ordered, truthy, unsupported_value,
        },
        response_format::{ResponseFormatSchema, unpack_defs, unpack_legacy_defs},
    },
    bedrock::chat::converse_params::bedrock_base_model,
};

/// `ANTHROPIC_MIN_THINKING_BUDGET_TOKENS`.
pub const MIN_THINKING_BUDGET_TOKENS: i64 = 1024;

/// The thinking budgets `_map_reasoning_effort` gives a legacy-thinking model.
pub fn reasoning_effort_budget(effort: &str) -> Option<i64> {
    Some(match effort {
        "low" => 1024,
        "medium" => 2048,
        "high" => 4096,
        "xhigh" => 8192,
        "max" => 16384,
        // `max(DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET, ANTHROPIC_MIN_THINKING_BUDGET_TOKENS)`
        "minimal" => 128.max(MIN_THINKING_BUDGET_TOKENS),
        _ => return None,
    })
}

/// `REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT`.
pub fn output_config_effort(effort: &str) -> Option<&'static str> {
    Some(match effort {
        "low" | "minimal" => "low",
        "medium" => "medium",
        "high" => "high",
        "xhigh" => "xhigh",
        "max" => "max",
        _ => return None,
    })
}

pub fn unmapped_reasoning_effort(effort: &str) -> Error {
    Error::InvalidRequest(ErrorDetail::InvalidChoice {
        field: "reasoning effort",
        actual: format!("'{effort}'"),
        choices: vec!["minimal", "low", "medium", "high", "xhigh", "max", "none"],
    })
}

/// Strips a trailing `-v<N>` or `-v<N>:<M>`.
fn strip_bedrock_version(model: &str) -> String {
    let bytes = model.as_bytes();
    let mut end = bytes.len();
    // Optional `:<digits>`.
    let mut cut = end;
    while cut > 0 && bytes[cut - 1].is_ascii_digit() {
        cut -= 1;
    }
    if cut < end && cut > 0 && bytes[cut - 1] == b':' {
        end = cut - 1;
    }
    let mut digits = end;
    while digits > 0 && bytes[digits - 1].is_ascii_digit() {
        digits -= 1;
    }
    if digits < end && digits >= 2 && &bytes[digits - 2..digits] == b"-v" {
        return model[..digits - 2].to_owned();
    }
    // The `:<M>` alone is not a version suffix, so only the full shape is removed.
    model.to_owned()
}

/// Strips a trailing `:<digits>`.
fn strip_profile_minor(model: &str) -> String {
    let trimmed = model.trim_end_matches(|c: char| c.is_ascii_digit());
    match trimmed.strip_suffix(':') {
        Some(head) if trimmed.len() < model.len() => head.to_owned(),
        _ => model.to_owned(),
    }
}

/// Strips a trailing `-<8 digits>`.
fn strip_dated_release(model: &str) -> String {
    let bytes = model.as_bytes();
    if bytes.len() >= 9 {
        let tail = &bytes[bytes.len() - 9..];
        if tail[0] == b'-' && tail[1..].iter().all(u8::is_ascii_digit) {
            return model[..model.len() - 9].to_owned();
        }
    }
    model.to_owned()
}

/// `(\d)\.(\d)` -> `\1-\2`, non-overlapping like `re.sub`.
fn hyphenate_dotted_versions(model: &str) -> String {
    let chars: Vec<char> = model.chars().collect();
    let mut out = String::with_capacity(model.len());
    let mut index = 0;
    while index < chars.len() {
        if index + 2 < chars.len()
            && chars[index].is_ascii_digit()
            && chars[index + 1] == '.'
            && chars[index + 2].is_ascii_digit()
        {
            out.push(chars[index]);
            out.push('-');
            out.push(chars[index + 2]);
            index += 3;
        } else {
            out.push(chars[index]);
            index += 1;
        }
    }
    out
}

fn strip_at_version(model: &str) -> String {
    match model.rfind('@') {
        Some(at) if at > 0 => model[..at].to_owned(),
        _ => model.to_owned(),
    }
}

/// `AnthropicModelInfo._model_map_lookup_candidates`.
pub fn model_map_candidates(model: &str) -> Vec<String> {
    let mut primary = vec![model.to_owned()];
    for prefix in [
        "bedrock/converse/",
        "bedrock/invoke/",
        "bedrock/",
        "vertex_ai/",
    ] {
        if let Some(rest) = model.strip_prefix(prefix) {
            primary.push(rest.to_owned());
        }
    }
    let base = bedrock_base_model(model);
    if !base.is_empty() {
        primary.push(base.clone());
        primary.push(format!("bedrock/{base}"));
    }
    let mut candidates: Vec<String> = Vec::new();
    let mut push = |candidate: String| {
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    };
    for candidate in &primary {
        push(candidate.clone());
    }
    for candidate in &primary {
        push(strip_bedrock_version(candidate));
        push(strip_profile_minor(candidate));
        push(strip_dated_release(candidate));
        push(hyphenate_dotted_versions(candidate));
        push(strip_dated_release(&strip_bedrock_version(
            &hyphenate_dotted_versions(candidate),
        )));
        push(strip_at_version(candidate));
    }
    candidates
}

/// The host an Anthropic Messages request built from a chat call goes to. The request is the
/// same; the cost-map reads and a few params depend on the host, as Python's
/// `custom_llm_provider` on the config decides them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AnthropicHost {
    /// The Anthropic API, `AnthropicConfig`.
    #[default]
    Anthropic,
    /// Claude on Vertex AI, `VertexAIAnthropicConfig`.
    VertexAi,
}

impl AnthropicHost {
    /// The provider an unsupported-params error names.
    pub fn provider(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::VertexAi => "vertex_ai",
        }
    }
}

/// `_strip_stable_vertex_version`: a trailing `-<digits>` removed.
fn strip_stable_vertex_version(model: &str) -> String {
    let trimmed = model.trim_end_matches(|c: char| c.is_ascii_digit());
    match trimmed.strip_suffix('-') {
        Some(head) if trimmed.len() < model.len() => head.to_owned(),
        _ => model.to_owned(),
    }
}

/// The keys `get_model_info_helper` tries for a Vertex AI model, most specific first: the
/// `vertex_ai/` entry, the bare id, then both with a stable version suffix removed.
pub fn vertex_model_info_candidates(model: &str) -> Vec<String> {
    let stripped = strip_stable_vertex_version(model);
    let mut candidates: Vec<String> = Vec::new();
    for candidate in [
        format!("vertex_ai/{model}"),
        model.to_owned(),
        format!("vertex_ai/{stripped}"),
        stripped,
    ] {
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    }
    candidates
}

/// The cost-map reads a model's request rules make, through its lookup candidates.
pub struct ModelCapabilities<'a> {
    pub model: String,
    pub context: ChatRequestContext<'a>,
    pub candidates: Vec<String>,
    pub host: AnthropicHost,
    /// The keys a provider-aware lookup (`_get_provider_resolved_capability`) resolves the
    /// model through; the first entry found answers. Empty when the host has none.
    pub resolved: Vec<String>,
}

impl<'a> ModelCapabilities<'a> {
    pub fn new(model: &str, context: &ChatRequestContext<'a>) -> Self {
        Self::on(AnthropicHost::Anthropic, model, context)
    }

    pub fn on(host: AnthropicHost, model: &str, context: &ChatRequestContext<'a>) -> Self {
        Self {
            model: model.to_owned(),
            context: *context,
            candidates: model_map_candidates(model),
            host,
            resolved: match host {
                AnthropicHost::Anthropic => Vec::new(),
                AnthropicHost::VertexAi => vertex_model_info_candidates(model),
            },
        }
    }

    /// `_get_model_capability`: the flag the first candidate the map knows sets.
    pub fn walk_flag(&self, key: &str) -> Result<Option<bool>, Error> {
        capability(&self.context, &self.candidates, key)
    }

    /// `_get_provider_resolved_capability`: the flag of the entry the host's own lookup
    /// resolves the model to, `None` when that entry does not set it.
    pub fn resolved_flag(&self, key: &str) -> Result<Option<bool>, Error> {
        if self.resolved.is_empty() {
            return Ok(None);
        }
        let Some(info) = self.context.model_info else {
            return match self.context.drop_params {
                None => Err(Error::Unsupported(
                    "a request whose translation reads the model cost map",
                )),
                Some(_) => Ok(None),
            };
        };
        Ok(self
            .resolved
            .iter()
            .find_map(|candidate| info.entry(candidate))
            .and_then(|entry| entry.get(key).and_then(Value::as_bool)))
    }

    /// `_supports_model_capability`'s answer: the host's resolved flag when it sets one, else
    /// the model-map walk.
    pub fn flag(&self, key: &str) -> Result<Option<bool>, Error> {
        match self.resolved_flag(key)? {
            Some(flag) => Ok(Some(flag)),
            None => self.walk_flag(key),
        }
    }

    /// `_supports_model_capability`: the flag is set to `true`.
    pub fn supports(&self, key: &str) -> Result<bool, Error> {
        Ok(self.flag(key)? == Some(true))
    }

    /// `uses_native_structured_output`: Vertex AI reads only its own entry for it.
    pub fn native_structured_output(&self) -> Result<bool, Error> {
        match self.host {
            AnthropicHost::Anthropic => self.supports("supports_native_structured_output"),
            AnthropicHost::VertexAi => {
                Ok(self.resolved_flag("supports_native_structured_output")? == Some(true))
            }
        }
    }

    /// `_model_supports_effort_param`: `output_config` or any reasoning effort level.
    pub fn supports_effort(&self) -> Result<bool, Error> {
        if self.supports("supports_output_config")? {
            return Ok(true);
        }
        for level in ["low", "minimal", "medium", "high", "xhigh", "max"] {
            if self.supports(&format!("supports_{level}_reasoning_effort"))? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// `_supports_sampling_params`.
    pub fn supports_sampling(&self) -> Result<bool, Error> {
        if let Some(flag) = self.walk_flag("supports_sampling_params")? {
            return Ok(flag);
        }
        let lower = self.model.to_lowercase();
        Ok(![
            "fable", "opus-4-7", "opus_4_7", "opus-4.7", "opus_4.7", "opus-4-8", "opus_4_8",
            "opus-4.8", "opus_4.8",
        ]
        .iter()
        .any(|name| lower.contains(name)))
    }

    pub fn forced_tool_use_unsupported(&self) -> Result<bool, Error> {
        Ok(self.walk_flag("supports_forced_tool_use")? == Some(false))
    }

    pub fn is_adaptive(&self) -> Result<bool, Error> {
        self.supports("supports_adaptive_thinking")
    }
}

/// Python's `value` in an error message: its `str()`.
pub fn python_str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Null => "None".into(),
        other => other.to_string(),
    }
}

/// `_apply_sampling_param`: the param, unless the model removed sampling params.
pub fn apply_sampling_param(
    out: &mut Map<String, Value>,
    caps: &ModelCapabilities<'_>,
    drop: DropPolicy,
    param: &str,
    value: &Value,
    output_key: &str,
) -> Result<(), Error> {
    if caps.supports_sampling()? || (param == "temperature" && value.as_f64() == Some(1.0)) {
        out.insert(output_key.to_owned(), value.clone());
        return Ok(());
    }
    drop.unsupported(|| {
        Error::InvalidRequest(ErrorDetail::UnsupportedParameter {
            model: caps.model.clone(),
            param: param.to_owned(),
            value: python_str(value),
            hint: match param {
                "temperature" => "Only temperature=1 is supported. ".into(),
                _ => String::new(),
            },
        })
    })
}

/// `_valid_user_id`: Anthropic rejects an email or a phone number as `metadata.user_id`.
pub fn valid_user_id(user_id: &str) -> bool {
    !(looks_like_email(user_id) || looks_like_phone(user_id))
}

fn looks_like_email(text: &str) -> bool {
    let Some((local, domain)) = text.split_once('@') else {
        return false;
    };
    let local_ok = !local.is_empty()
        && local
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "._%+-".contains(c));
    let domain_ok = domain
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
        && domain.rsplit_once('.').is_some_and(|(head, tld)| {
            !head.is_empty() && tld.len() >= 2 && tld.chars().all(|c| c.is_ascii_alphabetic())
        });
    local_ok && domain_ok
}

fn looks_like_phone(text: &str) -> bool {
    let body = text.strip_prefix('+').unwrap_or(text);
    body.chars().count() >= 7
        && body
            .chars()
            .all(|c| c.is_numeric() || c.is_whitespace() || "()-".contains(c))
}

/// `_map_stop_sequences`: whitespace-only entries are dropped only under `drop_params`.
pub fn map_stop_sequences(stop: &Value, drop: DropPolicy) -> Option<Value> {
    let keep = |text: &str| !(is_python_space(text) && drop.drops());
    match stop {
        Value::String(text) => keep(text).then(|| json!([text])),
        Value::Array(items) => {
            let kept: Vec<Value> = items
                .iter()
                .filter(|item| item.as_str().is_none_or(keep))
                .cloned()
                .collect();
            (!kept.is_empty()).then_some(Value::Array(kept))
        }
        _ => None,
    }
}

/// `str.isspace()`: non-empty and all whitespace.
pub fn is_python_space(text: &str) -> bool {
    !text.is_empty() && text.chars().all(char::is_whitespace)
}

/// `sanitize_input_schema_for_anthropic`.
pub fn sanitize_input_schema(schema: Option<&Value>) -> Value {
    let mut normalized = match schema {
        Some(Value::Object(object)) => object.clone(),
        _ => Map::new(),
    };
    if normalized.get("type").and_then(Value::as_str) != Some("object") {
        normalized.insert("type".into(), json!("object"));
    }
    normalized
        .entry("properties")
        .or_insert_with(|| Value::Object(Map::new()));
    unpack_legacy_defs(&mut normalized);
    Value::Object(
        normalized
            .into_iter()
            .filter(|(key, _)| {
                matches!(
                    key.as_str(),
                    "type"
                        | "properties"
                        | "additionalProperties"
                        | "required"
                        | "$defs"
                        | "strict"
                )
            })
            .collect(),
    )
}

fn invalid_tool(message: &str) -> Error {
    unsupported_value(message.to_owned())
}

/// `_map_tool_helper` for the tool shapes chat translates: function and custom tools.
fn map_function_tool(tool: &Map<String, Value>) -> Result<Value, Error> {
    let function = tool.get("function").and_then(Value::as_object);
    let Some(function) = function else {
        return Err(invalid_tool("A function tool needs a `function` object."));
    };
    let name = function
        .get("name")
        .cloned()
        .ok_or_else(|| invalid_tool("A function tool needs a name."))?;
    let default_schema = json!({"type": "object", "properties": {}});
    let schema = sanitize_input_schema(Some(function.get("parameters").unwrap_or(&default_schema)));
    let mut mapped = Map::new();
    mapped.insert("name".into(), name);
    mapped.insert("input_schema".into(), schema);
    mapped.insert("type".into(), json!("custom"));
    let eager = match tool.get("eager_input_streaming") {
        Some(Value::Bool(flag)) => Some(*flag),
        Some(Value::Null) | None => match function.get("eager_input_streaming") {
            Some(Value::Bool(flag)) => Some(*flag),
            Some(Value::Null) | None => None,
            Some(_) => return Err(invalid_tool("eager_input_streaming must be a boolean")),
        },
        Some(_) => return Err(invalid_tool("eager_input_streaming must be a boolean")),
    };
    if let Some(eager) = eager {
        mapped.insert("eager_input_streaming".into(), json!(eager));
    }
    if let Some(description) = function.get("description").filter(|d| !d.is_null()) {
        mapped.insert("description".into(), description.clone());
    }
    match (tool.get("cache_control"), function.get("cache_control")) {
        (Some(cache_control), _) if !cache_control.is_null() => {
            mapped.insert("cache_control".into(), cache_control.clone());
        }
        (_, Some(Value::Object(cache_control))) => {
            mapped.insert("cache_control".into(), Value::Object(cache_control.clone()));
        }
        _ => {}
    }
    match (tool.get("defer_loading"), function.get("defer_loading")) {
        (Some(Value::Bool(flag)), _) => {
            mapped.insert("defer_loading".into(), json!(flag));
        }
        (Some(other), _) if !other.is_null() => {
            return Err(invalid_tool("defer_loading must be a boolean"));
        }
        (_, Some(Value::Bool(flag))) => {
            mapped.insert("defer_loading".into(), json!(flag));
        }
        (_, Some(other)) if !other.is_null() => {
            return Err(invalid_tool("defer_loading must be a boolean"));
        }
        _ => {}
    }
    let string_list = |value: &Value| {
        value
            .as_array()
            .is_some_and(|items| items.iter().all(Value::is_string))
    };
    for key in ["allowed_callers", "input_examples"] {
        let value = tool
            .get(key)
            .filter(|v| !v.is_null())
            .or_else(|| function.get(key).filter(|v| !v.is_null()));
        if let Some(value) = value {
            if key == "allowed_callers" && !string_list(value) {
                return Err(invalid_tool("allowed_callers must be a list of strings"));
            }
            if key == "input_examples" && !value.is_array() {
                return Err(invalid_tool("input_examples must be a list"));
            }
            mapped.insert(key.into(), value.clone());
        }
    }
    Ok(Value::Object(mapped))
}

/// `_map_tools` for function and custom tools and tools already in Anthropic's shape.
pub fn map_tools(tools: &Value) -> Result<Vec<Value>, Error> {
    let Some(tools) = tools.as_array() else {
        return Err(invalid_tool("tools must be a list"));
    };
    let mut mapped = Vec::new();
    for tool in tools {
        let Some(object) = tool.as_object() else {
            return Err(invalid_tool("each tool must be an object"));
        };
        if object.contains_key("input_schema") {
            mapped.push(tool.clone());
            continue;
        }
        match object.get("type").and_then(Value::as_str) {
            Some("function" | "custom") => mapped.push(map_function_tool(object)?),
            Some("namespace") => {
                for nested in object
                    .get("tools")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let Some(nested_object) = nested.as_object() else {
                        continue;
                    };
                    if nested_object.contains_key("input_schema") {
                        mapped.push(nested.clone());
                    } else if !nested_object.contains_key("function")
                        && nested_object.contains_key("name")
                    {
                        let kind = nested_object
                            .get("type")
                            .cloned()
                            .unwrap_or_else(|| json!("function"));
                        let function: Map<String, Value> = nested_object
                            .iter()
                            .filter(|(key, _)| key.as_str() != "type")
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();
                        let wrapped = Map::from_iter([
                            ("type".to_owned(), kind.clone()),
                            ("function".to_owned(), Value::Object(function)),
                        ]);
                        match kind.as_str() {
                            Some("function" | "custom") => {
                                mapped.push(map_function_tool(&wrapped)?)
                            }
                            _ => {
                                return Err(Error::Unsupported("a hosted tool inside a namespace"));
                            }
                        }
                    } else if nested_object.contains_key("function") {
                        match nested_object.get("type").and_then(Value::as_str) {
                            Some("function" | "custom") => {
                                mapped.push(map_function_tool(nested_object)?)
                            }
                            _ => {
                                return Err(Error::Unsupported("a hosted tool inside a namespace"));
                            }
                        }
                    }
                }
            }
            // Computer use, hosted tools, MCP servers and tool search each have their own
            // shape and betas; they stay on Python's path.
            _ => {
                return Err(Error::Unsupported(
                    "a chat tool other than a function or custom tool",
                ));
            }
        }
    }
    Ok(mapped)
}

/// `_map_tool_choice`.
pub fn map_tool_choice(tool_choice: Option<&Value>, parallel: Option<&Value>) -> Option<Value> {
    let mut choice: Option<Map<String, Value>> = match tool_choice {
        Some(Value::String(choice)) => match choice.as_str() {
            "auto" => Some(json!({"type": "auto"})),
            "required" => Some(json!({"type": "any"})),
            "none" => Some(json!({"type": "none"})),
            _ => None,
        },
        Some(Value::Object(choice))
            if choice.contains_key("type") && !choice.contains_key("function") =>
        {
            match choice.get("type").and_then(Value::as_str) {
                Some("auto") => Some(json!({"type": "auto"})),
                Some("required" | "any") => Some(json!({"type": "any"})),
                Some("none") => Some(json!({"type": "none"})),
                _ => None,
            }
        }
        Some(Value::Object(choice)) => choice
            .get("function")
            .and_then(|function| function.get("name"))
            .filter(|name| !name.is_null())
            .map(|name| json!({"type": "tool", "name": name})),
        _ => None,
    }
    .and_then(|value| value.as_object().cloned());
    if let Some(parallel) = parallel.filter(|parallel| !parallel.is_null()) {
        let disable = json!(!truthy(parallel));
        let choice_is_none = tool_choice.and_then(Value::as_str) == Some("none");
        match &mut choice {
            _ if choice_is_none => {}
            Some(choice) => {
                choice.insert("disable_parallel_tool_use".into(), disable);
            }
            None => {
                choice = Some(Map::from_iter([
                    ("type".to_owned(), json!("auto")),
                    ("disable_parallel_tool_use".to_owned(), disable),
                ]));
            }
        }
    }
    choice.map(Value::Object)
}

/// `_apply_forced_tool_choice`: `any` / `tool` on a model that refuses forced tool use.
pub fn apply_forced_tool_choice(
    choice: Value,
    caps: &ModelCapabilities<'_>,
    drop: DropPolicy,
) -> Result<Value, Error> {
    let kind = choice.get("type").and_then(Value::as_str);
    if !matches!(kind, Some("any" | "tool")) || !caps.forced_tool_use_unsupported()? {
        return Ok(choice);
    }
    drop.unsupported(|| {
        unsupported_value(format!(
            "{} does not support forced tool use (tool_choice='required' or a named tool). Use tool_choice='auto' and tell the model in the prompt when to call the tool, or set `litellm.drop_params = True` to downgrade to 'auto' automatically.",
            caps.model
        ))
    })?;
    let mut auto = Map::from_iter([("type".to_owned(), json!("auto"))]);
    if let Some(disable) = choice.get("disable_parallel_tool_use") {
        auto.insert("disable_parallel_tool_use".into(), disable.clone());
    }
    Ok(Value::Object(auto))
}

fn enum_conflicts_with_declared_type(schema: &Map<String, Value>) -> bool {
    let (Some(Value::Array(values)), Some(declared)) = (schema.get("enum"), schema.get("type"))
    else {
        return false;
    };
    if declared.is_array() {
        return true;
    }
    let check: fn(&Value) -> bool = match declared.as_str() {
        Some("null") => Value::is_null,
        Some("boolean") => Value::is_boolean,
        Some("integer") => |v| v.is_i64() || v.is_u64(),
        Some("number") => Value::is_number,
        Some("string") => Value::is_string,
        Some("array") => Value::is_array,
        Some("object") => Value::is_object,
        _ => return false,
    };
    !values.iter().all(check)
}

const CONSTRAINT_LABELS: [(&str, &str); 25] = [
    ("minItems", "minimum number of items: {}"),
    ("maxItems", "maximum number of items: {}"),
    ("uniqueItems", "all array items must be unique"),
    ("contains", "array must contain an item matching: {}"),
    ("minContains", "minimum number of matching items: {}"),
    ("maxContains", "maximum number of matching items: {}"),
    ("prefixItems", "leading items must match, in order: {}"),
    ("minimum", "minimum value: {}"),
    ("maximum", "maximum value: {}"),
    ("exclusiveMinimum", "exclusive minimum value: {}"),
    ("exclusiveMaximum", "exclusive maximum value: {}"),
    ("multipleOf", "must be a multiple of {}"),
    ("minLength", "minimum length: {}"),
    ("maxLength", "maximum length: {}"),
    ("minProperties", "minimum number of properties: {}"),
    ("maxProperties", "maximum number of properties: {}"),
    (
        "patternProperties",
        "properties whose names match each pattern must satisfy: {}",
    ),
    ("propertyNames", "property names must satisfy: {}"),
    ("dependentRequired", "dependent required properties: {}"),
    ("dependentSchemas", "dependent schemas: {}"),
    (
        "unevaluatedProperties",
        "unevaluated properties must satisfy: {}",
    ),
    ("if", "conditional (if): {}"),
    ("then", "conditional (then): {}"),
    ("else", "conditional (else): {}"),
    ("not", "must not match: {}"),
];

/// `filter_anthropic_output_schema`: the constraints structured output cannot enforce,
/// removed and described instead.
pub fn filter_output_schema(schema: &Value) -> Value {
    let Some(object) = schema.as_object() else {
        return schema.clone();
    };
    let notes: Vec<String> = CONSTRAINT_LABELS
        .iter()
        .filter_map(|(field, label)| {
            let value = object.get(*field)?;
            if value == &Value::Bool(false) {
                return None;
            }
            let rendered = match value {
                Value::Object(_) | Value::Array(_) => {
                    crate::base_llm::chat::params::python_dumps(value)
                }
                other => python_str(other),
            };
            Some(label.replace("{}", &rendered))
        })
        .collect();
    let mut result = Map::new();
    if !notes.is_empty() {
        let note = format!("Note: {}.", notes.join(", "));
        let existing = object
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let description = match existing.is_empty() {
            true => note,
            false => format!("{existing} {note}"),
        };
        result.insert("description".into(), json!(description));
    }
    let drops_type = enum_conflicts_with_declared_type(object);
    for (key, value) in object {
        if CONSTRAINT_LABELS.iter().any(|(field, _)| field == key)
            || (key == "type" && drops_type)
            || (key == "description" && result.contains_key("description"))
        {
            continue;
        }
        match (key.as_str(), value) {
            ("properties" | "$defs", Value::Object(nested)) => {
                result.insert(
                    key.clone(),
                    Value::Object(
                        nested
                            .iter()
                            .map(|(k, v)| (k.clone(), filter_output_schema(v)))
                            .collect(),
                    ),
                );
            }
            ("items", Value::Object(_)) => {
                result.insert(key.clone(), filter_output_schema(value));
            }
            ("anyOf" | "oneOf", Value::Array(items)) => {
                let mut any_of = match result.remove("anyOf") {
                    Some(Value::Array(existing)) => existing,
                    _ => Vec::new(),
                };
                any_of.extend(items.iter().map(filter_output_schema));
                result.insert("anyOf".into(), Value::Array(any_of));
            }
            ("allOf", Value::Array(items)) => {
                result.insert(
                    key.clone(),
                    Value::Array(items.iter().map(filter_output_schema).collect()),
                );
            }
            _ => {
                result.insert(key.clone(), value.clone());
            }
        }
    }
    if result.get("type").and_then(Value::as_str) == Some("object") {
        result.insert("additionalProperties".into(), json!(false));
    }
    Value::Object(result)
}

/// The schema a `response_format` names, if any. A `json_schema` without `schema` names none.
fn response_format_schema(value: &Map<String, Value>) -> Option<Value> {
    if let Some(schema) = value.get("response_schema") {
        return Some(schema.clone()).filter(|schema| !schema.is_null());
    }
    value
        .get("json_schema")
        .and_then(|json_schema| json_schema.get("schema"))
        .filter(|schema| !schema.is_null())
        .cloned()
}

/// `map_response_format_to_anthropic_output_format`.
fn output_format(value: &Map<String, Value>) -> Option<Value> {
    let mut schema = response_format_schema(value)?;
    let mut defs = Map::new();
    if let Value::Object(object) = &mut schema {
        let definitions = object.remove("definitions");
        let dollar_defs = object.remove("$defs");
        if let Some(Value::Object(found)) = dollar_defs.or(definitions) {
            defs = found;
        }
    }
    if !defs.is_empty() {
        unpack_defs(&mut schema, &defs);
    }
    Some(json!({"type": "json_schema", "schema": filter_output_schema(&schema)}))
}

/// `_create_json_tool_call_for_response_format`.
fn response_format_tool(schema: &Value) -> Value {
    let mut input_schema = Map::from_iter([("type".to_owned(), json!("object"))]);
    if let Value::Object(schema) = schema {
        input_schema.extend(schema.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    json!({"name": RESPONSE_FORMAT_TOOL_NAME, "input_schema": input_schema})
}

/// `cap_thinking_budget_to_max_tokens`.
pub fn cap_thinking_budget(thinking: Value, max_tokens: Option<&Value>) -> Option<Value> {
    let budget = thinking.get("budget_tokens").and_then(Value::as_i64);
    let (Some(max_tokens), Some(budget)) = (max_tokens.and_then(Value::as_f64), budget) else {
        return Some(thinking);
    };
    if max_tokens <= MIN_THINKING_BUDGET_TOKENS as f64 {
        return None;
    }
    if (budget as f64) < max_tokens {
        return Some(thinking);
    }
    let kind = thinking
        .get("type")
        .cloned()
        .unwrap_or_else(|| json!("enabled"));
    Some(json!({"type": kind, "budget_tokens": (max_tokens - 1.0) as i64}))
}

/// `_map_reasoning_effort`: `None` for `none`.
pub fn map_reasoning_effort(
    effort: &str,
    caps: &ModelCapabilities<'_>,
) -> Result<Option<Value>, Error> {
    if effort == "none" {
        return Ok(None);
    }
    if caps.is_adaptive()? {
        return Ok(Some(json!({"type": "adaptive", "display": "summarized"})));
    }
    let budget =
        reasoning_effort_budget(effort).ok_or_else(|| unmapped_reasoning_effort(effort))?;
    Ok(Some(json!({"type": "enabled", "budget_tokens": budget})))
}

/// `_legacy_budget_to_effort`.
fn legacy_budget_to_effort(
    budget: i64,
    caps: &ModelCapabilities<'_>,
) -> Result<&'static str, Error> {
    Ok(
        if budget >= 8192 && caps.supports("supports_xhigh_reasoning_effort")? {
            "xhigh"
        } else if budget >= 4096 {
            "high"
        } else if budget >= 2048 {
            "medium"
        } else {
            "low"
        },
    )
}

/// `translate_legacy_thinking_for_adaptive_model`.
pub fn translate_legacy_thinking(
    out: &mut Map<String, Value>,
    caps: &ModelCapabilities<'_>,
    drop: DropPolicy,
) -> Result<(), Error> {
    if !caps.is_adaptive()? || caps.supports("supports_legacy_thinking")? {
        return Ok(());
    }
    let Some(Value::Object(thinking)) = out.get("thinking").cloned() else {
        return Ok(());
    };
    if thinking.get("type").and_then(Value::as_str) != Some("enabled") {
        return Ok(());
    }
    let budget = match thinking.get("budget_tokens") {
        Some(Value::Number(number)) => number.as_f64().unwrap_or(0.0) as i64,
        Some(Value::String(text)) => text.trim().parse().unwrap_or(0),
        _ => 0,
    };
    let effort = legacy_budget_to_effort(budget, caps)?;
    let mut rebuilt: Map<String, Value> = thinking
        .iter()
        .filter(|(key, _)| {
            !matches!(key.as_str(), "type" | "budget_tokens" | "display")
                && (key.as_str() == "block_binding" || !drop.drops())
        })
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    rebuilt.insert("type".into(), json!("adaptive"));
    if let Some(display) = thinking
        .get("display")
        .filter(|display| matches!(display.as_str(), Some("summarized" | "omitted")))
    {
        rebuilt.insert("display".into(), display.clone());
    }
    let mut output_config = Map::from_iter([("effort".to_owned(), json!(effort))]);
    if let Some(Value::Object(existing)) = out.get("output_config") {
        output_config.extend(existing.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    out.insert("thinking".into(), Value::Object(rebuilt));
    out.insert("output_config".into(), Value::Object(output_config));
    Ok(())
}

/// `web_search_options` as Anthropic's hosted web search tool.
pub fn web_search_tool(options: &Map<String, Value>) -> Value {
    let mut tool = Map::from_iter([
        ("type".to_owned(), json!("web_search_20250305")),
        ("name".to_owned(), json!("web_search")),
    ]);
    if let Some(location) = options.get("user_location").filter(|v| !v.is_null()) {
        let mut user_location = Map::from_iter([("type".to_owned(), json!("approximate"))]);
        if let Some(Value::Object(approximate)) = location.get("approximate") {
            for (key, value) in approximate {
                if matches!(key.as_str(), "city" | "region" | "country" | "timezone") {
                    user_location.insert(key.clone(), value.clone());
                }
            }
        }
        tool.insert("user_location".into(), Value::Object(user_location));
    }
    if let Some(size) = options.get("search_context_size").and_then(Value::as_str) {
        let max_uses = match size {
            "low" => Some(1),
            "medium" => Some(5),
            "high" => Some(10),
            _ => None,
        };
        if let Some(max_uses) = max_uses {
            tool.insert("max_uses".into(), json!(max_uses));
        }
    }
    Value::Object(tool)
}

/// `map_openai_context_management_to_anthropic`.
pub fn map_context_management(value: &Value) -> Option<Value> {
    match value {
        Value::Object(object) if object.contains_key("edits") => Some(value.clone()),
        Value::Array(entries) => {
            let edits: Vec<Value> = entries
                .iter()
                .filter_map(Value::as_object)
                .filter(|entry| entry.get("type").and_then(Value::as_str) == Some("compaction"))
                .map(|entry| {
                    let mut edit = Map::from_iter([("type".to_owned(), json!("compact_20260112"))]);
                    if let Some(threshold) = entry.get("compact_threshold").and_then(Value::as_f64)
                    {
                        edit.insert(
                            "trigger".into(),
                            json!({"type": "input_tokens", "value": threshold as i64}),
                        );
                    }
                    for (key, value) in entry {
                        if !matches!(key.as_str(), "type" | "compact_threshold") {
                            edit.insert(key.clone(), value.clone());
                        }
                    }
                    Value::Object(edit)
                })
                .collect();
            (!edits.is_empty()).then(|| json!({"edits": edits}))
        }
        _ => None,
    }
}

/// `max_tokens` as Python keeps it: an int as sent, any other number rounded to at least 1.
pub fn max_tokens_value(value: &Value) -> Result<Value, Error> {
    match value {
        Value::Number(number) if number.is_i64() || number.is_u64() => Ok(value.clone()),
        Value::Number(number) => {
            let rounded = number.as_f64().unwrap_or(1.0).round_ties_even();
            Ok(json!((rounded as i64).max(1)))
        }
        Value::Bool(flag) => Ok(json!(i64::from(*flag))),
        other => Err(Error::InvalidRequest(ErrorDetail::InvalidValue {
            field: "max_tokens",
            expected: "a number",
            actual: other.clone(),
        })),
    }
}

/// What `map_openai_params` hands to `transform_request`.
#[derive(Debug, Default)]
pub struct MappedParams {
    pub params: Map<String, Value>,
    pub json_mode: bool,
}

/// `get_supported_openai_params`.
pub fn supported_params(caps: &ModelCapabilities<'_>) -> Result<Vec<&'static str>, Error> {
    let mut params = vec![
        "stream",
        "stop",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "tools",
        "tool_choice",
        "extra_headers",
        "parallel_tool_calls",
        "response_format",
        "user",
        "web_search_options",
        "speed",
        "context_management",
        "cache_control",
    ];
    let supports_reasoning = match caps.host {
        AnthropicHost::Anthropic => {
            let mut reasoning_keys = caps.candidates.clone();
            reasoning_keys.insert(1, format!("anthropic/{}", caps.model));
            capability(&caps.context, &reasoning_keys, "supports_reasoning")? == Some(true)
        }
        // `supports_reasoning(model, "vertex_ai")` reads the entry Vertex resolves to.
        AnthropicHost::VertexAi => caps.resolved_flag("supports_reasoning")? == Some(true),
    };
    if caps.model.contains("claude-3-7-sonnet") || caps.is_adaptive()? || supports_reasoning {
        params.extend(["thinking", "reasoning_effort"]);
    }
    // `SERVICE_TIER_PROVIDERS`: Vertex AI rejects the field.
    if caps.host == AnthropicHost::Anthropic {
        params.push("service_tier");
    }
    Ok(params)
}

/// `AnthropicConfig.map_openai_params`.
pub fn map_openai_params(
    openai: &Map<String, Value>,
    caps: &ModelCapabilities<'_>,
    drop: DropPolicy,
) -> Result<MappedParams, Error> {
    let thinking_enabled = crate::base_llm::chat::params::is_thinking_enabled(openai);
    let mut out = Map::new();
    let mut json_mode = false;
    let add_tools = |out: &mut Map<String, Value>, tools: Vec<Value>| match out.get_mut("tools") {
        Some(Value::Array(existing)) => existing.extend(tools),
        _ => {
            out.insert("tools".into(), Value::Array(tools));
        }
    };
    for (name, value) in ordered(openai) {
        match name {
            "max_tokens" | "max_completion_tokens" => {
                out.insert("max_tokens".into(), max_tokens_value(value)?);
            }
            "tools" => add_tools(&mut out, map_tools(value)?),
            "tool_choice" | "parallel_tool_calls" => {
                if let Some(choice) =
                    map_tool_choice(openai.get("tool_choice"), openai.get("parallel_tool_calls"))
                {
                    out.insert(
                        "tool_choice".into(),
                        apply_forced_tool_choice(choice, caps, drop)?,
                    );
                }
            }
            "stream" if value == &Value::Bool(true) => {
                out.insert("stream".into(), json!(true));
            }
            "stop" if value.is_string() || value.is_array() => {
                if let Some(stop) = map_stop_sequences(value, drop) {
                    out.insert("stop_sequences".into(), stop);
                }
            }
            "temperature" | "top_p" => {
                apply_sampling_param(&mut out, caps, drop, name, value, name)?
            }
            "response_format" => {
                let Some(format) = value.as_object() else {
                    continue;
                };
                if caps.native_structured_output()? {
                    if let Some(output_format) = output_format(format) {
                        out.insert("output_format".into(), output_format);
                    }
                } else {
                    if format.get("type").and_then(Value::as_str) == Some("text") {
                        continue;
                    }
                    let Some(schema) = response_format_schema(format) else {
                        continue;
                    };
                    if !thinking_enabled && !caps.forced_tool_use_unsupported()? {
                        out.insert(
                            "tool_choice".into(),
                            json!({"name": RESPONSE_FORMAT_TOOL_NAME, "type": "tool"}),
                        );
                    }
                    add_tools(&mut out, vec![response_format_tool(&schema)]);
                }
                json_mode = true;
            }
            "user" => {
                if let Some(user) = value.as_str().filter(|user| valid_user_id(user)) {
                    out.insert("metadata".into(), json!({"user_id": user}));
                }
            }
            "thinking" => {
                let adaptive_request =
                    value.get("type").and_then(Value::as_str) == Some("adaptive");
                if value.is_object() && adaptive_request && !caps.is_adaptive()? {
                    let max_tokens = openai
                        .get("max_completion_tokens")
                        .filter(|v| truthy(v))
                        .or_else(|| openai.get("max_tokens"))
                        .filter(|v| !v.is_null());
                    let legacy = map_reasoning_effort("medium", caps)?;
                    match legacy.and_then(|legacy| cap_thinking_budget(legacy, max_tokens)) {
                        Some(capped) => {
                            out.insert("thinking".into(), capped);
                        }
                        None => {
                            out.remove("thinking");
                        }
                    }
                } else {
                    out.insert("thinking".into(), value.clone());
                    translate_legacy_thinking(&mut out, caps, drop)?;
                }
            }
            "reasoning_effort" => {
                let effort = match value {
                    Value::Object(effort) => effort.get("effort"),
                    other => Some(other),
                };
                let Some(effort) = effort.and_then(Value::as_str) else {
                    continue;
                };
                match map_reasoning_effort(effort, caps)? {
                    None => {
                        out.remove("thinking");
                        out.remove("output_config");
                    }
                    Some(thinking) => {
                        out.insert("thinking".into(), thinking);
                        if caps.is_adaptive()? {
                            let mapped = output_config_effort(effort)
                                .ok_or_else(|| unmapped_reasoning_effort(effort))?;
                            out.insert("output_config".into(), json!({"effort": mapped}));
                        }
                    }
                }
            }
            "web_search_options" => {
                if let Some(options) = value.as_object() {
                    add_tools(&mut out, vec![web_search_tool(options)]);
                }
            }
            "context_management" => {
                if let Some(mapped) = map_context_management(value) {
                    out.insert("context_management".into(), mapped);
                }
            }
            "service_tier" => {
                let Some(tier) = value.as_str() else { continue };
                match tier {
                    "auto" => {
                        out.insert("service_tier".into(), json!("auto"));
                    }
                    "default" | "standard_only" => {
                        out.insert("service_tier".into(), json!("standard_only"));
                    }
                    other => drop.unsupported(|| {
                        unsupported_value(format!(
                            "{} does not support service_tier='{other}'. Anthropic accepts 'auto', 'standard_only', or OpenAI's 'default' (sent as 'standard_only'). To drop unsupported params, set `litellm.drop_params = True`.",
                            caps.model
                        ))
                    })?,
                }
            }
            _ => {}
        }
    }
    // `update_optional_params_with_thinking_tokens`.
    let thinking_on = out
        .get("thinking")
        .and_then(|thinking| thinking.get("type"))
        .and_then(Value::as_str)
        == Some("enabled");
    if thinking_on
        && !openai.contains_key("max_tokens")
        && !openai.contains_key("max_completion_tokens")
        && let Some(budget) = out
            .get("thinking")
            .and_then(|thinking| thinking.get("budget_tokens"))
            .and_then(Value::as_i64)
    {
        out.insert(
            "max_tokens".into(),
            json!(budget + DEFAULT_MAX_TOKENS as i64),
        );
    }
    Ok(MappedParams {
        params: out,
        json_mode,
    })
}

/// The schema of the request's `json_tool_call` tool (`JsonModeRequest.tool_schema`).
pub fn json_tool_schema(tools: Option<&Value>) -> Option<ResponseFormatSchema> {
    tools?
        .as_array()?
        .iter()
        .find(|tool| tool.get("name").and_then(Value::as_str) == Some(RESPONSE_FORMAT_TOOL_NAME))
        .and_then(|tool| tool.get("input_schema"))
        .and_then(ResponseFormatSchema::from_schema)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn object(value: Value) -> Map<String, Value> {
        value.as_object().unwrap().clone()
    }

    fn caps(model: &str) -> ModelCapabilities<'static> {
        ModelCapabilities::new(
            model,
            &ChatRequestContext {
                drop_params: Some(false),
                model_info: None,
            },
        )
    }

    fn map(params: Value) -> MappedParams {
        map_openai_params(
            &object(params),
            &caps("claude-sonnet-4-5"),
            DropPolicy::new(Some(false)),
        )
        .unwrap()
    }

    #[rstest]
    fn candidates_cover_bedrock_ids_and_dotted_versions() {
        let candidates = model_map_candidates("us.anthropic.claude-sonnet-4.6-20251101-v1:0");
        assert!(candidates.contains(&"anthropic.claude-sonnet-4.6-20251101-v1:0".to_string()));
        assert!(candidates.contains(&"us.anthropic.claude-sonnet-4-6".to_string()));
        assert!(
            candidates.contains(&"bedrock/anthropic.claude-sonnet-4.6-20251101-v1:0".to_string())
        );
        assert!(model_map_candidates("claude-x@20250101").contains(&"claude-x".to_string()));
    }

    #[rstest]
    fn tools_and_choices_map_like_python() {
        let mapped = map(json!({
            "tools": [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {"a": {}}, "title": "x"}}, "cache_control": {"type": "ephemeral"}}],
            "tool_choice": "required",
            "parallel_tool_calls": false
        }));
        assert_eq!(
            mapped.params,
            object(json!({
                "tools": [{"name": "f", "input_schema": {"type": "object", "properties": {"a": {}}}, "type": "custom", "description": "d", "cache_control": {"type": "ephemeral"}}],
                "tool_choice": {"type": "any", "disable_parallel_tool_use": true}
            }))
        );
        assert_eq!(
            map(json!({"tool_choice": {"type": "function", "function": {"name": "f"}}})).params["tool_choice"],
            json!({"type": "tool", "name": "f"})
        );
        assert_eq!(
            map(json!({"parallel_tool_calls": true})).params["tool_choice"],
            json!({"type": "auto", "disable_parallel_tool_use": false})
        );
    }

    #[rstest]
    fn max_tokens_stop_user_and_sampling() {
        let mapped = map(
            json!({"max_completion_tokens": 10.6, "stop": "END", "user": "u-1", "temperature": 0.2, "top_p": 0.5}),
        );
        assert_eq!(
            mapped.params,
            object(
                json!({"temperature": 0.2, "top_p": 0.5, "stop_sequences": ["END"], "max_tokens": 11, "metadata": {"user_id": "u-1"}})
            )
        );
        assert!(
            !map(json!({"user": "a@b.co"}))
                .params
                .contains_key("metadata")
        );
        assert!(
            !map(json!({"user": "+1 (555) 123-4567"}))
                .params
                .contains_key("metadata")
        );
    }

    #[rstest]
    fn removed_sampling_params_are_refused_or_dropped() {
        let refused = map_openai_params(
            &object(json!({"temperature": 0.3})),
            &caps("claude-opus-4-7"),
            DropPolicy::new(Some(false)),
        )
        .unwrap_err();
        assert_eq!(
            refused.to_string(),
            "invalid request: claude-opus-4-7 does not support temperature=0.3. Only temperature=1 is supported. To drop unsupported params, set `litellm.drop_params = True`."
        );
        let dropped = map_openai_params(
            &object(json!({"temperature": 0.3, "top_p": 0.1})),
            &caps("claude-opus-4-7"),
            DropPolicy::new(Some(true)),
        )
        .unwrap();
        assert!(dropped.params.is_empty());
    }

    #[rstest]
    fn response_format_becomes_a_forced_tool() {
        let mapped = map(
            json!({"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}}}),
        );
        assert!(mapped.json_mode);
        assert_eq!(
            mapped.params,
            object(json!({
                "tool_choice": {"name": "json_tool_call", "type": "tool"},
                "tools": [{"name": "json_tool_call", "input_schema": {"type": "object", "properties": {"a": {"type": "string"}}}}]
            }))
        );
        assert!(!map(json!({"response_format": {"type": "text"}})).json_mode);
    }

    #[rstest]
    fn reasoning_effort_sets_a_budget_and_max_tokens() {
        let mapped = map(json!({"reasoning_effort": "low"}));
        assert_eq!(
            mapped.params,
            object(
                json!({"thinking": {"type": "enabled", "budget_tokens": 1024}, "max_tokens": 5120})
            )
        );
        assert!(
            map_openai_params(
                &object(json!({"reasoning_effort": "huge"})),
                &caps("claude-sonnet-4-5"),
                DropPolicy::new(Some(false))
            )
            .is_err()
        );
    }

    #[rstest]
    fn output_schemas_lose_unenforceable_constraints() {
        assert_eq!(
            filter_output_schema(
                &json!({"type": "object", "properties": {"n": {"type": "integer", "minimum": 1, "description": "count"}, "o": {"oneOf": [{"type": "string"}]}}})
            ),
            json!({"type": "object", "properties": {"n": {"description": "count Note: minimum value: 1.", "type": "integer"}, "o": {"anyOf": [{"type": "string"}]}}, "additionalProperties": false})
        );
    }
}
