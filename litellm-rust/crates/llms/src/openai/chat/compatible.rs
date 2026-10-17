//! The chat flow every OpenAI-compatible provider shares, `litellm/llms/openai/chat/gpt_transformation.py`
//! (`OpenAIGPTConfig`) as the base each provider config overrides, run over the caller's
//! OpenAI-shaped body: Python's `get_optional_params` sorts the request's OpenAI params from the
//! rest, refuses or drops what the provider does not take, maps the rest by the provider's
//! rules, and passes every other key through as `extra_body` does. The messages go through the
//! provider's `_transform_messages`, and the response reaches the caller as the provider sent
//! it, changed only by the fixups each provider's `transform_response` and stream iterator make.

use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use serde_json::{Map, Value, json};

use crate::{
    Error, ErrorDetail,
    base_llm::{
        chat::params::{ChatRequestContext, sort_params, truthy},
        model_info::ModelInfo,
    },
    openai::chat::transformation::{OPENAI_CHAT_COMPLETIONS_PATH, is_stream, take_extra_body},
};

/// The `base_params` of `OpenAIGPTConfig.get_supported_openai_params`.
pub const GPT_BASE_PARAMS: &[&str] = &[
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "max_tokens",
    "max_completion_tokens",
    "modalities",
    "prediction",
    "n",
    "presence_penalty",
    "seed",
    "stop",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "function_call",
    "functions",
    "max_retries",
    "extra_headers",
    "parallel_tool_calls",
    "audio",
    "web_search_options",
    "service_tier",
    "safety_identifier",
    "prompt_cache_key",
    "prompt_cache_retention",
    "store",
];

/// `LITELLM_RESPONSE_ONLY_MESSAGE_FIELDS`: what LiteLLM adds to a returned message that no
/// OpenAI-compatible endpoint takes back.
const RESPONSE_ONLY_MESSAGE_FIELDS: &[&str] = &[
    "thinking_blocks",
    "reasoning_items",
    "provider_specific_fields",
];
/// `LITELLM_INTERNAL_MESSAGE_FIELDS`: the response-only fields and `reasoning_content`.
const INTERNAL_MESSAGE_FIELDS: &[&str] = &[
    "thinking_blocks",
    "reasoning_items",
    "provider_specific_fields",
    "reasoning_content",
];
/// `DEFAULT_MAX_RECURSE_DEPTH`.
const MAX_RECURSE_DEPTH: usize = 100;
const STREAM_OPTIONS: &str = "stream_options";
const INCLUDE_USAGE: &str = "include_usage";
/// `RESPONSE_FORMAT_TOOL_NAME`.
pub const RESPONSE_FORMAT_TOOL_NAME: &str = "json_tool_call";
const POTENTIAL_PDF_URL_STARTS: &[&str] = &["https://", "http://", "www."];
/// The content part types `_content_parts_contain_image` looks for.
const IMAGE_PART_TYPES: &[&str] = &["image_url", "input_image", "image"];
/// `DEFAULT_REASONING_EFFORT_{LOW,MEDIUM,HIGH}_THINKING_BUDGET` at their defaults.
const THINKING_BUDGETS: (u64, u64, u64) = (1024, 2048, 4096);
/// `REASONING_EFFORT_STRENGTH_ORDER`.
const EFFORT_STRENGTH_ORDER: &[&str] = &["minimal", "low", "medium", "high", "xhigh", "max"];

/// What a provider's param rules read about the call.
#[derive(Clone, Copy)]
pub struct CompatibleContext<'a> {
    /// The model as the provider names it, without the route prefix.
    pub model: &'a str,
    /// `drop_params` from the deployment, the proxy settings or the request.
    pub drop_params: bool,
    pub model_info: Option<&'a dyn ModelInfo>,
}

impl CompatibleContext<'_> {
    /// `supports_factory`: the `<provider>/<model>` entry's flag, else the bare model's.
    pub fn capability(&self, provider: &str, key: &str) -> Option<bool> {
        self.capability_of(
            &[format!("{provider}/{}", self.model), self.model.to_string()],
            key,
        )
    }

    /// The first of `candidates` whose cost map entry sets `key`.
    pub fn capability_of(&self, candidates: &[String], key: &str) -> Option<bool> {
        let info = self.model_info?;
        candidates
            .iter()
            .filter_map(|candidate| info.entry(candidate))
            .find_map(|entry| entry.get(key).and_then(Value::as_bool))
    }

    pub fn supports(&self, provider: &str, key: &str) -> bool {
        self.capability(provider, key) == Some(true)
    }

    /// `declared_reasoning_efforts_for_model`: the `reasoning_effort_levels` the entry for the
    /// model declares, read through its bare twin. `None` when the map says nothing.
    pub fn declared_reasoning_efforts(&self, provider: &str) -> Option<Vec<String>> {
        let info = self.model_info?;
        let prefixed = format!("{provider}/{}", self.model);
        let (entry, prefixed_hit) = match info.entry(&prefixed) {
            Some(entry) => (entry, true),
            None => (info.entry(self.model)?, false),
        };
        let levels = match entry.get("reasoning_effort_levels") {
            Some(levels) if !levels.is_null() => levels.clone(),
            _ if prefixed_hit => info
                .entry(self.model)
                .and_then(|bare| bare.get("reasoning_effort_levels").cloned())
                .unwrap_or(Value::Null),
            _ => Value::Null,
        };
        let Value::Array(levels) = levels else {
            return None;
        };
        Some(
            levels
                .into_iter()
                .filter_map(|level| level.as_str().map(str::to_owned))
                .collect(),
        )
    }
}

/// `nearest_declared_reasoning_effort`: the weakest declared level at least as strong as the
/// request, else the strongest declared one. `none` and unknown levels go as asked.
pub fn nearest_declared_reasoning_effort(requested: &str, declared: &[String]) -> String {
    let rank = |effort: &str| EFFORT_STRENGTH_ORDER.iter().position(|e| *e == effort);
    let mut ranked: Vec<&str> = declared
        .iter()
        .map(String::as_str)
        .filter(|effort| rank(effort).is_some())
        .collect();
    ranked.sort_by_key(|effort| rank(effort));
    let Some(requested_rank) = rank(requested) else {
        return requested.to_owned();
    };
    if ranked.is_empty() || ranked.contains(&requested) {
        return requested.to_owned();
    }
    ranked
        .iter()
        .find(|effort| rank(effort) >= Some(requested_rank))
        .or(ranked.last())
        .map_or_else(|| requested.to_owned(), |effort| (*effort).to_owned())
}

/// `reasoning_effort_from_thinking_budget` at the default budget thresholds.
pub fn reasoning_effort_from_thinking_budget(budget_tokens: i64) -> &'static str {
    let (low, medium, high) = THINKING_BUDGETS;
    match budget_tokens {
        tokens if tokens >= high as i64 => "high",
        tokens if tokens >= medium as i64 => "medium",
        tokens if tokens >= low as i64 => "low",
        _ => "minimal",
    }
}

/// What a request decided about its answer, which the response and stream fixups read.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResponseShape {
    pub stream: bool,
    /// The client asked to see the usage on a stream. When it did not, the usage the call asked
    /// for on its behalf is recorded and kept from the client.
    pub client_usage: bool,
    /// The provider answers a `response_format` with a `json_tool_call` tool call, which goes
    /// back as content.
    pub json_mode: bool,
    /// The names of the function tools the request sent.
    pub tool_names: Vec<String>,
    /// Whether the request sent `tools` at all.
    pub tools_sent: bool,
}

/// A request ready for an OpenAI-compatible provider.
#[derive(Debug, PartialEq)]
pub struct CompatibleChatBody {
    pub body: Map<String, Value>,
    pub shape: ResponseShape,
}

/// One OpenAI-compatible provider's config: its endpoint and credential fallbacks, and the
/// overrides it makes to `OpenAIGPTConfig`.
pub trait OpenAiCompatibleChatConfig: Sync {
    fn provider(&self) -> LlmProviders;

    /// The provider as a missing key error names it.
    fn label(&self) -> &'static str;

    /// The variables an unset `api_base` falls back to, in order.
    fn api_base_env(&self) -> &'static [&'static str];

    fn default_api_base(&self) -> Option<&'static str>;

    /// The variables an unset key falls back to, in order.
    fn api_key_env(&self) -> &'static [&'static str];

    /// The key sent when neither the deployment nor the environment has one; `None` means the
    /// call needs one.
    fn default_api_key(&self) -> Option<&'static str> {
        None
    }

    /// Other variables the provider's headers read.
    fn header_env(&self) -> &'static [&'static str] {
        &[]
    }

    fn name(&self) -> &'static str {
        self.provider().into()
    }

    /// Every variable a call may read.
    fn secret_names(&self) -> Vec<&'static str> {
        self.api_base_env()
            .iter()
            .chain(self.api_key_env())
            .chain(self.header_env())
            .copied()
            .collect()
    }

    /// The deployment's `api_base`, else the environment's, else the provider's own.
    fn api_base(
        &self,
        configured: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        configured
            .filter(|base| !base.is_empty())
            .map(str::to_owned)
            .or_else(|| self.api_base_env().iter().find_map(|name| lookup(name)))
            .or_else(|| self.default_api_base().map(str::to_owned))
            .ok_or_else(|| {
                Error::InvalidRequest(ErrorDetail::Message(format!(
                    "Missing API Base - set api_base on the {} deployment or {}",
                    self.name(),
                    self.api_base_env().join(" or ")
                )))
            })
    }

    /// `OpenAIGPTConfig.get_complete_url`: `/chat/completions` appended unless the base already
    /// names it.
    fn complete_url(&self, api_base: &str) -> String {
        gpt_complete_url(api_base)
    }

    /// The deployment's key, else the environment's, else the provider's default.
    fn api_key(
        &self,
        configured: Option<&str>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        configured
            .filter(|key| !key.is_empty())
            .map(str::to_owned)
            .or_else(|| {
                self.api_key_env()
                    .iter()
                    .find_map(|name| lookup(name).filter(|key| !key.is_empty()))
            })
            .or_else(|| self.default_api_key().map(str::to_owned))
            .ok_or(Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: self.label(),
                environment_variable: self.api_key_env().first().copied().unwrap_or_default(),
            }))
    }

    /// Headers the provider adds to every call, read off the caller's body before it is
    /// sorted.
    fn headers(
        &self,
        _body: &Map<String, Value>,
        _lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Vec<(String, String)> {
        Vec::new()
    }

    /// The model the body names.
    fn wire_model(&self, model: &str) -> String {
        model.to_owned()
    }

    /// `get_supported_openai_params`.
    fn supported_params(&self, context: &CompatibleContext<'_>) -> Vec<&'static str>;

    /// `map_openai_params`: `non_default` holds only supported params.
    fn map_params(
        &self,
        non_default: &Map<String, Value>,
        optional: &mut Map<String, Value>,
        context: &CompatibleContext<'_>,
    ) -> Result<(), Error> {
        copy_supported(non_default, optional, &self.supported_params(context));
        Ok(())
    }

    /// The keys sent as `extra_body`, which a provider may rewrite.
    fn map_extra_body(
        &self,
        _extra_body: &mut Map<String, Value>,
        _optional: &mut Map<String, Value>,
        _context: &CompatibleContext<'_>,
    ) -> Result<(), Error> {
        Ok(())
    }

    /// `_transform_messages`.
    fn transform_messages(
        &self,
        messages: Vec<Value>,
        _optional: &Map<String, Value>,
        _context: &CompatibleContext<'_>,
    ) -> Result<Vec<Value>, Error> {
        gpt_messages(messages)
    }

    /// Whether `cache_control` marks reach the provider; `OpenAIGPTConfig` strips them.
    fn keeps_cache_control(&self, _context: &CompatibleContext<'_>) -> bool {
        false
    }

    /// The provider's `transform_request` changes over the assembled body, before the
    /// `extra_body` keys are merged over it.
    fn finish_request(
        &self,
        _body: &mut Map<String, Value>,
        _context: &CompatibleContext<'_>,
    ) -> Result<(), Error> {
        Ok(())
    }

    /// `transform_response`'s fixups to a complete response. Returns whether it changed.
    fn fix_response(&self, _body: &mut Map<String, Value>, _shape: &ResponseShape) -> bool {
        false
    }

    /// The stream iterator's fixups to one chunk. Returns whether it changed.
    fn fix_chunk(&self, chunk: &mut Map<String, Value>, _shape: &ResponseShape) -> bool {
        rename_reasoning_in_deltas(chunk)
    }
}

/// `OpenAIGPTConfig.get_complete_url`.
pub fn gpt_complete_url(api_base: &str) -> String {
    let base = api_base.trim_end_matches('/');
    match base.contains("chat/completions") {
        true => base.to_owned(),
        false => format!("{base}{OPENAI_CHAT_COMPLETIONS_PATH}"),
    }
}

/// The URL the OpenAI SDK calls for a `base_url`.
pub fn sdk_complete_url(api_base: &str) -> String {
    format!(
        "{}{OPENAI_CHAT_COMPLETIONS_PATH}",
        api_base.trim_end_matches('/')
    )
}

/// `OpenAIGPTConfig._map_openai_params`: every supported param as sent.
pub fn copy_supported(
    non_default: &Map<String, Value>,
    optional: &mut Map<String, Value>,
    supported: &[&str],
) {
    for (name, value) in non_default {
        if supported.contains(&name.as_str()) {
            optional.insert(name.clone(), value.clone());
        }
    }
}

/// The request for `config`, built from the caller's body.
pub fn transform_request(
    config: &dyn OpenAiCompatibleChatConfig,
    context: &CompatibleContext<'_>,
    body: Map<String, Value>,
) -> Result<CompatibleChatBody, Error> {
    let mut body = body;
    body.remove("model");
    let messages = match body.remove("messages") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(messages)) => messages,
        Some(other) => {
            return Err(Error::InvalidRequest(ErrorDetail::InvalidValue {
                field: "messages",
                expected: "an array",
                actual: other,
            }));
        }
    };
    let mut user_extra_body = take_extra_body(&mut body)?;
    user_extra_body.remove("model");
    let stream = is_stream(&body)?;
    let client_usage = stream
        && body
            .get(STREAM_OPTIONS)
            .and_then(|options| options.get(INCLUDE_USAGE))
            == Some(&Value::Bool(true));
    let supported = config.supported_params(context);
    let sorted = sort_params(
        body,
        &[],
        &ChatRequestContext {
            drop_params: Some(context.drop_params),
            model_info: context.model_info,
        },
        |_| Ok(supported.clone()),
        config.name(),
        context.model,
    )?;
    let mut non_default = sorted.openai;
    // The proxy's stream usage tracking: a stream the provider can report usage on asks for
    // it, so the call is accounted for, and the client sees it only if it asked.
    if stream && !client_usage && supported.contains(&STREAM_OPTIONS) {
        let mut options = match non_default.remove(STREAM_OPTIONS) {
            Some(Value::Object(options)) => options,
            _ => Map::new(),
        };
        options.insert(INCLUDE_USAGE.into(), Value::Bool(true));
        non_default.insert(STREAM_OPTIONS.into(), Value::Object(options));
    }

    let mut optional = Map::new();
    config.map_params(&non_default, &mut optional, context)?;
    // `apply_openai_param_overrides`: what the caller allowed goes as sent.
    for name in &sorted.allowed {
        if !optional.contains_key(name)
            && let Some(value) = non_default.get(name)
        {
            optional.insert(name.clone(), value.clone());
        }
    }
    let json_mode = optional
        .remove("json_mode")
        .is_some_and(|value| value == Value::Bool(true));
    let mut extra_body = match optional.remove("extra_body") {
        Some(Value::Object(extra)) => extra,
        _ => Map::new(),
    };
    extra_body.extend(user_extra_body);
    extra_body.extend(sorted.provider);
    config.map_extra_body(&mut extra_body, &mut optional, context)?;

    let mut messages = config.transform_messages(messages, &optional, context)?;
    if !config.keeps_cache_control(context) {
        for message in &mut messages {
            filter_value(message, "cache_control", 0);
        }
        if let Some(Value::Array(tools)) = optional.get_mut("tools") {
            for tool in tools {
                filter_value(tool, "cache_control", 0);
            }
        }
    }
    if context.drop_params {
        for message in &mut messages {
            without_prompt_cache_breakpoints(message);
        }
        if let Some(Value::Array(tools)) = optional.get_mut("tools") {
            for tool in tools {
                without_prompt_cache_breakpoints(tool);
            }
        }
    }
    optional.remove("max_retries");
    if !optional.get("tools").is_some_and(truthy) && !optional.get("functions").is_some_and(truthy)
    {
        optional.remove("tool_choice");
    }

    let mut sent = Map::new();
    sent.insert(
        "model".into(),
        Value::String(config.wire_model(context.model)),
    );
    sent.insert("messages".into(), Value::Array(messages));
    sent.extend(optional);
    config.finish_request(&mut sent, context)?;
    extra_body.remove("model");
    sent.extend(extra_body);

    let tools = sent.get("tools").and_then(Value::as_array);
    let shape = ResponseShape {
        stream,
        client_usage,
        json_mode,
        tool_names: tools
            .into_iter()
            .flatten()
            .filter_map(|tool| tool.get("function")?.get("name")?.as_str())
            .map(str::to_owned)
            .collect(),
        tools_sent: sent.get("tools").is_some_and(|tools| !tools.is_null()),
    };
    Ok(CompatibleChatBody { body: sent, shape })
}

/// A complete response as the caller receives it. Returns whether it changed.
pub fn fix_response(
    config: &dyn OpenAiCompatibleChatConfig,
    body: &mut Map<String, Value>,
    shape: &ResponseShape,
) -> bool {
    let renamed = rename_reasoning_in_messages(body);
    config.fix_response(body, shape) || renamed
}

/// `OpenAIGPTConfig._transform_messages`: `tool_reference` parts dropped from tool messages,
/// and each user content part normalized.
pub fn gpt_messages(messages: Vec<Value>) -> Result<Vec<Value>, Error> {
    if messages.iter().any(tool_message_carries_image) {
        // `hoist_images_from_tool_messages` moves them into an inserted user message.
        return Err(Error::Unsupported(
            "images in tool messages, which Python moves into a user message",
        ));
    }
    messages
        .into_iter()
        .map(drop_tool_reference_parts)
        .map(|mut message| {
            if role(&message) == Some("user")
                && let Some(Value::Array(parts)) = message.get_mut("content")
            {
                for part in parts.iter_mut() {
                    transform_content_item(part)?;
                }
            }
            Ok(message)
        })
        .collect()
}

pub fn role(message: &Value) -> Option<&str> {
    message.get("role").and_then(Value::as_str)
}

fn is_tool_reference(part: &Value) -> bool {
    part.get("type").and_then(Value::as_str) == Some("tool_reference")
}

/// `drop_tool_reference_parts_from_tool_messages` for one message.
fn drop_tool_reference_parts(mut message: Value) -> Value {
    if role(&message) != Some("tool") {
        return message;
    }
    if let Some(content) = message.get_mut("content")
        && let Value::Array(parts) = content
        && parts.iter().any(is_tool_reference)
    {
        parts.retain(|part| !is_tool_reference(part));
        if parts.is_empty() {
            *content = Value::String(String::new());
        }
    }
    message
}

fn tool_message_carries_image(message: &Value) -> bool {
    fn has_image(parts: &[Value], depth: usize) -> bool {
        parts.iter().any(|part| {
            part.get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| IMAGE_PART_TYPES.contains(&kind))
                || (depth < 4
                    && part
                        .get("content")
                        .and_then(Value::as_array)
                        .is_some_and(|nested| has_image(nested, depth + 1)))
        })
    }
    role(message) == Some("tool")
        && message
            .get("content")
            .and_then(Value::as_array)
            .is_some_and(|parts| has_image(parts, 1))
}

/// `OpenAIGPTConfig._transform_content_item`.
fn transform_content_item(part: &mut Value) -> Result<(), Error> {
    let Some(object) = part.as_object_mut() else {
        return Ok(());
    };
    match object.get("type").and_then(Value::as_str) {
        Some("image_url") => match object.get_mut("image_url") {
            Some(Value::String(url)) => {
                let url = std::mem::take(url);
                object.insert("image_url".into(), json!({"url": url}));
            }
            Some(Value::Object(image)) => {
                image.remove("format");
            }
            _ => {}
        },
        Some("file") => {
            let Some(Value::Object(file)) = object.get_mut("file") else {
                return Err(Error::InvalidRequest(ErrorDetail::Message(
                    "Content block has type='file' but is missing the required 'file' field".into(),
                )));
            };
            file.remove("format");
            if file
                .get("file_id")
                .and_then(Value::as_str)
                .is_some_and(|id| POTENTIAL_PDF_URL_STARTS.iter().any(|s| id.starts_with(s)))
            {
                // `_handle_pdf_url` downloads it into `file_data`.
                return Err(Error::Unsupported(
                    "a file_id URL Python downloads into file_data",
                ));
            }
            if file.get("file_data").is_some_and(|data| !data.is_null())
                && file.get("filename").is_none_or(Value::is_null)
            {
                file.insert("filename".into(), Value::String("my_file.pdf".into()));
            }
        }
        _ => {}
    }
    Ok(())
}

/// `filter_value_from_dict`: `key` removed from every object, at any depth.
pub fn filter_value(value: &mut Value, key: &str, depth: usize) {
    if depth > MAX_RECURSE_DEPTH {
        return;
    }
    if let Value::Object(object) = value {
        object.remove(key);
        for nested in object.values_mut() {
            match nested {
                Value::Object(_) => filter_value(nested, key, depth + 1),
                Value::Array(items) => items
                    .iter_mut()
                    .filter(|item| item.is_object())
                    .for_each(|item| filter_value(item, key, depth + 1)),
                _ => {}
            }
        }
    }
}

/// `without_prompt_cache_breakpoints`: the marker removed from a message or tool, its content
/// parts, its tool calls and its `function`.
fn without_prompt_cache_breakpoints(block: &mut Value) {
    const MARKER: &str = "prompt_cache_breakpoint";
    let Some(object) = block.as_object_mut() else {
        return;
    };
    object.remove(MARKER);
    for key in ["content", "tool_calls"] {
        if let Some(Value::Array(items)) = object.get_mut(key) {
            for item in items.iter_mut().filter_map(Value::as_object_mut) {
                item.remove(MARKER);
            }
        }
    }
    if let Some(Value::Object(function)) = object.get_mut("function") {
        function.remove(MARKER);
    }
}

fn without_fields(message: &mut Value, fields: &[&str]) {
    let Some(object) = message.as_object_mut() else {
        return;
    };
    for field in fields {
        object.remove(*field);
    }
    if let Some(Value::Array(calls)) = object.get_mut("tool_calls") {
        for call in calls.iter_mut().filter_map(Value::as_object_mut) {
            call.remove("provider_specific_fields");
            if let Some(Value::Object(function)) = call.get_mut("function") {
                function.remove("provider_specific_fields");
            }
        }
    }
}

/// `strip_litellm_response_only_fields`.
pub fn strip_response_only_fields(message: &mut Value) {
    without_fields(message, RESPONSE_ONLY_MESSAGE_FIELDS);
}

/// `strip_litellm_internal_message_fields`.
pub fn strip_internal_fields(message: &mut Value) {
    without_fields(message, INTERNAL_MESSAGE_FIELDS);
}

/// `extract_search_results_text`.
pub fn search_results_text(search_results: Option<&Value>) -> String {
    let Some(Value::Array(results)) = search_results else {
        return String::new();
    };
    let mut text = String::new();
    for result in results.iter().filter_map(Value::as_object) {
        for key in ["source", "title"] {
            if let Some(value) = result.get(key).and_then(Value::as_str) {
                text.push_str(value);
            }
        }
        if let Some(Value::Array(blocks)) = result.get("content") {
            for block in blocks {
                if let Some(value) = block.get("text").and_then(Value::as_str) {
                    text.push_str(value);
                }
            }
        }
        if let Some(citations) = result.get("citations").filter(|c| !c.is_null()) {
            text.push_str(&serde_json::to_string(citations).unwrap_or_default());
        }
    }
    text
}

/// `convert_content_list_to_str`: the text parts joined, then any `search_results` text.
pub fn content_list_to_str(message: &Value) -> String {
    let mut text = String::new();
    match message.get("content") {
        Some(Value::Array(parts)) => {
            for part in parts {
                if let Some(value) = part.get("text").and_then(Value::as_str) {
                    text.push_str(value);
                }
            }
        }
        Some(Value::String(content)) => text.push_str(content),
        _ => {}
    }
    text.push_str(&search_results_text(message.get("search_results")));
    text
}

/// The `reasoning` some OpenAI-compatible servers send, under LiteLLM's `reasoning_content`.
pub fn rename_reasoning(message: &mut Map<String, Value>) -> bool {
    if message.contains_key("reasoning_content") || !message.contains_key("reasoning") {
        return false;
    }
    if let Some(reasoning) = message.remove("reasoning") {
        message.insert("reasoning_content".into(), reasoning);
    }
    true
}

/// `_map_reasoning_to_reasoning_content` over a chunk's deltas.
pub fn rename_reasoning_in_deltas(chunk: &mut Map<String, Value>) -> bool {
    let mut changed = false;
    if let Some(Value::Array(choices)) = chunk.get_mut("choices") {
        for delta in choices
            .iter_mut()
            .filter_map(|choice| choice.get_mut("delta")?.as_object_mut())
        {
            changed |= rename_reasoning(delta);
        }
    }
    changed
}

/// `extract_reasoning_content` over a complete response's messages.
pub fn rename_reasoning_in_messages(body: &mut Map<String, Value>) -> bool {
    let mut changed = false;
    for message in messages_mut(body) {
        changed |= rename_reasoning(message);
    }
    changed
}

/// Each choice's message in a complete response.
pub fn messages_mut(
    body: &mut Map<String, Value>,
) -> impl Iterator<Item = &mut Map<String, Value>> {
    body.get_mut("choices")
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
        .filter_map(|choice| choice.get_mut("message")?.as_object_mut())
}

/// `OpenAILikeChatConfig._sanitize_usage_obj`: a null `*_tokens` usage count becomes 0.
pub fn sanitize_usage(body: &mut Map<String, Value>) -> bool {
    let mut changed = false;
    if let Some(Value::Object(usage)) = body.get_mut("usage") {
        for (key, value) in usage.iter_mut() {
            if key.ends_with("_tokens") && value.is_null() {
                *value = json!(0);
                changed = true;
            }
        }
    }
    changed
}

/// `_json_mode_convert_tool_response_to_message`: a lone tool call's arguments become the
/// content.
pub fn json_mode_tool_call_to_content(body: &mut Map<String, Value>) -> bool {
    let mut changed = false;
    for message in messages_mut(body) {
        let arguments = match message.get("tool_calls") {
            Some(Value::Array(calls)) if calls.len() == 1 => calls[0]
                .get("function")
                .and_then(|function| function.get("arguments"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            _ => continue,
        };
        message.insert("content".into(), Value::String(arguments));
        message.insert("tool_calls".into(), Value::Null);
        changed = true;
    }
    changed
}

/// Python's `str` indices of `[n]` citation markers in `text`: `(number, start, end)` in code
/// points, as `re.finditer(r"\[(\d+)\]")` reports them.
pub fn citation_markers(text: &str) -> Vec<(usize, usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let mut markers = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        if chars[index] == '[' {
            let digits = chars[index + 1..]
                .iter()
                .take_while(|c| c.is_ascii_digit())
                .count();
            let close = index + 1 + digits;
            if digits > 0 && chars.get(close) == Some(&']') {
                let number: String = chars[index + 1..close].iter().collect();
                if let Ok(number) = number.parse() {
                    markers.push((number, index, close + 1));
                }
                index = close + 1;
                continue;
            }
        }
        index += 1;
    }
    markers
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn object(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(map) => map,
            other => panic!("expected an object, got {other}"),
        }
    }

    #[rstest]
    #[case::bare("https://h/v1", "https://h/v1/chat/completions")]
    #[case::slash("https://h/v1/", "https://h/v1/chat/completions")]
    #[case::named("https://h/v1/chat/completions", "https://h/v1/chat/completions")]
    fn the_gpt_url_appends_the_path_once(#[case] base: &str, #[case] expected: &str) {
        assert_eq!(gpt_complete_url(base), expected);
    }

    #[rstest]
    fn user_content_parts_are_normalized() {
        let messages = gpt_messages(vec![json!({"role": "user", "content": [
            {"type": "image_url", "image_url": "https://img"},
            {"type": "image_url", "image_url": {"url": "u", "format": "image/png", "detail": "low"}},
            {"type": "file", "file": {"file_data": "data:application/pdf;base64,AA", "format": "application/pdf"}},
        ]})])
        .unwrap();
        assert_eq!(
            messages[0]["content"],
            json!([
                {"type": "image_url", "image_url": {"url": "https://img"}},
                {"type": "image_url", "image_url": {"url": "u", "detail": "low"}},
                {"type": "file", "file": {"file_data": "data:application/pdf;base64,AA", "filename": "my_file.pdf"}},
            ])
        );
    }

    #[rstest]
    #[case::pdf_url(json!({"role": "user", "content": [{"type": "file", "file": {"file_id": "https://x/a.pdf"}}]}))]
    #[case::tool_image(json!({"role": "tool", "tool_call_id": "c", "content": [{"type": "image_url", "image_url": {"url": "u"}}]}))]
    fn shapes_python_rewrites_with_io_are_refused(#[case] message: Value) {
        assert!(matches!(
            gpt_messages(vec![message]),
            Err(Error::Unsupported(_))
        ));
    }

    #[rstest]
    fn a_file_part_without_its_file_is_rejected() {
        assert!(matches!(
            gpt_messages(vec![json!({"role": "user", "content": [{"type": "file"}]})]),
            Err(Error::InvalidRequest(_))
        ));
    }

    #[rstest]
    fn tool_reference_parts_leave_tool_messages() {
        let messages = gpt_messages(vec![json!({"role": "tool", "tool_call_id": "c",
            "content": [{"type": "tool_reference", "tool_name": "f"}]})])
        .unwrap();
        assert_eq!(messages[0]["content"], json!(""));
    }

    #[rstest]
    fn cache_control_is_filtered_at_any_depth() {
        let mut message = json!({"role": "user", "cache_control": {"type": "ephemeral"},
            "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]});
        filter_value(&mut message, "cache_control", 0);
        assert_eq!(
            message,
            json!({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        );
    }

    #[rstest]
    #[case::exact(1024, "low")]
    #[case::medium(3000, "medium")]
    #[case::high(5000, "high")]
    #[case::minimal(10, "minimal")]
    fn thinking_budgets_bucket_into_efforts(#[case] budget: i64, #[case] effort: &str) {
        assert_eq!(reasoning_effort_from_thinking_budget(budget), effort);
    }

    #[rstest]
    #[case::declared("high", &["low", "high"], "high")]
    #[case::rounds_up("medium", &["low", "high"], "high")]
    #[case::rounds_down("max", &["low", "high"], "high")]
    #[case::none("none", &["low"], "none")]
    fn efforts_round_to_a_declared_level(
        #[case] requested: &str,
        #[case] declared: &[&str],
        #[case] expected: &str,
    ) {
        let declared: Vec<String> = declared.iter().map(|d| (*d).to_owned()).collect();
        assert_eq!(
            nearest_declared_reasoning_effort(requested, &declared),
            expected
        );
    }

    #[rstest]
    fn citation_markers_count_code_points() {
        assert_eq!(
            citation_markers("é [1] and [22][x]"),
            vec![(1, 2, 5), (22, 10, 14)]
        );
    }

    #[rstest]
    fn json_mode_moves_a_lone_tool_call_into_the_content() {
        let mut body = object(
            json!({"choices": [{"message": {"role": "assistant", "content": null,
            "tool_calls": [{"id": "c", "type": "function", "function": {"name": "json_tool_call", "arguments": "{\"a\":1}"}}]},
            "finish_reason": "tool_calls"}]}),
        );
        assert!(json_mode_tool_call_to_content(&mut body));
        assert_eq!(
            body["choices"][0]["message"],
            json!({"role": "assistant", "content": "{\"a\":1}", "tool_calls": null})
        );
    }
}
