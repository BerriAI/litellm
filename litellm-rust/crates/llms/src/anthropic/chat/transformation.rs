//! Python's `AnthropicConfig` for `/chat/completions`: `get_optional_params` and
//! `transform_request` on the way in, `transform_parsed_response` on the way out.

use std::collections::{BTreeSet, HashMap, HashSet};

use litellm_auth::SecretValue;
use litellm_core_utils::core_helpers::{finish_reason_for, unix_now, usage_from_parts};
use litellm_llms_types::{
    formats::chat_completions::{
        ChatCompletionThinkingBlock, ChatCompletionToolCallChunk,
        ChatCompletionToolCallFunctionChunk, ChatCompletionsChoice, ChatCompletionsChoiceMessage,
        ChatCompletionsResponse, ChatMessage,
    },
    providers::anthropic::DEFAULT_HEADERS,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::{
    Error, ErrorDetail,
    anthropic::{
        chat::{
            handler::ModelResponseIterator,
            messages::{
                anthropic_messages_pt, place_mid_conversation_system, role,
                split_leading_system_run, system_blocks,
            },
            params::{
                MappedParams, ModelCapabilities, apply_sampling_param, json_tool_schema,
                map_openai_params, map_tools, supported_params, valid_user_id,
            },
        },
        common_utils::{
            API_KEY_PLACEMENT, complete_anthropic_url, forwarded_oauth_bearer,
            resolve_anthropic_api_key,
        },
    },
    base_llm::{
        auth::AuthScheme,
        chat::{
            media::{data_url, remote_urls},
            params::{
                ChatRequestContext, DEFAULT_MAX_TOKENS, RESPONSE_FORMAT_TOOL_NAME, capability,
                python_dumps, sort_params,
            },
            response_format::json_tool_answer,
            streaming::{ChatStream, StreamShape},
            transformation::{
                BaseConfig, Headers, ProviderChatRequestData, ProviderChatResponseData,
                STREAM_PARAM, Unsupported, ValidatedEnvironment,
            },
        },
        messages::streaming::event_stream as messages_event_stream,
    },
};

pub const BETA_HEADER: &str = "anthropic-beta";
const TOOL_NAME_MAX_LEN: usize = 128;
const VALID_EFFORTS: [&str; 5] = ["high", "medium", "low", "xhigh", "max"];

#[derive(Deserialize)]
struct ResponseProjection {
    model: String,
    content: Vec<Value>,
    usage: ResponseUsageProjection,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    stop_details: Option<Value>,
    #[serde(default)]
    context_management: Option<Value>,
    #[serde(default)]
    container: Option<Value>,
}

#[derive(Deserialize)]
struct ResponseUsageProjection {
    input_tokens: u64,
    output_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u64>,
}

pub struct AnthropicConfig;

pub const ANTHROPIC_CHAT_COMPLETIONS_CONFIG: AnthropicConfig = AnthropicConfig;

/// `add_dummy_tool`, the tool Anthropic needs when the history holds tool calls and the
/// request names no tools.
fn dummy_tool() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "dummy_tool",
            "description": "This is a dummy tool call",
            "parameters": {"type": "object", "properties": {}},
        },
    }])
}

/// `get_prefix_prompt`: the text of a trailing `{"role": "assistant", "prefix": true}`.
fn prefix_prompt(messages: &[Value]) -> Option<String> {
    let last = messages.last()?;
    let is_prefix = last
        .get("prefix")
        .is_some_and(crate::base_llm::chat::params::truthy);
    (last.get("role").and_then(Value::as_str) == Some("assistant") && is_prefix)
        .then(|| {
            last.get("content")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .flatten()
}

/// `_basic_sanitize_anthropic_tool_name`.
fn basic_sanitize_tool_name(name: &str) -> String {
    name.chars()
        .map(
            |c| match c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                true => c,
                false => '_',
            },
        )
        .take(TOOL_NAME_MAX_LEN)
        .collect()
}

/// `_build_anthropic_tool_name_maps`: `(forward, reverse)` for the names that change.
fn tool_name_maps(originals: &[String]) -> (HashMap<String, String>, HashMap<String, String>) {
    let mut forward: HashMap<String, String> = HashMap::new();
    let mut used: HashSet<String> = originals
        .iter()
        .filter(|original| basic_sanitize_tool_name(original) == **original)
        .cloned()
        .collect();
    for original in originals {
        let candidate = basic_sanitize_tool_name(original);
        if candidate == *original || forward.contains_key(original) {
            continue;
        }
        let mut unique = candidate.clone();
        let mut n = 1;
        while used.contains(&unique) {
            n += 1;
            let suffix = format!("_{n}");
            let head: String = candidate
                .chars()
                .take(TOOL_NAME_MAX_LEN - suffix.len())
                .collect();
            unique = format!("{head}{suffix}");
        }
        used.insert(unique.clone());
        forward.insert(original.clone(), unique);
    }
    let reverse = forward
        .iter()
        .map(|(original, sanitized)| (sanitized.clone(), original.clone()))
        .collect();
    (forward, reverse)
}

/// `_sanitize_tool_names_in_request` and `_rewrite_tool_names_in_messages`.
fn sanitize_tool_names(
    params: &mut Map<String, Value>,
    messages: &mut [Value],
) -> HashMap<String, String> {
    let Some(Value::Array(tools)) = params.get_mut("tools") else {
        return HashMap::new();
    };
    let originals: Vec<String> = tools
        .iter()
        .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("custom"))
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect();
    let (forward, reverse) = tool_name_maps(&originals);
    if forward.is_empty() {
        return reverse;
    }
    let rename = |value: Option<&mut Value>| {
        if let Some(Value::String(name)) = value
            && let Some(sanitized) = forward.get(name.as_str())
        {
            *name = sanitized.clone();
        }
    };
    for tool in tools.iter_mut() {
        if tool.get("type").and_then(Value::as_str) == Some("custom") {
            rename(tool.get_mut("name"));
        }
    }
    if let Some(choice) = params.get_mut("tool_choice")
        && choice.get("type").and_then(Value::as_str) == Some("tool")
    {
        rename(choice.get_mut("name"));
    }
    for message in messages.iter_mut() {
        if let Some(Value::Array(calls)) = message.get_mut("tool_calls") {
            for call in calls {
                rename(call.get_mut("function").and_then(|f| f.get_mut("name")));
            }
        }
        rename(
            message
                .get_mut("function_call")
                .and_then(|f| f.get_mut("name")),
        );
    }
    reverse
}

/// The `anthropic-beta` values the request's features need: the ones Python's
/// `validate_environment` collects and those `update_headers_with_optional_anthropic_beta`
/// adds. Python joins the first lot from a set, so their order carries no meaning.
fn request_betas(
    params: &Map<String, Value>,
    original_messages: &[Value],
    anthropic_messages: &[Value],
    caps: &ModelCapabilities<'_>,
) -> Result<Vec<String>, Error> {
    let tools: Vec<&Value> = params
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .collect();
    let mut environment: BTreeSet<&'static str> = BTreeSet::new();
    let tool_field = |tool: &Value, key: &str| -> Option<Value> {
        tool.get(key)
            .or_else(|| tool.get("function").and_then(|f| f.get(key)))
            .cloned()
    };
    if params
        .get("mcp_servers")
        .is_some_and(crate::base_llm::chat::params::truthy)
    {
        environment.insert("mcp-client-2025-04-04");
    }
    if tools.iter().any(|tool| {
        tool_field(tool, "allowed_callers")
            .and_then(|callers| callers.as_array().cloned())
            .is_some_and(|callers| callers.iter().any(|c| c == "code_execution_20250825"))
            || tool_field(tool, "input_examples")
                .and_then(|examples| examples.as_array().map(|e| !e.is_empty()))
                .unwrap_or(false)
    }) {
        environment.insert("advanced-tool-use-2025-11-20");
    }
    let output_config = params.get("output_config");
    if output_config
        .and_then(|config| config.get("effort"))
        .and_then(Value::as_str)
        .is_some_and(|effort| !effort.is_empty())
        && !caps.is_adaptive()?
    {
        environment.insert("effort-2025-11-24");
    }
    if original_messages
        .iter()
        .any(|message| message.get("output_config").is_some())
    {
        environment.insert("mid-conversation-output-config-2026-07-01");
    }
    let thinking = params.get("thinking");
    if thinking
        .and_then(|t| t.get("type"))
        .and_then(Value::as_str)
        .is_some_and(|kind| matches!(kind, "adaptive" | "enabled"))
        && thinking
            .and_then(|t| t.get("display"))
            .and_then(Value::as_str)
            == Some("updates")
    {
        environment.insert("thinking-display-updates-2026-08-18");
    }
    if original_messages.iter().any(|message| {
        role(message) == "system"
            && message
                .get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .any(|block| {
                    matches!(
                        block.get("type").and_then(Value::as_str),
                        Some("tool_addition" | "tool_removal")
                    ) && block
                        .get("tool")
                        .and_then(|t| t.get("type"))
                        .and_then(Value::as_str)
                        == Some("tool_reference")
                })
    }) {
        environment.insert("mid-conversation-tool-changes-2026-07-01");
    }

    let mut betas: Vec<String> = environment.into_iter().map(str::to_owned).collect();
    let mut add = |beta: &str| {
        if !betas.iter().any(|existing| existing == beta) {
            betas.push(beta.to_owned());
        }
    };
    let signed_compaction = anthropic_messages.iter().any(|message| {
        message
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .any(|block| {
                block.get("type").and_then(Value::as_str) == Some("compaction")
                    && block
                        .get("signature")
                        .and_then(Value::as_str)
                        .is_some_and(|s| !s.is_empty())
            })
    });
    if params.get("compaction").is_some_and(|v| !v.is_null()) || signed_compaction {
        add("compact-2026-09-04");
    }
    if let Some(context_management) = params.get("context_management").filter(|v| !v.is_null()) {
        let edits = match context_management {
            Value::Object(object) if object.contains_key("edits") => {
                object.get("edits").and_then(Value::as_array).cloned()
            }
            Value::Array(entries) => Some(entries.clone()),
            _ => None,
        };
        if let Some(edits) = edits {
            let compact = |edit: &Value| {
                matches!(
                    edit.get("type").and_then(Value::as_str),
                    Some("compact_20260112" | "compaction")
                )
            };
            if edits.iter().any(compact) {
                add("compact-2026-01-12");
            }
            if edits.iter().any(|edit| !compact(edit)) {
                add("context-management-2025-06-27");
            }
        }
    }
    if params.get("output_format").is_some_and(|v| !v.is_null())
        || output_config
            .and_then(|config| config.get("format"))
            .is_some_and(|v| !v.is_null())
    {
        add("structured-outputs-2025-11-13");
    }
    if params.get("speed").and_then(Value::as_str) == Some("fast") {
        add("fast-mode-2026-02-01");
    }
    Ok(betas)
}

/// `_apply_output_config`: the effort check Python makes before sending `output_config`.
fn apply_output_config(
    body: &mut Map<String, Value>,
    caps: &ModelCapabilities<'_>,
    drops: bool,
) -> Result<(), Error> {
    let Some(Value::Object(output_config)) = body.get("output_config").cloned() else {
        return Ok(());
    };
    if output_config.is_empty() {
        return Ok(());
    }
    let supports_effort = || -> Result<bool, Error> {
        if caps.supports("supports_output_config")? {
            return Ok(true);
        }
        for level in ["low", "minimal", "medium", "high", "xhigh", "max"] {
            if caps.supports(&format!("supports_{level}_reasoning_effort"))? {
                return Ok(true);
            }
        }
        Ok(false)
    };
    if drops && output_config.keys().any(|key| key != "format") && !supports_effort()? {
        match output_config.get("format") {
            None | Some(Value::Null) => {
                body.remove("output_config");
            }
            Some(format) => {
                body.insert("output_config".into(), json!({"format": format}));
            }
        }
        return Ok(());
    }
    let effort = output_config
        .get("effort")
        .filter(|effort| !effort.is_null());
    if let Some(effort) = effort {
        let text = effort.as_str().filter(|text| VALID_EFFORTS.contains(text));
        if text.is_none() {
            let shown = match effort {
                Value::String(text) => format!("'{text}'"),
                other => python_dumps(other),
            };
            return Err(Error::InvalidRequest(ErrorDetail::Message(format!(
                "Invalid effort value: {shown}. Must be one of: 'high', 'medium', 'low', 'xhigh', 'max'"
            ))));
        }
        let value = match text {
            Some("max")
                if !(caps.is_adaptive()? || caps.supports("supports_max_reasoning_effort")?) =>
            {
                Some("max")
            }
            Some("xhigh") if !caps.supports("supports_xhigh_reasoning_effort")? => Some("xhigh"),
            _ => None,
        };
        if let Some(value) = value {
            return Err(Error::InvalidRequest(ErrorDetail::UnsupportedValue {
                field: "effort",
                value,
                model: caps.model.clone(),
            }));
        }
    }
    Ok(())
}

/// `get_max_tokens_for_model`: the exact cost-map entry's limit, else 4096.
fn default_max_tokens(model: &str, context: &ChatRequestContext<'_>) -> Result<Value, Error> {
    let Some(info) = context.model_info else {
        return match context.drop_params {
            None => Err(Error::Unsupported(
                "a request whose default max_tokens comes from the model cost map",
            )),
            Some(_) => Ok(json!(DEFAULT_MAX_TOKENS)),
        };
    };
    let limit = info
        .entry(model)
        .and_then(|entry| match entry.get("max_output_tokens") {
            Some(value) => Some(value.clone()),
            None => entry.get("max_tokens").cloned(),
        });
    Ok(match limit {
        Some(value) if value.is_number() => value,
        _ => json!(DEFAULT_MAX_TOKENS),
    })
}

impl AnthropicConfig {
    fn build_request(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        optional_params: Map<String, Value>,
        context: &ChatRequestContext<'_>,
    ) -> Result<ProviderChatRequestData, Error> {
        let caps = ModelCapabilities::new(model, context);
        let request_metadata = optional_params.get("metadata").cloned();
        let sorted = sort_params(
            optional_params,
            &[],
            context,
            |_| supported_params(&caps),
            "anthropic",
            model,
        )?;
        let drop = sorted.drop;
        let MappedParams {
            mut params,
            json_mode,
        } = map_openai_params(&sorted.openai, &caps, drop)?;
        for name in &sorted.allowed {
            if let Some(value) = sorted.openai.get(name)
                && !params.contains_key(name)
            {
                params.insert(name.clone(), value.clone());
            }
        }
        params.extend(sorted.provider);

        let mut messages: Vec<Value> = messages
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<_, _>>()
            .map_err(|error| Error::InvalidRequest(ErrorDetail::invalid("messages", error)))?;
        let original_messages = messages.clone();
        let prefix_prompt = prefix_prompt(&messages);

        if !params.contains_key("tools")
            && messages.iter().any(|message| {
                message
                    .get("tool_calls")
                    .is_some_and(|calls| !calls.is_null())
            })
        {
            params.insert("tools".into(), Value::Array(map_tools(&dummy_tool())?));
        }

        // `_maybe_drop_speed_param`: fast mode is the Anthropic API's own, per exact entry.
        if let Some(speed) = params.get("speed").cloned()
            && capability(context, &[model.to_owned()], "supports_speed")? != Some(true)
        {
            drop.unsupported(|| {
                let shown = match &speed {
                    Value::String(text) => format!("'{text}'"),
                    other => python_dumps(other),
                };
                Error::InvalidRequest(ErrorDetail::Message(format!(
                    "{model} does not support speed={shown}. To drop unsupported params, set `litellm.drop_params = True`."
                )))
            })?;
            params.remove("speed");
        }

        // `maybe_drop_disabled_thinking`.
        if params
            .get("thinking")
            .and_then(|thinking| thinking.get("type"))
            .and_then(Value::as_str)
            == Some("disabled")
            && caps.supports("thinking_always_on")?
        {
            params.remove("thinking");
        }

        let tool_name_reverse_map = sanitize_tool_names(&mut params, &mut messages);

        let (leading, later) = split_leading_system_run(messages);
        let system = system_blocks(&leading);
        if !system.is_empty() {
            params.insert("system".into(), Value::Array(system));
        }
        let anthropic_messages = anthropic_messages_pt(place_mid_conversation_system(later))?;

        let betas = request_betas(&params, &original_messages, &anthropic_messages, &caps)?;

        if !params.contains_key("max_tokens") {
            params.insert("max_tokens".into(), default_max_tokens(model, context)?);
        }

        if let Some(user_id) = request_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("user_id"))
            .and_then(Value::as_str)
            .filter(|user_id| valid_user_id(user_id))
        {
            params.insert("metadata".into(), json!({"user_id": user_id}));
        }
        if let Some(Value::Object(metadata)) = params.get("metadata") {
            match metadata.get("user_id").filter(|user_id| !user_id.is_null()) {
                Some(user_id) => {
                    let reduced = json!({"user_id": user_id});
                    params.insert("metadata".into(), reduced);
                }
                None => {
                    params.remove("metadata");
                }
            }
        }
        params.remove("is_vertex_request");
        params.remove("client_metadata");
        params.remove("json_mode");

        if let Some(top_k) = params.remove("top_k").filter(|top_k| !top_k.is_null()) {
            apply_sampling_param(&mut params, &caps, drop, "top_k", &top_k, "top_k")?;
        }

        let speed = params
            .get("speed")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let response_format_schema = json_mode
            .then(|| json_tool_schema(params.get("tools")))
            .flatten();

        let mut body = Map::from_iter([
            ("model".to_owned(), json!(model)),
            ("messages".to_owned(), Value::Array(anthropic_messages)),
        ]);
        body.extend(params);
        apply_output_config(&mut body, &caps, drop.drops())?;

        Ok(ProviderChatRequestData {
            body: Value::Object(body),
            stream_shape: StreamShape {
                json_mode,
                speed,
                tool_name_reverse_map,
                response_format_schema,
                prefix_prompt,
            },
            headers: match betas.is_empty() {
                true => Vec::new(),
                false => vec![(BETA_HEADER.to_owned(), betas.join(", "))],
            },
        })
    }
}

/// `convert_tool_use_to_openai_format`, the name put back the way the caller spelled it.
fn tool_call(block: &Value, index: usize, shape: &StreamShape) -> ChatCompletionToolCallChunk {
    let name = block.get("name").and_then(Value::as_str).map(|name| {
        shape
            .tool_name_reverse_map
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_owned())
    });
    ChatCompletionToolCallChunk {
        id: block.get("id").and_then(Value::as_str).map(str::to_owned),
        tool_type: "function".into(),
        function: ChatCompletionToolCallFunctionChunk {
            name,
            arguments: python_dumps(block.get("input").unwrap_or(&Value::Null)),
            provider_specific_fields: None,
        },
        index: index as i64,
        provider_specific_fields: None,
    }
}

/// Python's `refusal_provider_specific_fields`.
fn refusal_fields(stop_reason: Option<&str>, stop_details: Option<&Value>) -> Map<String, Value> {
    let mut fields = Map::new();
    match stop_details.filter(|details| !details.is_null()) {
        Some(details) => {
            fields.insert("stop_reason".into(), json!(stop_reason));
            fields.insert("stop_details".into(), details.clone());
        }
        None if stop_reason == Some("refusal") => {
            fields.insert("stop_reason".into(), json!("refusal"));
        }
        None => {}
    }
    fields
}

fn anthropic_response(
    body: ResponseProjection,
    shape: &StreamShape,
) -> Result<ChatCompletionsResponse, Error> {
    let mut text = String::new();
    let mut citations: Option<Vec<Value>> = None;
    let mut thinking_blocks: Option<Vec<Value>> = None;
    let mut tool_calls: Vec<ChatCompletionToolCallChunk> = Vec::new();
    let mut web_search_results: Option<Vec<Value>> = None;
    let mut tool_results: Option<Vec<Value>> = None;
    let mut compaction_blocks: Option<Vec<Value>> = None;
    for (index, block) in body.content.iter().enumerate() {
        let kind = block
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match kind {
            "text" => text.push_str(
                block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            ),
            "tool_use" | "server_tool_use" => tool_calls.push(tool_call(block, index, shape)),
            result if result.ends_with("_tool_result") => {
                let list = match result {
                    "web_search_tool_result" | "web_fetch_tool_result" => &mut web_search_results,
                    _ => &mut tool_results,
                };
                list.get_or_insert_with(Vec::new).push(block.clone());
            }
            "thinking" | "redacted_thinking" => thinking_blocks
                .get_or_insert_with(Vec::new)
                .push(block.clone()),
            "compaction" => compaction_blocks
                .get_or_insert_with(Vec::new)
                .push(block.clone()),
            _ => {}
        }
        if let Some(Value::Array(block_citations)) = block.get("citations") {
            let supported_text = block.get("text").cloned().unwrap_or_else(|| json!(""));
            citations.get_or_insert_with(Vec::new).push(Value::Array(
                block_citations
                    .iter()
                    .map(|citation| {
                        let mut citation = citation.as_object().cloned().unwrap_or_default();
                        citation.insert("supported_text".into(), supported_text.clone());
                        Value::Object(citation)
                    })
                    .collect(),
            ));
        }
    }
    let reasoning_content = thinking_blocks.as_ref().map(|blocks| {
        blocks
            .iter()
            .filter_map(|block| block.get("thinking").and_then(Value::as_str))
            .collect::<String>()
    });
    if let Some(prefix) = &shape.prefix_prompt
        && !text.starts_with(prefix.as_str())
    {
        text = format!("{prefix}{text}");
    }

    // `_build_provider_specific_fields`, unset fields left out.
    let mut provider_fields = Map::new();
    let mut put = |key: &str, value: Option<Value>| {
        if let Some(value) = value.filter(|value| !value.is_null()) {
            provider_fields.insert(key.to_owned(), value);
        }
    };
    put("citations", citations.map(Value::Array));
    put("thinking_blocks", thinking_blocks.clone().map(Value::Array));
    put("context_management", body.context_management.clone());
    put("web_search_results", web_search_results.map(Value::Array));
    put("tool_results", tool_results.map(Value::Array));
    put("container", body.container.clone());
    put("compaction_blocks", compaction_blocks.map(Value::Array));
    let refusal = refusal_fields(body.stop_reason.as_deref(), body.stop_details.as_ref());
    provider_fields.extend(refusal.clone());

    let thinking_blocks: Option<Vec<ChatCompletionThinkingBlock>> = thinking_blocks
        .map(|blocks| serde_json::from_value(Value::Array(blocks)))
        .transpose()
        .map_err(|error| Error::InvalidResponse(ErrorDetail::invalid("thinking block", error)))?;

    let mut stop_reason = body.stop_reason.clone().unwrap_or_default();
    let json_indices: Vec<usize> = match shape.json_mode {
        true => tool_calls
            .iter()
            .enumerate()
            .filter(|(_, call)| call.function.name.as_deref() == Some(RESPONSE_FORMAT_TOOL_NAME))
            .map(|(index, _)| index)
            .collect(),
        false => Vec::new(),
    };
    let schema = shape.response_format_schema.as_ref();
    let message = if !json_indices.is_empty() && json_indices.len() == tool_calls.len() {
        stop_reason = "stop".into();
        let arguments: Vec<String> = tool_calls
            .iter()
            .map(|call| call.function.arguments.clone())
            .collect();
        ChatCompletionsChoiceMessage {
            role: "assistant".into(),
            content: Some(json_tool_answer(&arguments, schema)),
            provider_specific_fields: (!refusal.is_empty()).then_some(refusal),
            ..ChatCompletionsChoiceMessage::default()
        }
    } else {
        if !json_indices.is_empty() {
            let arguments: Vec<String> = json_indices
                .iter()
                .map(|index| tool_calls[*index].function.arguments.clone())
                .collect();
            text.push_str(&json_tool_answer(&arguments, schema));
            let mut position = 0;
            tool_calls.retain(|_| {
                let keep = !json_indices.contains(&position);
                position += 1;
                keep
            });
        }
        ChatCompletionsChoiceMessage {
            role: "assistant".into(),
            content: (!text.is_empty()).then_some(text),
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            reasoning_content,
            thinking_blocks,
            provider_specific_fields: (!provider_fields.is_empty()).then_some(provider_fields),
            annotations: None,
        }
    };

    Ok(ChatCompletionsResponse {
        created: unix_now(),
        model: body.model,
        choices: vec![ChatCompletionsChoice {
            index: 0,
            message,
            finish_reason: finish_reason_for(&stop_reason).to_string(),
            logprobs: None,
        }],
        usage: usage_from_parts(
            body.usage.input_tokens,
            body.usage.output_tokens,
            body.usage.cache_read_input_tokens.unwrap_or(0),
            body.usage.cache_creation_input_tokens.unwrap_or(0),
        ),
        extra: Map::new(),
    })
}

impl BaseConfig for AnthropicConfig {
    fn secret_names(&self) -> Vec<&'static str> {
        use crate::anthropic::common_utils::{
            ANTHROPIC_API_BASE_ENV, ANTHROPIC_API_KEY_ENV, ANTHROPIC_BASE_URL_ENV,
        };
        vec![
            ANTHROPIC_API_KEY_ENV,
            ANTHROPIC_API_BASE_ENV,
            ANTHROPIC_BASE_URL_ENV,
        ]
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(complete_anthropic_url(api_base, env_lookup))
    }

    fn transform_request(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        optional_params: Map<String, Value>,
        context: &ChatRequestContext<'_>,
    ) -> Result<ProviderChatRequestData, Error> {
        self.build_request(model, messages, optional_params, context)
    }

    fn transform_response(
        &self,
        _model: &str,
        response: ProviderChatResponseData,
        shape: &StreamShape,
    ) -> Result<ChatCompletionsResponse, Error> {
        let body: ResponseProjection = serde_json::from_value(response.body).map_err(|error| {
            Error::InvalidResponse(ErrorDetail::invalid("messages response", error))
        })?;
        anthropic_response(body, shape)
    }

    /// `inlines_remote_media`: Anthropic fetches an `https` image itself, not an `http` one.
    fn remote_media(&self, messages: &[ChatMessage]) -> Vec<String> {
        remote_urls(messages, &["user"], false, |url| {
            url.starts_with("http://") && data_url(url).is_none()
        })
    }

    /// A forwarded OAuth bearer is the whole credential: Python pops `x-api-key` for it,
    /// so the resolved key is not applied over it. Any other forwarded header loses to
    /// the deployment's key, which Python writes last.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        if forwarded_oauth_bearer(&headers).is_some() {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            });
        }
        let auth = AuthScheme::Credential {
            placement: API_KEY_PLACEMENT,
            secret: SecretValue::new(resolve_anthropic_api_key(api_key, env_lookup)?),
        };
        Ok(ValidatedEnvironment { headers, auth })
    }

    fn model_response_iterator(&self, shape: StreamShape) -> Option<ChatStream> {
        Some(ChatStream::new(
            messages_event_stream,
            ModelResponseIterator::new(shape),
        ))
    }

    /// Python sends `stream: true` in the body, after the other params.
    fn streaming_request(&self, url: String, mut body: Value) -> Result<(String, Value), Error> {
        if let Value::Object(fields) = &mut body {
            fields.insert(STREAM_PARAM.into(), Value::Bool(true));
        }
        Ok((url, body))
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_HEADERS
    }

    /// Every request is translated; a param or message Python refuses is refused here too.
    fn unsupported_reason(
        &self,
        _messages: &[ChatMessage],
        _optional_params: &Map<String, Value>,
    ) -> Option<Unsupported> {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use litellm_llms_types::formats::chat_completions::{ChatCompletionsResponse, ChatMessage};
    use rstest::rstest;
    use serde_json::{Map, Value, json};

    use super::*;
    use crate::base_llm::model_info::ModelInfo;

    struct Map1(HashMap<String, Map<String, Value>>);

    impl ModelInfo for Map1 {
        fn entry(&self, model: &str) -> Option<std::borrow::Cow<'_, Map<String, Value>>> {
            self.0.get(model).map(std::borrow::Cow::Borrowed)
        }
        fn declares(&self, key: &str) -> bool {
            self.0.values().any(|entry| entry.contains_key(key))
        }
    }

    fn cost_map() -> Map1 {
        let entry = |value: Value| value.as_object().cloned().unwrap();
        Map1(HashMap::from([
            (
                "claude-sonnet-4-5".to_owned(),
                entry(json!({"max_output_tokens": 64000, "supports_reasoning": true})),
            ),
            (
                "claude-opus-4-8".to_owned(),
                entry(
                    json!({"max_output_tokens": 128000, "supports_adaptive_thinking": true,
                    "supports_speed": true, "supports_sampling_params": false,
                    "supports_native_structured_output": true}),
                ),
            ),
        ]))
    }

    fn messages(value: Value) -> Vec<ChatMessage> {
        serde_json::from_value(value).expect("valid messages")
    }

    fn params(value: Value) -> Map<String, Value> {
        value.as_object().cloned().expect("params object")
    }

    fn request(
        model: &str,
        msgs: Value,
        opts: Value,
        drop: bool,
    ) -> Result<ProviderChatRequestData, Error> {
        let info = cost_map();
        let context = ChatRequestContext {
            drop_params: Some(drop),
            model_info: Some(&info),
        };
        ANTHROPIC_CHAT_COMPLETIONS_CONFIG.transform_request(
            model,
            messages(msgs),
            params(opts),
            &context,
        )
    }

    fn transform(model: &str, msgs: Value, opts: Value) -> Value {
        request(model, msgs, opts, false)
            .expect("request transforms")
            .body
    }

    fn transform_response(body: Value) -> Result<ChatCompletionsResponse, Error> {
        transform_response_with(body, &StreamShape::default())
    }

    fn transform_response_with(
        body: Value,
        shape: &StreamShape,
    ) -> Result<ChatCompletionsResponse, Error> {
        ANTHROPIC_CHAT_COMPLETIONS_CONFIG.transform_response(
            "claude-sonnet-4-5",
            ProviderChatResponseData { body },
            shape,
        )
    }

    #[rstest]
    fn builds_the_messages_body_python_builds() {
        let body = transform(
            "claude-sonnet-4-5",
            json!([
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"}
            ]),
            json!({"max_tokens": 128, "temperature": 0.2, "stop": "END", "user": "u-1"}),
        );
        assert_eq!(
            body,
            json!({
                "model": "claude-sonnet-4-5",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]}
                ],
                "temperature": 0.2,
                "stop_sequences": ["END"],
                "max_tokens": 128,
                "metadata": {"user_id": "u-1"},
                "system": [{"type": "text", "text": "be terse"}]
            })
        );
    }

    #[rstest]
    #[case::from_the_cost_map("claude-sonnet-4-5", 64000)]
    #[case::unknown_model("claude-unknown", 4096)]
    fn max_tokens_defaults_like_python(#[case] model: &str, #[case] expected: u64) {
        let body = transform(model, json!([{"role": "user", "content": "hi"}]), json!({}));
        assert_eq!(body["max_tokens"], json!(expected));
        let body = transform(
            model,
            json!([{"role": "user", "content": "hi"}]),
            json!({"max_completion_tokens": 7}),
        );
        assert_eq!(body["max_tokens"], json!(7));
    }

    #[rstest]
    fn tools_tool_choice_and_history_round_trip() {
        let body = transform(
            "claude-sonnet-4-5",
            json!([
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get.weather", "arguments": "{\"city\":\"Paris\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
            ]),
            json!({
                "max_tokens": 10,
                "tools": [{"type": "function", "function": {
                    "name": "get.weather", "description": "d",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }}],
                "tool_choice": {"type": "function", "function": {"name": "get.weather"}},
                "parallel_tool_calls": false
            }),
        );
        assert_eq!(
            body["tools"],
            json!([{"name": "get_weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}, "type": "custom", "description": "d"}])
        );
        assert_eq!(
            body["tool_choice"],
            json!({"type": "tool", "name": "get_weather", "disable_parallel_tool_use": true})
        );
        assert_eq!(
            body["messages"][1],
            json!({"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Paris"}}
            ]})
        );
        assert_eq!(
            body["messages"][2],
            json!({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "sunny"}
            ]})
        );
    }

    #[rstest]
    fn tool_history_without_tools_gets_the_dummy_tool() {
        let body = transform(
            "claude-sonnet-4-5",
            json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [
                    {"id": "c", "type": "function", "function": {"name": "f", "arguments": ""}}
                ]},
                {"role": "tool", "tool_call_id": "c", "content": "ok"}
            ]),
            json!({"max_tokens": 5}),
        );
        assert_eq!(body["tools"][0]["name"], "dummy_tool");
        assert_eq!(body["tools"][0]["description"], "This is a dummy tool call");
    }

    #[rstest]
    fn response_format_becomes_a_forced_json_tool() {
        let data = request(
            "claude-sonnet-4-5",
            json!([{"role": "user", "content": "hi"}]),
            json!({"max_tokens": 5, "response_format": {"type": "json_schema", "json_schema": {
                "name": "x", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}
            }}}),
            false,
        )
        .unwrap();
        assert_eq!(
            data.body["tool_choice"],
            json!({"name": "json_tool_call", "type": "tool"})
        );
        assert_eq!(data.body["tools"][0]["name"], "json_tool_call");
        assert!(data.body.get("json_mode").is_none());
        assert!(data.stream_shape.json_mode);
        assert!(data.stream_shape.response_format_schema.is_some());
    }

    #[rstest]
    fn native_structured_output_needs_the_beta() {
        let data = request(
            "claude-opus-4-8",
            json!([{"role": "user", "content": "hi"}]),
            json!({"max_tokens": 5, "response_format": {"type": "json_schema", "json_schema": {
                "name": "x", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}
            }}}),
            false,
        )
        .unwrap();
        assert!(data.body.get("output_format").is_some());
        assert_eq!(
            data.headers,
            vec![(
                "anthropic-beta".to_owned(),
                "structured-outputs-2025-11-13".to_owned()
            )]
        );
    }

    #[rstest]
    fn unsupported_params_are_refused_or_dropped() {
        let msgs = json!([{"role": "user", "content": "hi"}]);
        let error = request("claude-sonnet-4-5", msgs.clone(), json!({"seed": 1}), false)
            .map(|_| ())
            .unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidRequest(ErrorDetail::UnsupportedParams { ref params, .. }) if params == &["seed"]
        ));
        let body = request("claude-sonnet-4-5", msgs.clone(), json!({"seed": 1}), true)
            .unwrap()
            .body;
        assert!(body.get("seed").is_none());
        let body = request(
            "claude-sonnet-4-5",
            msgs.clone(),
            json!({"seed": 1, "drop_params": true}),
            false,
        )
        .unwrap()
        .body;
        assert!(body.get("seed").is_none());
        let body = request(
            "claude-sonnet-4-5",
            msgs.clone(),
            json!({"seed": 1, "allowed_openai_params": ["seed"]}),
            false,
        )
        .unwrap()
        .body;
        assert_eq!(body["seed"], 1);
        let body = request(
            "claude-sonnet-4-5",
            msgs,
            json!({"temperature": 0.5, "additional_drop_params": ["temperature"]}),
            false,
        )
        .unwrap()
        .body;
        assert!(body.get("temperature").is_none());
    }

    #[rstest]
    fn n_above_one_is_refused() {
        let error = request(
            "claude-sonnet-4-5",
            json!([{"role": "user", "content": "hi"}]),
            json!({"n": 2}),
            false,
        )
        .map(|_| ())
        .unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidRequest(ErrorDetail::UnsupportedParams { .. })
        ));
    }

    #[rstest]
    fn sampling_params_follow_the_model() {
        let msgs = json!([{"role": "user", "content": "hi"}]);
        let error = request("claude-opus-4-8", msgs.clone(), json!({"top_k": 3}), false)
            .map(|_| ())
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "invalid request: claude-opus-4-8 does not support top_k=3. To drop unsupported params, set `litellm.drop_params = True`."
        );
        let body = request("claude-opus-4-8", msgs.clone(), json!({"top_k": 3}), true)
            .unwrap()
            .body;
        assert!(body.get("top_k").is_none());
        let body = transform("claude-sonnet-4-5", msgs, json!({"top_k": 3}));
        assert_eq!(body["top_k"], 3);
    }

    #[rstest]
    fn speed_is_checked_against_the_exact_entry() {
        let msgs = json!([{"role": "user", "content": "hi"}]);
        let data = request(
            "claude-opus-4-8",
            msgs.clone(),
            json!({"speed": "fast"}),
            false,
        )
        .unwrap();
        assert_eq!(data.body["speed"], "fast");
        assert_eq!(data.stream_shape.speed.as_deref(), Some("fast"));
        assert_eq!(data.headers[0].1, "fast-mode-2026-02-01");
        let error = request("claude-sonnet-4-5", msgs, json!({"speed": "fast"}), false)
            .map(|_| ())
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "invalid request: claude-sonnet-4-5 does not support speed='fast'. To drop unsupported params, set `litellm.drop_params = True`."
        );
    }

    #[rstest]
    fn reasoning_effort_maps_to_thinking() {
        let body = transform(
            "claude-sonnet-4-5",
            json!([{"role": "user", "content": "hi"}]),
            json!({"reasoning_effort": "low"}),
        );
        assert_eq!(
            body["thinking"],
            json!({"type": "enabled", "budget_tokens": 1024})
        );
        assert_eq!(body["max_tokens"], json!(1024 + 4096));
    }

    #[rstest]
    fn output_config_effort_is_validated() {
        let error = request(
            "claude-sonnet-4-5",
            json!([{"role": "user", "content": "hi"}]),
            json!({"output_config": {"effort": "max"}}),
            false,
        )
        .map(|_| ())
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "invalid request: effort='max' is not supported by this model. Got model: claude-sonnet-4-5"
        );
        let error = request(
            "claude-sonnet-4-5",
            json!([{"role": "user", "content": "hi"}]),
            json!({"output_config": {"effort": "huge"}}),
            false,
        )
        .map(|_| ())
        .unwrap_err();
        assert!(error.to_string().contains("Invalid effort value: 'huge'"));
    }

    #[rstest]
    fn metadata_keeps_only_a_user_id() {
        let msgs = json!([{"role": "user", "content": "hi"}]);
        let body = transform(
            "claude-sonnet-4-5",
            msgs.clone(),
            json!({"metadata": {"user_id": "abc", "tags": ["x"]}}),
        );
        assert_eq!(body["metadata"], json!({"user_id": "abc"}));
        let body = transform("claude-sonnet-4-5", msgs, json!({"user": "a@b.com"}));
        assert!(body.get("metadata").is_none());
    }

    #[rstest]
    fn without_a_known_drop_policy_refusals_go_back_to_python() {
        let context = ChatRequestContext::default();
        let error = ANTHROPIC_CHAT_COMPLETIONS_CONFIG
            .transform_request(
                "claude-sonnet-4-5",
                messages(json!([{"role": "user", "content": "hi"}])),
                params(json!({"max_tokens": 3})),
                &context,
            )
            .map(|_| ())
            .unwrap_err();
        assert!(matches!(error, Error::Unsupported(_)));
    }

    #[rstest]
    fn only_http_images_are_downloaded() {
        let urls = ANTHROPIC_CHAT_COMPLETIONS_CONFIG.remote_media(&messages(json!([
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "http://x/a.png"}},
                {"type": "image_url", "image_url": {"url": "https://x/b.png"}}
            ]}
        ])));
        assert_eq!(urls, vec!["http://x/a.png"]);
    }

    #[rstest]
    fn a_data_url_image_becomes_a_base64_block() {
        let body = transform(
            "claude-sonnet-4-5",
            json!([{"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
            ]}]),
            json!({"max_tokens": 5}),
        );
        assert_eq!(
            body["messages"][0]["content"][1],
            json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}})
        );
    }

    #[rstest]
    fn normalizes_a_text_response_into_openai_shape() {
        let response = transform_response(json!({
            "id": "msg_123",
            "model": "claude-sonnet-4-5-20260101",
            "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": " there"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 11, "output_tokens": 4}
        }))
        .expect("response transforms");
        assert_eq!(response.model, "claude-sonnet-4-5-20260101");
        assert_eq!(
            response.choices[0].message.content.as_deref(),
            Some("hello there")
        );
        assert_eq!(response.choices[0].finish_reason, "stop");
        assert_eq!(response.usage.total_tokens, 15);
        let value = serde_json::to_value(response).unwrap();
        assert!(value.get("id").is_none());
        assert!(value["choices"][0]["message"].get("tool_calls").is_none());
    }

    #[rstest]
    fn folds_cache_tokens_into_prompt_tokens_like_python() {
        let response = transform_response(json!({
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 2,
                      "cache_read_input_tokens": 5, "cache_creation_input_tokens": 3}
        }))
        .expect("response transforms");
        assert_eq!(response.usage.prompt_tokens, 18);
        assert_eq!(response.usage.prompt_tokens_details.cached_tokens, 5);
        assert_eq!(
            response.usage.prompt_tokens_details.cache_creation_tokens,
            3
        );
    }

    #[rstest]
    #[case::refusal("refusal", "content_filter")]
    #[case::max_tokens("max_tokens", "length")]
    #[case::tool_use("tool_use", "tool_calls")]
    fn maps_stop_reasons(#[case] stop_reason: &str, #[case] expected: &str) {
        let response = transform_response(json!({
            "model": "m", "content": [], "stop_reason": stop_reason,
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }))
        .unwrap();
        assert_eq!(response.choices[0].finish_reason, expected);
        assert_eq!(response.choices[0].message.content, None);
    }

    #[rstest]
    fn tool_use_and_thinking_come_back_as_openai_fields() {
        let shape = StreamShape {
            tool_name_reverse_map: HashMap::from([("get_weather".into(), "get.weather".into())]),
            ..StreamShape::default()
        };
        let response = transform_response_with(
            json!({
                "model": "m",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                    {"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1}
            }),
            &shape,
        )
        .unwrap();
        let message = serde_json::to_value(&response.choices[0].message).unwrap();
        assert_eq!(
            message["tool_calls"],
            json!([{"id": "toolu_1", "type": "function", "function": {"name": "get.weather", "arguments": "{\"city\": \"Paris\"}"}, "index": 2}])
        );
        assert_eq!(message["reasoning_content"], "hmm");
        assert_eq!(
            message["thinking_blocks"],
            json!([{"type": "thinking", "thinking": "hmm", "signature": "sig"}])
        );
        assert_eq!(message["content"], "checking");
        assert_eq!(response.choices[0].finish_reason, "tool_calls");
    }

    #[rstest]
    fn a_json_tool_answer_becomes_the_content() {
        let shape = StreamShape {
            json_mode: true,
            ..StreamShape::default()
        };
        let response = transform_response_with(
            json!({
                "model": "m",
                "content": [{"type": "tool_use", "id": "t", "name": "json_tool_call", "input": {"a": "b"}}],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1}
            }),
            &shape,
        )
        .unwrap();
        assert_eq!(
            response.choices[0].message.content.as_deref(),
            Some("{\"a\": \"b\"}")
        );
        assert_eq!(response.choices[0].message.tool_calls, None);
        assert_eq!(response.choices[0].finish_reason, "stop");
    }

    #[rstest]
    fn a_prefix_prompt_is_put_back() {
        let data = request(
            "claude-sonnet-4-5",
            json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Argentina", "prefix": true}
            ]),
            json!({"max_tokens": 5}),
            false,
        )
        .unwrap();
        assert_eq!(
            data.stream_shape.prefix_prompt.as_deref(),
            Some("Argentina")
        );
        let response = transform_response_with(
            json!({"model": "m", "content": [{"type": "text", "text": " is"}],
                   "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}),
            &data.stream_shape,
        )
        .unwrap();
        assert_eq!(
            response.choices[0].message.content.as_deref(),
            Some("Argentina is")
        );
    }

    #[rstest]
    #[case::not_an_object(json!("nope"))]
    #[case::missing_content(json!({"model": "test-model", "usage": {"input_tokens": 1, "output_tokens": 1}}))]
    #[case::missing_usage(json!({"model": "test-model", "content": []}))]
    #[case::missing_model(json!({"content": [], "usage": {"input_tokens": 1, "output_tokens": 1}}))]
    fn errors_on_a_response_missing_required_fields(#[case] body: Value) {
        let error = transform_response(body).expect_err("invalid response");
        assert!(matches!(error, Error::InvalidResponse(_)));
        let source = std::iter::successors(Some(&error as &dyn std::error::Error), |error| {
            error.source()
        })
        .find_map(|error| error.downcast_ref::<serde_json::Error>())
        .expect("the JSON decoding source is preserved");
        assert_eq!(
            error.to_string(),
            format!("invalid response: invalid messages response: {source}")
        );
    }

    #[rstest]
    fn resolves_the_messages_url_and_x_api_key_auth() {
        let config = &ANTHROPIC_CHAT_COMPLETIONS_CONFIG;
        assert_eq!(
            config
                .get_complete_url(None, "claude-sonnet-4-5", &Map::new(), &|_| None)
                .expect("url builds"),
            "https://api.anthropic.com/v1/messages"
        );
        let validated = config
            .validate_environment(
                Vec::new(),
                Some("sk-x"),
                "claude-sonnet-4-5",
                &Map::new(),
                &|_| None,
            )
            .expect("auth resolves");
        assert!(matches!(
            validated.auth,
            AuthScheme::Credential {
                placement: litellm_auth::CredentialPlacement::Header("x-api-key"),
                ref secret
            } if secret.expose() == "sk-x"
        ));
        assert_eq!(
            config.default_headers(),
            &[
                ("anthropic-version", "2023-06-01"),
                ("content-type", "application/json"),
            ]
        );
    }

    #[rstest]
    fn tool_names_are_sanitized_without_collisions() {
        let (forward, reverse) =
            tool_name_maps(&["a.b".to_owned(), "a_b".to_owned(), "c d".to_owned()]);
        assert_eq!(forward["a.b"], "a_b_2");
        assert_eq!(forward["c d"], "c_d");
        assert!(!forward.contains_key("a_b"));
        assert_eq!(reverse["a_b_2"], "a.b");
    }
}
