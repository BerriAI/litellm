//! `litellm/llms/fireworks_ai/chat/transformation.py`: Fireworks AI at
//! `https://api.fireworks.ai/inference/v1`, with models named by their Fireworks resource,
//! capability gated tool and reasoning params, vLLM style `extra_body` keys translated or
//! dropped, and tool calls Fireworks writes into the content moved back to `tool_calls`.

use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use serde_json::{Map, Value, json};

use crate::{
    Error, ErrorDetail,
    base_llm::chat::{
        params::{python_dumps, truthy},
        response_format::unpack_legacy_defs,
    },
    openai::chat::compatible::{
        CompatibleContext, OpenAiCompatibleChatConfig, RESPONSE_FORMAT_TOOL_NAME, ResponseShape,
        gpt_messages, messages_mut, role, strip_response_only_fields,
    },
};

const SUPPORTED_PARAMS: &[&str] = &[
    "stream",
    "max_completion_tokens",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "n",
    "stop",
    "response_format",
    "user",
    "logprobs",
    "prompt_truncate_len",
    "context_length_exceeded_behavior",
    "seed",
    "top_logprobs",
    "min_p",
    "typical_p",
    "repetition_penalty",
    "mirostat_target",
    "mirostat_lr",
    "logit_bias",
    "echo",
    "echo_last",
    "ignore_eos",
    "prompt_cache_key",
    "prompt_cache_isolation_key",
    "raw_output",
    "perf_metrics_in_response",
    "return_token_ids",
    "safe_tokenization",
    "service_tier",
    "speculation",
    "prediction",
    "stream_options",
    "sampling_mask",
];
const NIM_VLLM_STRIP_PARAMS: &[&str] = &[
    "stop_token_ids",
    "include_stop_str_in_output",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "best_of",
    "use_beam_search",
    "guided_decoding_backend",
    "guided_regex",
    "add_generation_prompt",
    "continue_final_message",
    "add_special_tokens",
    "detokenize",
    "allowed_token_ids",
    "bad_words",
    "include_reasoning",
    "nvext",
];
const EXTRA_BODY_CONSUMED_PARAMS: &[&str] = &[
    "truncate_prompt_tokens",
    "chat_template_kwargs",
    "guided_json",
    "guided_grammar",
    "guided_choice",
];
const FIREROUTER: &str = "firerouter";
const ROUTER_SHORT_NAMES: &[&str] = &[FIREROUTER, "auto", "auto-instant"];
const SESSION_AFFINITY_HEADER: &str = "x-session-affinity";

pub struct FireworksAiChatConfig;

pub const FIREWORKS_AI_CHAT_CONFIG: FireworksAiChatConfig = FireworksAiChatConfig;

impl OpenAiCompatibleChatConfig for FireworksAiChatConfig {
    fn provider(&self) -> LlmProviders {
        LlmProviders::FireworksAi
    }

    fn label(&self) -> &'static str {
        "Fireworks AI"
    }

    fn api_base_env(&self) -> &'static [&'static str] {
        &["FIREWORKS_API_BASE"]
    }

    fn default_api_base(&self) -> Option<&'static str> {
        Some("https://api.fireworks.ai/inference/v1")
    }

    fn api_key_env(&self) -> &'static [&'static str] {
        &[
            "FIREWORKS_API_KEY",
            "FIREWORKS_AI_API_KEY",
            "FIREWORKSAI_API_KEY",
            "FIREWORKS_AI_TOKEN",
        ]
    }

    /// `with_fireworks_session_affinity`: the caller's session pins its calls to one node so
    /// prompt caching hits.
    fn headers(
        &self,
        body: &Map<String, Value>,
        _lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Vec<(String, String)> {
        let session = body
            .get("litellm_session_id")
            .filter(|id| truthy(id))
            .or_else(|| {
                body.get("metadata")
                    .and_then(|metadata| metadata.get("session_id"))
                    .filter(|id| truthy(id))
            });
        session
            .map(|id| match id {
                Value::String(id) => id.clone(),
                other => other.to_string(),
            })
            .map(|id| vec![(SESSION_AFFINITY_HEADER.to_owned(), id)])
            .unwrap_or_default()
    }

    fn wire_model(&self, model: &str) -> String {
        resolve_fireworks_resource_name(model)
    }

    fn supported_params(&self, context: &CompatibleContext<'_>) -> Vec<&'static str> {
        let mut params = SUPPORTED_PARAMS.to_vec();
        if exact_capability(context, "supports_function_calling") != Some(false) {
            params.extend(["tools", "parallel_tool_calls"]);
        }
        if capability(context, "supports_tool_choice") {
            params.push("tool_choice");
        }
        if capability(context, "supports_reasoning") {
            params.extend(["reasoning_effort", "reasoning_history", "thinking"]);
        }
        params
    }

    fn map_params(
        &self,
        non_default: &Map<String, Value>,
        optional: &mut Map<String, Value>,
        context: &CompatibleContext<'_>,
    ) -> Result<(), Error> {
        if non_default.get("thinking").is_some_and(|v| !v.is_null())
            && non_default
                .get("reasoning_effort")
                .is_some_and(|v| !v.is_null())
        {
            return Err(Error::InvalidRequest(ErrorDetail::Message(
                "Fireworks AI chat completions does not support specifying both `thinking` and \
                 `reasoning_effort` in the same request."
                    .into(),
            )));
        }
        let supported = self.supported_params(context);
        let tools_set = non_default
            .get("tools")
            .is_some_and(|tools| !tools.is_null());
        let mut response_format_tool = None;
        for (name, value) in non_default {
            match name.as_str() {
                "tool_choice" => {
                    let choice = match value.as_str() {
                        Some("required") => json!("any"),
                        _ => value.clone(),
                    };
                    optional.insert("tool_choice".into(), choice);
                }
                "response_format" if tools_set => {
                    if let Some(schema) = response_schema(value) {
                        response_format_tool = Some(json!({"type": "function",
                            "function": {"name": RESPONSE_FORMAT_TOOL_NAME, "parameters": schema}}));
                        optional.insert("json_mode".into(), Value::Bool(true));
                    }
                }
                "max_completion_tokens" => {
                    optional.insert("max_tokens".into(), value.clone());
                }
                "reasoning_effort" => {
                    if let Some(effort) = map_reasoning_effort(value) {
                        optional.insert("reasoning_effort".into(), effort);
                    }
                }
                name if supported.contains(&name) && !value.is_null() => {
                    optional.insert(name.to_owned(), value.clone());
                }
                _ => {}
            }
        }
        if let Some(tool) = response_format_tool
            && let Some(Value::Array(tools)) = optional.get_mut("tools")
        {
            tools.push(tool);
        }
        Ok(())
    }

    /// `map_extra_body_params`.
    fn map_extra_body(
        &self,
        extra_body: &mut Map<String, Value>,
        optional: &mut Map<String, Value>,
        context: &CompatibleContext<'_>,
    ) -> Result<(), Error> {
        let mut promoted = Map::new();
        if let Some(tokens) = extra_body
            .get("truncate_prompt_tokens")
            .filter(|v| !v.is_null())
            && !extra_body.contains_key("prompt_truncate_len")
            && !optional.contains_key("prompt_truncate_len")
        {
            promoted.insert("prompt_truncate_len".into(), tokens.clone());
        }
        let explicit_reasoning = ["reasoning_effort", "thinking"]
            .iter()
            .any(|key| optional.contains_key(*key) || extra_body.contains_key(*key));
        if let Some(Value::Object(kwargs)) = extra_body.get("chat_template_kwargs")
            && !explicit_reasoning
            && let Some(effort) = effort_from_chat_template_kwargs(kwargs)
            && context.supports("fireworks_ai", "supports_reasoning")
        {
            promoted.insert("reasoning_effort".into(), effort);
        }
        let has_response_format =
            optional.contains_key("response_format") || extra_body.contains_key("response_format");
        if !has_response_format && let Some(format) = guided_response_format(extra_body) {
            promoted.insert("response_format".into(), format);
        }
        extra_body.retain(|key, _| {
            !NIM_VLLM_STRIP_PARAMS.contains(&key.as_str())
                && !EXTRA_BODY_CONSUMED_PARAMS.contains(&key.as_str())
                && (key != "response_format" || !optional.contains_key("response_format"))
        });
        optional.extend(promoted);
        Ok(())
    }

    /// `_transform_messages_helper`, then the base transform.
    fn transform_messages(
        &self,
        messages: Vec<Value>,
        _optional: &Map<String, Value>,
        context: &CompatibleContext<'_>,
    ) -> Result<Vec<Value>, Error> {
        let vision = exact_capability(context, "supports_vision");
        for message in messages
            .iter()
            .filter(|message| role(message) == Some("user"))
        {
            for kind in message
                .get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|part| part.get("type").and_then(Value::as_str))
            {
                if kind == "file" {
                    return Err(Error::InvalidRequest(ErrorDetail::Message(
                        "Fireworks AI chat completions does not support file content blocks. For \
                         PDFs, convert pages to images and send image_url blocks to a Fireworks \
                         vision model, or extract text before calling a text-only model."
                            .into(),
                    )));
                }
                if kind == "image_url" && vision == Some(false) {
                    return Err(Error::InvalidRequest(ErrorDetail::Message(format!(
                        "Fireworks AI model {} does not support image inputs. Use a Fireworks \
                         vision model or remove image_url content blocks.",
                        resolve_fireworks_resource_name(context.model)
                    ))));
                }
            }
        }
        gpt_messages(
            messages
                .into_iter()
                .map(|mut message| {
                    strip_response_only_fields(&mut message);
                    message
                })
                .collect(),
        )
    }

    fn finish_request(
        &self,
        body: &mut Map<String, Value>,
        _context: &CompatibleContext<'_>,
    ) -> Result<(), Error> {
        if let Some(Value::Array(tools)) = body.get_mut("tools") {
            for function in tools
                .iter_mut()
                .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("function"))
                .filter_map(|tool| tool.get_mut("function")?.as_object_mut())
            {
                function.remove("strict");
                if let Some(Value::Object(parameters)) = function.get_mut("parameters") {
                    unpack_legacy_defs(parameters);
                }
            }
        }
        if body.get("stream").is_some_and(truthy) {
            match body.get_mut("stream_options") {
                Some(Value::Object(options)) => {
                    if options.get("include_usage") != Some(&Value::Bool(false)) {
                        options.insert("include_usage".into(), Value::Bool(true));
                    }
                }
                _ => {
                    body.insert("stream_options".into(), json!({"include_usage": true}));
                }
            }
        }
        Ok(())
    }

    fn fix_response(&self, body: &mut Map<String, Value>, shape: &ResponseShape) -> bool {
        let mut changed = false;
        if let Some(Value::String(model)) = body.get_mut("model") {
            *model = format!("fireworks_ai/{model}");
            changed = true;
        }
        if !shape.tools_sent {
            return changed;
        }
        for message in messages_mut(body) {
            changed |= content_tool_call_to_tool_calls(message, &shape.tool_names);
        }
        changed
    }
}

/// `resolve_fireworks_resource_name`: the full Fireworks resource a short model name means.
pub fn resolve_fireworks_resource_name(model: &str) -> String {
    let stripped = model.strip_prefix("fireworks_ai/").unwrap_or(model);
    if stripped.starts_with("accounts/") || stripped.starts_with("FW-") || stripped.contains('#') {
        return stripped.to_owned();
    }
    if stripped.starts_with("routers/") || stripped.starts_with("models/") {
        return format!("accounts/fireworks/{stripped}");
    }
    if stripped.ends_with("-fast")
        || ROUTER_SHORT_NAMES.contains(&stripped)
        || stripped.starts_with(&format!("{FIREROUTER}/"))
    {
        return format!("accounts/fireworks/routers/{stripped}");
    }
    format!("accounts/fireworks/models/{stripped}")
}

/// `_get_model_cost_capability_exact`.
fn exact_capability(context: &CompatibleContext<'_>, key: &str) -> Option<bool> {
    let short = context
        .model
        .strip_prefix("fireworks_ai/")
        .unwrap_or(context.model);
    let short = short
        .strip_prefix("accounts/fireworks/models/")
        .unwrap_or(short);
    let firerouter = format!("accounts/fireworks/routers/{FIREROUTER}");
    let mut candidates = vec![
        context.model.to_owned(),
        format!("fireworks_ai/{short}"),
        format!("fireworks_ai/accounts/fireworks/models/{short}"),
    ];
    if resolve_fireworks_resource_name(context.model).starts_with(&format!("{firerouter}/")) {
        candidates.push(format!("fireworks_ai/{firerouter}"));
    }
    context.capability_of(&candidates, key)
}

fn capability(context: &CompatibleContext<'_>, key: &str) -> bool {
    exact_capability(context, key) == Some(true) || context.supports("fireworks_ai", key)
}

/// `_map_reasoning_effort`: booleans as levels, `auto` left to Fireworks.
fn map_reasoning_effort(value: &Value) -> Option<Value> {
    let effort = match value {
        Value::Object(object) => object.get("effort").cloned().unwrap_or(Value::Null),
        other => other.clone(),
    };
    match effort {
        Value::Bool(true) => Some(json!("medium")),
        Value::Bool(false) => Some(json!("none")),
        Value::String(effort) if effort == "auto" => None,
        Value::Null => None,
        other => Some(other),
    }
}

fn response_schema(value: &Value) -> Option<Value> {
    let schema = match value.get("response_schema") {
        Some(schema) => schema.clone(),
        None => value.get("json_schema")?.get("schema")?.clone(),
    };
    truthy(&schema).then_some(schema)
}

/// `effort_from_chat_template_kwargs`.
fn effort_from_chat_template_kwargs(kwargs: &Map<String, Value>) -> Option<Value> {
    let flag = |keys: &[&str]| keys.iter().find_map(|key| kwargs.get(*key)?.as_bool());
    if flag(&["enable_thinking", "thinking"]) == Some(false) {
        return Some(json!("none"));
    }
    if let Some(budget) = kwargs.get("reasoning_budget").and_then(Value::as_f64)
        && budget > 0.0
    {
        return Some(json!(budget as i64));
    }
    (flag(&["low_effort"]) == Some(true)).then(|| json!("low"))
}

/// `translate_guided_params`.
fn guided_response_format(extra_body: &Map<String, Value>) -> Option<Value> {
    let present = |key: &str| extra_body.get(key).filter(|value| !value.is_null());
    let schema = |schema: &Value, name: &str| json!({"type": "json_schema", "json_schema": {"name": name, "schema": schema}});
    if let Some(guided) = present("guided_json") {
        return Some(schema(guided, "response"));
    }
    if let Some(grammar) = present("guided_grammar") {
        return Some(json!({"type": "grammar", "grammar": grammar}));
    }
    present("guided_choice")
        .map(|choices| schema(&json!({"type": "string", "enum": choices}), "choice"))
}

/// `_handle_message_content_with_tool_calls`: a call to a sent tool, written as the content.
fn content_tool_call_to_tool_calls(
    message: &mut Map<String, Value>,
    tool_names: &[String],
) -> bool {
    if message
        .get("tool_calls")
        .is_some_and(|calls| !calls.is_null())
    {
        return false;
    }
    let Some(Value::Object(call)) = message
        .get("content")
        .and_then(Value::as_str)
        .and_then(|content| serde_json::from_str::<Value>(content).ok())
    else {
        return false;
    };
    let Some(name) = call.get("name").and_then(Value::as_str) else {
        return false;
    };
    if name == RESPONSE_FORMAT_TOOL_NAME || !tool_names.iter().any(|tool| tool == name) {
        return false;
    }
    let arguments = match call.get("arguments") {
        Some(Value::String(arguments)) => arguments.clone(),
        Some(Value::Object(arguments)) => python_dumps(&Value::Object(arguments.clone())),
        Some(Value::Null) | None => match call.get("parameters") {
            Some(parameters @ Value::Object(_)) => python_dumps(parameters),
            _ => String::new(),
        },
        Some(_) => return false,
    };
    message.insert(
        "tool_calls".into(),
        json!([{"id": uuid::Uuid::new_v4().to_string(), "type": "function",
            "function": {"name": name, "arguments": arguments}}]),
    );
    message.insert("content".into(), Value::Null);
    true
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use rstest::rstest;

    use super::*;
    use crate::{
        base_llm::model_info::ModelInfo,
        openai::chat::compatible::{CompatibleChatBody, fix_response, transform_request},
    };

    struct Registry;

    impl ModelInfo for Registry {
        fn entry(&self, model: &str) -> Option<Cow<'_, Map<String, Value>>> {
            let entry = match model {
                "fireworks_ai/accounts/fireworks/models/thinker" => {
                    json!({"supports_reasoning": true, "supports_tool_choice": true})
                }
                "fireworks_ai/text-only" => {
                    json!({"supports_vision": false, "supports_function_calling": false})
                }
                _ => return None,
            };
            let Value::Object(entry) = entry else {
                unreachable!()
            };
            Some(Cow::Owned(entry))
        }

        fn declares(&self, _key: &str) -> bool {
            true
        }
    }

    fn send(model: &str, params: Value) -> Result<CompatibleChatBody, Error> {
        let Value::Object(body) = params else {
            panic!("an object")
        };
        transform_request(
            &FIREWORKS_AI_CHAT_CONFIG,
            &CompatibleContext {
                model,
                drop_params: false,
                model_info: Some(&Registry),
            },
            body,
        )
    }

    #[rstest]
    #[case::short(
        "llama-v3p1-8b-instruct",
        "accounts/fireworks/models/llama-v3p1-8b-instruct"
    )]
    #[case::router("glm-4p6-fast", "accounts/fireworks/routers/glm-4p6-fast")]
    #[case::models_prefix("models/x", "accounts/fireworks/models/x")]
    #[case::full("accounts/me/models/x", "accounts/me/models/x")]
    #[case::deployment("x#accounts/me/deployments/d", "x#accounts/me/deployments/d")]
    fn models_are_named_by_their_resource(#[case] model: &str, #[case] resource: &str) {
        assert_eq!(resolve_fireworks_resource_name(model), resource);
    }

    #[rstest]
    fn reasoning_models_take_tool_choice_and_mapped_efforts() {
        let sent = send("thinker", json!({"messages": [], "tool_choice": "required",
            "tools": [{"type": "function", "function": {"name": "f", "strict": true,
                "parameters": {"type": "object", "properties": {"a": {"$ref": "#/definitions/A"}}, "definitions": {"A": {"type": "string"}}}}}],
            "reasoning_effort": true, "max_completion_tokens": 8}))
        .unwrap();
        assert_eq!(
            Value::Object(sent.body),
            json!({"model": "accounts/fireworks/models/thinker", "messages": [],
                "tool_choice": "any",
                "tools": [{"type": "function", "function": {"name": "f",
                    "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}}],
                "reasoning_effort": "medium", "max_tokens": 8})
        );
    }

    #[rstest]
    fn thinking_and_effort_together_are_rejected() {
        assert!(matches!(
            send(
                "thinker",
                json!({"messages": [], "thinking": {"type": "enabled"}, "reasoning_effort": "low"})
            ),
            Err(Error::InvalidRequest(ErrorDetail::Message(_)))
        ));
    }

    #[rstest]
    #[case::file(json!({"type": "file", "file": {"file_id": "f"}}))]
    #[case::image_on_text_model(json!({"type": "image_url", "image_url": {"url": "https://i"}}))]
    fn content_fireworks_cannot_read_is_rejected(#[case] part: Value) {
        assert!(matches!(
            send(
                "text-only",
                json!({"messages": [{"role": "user", "content": [part]}]})
            ),
            Err(Error::InvalidRequest(ErrorDetail::Message(_)))
        ));
    }

    #[rstest]
    fn tools_are_refused_where_the_registry_rules_out_function_calling() {
        assert!(matches!(
            send(
                "text-only",
                json!({"messages": [], "tools": [{"type": "function", "function": {"name": "f"}}]})
            ),
            Err(Error::InvalidRequest(ErrorDetail::UnsupportedParams { .. }))
        ));
    }

    #[rstest]
    fn a_response_format_beside_tools_becomes_one_more_tool() {
        let sent = send("unknown", json!({"messages": [],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}}))
        .unwrap();
        assert_eq!(
            sent.body["tools"],
            json!([{"type": "function", "function": {"name": "f"}},
                {"type": "function", "function": {"name": "json_tool_call", "parameters": {"type": "object"}}}])
        );
        assert!(!sent.body.contains_key("response_format"));
    }

    #[rstest]
    fn vllm_extra_body_keys_are_translated_or_dropped() {
        let sent = send(
            "unknown",
            json!({"messages": [], "truncate_prompt_tokens": 100,
            "guided_choice": ["a", "b"], "stop_token_ids": [1], "custom": 1}),
        )
        .unwrap();
        assert_eq!(
            Value::Object(sent.body),
            json!({"model": "accounts/fireworks/models/unknown", "messages": [],
                "prompt_truncate_len": 100,
                "response_format": {"type": "json_schema", "json_schema": {"name": "choice", "schema": {"type": "string", "enum": ["a", "b"]}}},
                "custom": 1})
        );
    }

    #[rstest]
    fn a_session_pins_the_call() {
        let body = Map::from_iter([("metadata".to_owned(), json!({"session_id": "s1"}))]);
        assert_eq!(
            FIREWORKS_AI_CHAT_CONFIG.headers(&body, &|_| None),
            vec![("x-session-affinity".to_owned(), "s1".to_owned())]
        );
    }

    #[rstest]
    #[case::sent_tool("f", true)]
    #[case::json_tool("json_tool_call", false)]
    #[case::unknown_tool("g", false)]
    fn tool_calls_in_the_content_move_to_tool_calls(#[case] name: &str, #[case] moved: bool) {
        let content = json!({"name": name, "arguments": {"a": 1}}).to_string();
        let mut body = Map::from_iter([
            ("model".to_owned(), json!("accounts/fireworks/models/x")),
            (
                "choices".to_owned(),
                json!([{"message": {"role": "assistant", "content": content}}]),
            ),
        ]);
        let shape = ResponseShape {
            tools_sent: true,
            tool_names: vec!["f".into(), "json_tool_call".into()],
            ..ResponseShape::default()
        };
        fix_response(&FIREWORKS_AI_CHAT_CONFIG, &mut body, &shape);
        assert_eq!(
            body["model"],
            json!("fireworks_ai/accounts/fireworks/models/x")
        );
        let message = &body["choices"][0]["message"];
        assert_eq!(message["content"].is_null(), moved);
        if moved {
            assert_eq!(
                message["tool_calls"][0]["function"],
                json!({"name": "f", "arguments": "{\"a\": 1}"})
            );
        }
    }
}
