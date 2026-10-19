//! `litellm/llms/groq/chat/transformation.py`: Groq at `https://api.groq.com/openai/v1`. A
//! JSON schema a model cannot take natively goes as a forced `json_tool_call` tool whose call
//! comes back as content, `web_search_options` turns on the `browser_search` tool, and the
//! response's `service_tier` is held to the values OpenAI defines.

use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use serde_json::{Map, Value, json};

use crate::{
    Error, ErrorDetail,
    openai::chat::compatible::{
        CompatibleContext, GPT_BASE_PARAMS, OpenAiCompatibleChatConfig, RESPONSE_FORMAT_TOOL_NAME,
        ResponseShape, copy_supported, gpt_messages, json_mode_tool_call_to_content, messages_mut,
        role, sanitize_usage, strip_response_only_fields,
    },
};

const COMPOUND_MODELS: &[&str] = &["compound", "compound-mini"];
const SERVICE_TIERS: &[&str] = &["auto", "default", "flex"];

pub struct GroqChatConfig;

pub const GROQ_CHAT_CONFIG: GroqChatConfig = GroqChatConfig;

impl OpenAiCompatibleChatConfig for GroqChatConfig {
    fn provider(&self) -> LlmProviders {
        LlmProviders::Groq
    }

    fn label(&self) -> &'static str {
        "Groq"
    }

    fn api_base_env(&self) -> &'static [&'static str] {
        &["GROQ_API_BASE"]
    }

    fn default_api_base(&self) -> Option<&'static str> {
        Some("https://api.groq.com/openai/v1")
    }

    fn api_key_env(&self) -> &'static [&'static str] {
        &["GROQ_API_KEY"]
    }

    fn supported_params(&self, context: &CompatibleContext<'_>) -> Vec<&'static str> {
        let web_search =
            is_compound(context.model) || context.supports(self.name(), "supports_web_search");
        let mut params: Vec<&'static str> = GPT_BASE_PARAMS
            .iter()
            .copied()
            .filter(|param| *param != "max_retries")
            .filter(|param| web_search || *param != "web_search_options")
            .chain(["response_format"])
            .collect();
        if context.supports(self.name(), "supports_reasoning") {
            params.push("reasoning_effort");
        }
        params
    }

    fn map_params(
        &self,
        non_default: &Map<String, Value>,
        optional: &mut Map<String, Value>,
        context: &CompatibleContext<'_>,
    ) -> Result<(), Error> {
        let mut params = non_default.clone();
        if params.contains_key("response_format")
            && params.get("stream") == Some(&Value::Bool(true))
        {
            return Err(Error::Unsupported(
                "streaming a groq response_format, which Python fakes from a complete call",
            ));
        }
        if let Some(schema) = response_schema(params.get("response_format"))
            && context.capability(self.name(), "supports_response_schema") != Some(true)
        {
            if params.contains_key("tools") {
                return Err(Error::InvalidRequest(ErrorDetail::Message(format!(
                    "Groq model '{}' does not support native structured outputs. LiteLLM uses a \
                     tool-calling workaround for structured outputs on this model, which is \
                     incompatible with user-provided tools. Either use a model that supports \
                     native structured outputs (e.g., gpt-oss-120b, llama-4, kimi-k2), or remove \
                     the tools parameter. See: \
                     https://console.groq.com/docs/structured-outputs#supported-models",
                    context.model
                ))));
            }
            optional.insert(
                "tools".into(),
                json!([{"type": "function", "function": {"name": RESPONSE_FORMAT_TOOL_NAME, "parameters": schema}}]),
            );
            optional.insert(
                "tool_choice".into(),
                json!({"type": "function", "function": {"name": RESPONSE_FORMAT_TOOL_NAME}}),
            );
            optional.insert("json_mode".into(), Value::Bool(true));
            params.remove("response_format");
        }
        let web_search = params.remove("web_search_options");
        copy_supported(&params, optional, &self.supported_params(context));
        if let Some(max_tokens) = optional.remove("max_completion_tokens") {
            optional.insert("max_tokens".into(), max_tokens);
        }
        if web_search.is_none() || is_compound(context.model) {
            return Ok(());
        }
        let tools = optional
            .entry("tools")
            .or_insert_with(|| Value::Array(Vec::new()));
        if tools.is_null() {
            *tools = Value::Array(Vec::new());
        }
        if let Value::Array(tools) = tools
            && !tools
                .iter()
                .any(|tool| tool.get("type").and_then(Value::as_str) == Some("browser_search"))
        {
            tools.push(json!({"type": "browser_search"}));
        }
        Ok(())
    }

    fn transform_messages(
        &self,
        messages: Vec<Value>,
        _optional: &Map<String, Value>,
        _context: &CompatibleContext<'_>,
    ) -> Result<Vec<Value>, Error> {
        gpt_messages(
            messages
                .into_iter()
                .map(|mut message| {
                    if role(&message) == Some("assistant")
                        && let Value::Object(object) = &mut message
                    {
                        object.retain(|_, value| !value.is_null());
                        strip_response_only_fields(&mut message);
                    }
                    message
                })
                .collect(),
        )
    }

    fn fix_response(&self, body: &mut Map<String, Value>, shape: &ResponseShape) -> bool {
        let mut changed = sanitize_usage(body);
        if shape.json_mode {
            changed |= json_mode_tool_call_to_content(body);
        }
        let tier = body.get("service_tier").and_then(Value::as_str);
        if !tier.is_some_and(|tier| SERVICE_TIERS.contains(&tier)) {
            body.insert("service_tier".into(), json!("auto"));
            changed = true;
        }
        changed | add_web_search_usage(body)
    }
}

fn is_compound(model: &str) -> bool {
    COMPOUND_MODELS.contains(&model.strip_prefix("groq/").unwrap_or(model))
}

fn response_schema(response_format: Option<&Value>) -> Option<Value> {
    let format = response_format?.as_object()?;
    match format.get("response_schema") {
        Some(schema) => Some(schema.clone()),
        None => format.get("json_schema")?.get("schema").cloned(),
    }
    .filter(|schema| !schema.is_null())
}

/// `_add_web_search_usage`: the searches and page opens Groq ran, from each message's
/// `executed_tools`.
fn add_web_search_usage(body: &mut Map<String, Value>) -> bool {
    if !body.get("usage").is_some_and(Value::is_object) {
        return false;
    }
    let actions: Vec<String> = messages_mut(body)
        .filter_map(|message| message.get("executed_tools")?.as_array())
        .flatten()
        .filter_map(|tool| {
            tool.get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .or_else(|| tool.get("type").and_then(Value::as_str))
                .map(str::to_owned)
        })
        .collect();
    let count = |names: &[&str]| {
        actions
            .iter()
            .filter(|a| names.contains(&a.as_str()))
            .count()
    };
    let searches = count(&["browser.search", "browser_search"]);
    let opens = count(&["browser.open"]);
    if searches == 0 && opens == 0 {
        return false;
    }
    if let Some(Value::Object(usage)) = body.get_mut("usage") {
        usage.insert(
            "server_tool_use".into(),
            json!({"web_search_requests": searches, "browser_open_requests": opens}),
        );
    }
    true
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use rstest::rstest;

    use super::*;
    use crate::{
        base_llm::model_info::ModelInfo,
        openai::chat::compatible::{fix_response, transform_request},
    };

    fn object(value: Value) -> Map<String, Value> {
        let Value::Object(map) = value else {
            panic!("an object")
        };
        map
    }

    fn send(model: &str, params: Value) -> Result<Map<String, Value>, Error> {
        transform_request(
            &GROQ_CHAT_CONFIG,
            &CompatibleContext {
                model,
                drop_params: false,
                model_info: None,
            },
            object(params),
        )
        .map(|sent| sent.body)
    }

    #[rstest]
    fn a_schema_without_native_support_goes_as_a_forced_tool() {
        let sent = transform_request(
            &GROQ_CHAT_CONFIG,
            &CompatibleContext {
                model: "llama-3.1-8b-instant",
                drop_params: false,
                model_info: None,
            },
            object(json!({"messages": [], "max_completion_tokens": 3,
                "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}})),
        )
        .unwrap();
        assert_eq!(
            Value::Object(sent.body),
            json!({"model": "llama-3.1-8b-instant", "messages": [],
                "tools": [{"type": "function", "function": {"name": "json_tool_call", "parameters": {"type": "object"}}}],
                "tool_choice": {"type": "function", "function": {"name": "json_tool_call"}},
                "max_tokens": 3})
        );
        assert!(sent.shape.json_mode);
    }

    #[rstest]
    fn a_schema_workaround_with_user_tools_is_rejected() {
        assert!(matches!(
            send(
                "llama-3.1-8b-instant",
                json!({"messages": [], "tools": [{"type": "function", "function": {"name": "f"}}],
                "response_format": {"type": "json_schema", "json_schema": {"schema": {}}}})
            ),
            Err(Error::InvalidRequest(ErrorDetail::Message(_)))
        ));
    }

    #[rstest]
    fn a_streamed_response_format_is_refused() {
        assert!(matches!(
            send(
                "llama-3.1-8b-instant",
                json!({"messages": [], "stream": true, "response_format": {"type": "json_object"}})
            ),
            Err(Error::Unsupported(_))
        ));
    }

    struct WebSearch;

    impl ModelInfo for WebSearch {
        fn entry(&self, model: &str) -> Option<Cow<'_, Map<String, Value>>> {
            (model == "groq/openai/gpt-oss-20b").then(|| {
                Cow::Owned(Map::from_iter([(
                    "supports_web_search".to_owned(),
                    json!(true),
                )]))
            })
        }

        fn declares(&self, _key: &str) -> bool {
            true
        }
    }

    #[rstest]
    #[case::adds_browser_search("openai/gpt-oss-20b", json!([{"type": "browser_search"}]))]
    #[case::compound_searches_itself("compound", Value::Null)]
    fn web_search_turns_on_browser_search(#[case] model: &str, #[case] tools: Value) {
        let sent = transform_request(
            &GROQ_CHAT_CONFIG,
            &CompatibleContext {
                model,
                drop_params: false,
                model_info: Some(&WebSearch),
            },
            object(json!({"messages": [], "web_search_options": {"search_context_size": "low"}})),
        )
        .unwrap()
        .body;
        assert_eq!(sent.get("tools").cloned().unwrap_or(Value::Null), tools);
        assert!(!sent.contains_key("web_search_options"));
    }

    #[rstest]
    fn assistant_turns_lose_nulls_and_response_only_fields() {
        let sent = send(
            "llama-3.1-8b-instant",
            json!({"messages": [
            {"role": "assistant", "content": "a", "function_call": null, "thinking_blocks": []}]}),
        )
        .unwrap();
        assert_eq!(
            sent["messages"],
            json!([{"role": "assistant", "content": "a"}])
        );
    }

    #[rstest]
    #[case::unknown(json!("on_demand"), json!("auto"))]
    #[case::missing(Value::Null, json!("auto"))]
    #[case::kept(json!("flex"), json!("flex"))]
    fn service_tier_is_held_to_openai_values(#[case] tier: Value, #[case] expected: Value) {
        let mut body =
            object(json!({"choices": [], "service_tier": tier, "usage": {"prompt_tokens": null}}));
        fix_response(&GROQ_CHAT_CONFIG, &mut body, &ResponseShape::default());
        assert_eq!(body["service_tier"], expected);
        assert_eq!(body["usage"]["prompt_tokens"], json!(0));
    }

    #[rstest]
    fn executed_browser_tools_reach_the_usage() {
        let mut body = object(
            json!({"service_tier": "auto", "usage": {"prompt_tokens": 1},
            "choices": [{"message": {"role": "assistant", "content": "x",
                "executed_tools": [{"type": "browser_search"}, {"name": "browser.open"}, {"name": "browser.search"}]}}]}),
        );
        assert!(fix_response(
            &GROQ_CHAT_CONFIG,
            &mut body,
            &ResponseShape::default()
        ));
        assert_eq!(
            body["usage"]["server_tool_use"],
            json!({"web_search_requests": 2, "browser_open_requests": 1})
        );
    }
}
