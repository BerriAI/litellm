//! `litellm/llms/deepseek/chat/transformation.py`: DeepSeek at `https://api.deepseek.com/beta`.
//! `thinking` and `reasoning_effort` become DeepSeek's thinking toggle, only function tools go,
//! content lists collapse to strings unless a vision model can take them, and a thinking turn
//! replays `reasoning_content` on every assistant message.

use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use serde_json::{Map, Value, json};

use crate::{
    Error,
    base_llm::chat::params::truthy,
    openai::chat::{
        compatible::{
            CompatibleContext, GPT_BASE_PARAMS, OpenAiCompatibleChatConfig, content_list_to_str,
            copy_supported, gpt_messages, role, search_results_text,
        },
        transformation::OPENAI_CHAT_COMPLETIONS_PATH,
    },
};

pub struct DeepSeekChatConfig;

pub const DEEPSEEK_CHAT_CONFIG: DeepSeekChatConfig = DeepSeekChatConfig;

impl OpenAiCompatibleChatConfig for DeepSeekChatConfig {
    fn provider(&self) -> LlmProviders {
        LlmProviders::Deepseek
    }

    fn label(&self) -> &'static str {
        "DeepSeek"
    }

    fn api_base_env(&self) -> &'static [&'static str] {
        &["DEEPSEEK_API_BASE"]
    }

    fn default_api_base(&self) -> Option<&'static str> {
        Some("https://api.deepseek.com/beta")
    }

    fn api_key_env(&self) -> &'static [&'static str] {
        &["DEEPSEEK_API_KEY"]
    }

    fn complete_url(&self, api_base: &str) -> String {
        let base = api_base.trim_end_matches('/');
        match base.ends_with(OPENAI_CHAT_COMPLETIONS_PATH) {
            true => base.to_owned(),
            false => format!("{base}{OPENAI_CHAT_COMPLETIONS_PATH}"),
        }
    }

    fn supported_params(&self, _context: &CompatibleContext<'_>) -> Vec<&'static str> {
        GPT_BASE_PARAMS
            .iter()
            .copied()
            .chain(["response_format", "thinking", "reasoning_effort"])
            .collect()
    }

    fn map_params(
        &self,
        non_default: &Map<String, Value>,
        optional: &mut Map<String, Value>,
        context: &CompatibleContext<'_>,
    ) -> Result<(), Error> {
        copy_supported(non_default, optional, &self.supported_params(context));
        let thinking = optional.remove("thinking");
        let effort = optional.remove("reasoning_effort");
        let thinking_type = thinking
            .as_ref()
            .and_then(|thinking| thinking.get("type"))
            .and_then(Value::as_str)
            .filter(|kind| ["enabled", "disabled"].contains(kind));
        if let Some(kind) = thinking_type {
            optional.insert("thinking".into(), json!({"type": kind}));
        } else if let Some(effort) = effort.filter(|effort| !effort.is_null()) {
            let kind = match effort.as_str() {
                Some("none") => "disabled",
                _ => "enabled",
            };
            optional.insert("thinking".into(), json!({"type": kind}));
        }
        drop_unsupported_tools(optional);
        Ok(())
    }

    fn transform_messages(
        &self,
        messages: Vec<Value>,
        optional: &Map<String, Value>,
        context: &CompatibleContext<'_>,
    ) -> Result<Vec<Value>, Error> {
        let thinking = context.supports(self.name(), "supports_reasoning")
            && optional
                .get("thinking")
                .and_then(|thinking| thinking.get("type"))
                == Some(&json!("enabled"));
        let messages: Vec<Value> = match thinking {
            true => messages.into_iter().map(fill_reasoning_content).collect(),
            false => messages,
        };
        let forward_images = messages
            .iter()
            .any(|message| message.get("content").is_some_and(Value::is_array))
            && context.supports(self.name(), "supports_vision");
        gpt_messages(
            messages
                .into_iter()
                .map(|message| forward_or_collapse_content(message, forward_images))
                .collect(),
        )
    }
}

/// `_drop_unsupported_tools`: DeepSeek takes only function tools.
fn drop_unsupported_tools(optional: &mut Map<String, Value>) {
    let Some(Value::Array(tools)) = optional.get("tools") else {
        return;
    };
    let is_function = |tool: &Value| tool.get("type").and_then(Value::as_str) == Some("function");
    if tools.is_empty() || tools.iter().all(is_function) {
        return;
    }
    let functions: Vec<Value> = tools
        .iter()
        .filter(|tool| is_function(tool))
        .cloned()
        .collect();
    if functions.is_empty() {
        for key in ["tools", "tool_choice", "parallel_tool_calls"] {
            optional.remove(key);
        }
        return;
    }
    let names: Vec<&str> = functions
        .iter()
        .filter_map(|tool| tool.get("function")?.get("name")?.as_str())
        .collect();
    let choice_matches = match optional.get("tool_choice") {
        Some(Value::Object(choice)) => {
            choice.get("type").and_then(Value::as_str) == Some("function")
                && choice
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                    .is_some_and(|name| names.contains(&name))
        }
        _ => true,
    };
    if !choice_matches {
        optional.remove("tool_choice");
    }
    optional.insert("tools".into(), Value::Array(functions));
}

/// `_fill_reasoning_content`.
fn fill_reasoning_content(mut message: Value) -> Value {
    if role(&message) != Some("assistant") || message.get("reasoning_content").is_some_and(truthy) {
        return message;
    }
    let Some(object) = message.as_object_mut() else {
        return message;
    };
    let stored = object
        .get("provider_specific_fields")
        .and_then(|fields| fields.get("reasoning_content"))
        .filter(|stored| truthy(stored))
        .cloned();
    match stored {
        Some(stored) => {
            object.insert("reasoning_content".into(), stored);
            if let Some(Value::Object(fields)) = object.get_mut("provider_specific_fields") {
                fields.remove("reasoning_content");
            }
        }
        None => {
            object.insert("reasoning_content".into(), json!(" "));
        }
    }
    message
}

fn forward_or_collapse_content(message: Value, forward_images: bool) -> Value {
    if forward_images
        && let Some(Value::Array(content)) = message.get("content")
        && is_vision_forwardable(&message, content)
    {
        return with_search_results_text_block(message);
    }
    let collapsed = content_list_to_str(&message);
    if collapsed.is_empty() || message.get("content") == Some(&Value::String(collapsed.clone())) {
        return message;
    }
    let mut message = message;
    if let Some(object) = message.as_object_mut() {
        object.insert("content".into(), Value::String(collapsed));
    }
    message
}

fn is_vision_forwardable(message: &Value, content: &[Value]) -> bool {
    let forwardable = |block: &Value| match block.get("type").and_then(Value::as_str) {
        Some("image_url") => match block.get("image_url") {
            Some(Value::String(url)) => !url.is_empty(),
            Some(Value::Object(image)) => image
                .get("url")
                .and_then(Value::as_str)
                .is_some_and(|url| !url.is_empty()),
            _ => false,
        },
        Some("text") => block.get("text").is_some_and(Value::is_string),
        _ => false,
    };
    role(message) == Some("user")
        && content.iter().all(forwardable)
        && content
            .iter()
            .any(|block| block.get("type").and_then(Value::as_str) == Some("image_url"))
}

fn with_search_results_text_block(mut message: Value) -> Value {
    let text = search_results_text(message.get("search_results"));
    if text.is_empty() {
        return message;
    }
    if let Some(object) = message.as_object_mut() {
        object.remove("search_results");
        if let Some(Value::Array(content)) = object.get_mut("content") {
            content.push(json!({"type": "text", "text": text}));
        }
    }
    message
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use rstest::rstest;

    use super::*;
    use crate::{base_llm::model_info::ModelInfo, openai::chat::compatible::transform_request};

    struct Capable;

    impl ModelInfo for Capable {
        fn entry(&self, model: &str) -> Option<Cow<'_, Map<String, Value>>> {
            let entry = match model {
                "deepseek/deepseek-reasoner" => json!({"supports_reasoning": true}),
                "deepseek/deepseek-vl" => json!({"supports_vision": true}),
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

    fn send(model: &str, params: Value) -> Map<String, Value> {
        let Value::Object(body) = params else {
            panic!("an object")
        };
        transform_request(
            &DEEPSEEK_CHAT_CONFIG,
            &CompatibleContext {
                model,
                drop_params: false,
                model_info: Some(&Capable),
            },
            body,
        )
        .unwrap()
        .body
    }

    #[rstest]
    #[case::base(
        "https://api.deepseek.com/beta",
        "https://api.deepseek.com/beta/chat/completions"
    )]
    #[case::named("https://h/chat/completions/", "https://h/chat/completions")]
    fn the_url_ends_in_chat_completions_once(#[case] base: &str, #[case] expected: &str) {
        assert_eq!(DEEPSEEK_CHAT_CONFIG.complete_url(base), expected);
    }

    #[rstest]
    #[case::thinking_wins(json!({"thinking": {"type": "disabled", "budget_tokens": 9}, "reasoning_effort": "high"}), json!({"type": "disabled"}))]
    #[case::effort_none(json!({"reasoning_effort": "none"}), json!({"type": "disabled"}))]
    #[case::effort_on(json!({"reasoning_effort": "low"}), json!({"type": "enabled"}))]
    fn reasoning_becomes_the_thinking_toggle(#[case] params: Value, #[case] thinking: Value) {
        let Value::Object(mut params) = params else {
            unreachable!()
        };
        params.insert("messages".into(), json!([]));
        let sent = send("deepseek-chat", Value::Object(params));
        assert_eq!(sent["thinking"], thinking);
        assert!(!sent.contains_key("reasoning_effort"));
    }

    #[rstest]
    fn only_function_tools_go_and_a_choice_naming_a_dropped_tool_goes_too() {
        let sent = send(
            "deepseek-chat",
            json!({"messages": [],
            "tools": [{"type": "function", "function": {"name": "f"}}, {"type": "namespace", "name": "n"}],
            "tool_choice": {"type": "function", "function": {"name": "n"}}}),
        );
        assert_eq!(
            sent["tools"],
            json!([{"type": "function", "function": {"name": "f"}}])
        );
        assert!(!sent.contains_key("tool_choice"));
    }

    #[rstest]
    fn a_thinking_turn_replays_reasoning_on_assistant_messages() {
        let sent = send(
            "deepseek-reasoner",
            json!({"thinking": {"type": "enabled"}, "messages": [
            {"role": "assistant", "content": "a", "provider_specific_fields": {"reasoning_content": "r"}},
            {"role": "assistant", "content": "b"}]}),
        );
        assert_eq!(
            sent["messages"],
            json!([
                {"role": "assistant", "content": "a", "provider_specific_fields": {}, "reasoning_content": "r"},
                {"role": "assistant", "content": "b", "reasoning_content": " "}])
        );
    }

    #[rstest]
    #[case::collapsed("deepseek-chat", json!("look"))]
    #[case::forwarded("deepseek-vl", json!([{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "https://i"}}]))]
    fn content_lists_collapse_unless_vision_takes_them(
        #[case] model: &str,
        #[case] content: Value,
    ) {
        let sent = send(
            model,
            json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "look"}, {"type": "image_url", "image_url": "https://i"}]}]}),
        );
        assert_eq!(sent["messages"][0]["content"], content);
    }
}
