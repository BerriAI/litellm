//! `litellm/llms/openrouter/chat/transformation.py`: OpenRouter at
//! `https://openrouter.ai/api/v1`, with LiteLLM's attribution headers, `cache_control` kept
//! for the model families that cache, moved into the last content block, and the usage
//! accounting OpenRouter reports cost in always asked for.

use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use serde_json::{Map, Value, json};

use crate::{
    Error,
    openai::chat::compatible::{
        CompatibleContext, GPT_BASE_PARAMS, OpenAiCompatibleChatConfig, copy_supported,
        gpt_messages,
    },
};

const CACHE_CONTROL_MODEL_FAMILIES: &[&str] = &["claude", "gemini", "minimax", "glm", "z-ai"];

pub struct OpenRouterChatConfig;

pub const OPENROUTER_CHAT_CONFIG: OpenRouterChatConfig = OpenRouterChatConfig;

impl OpenAiCompatibleChatConfig for OpenRouterChatConfig {
    fn provider(&self) -> LlmProviders {
        LlmProviders::Openrouter
    }

    fn label(&self) -> &'static str {
        "OpenRouter"
    }

    fn api_base_env(&self) -> &'static [&'static str] {
        &["OPENROUTER_API_BASE"]
    }

    fn default_api_base(&self) -> Option<&'static str> {
        Some("https://openrouter.ai/api/v1")
    }

    fn api_key_env(&self) -> &'static [&'static str] {
        &["OPENROUTER_API_KEY", "OR_API_KEY"]
    }

    fn header_env(&self) -> &'static [&'static str] {
        &["OR_SITE_URL", "OR_APP_NAME"]
    }

    fn headers(
        &self,
        _body: &Map<String, Value>,
        lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Vec<(String, String)> {
        let setting = |name: &str, default: &str| {
            lookup(name)
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| default.to_owned())
        };
        vec![
            (
                "HTTP-Referer".into(),
                setting("OR_SITE_URL", "https://litellm.ai"),
            ),
            ("X-Title".into(), setting("OR_APP_NAME", "liteLLM")),
        ]
    }

    fn supported_params(&self, _context: &CompatibleContext<'_>) -> Vec<&'static str> {
        GPT_BASE_PARAMS
            .iter()
            .copied()
            .chain(["response_format", "reasoning_effort", "thinking"])
            .collect()
    }

    fn map_params(
        &self,
        non_default: &Map<String, Value>,
        optional: &mut Map<String, Value>,
        context: &CompatibleContext<'_>,
    ) -> Result<(), Error> {
        copy_supported(non_default, optional, &self.supported_params(context));
        if optional.get("reasoning_effort") == Some(&json!("max")) {
            optional.insert("reasoning_effort".into(), json!("xhigh"));
        }
        Ok(())
    }

    fn keeps_cache_control(&self, context: &CompatibleContext<'_>) -> bool {
        caches(context.model)
    }

    fn transform_messages(
        &self,
        messages: Vec<Value>,
        _optional: &Map<String, Value>,
        context: &CompatibleContext<'_>,
    ) -> Result<Vec<Value>, Error> {
        match caches(context.model) {
            true => gpt_messages(
                messages
                    .into_iter()
                    .map(move_cache_control_to_content)
                    .collect(),
            ),
            false => gpt_messages(messages),
        }
    }

    fn finish_request(
        &self,
        body: &mut Map<String, Value>,
        _context: &CompatibleContext<'_>,
    ) -> Result<(), Error> {
        body.entry("usage")
            .or_insert_with(|| json!({"include": true}));
        Ok(())
    }
}

fn caches(model: &str) -> bool {
    let model = model.to_lowercase();
    CACHE_CONTROL_MODEL_FAMILIES
        .iter()
        .any(|family| model.contains(family))
}

/// `_move_cache_control_to_content`: only the last block carries it, to stay within
/// Anthropic's four cache breakpoints.
fn move_cache_control_to_content(mut message: Value) -> Value {
    let Some(object) = message.as_object_mut() else {
        return message;
    };
    let Some(cache_control) = object.remove("cache_control").filter(|c| !c.is_null()) else {
        return message;
    };
    match object.get_mut("content") {
        Some(Value::Array(blocks)) => {
            if let Some(Value::Object(last)) = blocks.last_mut() {
                last.insert("cache_control".into(), cache_control);
            }
        }
        content => {
            let text = content.map(Value::take).unwrap_or(Value::Null);
            object.insert(
                "content".into(),
                json!([{"type": "text", "text": text, "cache_control": cache_control}]),
            );
        }
    }
    message
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::openai::chat::compatible::transform_request;

    fn send(model: &str, params: Value) -> Map<String, Value> {
        let Value::Object(body) = params else {
            panic!("an object")
        };
        transform_request(
            &OPENROUTER_CHAT_CONFIG,
            &CompatibleContext {
                model,
                drop_params: false,
                model_info: None,
            },
            body,
        )
        .unwrap()
        .body
    }

    #[rstest]
    fn openrouter_params_and_usage_accounting_go_along() {
        let sent = send(
            "openai/gpt-4o",
            json!({"messages": [], "reasoning_effort": "max",
            "transforms": ["middle-out"], "route": "fallback"}),
        );
        assert_eq!(
            Value::Object(sent),
            json!({"model": "openai/gpt-4o", "messages": [], "reasoning_effort": "xhigh",
                "usage": {"include": true}, "transforms": ["middle-out"], "route": "fallback"})
        );
    }

    #[rstest]
    fn a_callers_usage_setting_wins() {
        let sent = send(
            "openai/gpt-4o",
            json!({"messages": [], "usage": {"include": false}}),
        );
        assert_eq!(sent["usage"], json!({"include": false}));
    }

    #[rstest]
    #[case::caching_family("anthropic/claude-sonnet-4", json!([
        {"role": "system", "content": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}]},
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}}]}]))]
    #[case::other_family("openai/gpt-4o", json!([
        {"role": "system", "content": "s"},
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]))]
    fn cache_control_reaches_only_families_that_cache(
        #[case] model: &str,
        #[case] messages: Value,
    ) {
        let sent = send(
            model,
            json!({"messages": [
            {"role": "system", "content": "s", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "cache_control": {"type": "ephemeral"}, "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]}),
        );
        assert_eq!(sent["messages"], messages);
    }

    #[rstest]
    fn attribution_headers_default_to_litellm() {
        assert_eq!(
            OPENROUTER_CHAT_CONFIG.headers(&Map::new(), &|name| (name == "OR_APP_NAME")
                .then(|| "app".to_owned())),
            vec![
                ("HTTP-Referer".to_owned(), "https://litellm.ai".to_owned()),
                ("X-Title".to_owned(), "app".to_owned()),
            ]
        );
    }
}
