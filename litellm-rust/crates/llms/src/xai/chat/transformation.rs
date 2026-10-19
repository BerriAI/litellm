//! `litellm/llms/xai/chat/transformation.py`: xAI at `https://api.x.ai/v1`. Tools lose
//! `strict`, `web_search_options` and non-user `name`s are dropped, and the usage folds the
//! separately counted reasoning tokens into `completion_tokens` as OpenAI counts them.

use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use serde_json::{Map, Value, json};

use crate::{
    Error,
    openai::chat::compatible::{
        CompatibleContext, OpenAiCompatibleChatConfig, ResponseShape, filter_value, role,
    },
};

const SUPPORTED_PARAMS: &[&str] = &[
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "n",
    "parallel_tool_calls",
    "presence_penalty",
    "response_format",
    "seed",
    "stream",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "user",
    "web_search_options",
];

pub struct XaiChatConfig;

pub const XAI_CHAT_CONFIG: XaiChatConfig = XaiChatConfig;

impl OpenAiCompatibleChatConfig for XaiChatConfig {
    fn provider(&self) -> LlmProviders {
        LlmProviders::Xai
    }

    fn label(&self) -> &'static str {
        "xAI"
    }

    fn api_base_env(&self) -> &'static [&'static str] {
        &["XAI_API_BASE"]
    }

    fn default_api_base(&self) -> Option<&'static str> {
        Some("https://api.x.ai/v1")
    }

    fn api_key_env(&self) -> &'static [&'static str] {
        &["XAI_API_KEY"]
    }

    fn supported_params(&self, context: &CompatibleContext<'_>) -> Vec<&'static str> {
        let model = context.model;
        let mut params = SUPPORTED_PARAMS.to_vec();
        if !["grok-3-mini", "grok-4", "grok-code-fast"]
            .iter()
            .any(|family| model.contains(family))
        {
            params.push("stop");
        }
        if !["grok-4", "grok-code-fast"]
            .iter()
            .any(|family| model.contains(family))
        {
            params.push("frequency_penalty");
        }
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
        let supported = self.supported_params(context);
        for (name, value) in non_default {
            match name.as_str() {
                "max_completion_tokens" => {
                    optional.insert("max_tokens".into(), value.clone());
                }
                "tools" => {
                    let Value::Array(tools) = value else { continue };
                    let tools: Vec<Value> = tools
                        .iter()
                        .cloned()
                        .map(|mut tool| {
                            filter_value(&mut tool, "strict", 0);
                            tool
                        })
                        .collect();
                    if !tools.is_empty() {
                        optional.insert("tools".into(), Value::Array(tools));
                    }
                }
                name if supported.contains(&name) && !value.is_null() => {
                    optional.insert(name.to_owned(), value.clone());
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn finish_request(
        &self,
        body: &mut Map<String, Value>,
        _context: &CompatibleContext<'_>,
    ) -> Result<(), Error> {
        body.remove("web_search_options");
        if let Some(Value::Array(messages)) = body.get_mut("messages") {
            for message in messages.iter_mut() {
                if role(message) != Some("user")
                    && let Some(object) = message.as_object_mut()
                {
                    object.remove("name");
                }
            }
        }
        Ok(())
    }

    fn fix_response(&self, body: &mut Map<String, Value>, _shape: &ResponseShape) -> bool {
        let mut changed = false;
        if let Some(Value::Array(choices)) = body.get_mut("choices") {
            for choice in choices.iter_mut().filter_map(Value::as_object_mut) {
                let has_tool_calls = choice
                    .get("message")
                    .and_then(|message| message.get("tool_calls"))
                    .and_then(Value::as_array)
                    .is_some_and(|calls| !calls.is_empty());
                if has_tool_calls && choice.get("finish_reason") == Some(&json!("")) {
                    choice.insert("finish_reason".into(), json!("tool_calls"));
                    changed = true;
                }
            }
        }
        changed | fix_usage(body)
    }

    fn fix_chunk(&self, chunk: &mut Map<String, Value>, _shape: &ResponseShape) -> bool {
        fix_usage(chunk)
    }
}

fn count(usage: &Map<String, Value>, key: &str) -> i64 {
    usage.get(key).and_then(Value::as_i64).unwrap_or(0)
}

/// `fold_reasoning_tokens_into_completion` then `_normalize_openai_compatible_usage_totals`.
fn fix_usage(body: &mut Map<String, Value>) -> bool {
    let Some(Value::Object(usage)) = body.get_mut("usage") else {
        return false;
    };
    let mut changed = false;
    let reasoning = usage
        .get("completion_tokens_details")
        .and_then(|details| details.get("reasoning_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let (prompt, completion, total) = (
        count(usage, "prompt_tokens"),
        count(usage, "completion_tokens"),
        count(usage, "total_tokens"),
    );
    if reasoning > 0 && total != prompt + completion && total == prompt + completion + reasoning {
        usage.insert("completion_tokens".into(), json!(completion + reasoning));
        changed = true;
    }
    let expected = prompt + count(usage, "completion_tokens");
    if count(usage, "total_tokens") < expected {
        usage.insert("total_tokens".into(), json!(expected));
        changed = true;
    }
    changed
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::openai::chat::compatible::{fix_response, transform_request};

    fn object(value: Value) -> Map<String, Value> {
        let Value::Object(map) = value else {
            panic!("an object")
        };
        map
    }

    fn send(model: &str, params: Value) -> Result<Map<String, Value>, Error> {
        transform_request(
            &XAI_CHAT_CONFIG,
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
    fn tools_lose_strict_and_web_search_and_names_are_dropped() {
        let sent = send("grok-3", json!({"messages": [
                {"role": "user", "name": "u", "content": "hi"},
                {"role": "assistant", "name": "a", "content": "yo"}],
            "tools": [{"type": "function", "function": {"name": "f", "strict": true, "parameters": {"type": "object"}}}],
            "web_search_options": {}, "max_completion_tokens": 4, "stop": ["x"]}))
        .unwrap();
        assert_eq!(
            Value::Object(sent),
            json!({"model": "grok-3", "messages": [
                    {"role": "user", "name": "u", "content": "hi"},
                    {"role": "assistant", "content": "yo"}],
                "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
                "max_tokens": 4, "stop": ["x"]})
        );
    }

    #[rstest]
    #[case::stop("grok-4", json!({"messages": [], "stop": ["x"]}))]
    #[case::frequency_penalty("grok-code-fast-1", json!({"messages": [], "frequency_penalty": 1}))]
    fn params_some_grok_models_reject_are_refused(#[case] model: &str, #[case] params: Value) {
        assert!(send(model, params).is_err());
    }

    #[rstest]
    #[case::folds(json!({"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 9,
        "completion_tokens_details": {"reasoning_tokens": 4}}), json!(7), json!(9))]
    #[case::already_folded(json!({"prompt_tokens": 2, "completion_tokens": 7, "total_tokens": 9,
        "completion_tokens_details": {"reasoning_tokens": 4}}), json!(7), json!(9))]
    #[case::raises_total(json!({"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 1}), json!(3), json!(5))]
    fn usage_counts_reasoning_as_completion(
        #[case] usage: Value,
        #[case] completion: Value,
        #[case] total: Value,
    ) {
        let mut body = object(json!({"choices": [], "usage": usage}));
        XAI_CHAT_CONFIG.fix_chunk(&mut body, &ResponseShape::default());
        assert_eq!(body["usage"]["completion_tokens"], completion);
        assert_eq!(body["usage"]["total_tokens"], total);
    }

    #[rstest]
    fn an_empty_finish_reason_with_tool_calls_becomes_tool_calls() {
        let mut body = object(
            json!({"choices": [{"finish_reason": "", "message": {"role": "assistant",
            "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}}]}),
        );
        assert!(fix_response(
            &XAI_CHAT_CONFIG,
            &mut body,
            &ResponseShape::default()
        ));
        assert_eq!(body["choices"][0]["finish_reason"], json!("tool_calls"));
    }
}
