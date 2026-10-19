//! `litellm/llms/together_ai/chat/transformation.py`: Together AI at
//! `https://api.together.ai/v1`. Tool and structured output params the cost map rules out are
//! refused or dropped, and `reasoning_effort` is translated to the levels each reasoning model
//! family takes.

use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use serde_json::{Map, Value, json};

use crate::{
    Error, ErrorDetail,
    openai::chat::compatible::{
        CompatibleContext, GPT_BASE_PARAMS, OpenAiCompatibleChatConfig, copy_supported,
        gpt_messages, role, strip_response_only_fields,
    },
};

const TOOL_CALLING_PARAMS: &[&str] = &["tools", "tool_choice", "function_call"];
const ADJUSTABLE_EFFORT_REASONING_MODELS: &[&str] = &["openai/gpt-oss-120b", "openai/gpt-oss-20b"];
const HYBRID_REASONING_MODELS: &[&str] = &[
    "MiniMaxAI/MiniMax-M3",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.6-Plus",
    "deepseek-ai/DeepSeek-V4-Pro",
    "moonshotai/Kimi-K3",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "zai-org/GLM-5.2",
];
const HIGH_MAX_EFFORT_MODEL_PREFIX: &str = "deepseek-ai/DeepSeek-V4-Pro";

pub struct TogetherAiChatConfig;

pub const TOGETHER_AI_CHAT_CONFIG: TogetherAiChatConfig = TogetherAiChatConfig;

impl OpenAiCompatibleChatConfig for TogetherAiChatConfig {
    fn provider(&self) -> LlmProviders {
        LlmProviders::TogetherAi
    }

    fn label(&self) -> &'static str {
        "Together AI"
    }

    fn api_base_env(&self) -> &'static [&'static str] {
        &["TOGETHER_AI_API_BASE"]
    }

    fn default_api_base(&self) -> Option<&'static str> {
        Some("https://api.together.ai/v1")
    }

    fn api_key_env(&self) -> &'static [&'static str] {
        &[
            "TOGETHER_API_KEY",
            "TOGETHER_AI_API_KEY",
            "TOGETHERAI_API_KEY",
            "TOGETHER_AI_TOKEN",
        ]
    }

    fn supported_params(&self, context: &CompatibleContext<'_>) -> Vec<&'static str> {
        let mut params: Vec<&'static str> = GPT_BASE_PARAMS
            .iter()
            .copied()
            .chain(["response_format"])
            .collect();
        if supports_reasoning(context) {
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
        copy_supported(non_default, optional, &self.supported_params(context));
        let passed_tools: Vec<&str> = TOOL_CALLING_PARAMS
            .iter()
            .copied()
            .filter(|param| optional.contains_key(*param))
            .collect();
        if !passed_tools.is_empty()
            && registry_rules_out(context, "supports_function_calling", &passed_tools)?
        {
            for param in passed_tools {
                optional.remove(param);
            }
        }
        if optional.contains_key("response_format")
            && registry_rules_out(context, "supports_response_schema", &["response_format"])?
        {
            optional.remove("response_format");
        }
        let Some(Value::String(effort)) = optional.get("reasoning_effort").cloned() else {
            return Ok(());
        };
        optional.remove("reasoning_effort");
        for (key, value) in reasoning_effort_payload(&effort, context) {
            optional.entry(key).or_insert(value);
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
                    if role(&message) == Some("assistant") {
                        strip_response_only_fields(&mut message);
                    }
                    message
                })
                .collect(),
        )
    }
}

fn supports_reasoning(context: &CompatibleContext<'_>) -> bool {
    let model = context.model;
    ADJUSTABLE_EFFORT_REASONING_MODELS.contains(&model)
        || HYBRID_REASONING_MODELS.contains(&model)
        || model.starts_with(HIGH_MAX_EFFORT_MODEL_PREFIX)
        || context.supports("together_ai", "supports_reasoning")
}

/// `_registry_verdict`: true when the `together_ai/<model>` entry sets `flag` to false and the
/// params are to be dropped. Without `drop_params` that is a refusal.
fn registry_rules_out(
    context: &CompatibleContext<'_>,
    flag: &str,
    params: &[&str],
) -> Result<bool, Error> {
    if context.supports("together_ai", flag)
        || context.capability_of(&[format!("together_ai/{}", context.model)], flag) != Some(false)
    {
        return Ok(false);
    }
    if context.drop_params {
        return Ok(true);
    }
    Err(Error::InvalidRequest(ErrorDetail::UnsupportedParams {
        provider: "together_ai",
        params: params.iter().map(|param| (*param).to_owned()).collect(),
        model: context.model.to_owned(),
    }))
}

fn reasoning_effort_payload(effort: &str, context: &CompatibleContext<'_>) -> Vec<(String, Value)> {
    let model = context.model;
    let effort_only = |effort: &str| vec![("reasoning_effort".to_owned(), json!(effort))];
    if effort == "default" {
        return Vec::new();
    }
    if ADJUSTABLE_EFFORT_REASONING_MODELS.contains(&model) {
        return effort_only(match effort {
            "none" => "low",
            other => translate(other),
        });
    }
    if effort == "none" {
        return vec![("reasoning".to_owned(), json!({"enabled": false}))];
    }
    if context
        .declared_reasoning_efforts("together_ai")
        .is_some_and(|declared| declared.iter().any(|level| level == effort))
    {
        return effort_only(effort);
    }
    if model.starts_with(HIGH_MAX_EFFORT_MODEL_PREFIX) {
        return effort_only(match effort {
            "minimal" | "low" | "medium" => "high",
            "xhigh" => "max",
            other => other,
        });
    }
    effort_only(translate(effort))
}

fn translate(effort: &str) -> &str {
    match effort {
        "minimal" => "low",
        "xhigh" | "max" => "high",
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use rstest::rstest;

    use super::*;
    use crate::{base_llm::model_info::ModelInfo, openai::chat::compatible::transform_request};

    struct Registry;

    impl ModelInfo for Registry {
        fn entry(&self, model: &str) -> Option<Cow<'_, Map<String, Value>>> {
            let entry = match model {
                "together_ai/no-tools" => {
                    json!({"supports_function_calling": false, "supports_response_schema": false})
                }
                "together_ai/declared" => {
                    json!({"supports_reasoning": true, "reasoning_effort_levels": ["low", "xhigh"]})
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

    fn send(model: &str, drop_params: bool, params: Value) -> Result<Map<String, Value>, Error> {
        let Value::Object(body) = params else {
            panic!("an object")
        };
        transform_request(
            &TOGETHER_AI_CHAT_CONFIG,
            &CompatibleContext {
                model,
                drop_params,
                model_info: Some(&Registry),
            },
            body,
        )
        .map(|sent| sent.body)
    }

    #[rstest]
    #[case::adjustable_none("openai/gpt-oss-20b", "none", json!({"reasoning_effort": "low"}))]
    #[case::adjustable_max("openai/gpt-oss-20b", "max", json!({"reasoning_effort": "high"}))]
    #[case::hybrid_off("zai-org/GLM-5.2", "none", json!({"reasoning": {"enabled": false}}))]
    #[case::declared("declared", "xhigh", json!({"reasoning_effort": "xhigh"}))]
    #[case::high_max_floor("deepseek-ai/DeepSeek-V4-Pro", "low", json!({"reasoning_effort": "high"}))]
    #[case::high_max_top("deepseek-ai/DeepSeek-V4-Pro", "xhigh", json!({"reasoning_effort": "max"}))]
    #[case::hybrid_minimal("zai-org/GLM-5.2", "minimal", json!({"reasoning_effort": "low"}))]
    #[case::default("zai-org/GLM-5.2", "default", json!({}))]
    fn reasoning_effort_fits_the_model_family(
        #[case] model: &str,
        #[case] effort: &str,
        #[case] expected: Value,
    ) {
        let mut sent = send(
            model,
            false,
            json!({"messages": [], "reasoning_effort": effort}),
        )
        .unwrap();
        sent.remove("model");
        sent.remove("messages");
        assert_eq!(Value::Object(sent), expected);
    }

    #[rstest]
    #[case::refused(false)]
    #[case::dropped(true)]
    fn params_the_registry_rules_out_are_refused_or_dropped(#[case] drop_params: bool) {
        let result = send(
            "no-tools",
            drop_params,
            json!({"messages": [],
            "tools": [{"type": "function", "function": {"name": "f"}}], "tool_choice": "auto",
            "response_format": {"type": "json_object"}}),
        );
        match drop_params {
            false => assert!(matches!(
                result,
                Err(Error::InvalidRequest(ErrorDetail::UnsupportedParams {
                    provider: "together_ai",
                    ..
                }))
            )),
            true => assert_eq!(
                Value::Object(result.unwrap()),
                json!({"model": "no-tools", "messages": []})
            ),
        }
    }

    #[rstest]
    fn unknown_models_pass_tools_through() {
        let sent = send(
            "unknown",
            false,
            json!({"messages": [],
            "tools": [{"type": "function", "function": {"name": "f"}}]}),
        )
        .unwrap();
        assert_eq!(
            sent["tools"],
            json!([{"type": "function", "function": {"name": "f"}}])
        );
    }

    #[rstest]
    fn replayed_reasoning_stays_while_response_only_fields_go() {
        let sent = send(
            "unknown",
            false,
            json!({"messages": [{"role": "assistant", "content": "a",
            "reasoning_content": "r", "thinking_blocks": [], "provider_specific_fields": {}}]}),
        )
        .unwrap();
        assert_eq!(
            sent["messages"],
            json!([{"role": "assistant", "content": "a", "reasoning_content": "r"}])
        );
    }
}
