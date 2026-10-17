//! `litellm/llms/nvidia_nim/chat/transformation.py`: NVIDIA NIM through the OpenAI SDK at
//! `https://integrate.api.nvidia.com/v1`, with the params each model family takes and
//! `max_completion_tokens` sent as `max_tokens`.

use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use serde_json::{Map, Value};

use crate::{
    Error,
    openai::chat::compatible::{CompatibleContext, OpenAiCompatibleChatConfig, sdk_complete_url},
};

const GEMMA_MODELS: &[&str] = &[
    "google/recurrentgemma-2b",
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
    "gemma-2-9b-it",
];

pub struct NvidiaNimChatConfig;

pub const NVIDIA_NIM_CHAT_CONFIG: NvidiaNimChatConfig = NvidiaNimChatConfig;

impl OpenAiCompatibleChatConfig for NvidiaNimChatConfig {
    fn provider(&self) -> LlmProviders {
        LlmProviders::NvidiaNim
    }

    fn label(&self) -> &'static str {
        "NVIDIA NIM"
    }

    fn api_base_env(&self) -> &'static [&'static str] {
        &["NVIDIA_NIM_API_BASE"]
    }

    fn default_api_base(&self) -> Option<&'static str> {
        Some("https://integrate.api.nvidia.com/v1")
    }

    fn api_key_env(&self) -> &'static [&'static str] {
        &["NVIDIA_NIM_API_KEY"]
    }

    fn complete_url(&self, api_base: &str) -> String {
        sdk_complete_url(api_base)
    }

    fn supported_params(&self, context: &CompatibleContext<'_>) -> Vec<&'static str> {
        let model = context.model;
        if GEMMA_MODELS.contains(&model) {
            return vec![
                "stream",
                "temperature",
                "top_p",
                "max_tokens",
                "stop",
                "seed",
            ];
        }
        match model {
            "nvidia/nemotron-4-340b-instruct" => vec![
                "stream",
                "temperature",
                "top_p",
                "max_tokens",
                "max_completion_tokens",
            ],
            "nvidia/nemotron-4-340b-reward" => vec!["stream"],
            "google/codegemma-1.1-7b" => vec![
                "stream",
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "max_tokens",
                "max_completion_tokens",
                "stop",
            ],
            _ => vec![
                "stream",
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "max_tokens",
                "max_completion_tokens",
                "stop",
                "seed",
                "tools",
                "tool_choice",
                "parallel_tool_calls",
                "response_format",
            ],
        }
    }

    fn map_params(
        &self,
        non_default: &Map<String, Value>,
        optional: &mut Map<String, Value>,
        context: &CompatibleContext<'_>,
    ) -> Result<(), Error> {
        let supported = self.supported_params(context);
        for (name, value) in non_default {
            if name == "max_completion_tokens" {
                optional.insert("max_tokens".into(), value.clone());
            } else if supported.contains(&name.as_str()) {
                optional.insert(name.clone(), value.clone());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::{ErrorDetail, openai::chat::compatible::transform_request};

    fn context(model: &str, drop_params: bool) -> CompatibleContext<'_> {
        CompatibleContext {
            model,
            drop_params,
            model_info: None,
        }
    }

    fn body(value: Value) -> Map<String, Value> {
        let Value::Object(map) = value else {
            panic!("an object")
        };
        map
    }

    #[rstest]
    fn max_completion_tokens_goes_as_max_tokens() {
        let sent = transform_request(
            &NVIDIA_NIM_CHAT_CONFIG,
            &context("meta/llama3-70b-instruct", false),
            body(
                json!({"model": "x", "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 9, "seed": 1, "top_k": 4}),
            ),
        )
        .unwrap();
        assert_eq!(
            Value::Object(sent.body),
            json!({"model": "meta/llama3-70b-instruct", "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 9, "seed": 1, "top_k": 4})
        );
    }

    #[rstest]
    #[case::refused(false, true)]
    #[case::dropped(true, false)]
    fn a_param_the_model_family_lacks_is_refused_or_dropped(
        #[case] drop_params: bool,
        #[case] refused: bool,
    ) {
        let result = transform_request(
            &NVIDIA_NIM_CHAT_CONFIG,
            &context("google/gemma-2-9b-it", drop_params),
            body(
                json!({"messages": [], "tools": [{"type": "function", "function": {"name": "f"}}]}),
            ),
        );
        match refused {
            true => assert!(matches!(
                result,
                Err(Error::InvalidRequest(ErrorDetail::UnsupportedParams {
                    provider: "nvidia_nim",
                    ..
                }))
            )),
            false => assert!(!result.unwrap().body.contains_key("tools")),
        }
    }

    #[rstest]
    fn stream_options_are_not_sent_since_nim_does_not_list_them() {
        let sent = transform_request(
            &NVIDIA_NIM_CHAT_CONFIG,
            &context("meta/llama3-70b-instruct", false),
            body(
                json!({"messages": [], "stream": true, "stream_options": {"include_usage": true}}),
            ),
        )
        .unwrap();
        assert!(!sent.body.contains_key("stream_options"));
        assert!(sent.shape.client_usage);
    }
}
