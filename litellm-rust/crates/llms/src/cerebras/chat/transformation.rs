//! `litellm/llms/cerebras/chat.py`: Cerebras through the OpenAI SDK at
//! `https://api.cerebras.ai/v1`, with `reasoning_effort` for the models the cost map marks as
//! reasoning ones and `max_completion_tokens` sent as `max_tokens`.

use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use serde_json::{Map, Value};

use crate::{
    Error,
    openai::chat::compatible::{CompatibleContext, OpenAiCompatibleChatConfig, sdk_complete_url},
};

const SUPPORTED_PARAMS: &[&str] = &[
    "max_tokens",
    "max_completion_tokens",
    "response_format",
    "seed",
    "stop",
    "stream",
    "temperature",
    "top_p",
    "tool_choice",
    "tools",
    "user",
    "max_retries",
    "extra_headers",
];

pub struct CerebrasChatConfig;

pub const CEREBRAS_CHAT_CONFIG: CerebrasChatConfig = CerebrasChatConfig;

impl OpenAiCompatibleChatConfig for CerebrasChatConfig {
    fn provider(&self) -> LlmProviders {
        LlmProviders::Cerebras
    }

    fn label(&self) -> &'static str {
        "Cerebras"
    }

    fn api_base_env(&self) -> &'static [&'static str] {
        &["CEREBRAS_API_BASE"]
    }

    fn default_api_base(&self) -> Option<&'static str> {
        Some("https://api.cerebras.ai/v1")
    }

    fn api_key_env(&self) -> &'static [&'static str] {
        &["CEREBRAS_API_KEY"]
    }

    fn complete_url(&self, api_base: &str) -> String {
        sdk_complete_url(api_base)
    }

    fn supported_params(&self, context: &CompatibleContext<'_>) -> Vec<&'static str> {
        let mut params = SUPPORTED_PARAMS.to_vec();
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
    use std::borrow::Cow;

    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::{
        ErrorDetail, base_llm::model_info::ModelInfo, openai::chat::compatible::transform_request,
    };

    struct Reasoning;

    impl ModelInfo for Reasoning {
        fn entry(&self, model: &str) -> Option<Cow<'_, Map<String, Value>>> {
            (model == "cerebras/gpt-oss-120b").then(|| {
                Cow::Owned(Map::from_iter([(
                    "supports_reasoning".to_owned(),
                    json!(true),
                )]))
            })
        }

        fn declares(&self, _key: &str) -> bool {
            true
        }
    }

    fn send(model: &str, params: Value) -> Result<Map<String, Value>, Error> {
        let Value::Object(body) = params else {
            panic!("an object")
        };
        transform_request(
            &CEREBRAS_CHAT_CONFIG,
            &CompatibleContext {
                model,
                drop_params: false,
                model_info: Some(&Reasoning),
            },
            body,
        )
        .map(|sent| sent.body)
    }

    #[rstest]
    fn reasoning_effort_goes_to_reasoning_models() {
        let sent = send(
            "gpt-oss-120b",
            json!({"messages": [], "reasoning_effort": "low", "max_completion_tokens": 5}),
        )
        .unwrap();
        assert_eq!(
            Value::Object(sent),
            json!({"model": "gpt-oss-120b", "messages": [], "reasoning_effort": "low", "max_tokens": 5})
        );
    }

    #[rstest]
    fn reasoning_effort_is_refused_elsewhere() {
        assert!(matches!(
            send(
                "llama3.1-8b",
                json!({"messages": [], "reasoning_effort": "low"})
            ),
            Err(Error::InvalidRequest(ErrorDetail::UnsupportedParams { .. }))
        ));
    }

    #[rstest]
    fn frequency_penalty_is_refused() {
        assert!(
            send(
                "llama3.1-8b",
                json!({"messages": [], "frequency_penalty": 1})
            )
            .is_err()
        );
    }
}
