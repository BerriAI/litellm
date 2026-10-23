use std::{collections::BTreeMap, time::Duration};

use litellm_auth::{SecretValue, Sourced};
use litellm_llms_types::formats::tokenize::{
    TokenizeInclude, TokenizeRequest as PublicTokenizeRequest,
};
use litellm_token_counter::raw_text::LocalTokenizerConfig;
use serde_json::Value;
use std::sync::Arc;

#[derive(Clone, Default)]
pub enum TokenizeBackend {
    #[default]
    Provider,
    Local {
        config: Sourced<Arc<LocalTokenizerConfig>>,
    },
}

#[derive(Clone, Default)]
pub struct TokenizeRequest {
    pub model: String,
    pub prompt: String,
    pub include: TokenizeInclude,
    pub add_special_tokens: Option<bool>,
    pub backend: TokenizeBackend,
    pub custom_llm_provider: Option<String>,
    pub api_key: Option<Sourced<SecretValue>>,
    pub api_base: Option<Sourced<String>>,
    pub extra_headers: BTreeMap<String, Sourced<Value>>,
    pub deployment_params: BTreeMap<String, Sourced<Value>>,
    pub optional_params: BTreeMap<String, Sourced<Value>>,
    pub timeout: Option<Duration>,
}

impl From<PublicTokenizeRequest> for TokenizeRequest {
    fn from(request: PublicTokenizeRequest) -> Self {
        Self {
            model: request.model,
            prompt: request.prompt,
            include: request.include,
            add_special_tokens: request.add_special_tokens,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use litellm_llms_types::formats::tokenize::TokenizeOutput;
    use rstest::rstest;
    use serde_json::json;

    use super::{TokenizeBackend, TokenizeRequest};

    #[rstest]
    fn public_request_conversion_carries_only_public_fields() {
        let public: litellm_llms_types::formats::tokenize::TokenizeRequest =
            serde_json::from_value(json!({
                "model": "model-alias",
                "prompt": "hello",
                "include": ["token_ids"],
                "add_special_tokens": false
            }))
            .unwrap();
        let request: TokenizeRequest = public.into();

        assert_eq!(request.model, "model-alias");
        assert_eq!(request.prompt, "hello");
        assert_eq!(request.include.output(), TokenizeOutput::Ids);
        assert_eq!(request.add_special_tokens, Some(false));
        assert!(matches!(request.backend, TokenizeBackend::Provider));
        assert!(request.optional_params.is_empty());
    }
}
