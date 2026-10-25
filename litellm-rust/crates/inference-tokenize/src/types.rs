use std::{collections::BTreeMap, time::Duration};

use litellm_auth::{SecretValue, Sourced};
use litellm_llms_types::formats::tokenize::{
    TokenizeInclude, TokenizeOutput, TokenizeRequest as PublicTokenizeRequest,
};
use serde_json::Value;

#[derive(Clone, Default)]
pub struct TokenizeRequest {
    pub model: String,
    pub prompt: String,
    pub include: Option<TokenizeInclude>,
    pub add_special_tokens: Option<bool>,
    pub custom_llm_provider: Option<String>,
    pub api_key: Option<Sourced<SecretValue>>,
    pub api_base: Option<Sourced<String>>,
    pub extra_headers: BTreeMap<String, Sourced<Value>>,
    pub deployment_params: BTreeMap<String, Sourced<Value>>,
    pub optional_params: BTreeMap<String, Sourced<Value>>,
    pub timeout: Option<Duration>,
}

impl TokenizeRequest {
    pub fn output(&self) -> TokenizeOutput {
        self.include
            .as_ref()
            .map_or(TokenizeOutput::Available, TokenizeInclude::output)
    }
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

    use super::TokenizeRequest;

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
        assert_eq!(request.output(), TokenizeOutput::Ids);
        assert_eq!(request.add_special_tokens, Some(false));
        assert!(request.optional_params.is_empty());
    }
}
