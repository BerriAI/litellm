use litellm_llms_types::{
    formats::messages::{Message, SystemPrompt},
    providers::anthropic::{
        API_BASE, API_KEY_HEADER, API_VERSION, BETA_HEADER, COUNT_TOKENS_PATH, VERSION_HEADER,
        count_tokens::AnthropicCountTokensRequest,
    },
};
use serde_json::Value;

use crate::{Error, anthropic::ANTHROPIC_OAUTH_TOKEN_PREFIX};

const TOKEN_COUNTING_BETA: &str = "token-counting-2024-11-01";

pub trait AnthropicCountTokensConfig {
    fn endpoint(&self) -> String;

    fn validate_request(&self, model: &str, messages: &[Message]) -> Result<(), Error>;

    fn transform_request(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<Value>>,
        system: Option<SystemPrompt>,
    ) -> Result<AnthropicCountTokensRequest, Error>;

    fn required_headers(&self, api_key: &str) -> Vec<(&'static str, String)>;
}

pub struct AnthropicCountTokensTransformation;

pub const ANTHROPIC_COUNT_TOKENS_TRANSFORMATION: AnthropicCountTokensTransformation =
    AnthropicCountTokensTransformation;

impl AnthropicCountTokensConfig for AnthropicCountTokensTransformation {
    fn endpoint(&self) -> String {
        format!("{API_BASE}{COUNT_TOKENS_PATH}")
    }

    fn transform_request(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<Value>>,
        system: Option<SystemPrompt>,
    ) -> Result<AnthropicCountTokensRequest, Error> {
        self.validate_request(model, &messages)?;

        Ok(AnthropicCountTokensRequest {
            model: model.to_string(),
            messages,
            tools,
            system,
        })
    }

    fn validate_request(&self, model: &str, messages: &[Message]) -> Result<(), Error> {
        if model.is_empty() {
            return Err(Error::MissingField("model"));
        }
        if messages.is_empty() {
            return Err(Error::MissingField("messages"));
        }
        Ok(())
    }

    fn required_headers(&self, api_key: &str) -> Vec<(&'static str, String)> {
        let auth = if api_key.starts_with(ANTHROPIC_OAUTH_TOKEN_PREFIX) {
            ("authorization", format!("Bearer {api_key}"))
        } else {
            (API_KEY_HEADER, api_key.to_string())
        };
        vec![
            ("content-type", "application/json".to_string()),
            auth,
            (VERSION_HEADER, API_VERSION.to_string()),
            (BETA_HEADER, TOKEN_COUNTING_BETA.to_string()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use litellm_llms_types::formats::messages::MessageContent;
    use serde_json::{Map, json};

    use super::*;

    fn message() -> Message {
        Message {
            role: "user".into(),
            content: MessageContent::Text("hello".into()),
            extra: Map::new(),
        }
    }

    #[test]
    fn maps_the_python_count_tokens_contract() {
        let request = ANTHROPIC_COUNT_TOKENS_TRANSFORMATION
            .transform_request(
                "claude-test",
                vec![message()],
                Some(vec![json!({"name": "lookup"})]),
                Some(SystemPrompt::Text("system".into())),
            )
            .unwrap();

        assert_eq!(
            serde_json::to_value(request).unwrap(),
            json!({
                "model": "claude-test",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"name": "lookup"}],
                "system": "system"
            })
        );
        assert_eq!(
            ANTHROPIC_COUNT_TOKENS_TRANSFORMATION.endpoint(),
            "https://api.anthropic.com/v1/messages/count_tokens"
        );
    }

    #[test]
    fn rejects_the_invalid_requests_python_rejects() {
        assert!(matches!(
            ANTHROPIC_COUNT_TOKENS_TRANSFORMATION.transform_request(
                "",
                vec![message()],
                None,
                None
            ),
            Err(Error::MissingField("model"))
        ));
        assert!(matches!(
            ANTHROPIC_COUNT_TOKENS_TRANSFORMATION.transform_request(
                "claude-test",
                vec![],
                None,
                None
            ),
            Err(Error::MissingField("messages"))
        ));
    }

    #[test]
    fn uses_api_key_or_oauth_headers_without_combining_credentials() {
        let api_key = ANTHROPIC_COUNT_TOKENS_TRANSFORMATION.required_headers("sk-ant-api");
        assert!(api_key.contains(&("x-api-key", "sk-ant-api".into())));
        assert!(!api_key.iter().any(|(name, _)| *name == "authorization"));

        let oauth = ANTHROPIC_COUNT_TOKENS_TRANSFORMATION.required_headers("sk-ant-oat-test");
        assert!(oauth.contains(&("authorization", "Bearer sk-ant-oat-test".into())));
        assert!(!oauth.iter().any(|(name, _)| *name == "x-api-key"));
        assert!(oauth.contains(&("anthropic-beta", TOKEN_COUNTING_BETA.into())));
    }
}
