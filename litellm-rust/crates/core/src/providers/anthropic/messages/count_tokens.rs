use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::constants::ANTHROPIC_OAUTH_TOKEN_PREFIX;
use crate::messages::Error;
use crate::messages::types::{AnthropicMessage, SystemPrompt};

const COUNT_TOKENS_ENDPOINT: &str = "https://api.anthropic.com/v1/messages/count_tokens";
const TOKEN_COUNTING_BETA: &str = "token-counting-2024-11-01";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnthropicCountTokensRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicCountTokensResponse {
    pub input_tokens: u64,
}

pub trait AnthropicCountTokensConfig {
    fn endpoint(&self) -> &'static str;

    fn validate_request(&self, model: &str, messages: &[AnthropicMessage]) -> Result<(), Error>;

    fn transform_request(
        &self,
        model: &str,
        messages: Vec<AnthropicMessage>,
        tools: Option<Vec<Value>>,
        system: Option<SystemPrompt>,
    ) -> Result<AnthropicCountTokensRequest, Error>;

    fn required_headers(&self, api_key: &str) -> Vec<(&'static str, String)>;
}

pub struct AnthropicCountTokensTransformation;

pub const ANTHROPIC_COUNT_TOKENS_TRANSFORMATION: AnthropicCountTokensTransformation =
    AnthropicCountTokensTransformation;

impl AnthropicCountTokensConfig for AnthropicCountTokensTransformation {
    fn endpoint(&self) -> &'static str {
        COUNT_TOKENS_ENDPOINT
    }

    fn transform_request(
        &self,
        model: &str,
        messages: Vec<AnthropicMessage>,
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

    fn validate_request(&self, model: &str, messages: &[AnthropicMessage]) -> Result<(), Error> {
        if model.is_empty() {
            return Err(Error::InvalidRequest("model parameter is required".into()));
        }
        if messages.is_empty() {
            return Err(Error::InvalidRequest(
                "messages parameter is required".into(),
            ));
        }
        Ok(())
    }

    fn required_headers(&self, api_key: &str) -> Vec<(&'static str, String)> {
        let auth = if api_key.starts_with(ANTHROPIC_OAUTH_TOKEN_PREFIX) {
            ("authorization", format!("Bearer {api_key}"))
        } else {
            ("x-api-key", api_key.to_string())
        };
        vec![
            ("content-type", "application/json".to_string()),
            auth,
            ("anthropic-version", "2023-06-01".to_string()),
            ("anthropic-beta", TOKEN_COUNTING_BETA.to_string()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, json};

    use super::*;
    use crate::messages::types::MessageContent;

    fn message() -> AnthropicMessage {
        AnthropicMessage {
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
            COUNT_TOKENS_ENDPOINT
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
            Err(Error::InvalidRequest(message)) if message == "model parameter is required"
        ));
        assert!(matches!(
            ANTHROPIC_COUNT_TOKENS_TRANSFORMATION.transform_request(
                "claude-test",
                vec![],
                None,
                None
            ),
            Err(Error::InvalidRequest(message)) if message == "messages parameter is required"
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
