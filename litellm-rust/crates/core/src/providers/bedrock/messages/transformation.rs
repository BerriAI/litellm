use crate::error::CoreResult;
use crate::messages::transformation::{AnthropicMessagesProviderConfig, MessagesAuthStrategy};
use crate::messages::types::{AnthropicMessagesRequest, AnthropicMessagesResponse};
pub struct BedrockMessagesConfig;

pub const BEDROCK_MESSAGES_CONFIG: BedrockMessagesConfig = BedrockMessagesConfig;

pub fn complete_bedrock_url(
    api_base: Option<&str>,
    model: &str,
    stream: bool,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    let _ = (api_base, model, stream, env_lookup);
    todo!("implement Bedrock InvokeModel URL construction and model path encoding")
}

impl AnthropicMessagesProviderConfig for BedrockMessagesConfig {
    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        let _ = (api_base, model, env_lookup);
        todo!("extend the shared URL contract for streaming, then implement Bedrock URLs")
    }

    fn signing_region(
        &self,
        api_base: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Option<String> {
        let _ = (api_base, env_lookup);
        todo!("implement Bedrock region precedence and endpoint parsing")
    }

    fn resolve_api_key(
        &self,
        _api_key: Option<&str>,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        Ok(String::new())
    }

    fn auth_strategy(&self) -> MessagesAuthStrategy {
        todo!("extend MessagesAuthStrategy for Bedrock bearer and SigV4 authentication")
    }

    fn transform_request(
        &self,
        request: AnthropicMessagesRequest,
    ) -> CoreResult<AnthropicMessagesRequest> {
        let _ = request;
        todo!("implement Bedrock request body filtering and anthropic_version injection")
    }

    fn transform_response(
        &self,
        model: &str,
        response: AnthropicMessagesResponse,
    ) -> CoreResult<AnthropicMessagesResponse> {
        let _ = (model, response);
        todo!("restamp the requested model when Bedrock omits it")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::bedrock::constants::{AWS_REGION, AWS_REGION_NAME};
    use serde_json::{Value, json};

    const ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";

    fn request(value: Value) -> AnthropicMessagesRequest {
        serde_json::from_value(value).expect("valid request")
    }

    #[test]
    fn builds_default_and_streaming_urls_with_encoded_arn() {
        let env = |key: &str| (key == AWS_REGION).then(|| "eu-west-1".to_string());
        let model = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/foo/bar";
        assert_eq!(
            complete_bedrock_url(None, model, false, &env).expect("url"),
            "https://bedrock-runtime.eu-west-1.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Ainference-profile%2Ffoo%2Fbar/invoke"
        );
        assert!(
            complete_bedrock_url(None, "claude", true, &env)
                .expect("url")
                .ends_with("/invoke-with-response-stream")
        );
    }

    #[test]
    fn api_base_region_wins_over_environment() {
        let env = |key: &str| (key == AWS_REGION_NAME).then(|| "us-west-2".to_string());
        assert_eq!(
            BEDROCK_MESSAGES_CONFIG.signing_region(
                Some("https://bedrock-runtime.ap-south-1.amazonaws.com"),
                &env
            ),
            Some("ap-south-1".to_string())
        );
    }

    #[test]
    fn non_bedrock_api_base_does_not_supply_a_region() {
        let env = |key: &str| (key == AWS_REGION).then(|| "eu-west-1".to_string());
        assert_eq!(
            BEDROCK_MESSAGES_CONFIG.signing_region(Some("http://127.0.0.1:8080"), &env),
            Some("eu-west-1".to_string())
        );
    }

    #[test]
    fn request_removes_path_and_unsupported_fields() {
        let transformed = BEDROCK_MESSAGES_CONFIG
            .transform_request(request(json!({
                "model": "claude",
                "stream": true,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hello"}],
                "metadata": {"user_id": "ignored"},
                "tools": [{"name": "search"}]
            })))
            .expect("transform");
        let value = serde_json::to_value(transformed).expect("json");
        assert!(value.get("model").is_none());
        assert!(value.get("stream").is_none());
        assert!(value.get("metadata").is_none());
        assert_eq!(value["anthropic_version"], ANTHROPIC_VERSION);
        assert!(value.get("tools").is_some());
    }

    #[test]
    fn rejects_empty_model_and_restamps_empty_response_model() {
        assert!(complete_bedrock_url(None, " ", false, &|_| None).is_err());
        let response: AnthropicMessagesResponse = serde_json::from_value(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "",
            "content": [],
            "stop_reason": null,
            "stop_sequence": null
        }))
        .expect("response");
        assert_eq!(
            BEDROCK_MESSAGES_CONFIG
                .transform_response("claude", response)
                .expect("response")
                .model,
            "claude"
        );
    }
}
