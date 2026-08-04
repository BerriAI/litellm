use crate::error::CoreResult;
use crate::messages::transformation::{AnthropicMessagesProviderConfig, MessagesAuthStrategy};
use crate::messages::types::{AnthropicMessagesRequest, AnthropicMessagesResponse};
use crate::providers::bedrock::constants::{AWS_REGION, AWS_REGION_NAME, DEFAULT_BEDROCK_REGION};
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
        todo!("Bedrock InvokeModel URLs carry the model id and the streaming mode in the path")
    }

    fn signing_region(
        &self,
        _api_base: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Option<String> {
        env_lookup(AWS_REGION_NAME)
            .or_else(|| env_lookup(AWS_REGION))
            .or_else(|| Some(DEFAULT_BEDROCK_REGION.to_string()))
    }

    fn resolve_api_key(
        &self,
        _api_key: Option<&str>,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        Ok(String::new())
    }

    fn auth_strategy(&self) -> MessagesAuthStrategy {
        todo!(
            "Bedrock authenticates with a bearer token or a signed request, not a static api key header"
        )
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
    use crate::providers::bedrock::constants::{
        AWS_REGION, AWS_REGION_NAME, DEFAULT_BEDROCK_REGION,
    };
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
    fn signing_region_uses_aws_region_name_before_aws_region_and_ignores_api_base() {
        let env = |key: &str| match key {
            AWS_REGION_NAME => Some("us-west-2".to_string()),
            AWS_REGION => Some("eu-west-1".to_string()),
            _ => None,
        };
        assert_eq!(
            BEDROCK_MESSAGES_CONFIG.signing_region(
                Some("https://bedrock-runtime.ap-south-1.amazonaws.com"),
                &env
            ),
            Some("us-west-2".to_string())
        );
    }

    #[test]
    fn signing_region_falls_back_to_default_bedrock_region() {
        assert_eq!(
            BEDROCK_MESSAGES_CONFIG.signing_region(None, &|_| None),
            Some(DEFAULT_BEDROCK_REGION.to_string())
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
