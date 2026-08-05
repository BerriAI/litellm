use crate::error::{CoreError, CoreResult};
use crate::messages::transformation::{AnthropicMessagesProviderConfig, MessagesAuthStrategy};
use crate::messages::types::{
    AnthropicMessage, AnthropicMessagesRequest, AnthropicMessagesResponse, SystemPrompt,
};
use crate::providers::anthropic::messages::transformation::ANTHROPIC_MESSAGES_CONFIG;
use crate::providers::bedrock::constants::{
    AWS_REGION, AWS_REGION_NAME, BEDROCK_ANTHROPIC_VERSION, BEDROCK_RUNTIME_ENDPOINT_TEMPLATE,
    DEFAULT_BEDROCK_REGION,
};

const AWS_BEARER_TOKEN_BEDROCK: &str = "AWS_BEARER_TOKEN_BEDROCK";
use serde::Serialize;
use serde_json::Value;

#[derive(Serialize)]
struct BedrockInvokeAnthropicMessagesRequest {
    anthropic_version: Value,
    max_tokens: u64,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    anthropic_beta: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<SystemPrompt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_management: Option<Value>,
}

impl TryFrom<AnthropicMessagesRequest> for BedrockInvokeAnthropicMessagesRequest {
    type Error = CoreError;

    fn try_from(request: AnthropicMessagesRequest) -> CoreResult<Self> {
        let AnthropicMessagesRequest {
            max_tokens,
            messages,
            system,
            metadata,
            stop_sequences,
            temperature,
            top_p,
            top_k,
            tools,
            tool_choice,
            thinking,
            output_config,
            context_management,
            mut extra,
            ..
        } = request;
        let anthropic_version = extra
            .remove("anthropic_version")
            .unwrap_or_else(|| Value::String(BEDROCK_ANTHROPIC_VERSION.to_string()));

        Ok(Self {
            anthropic_version,
            max_tokens: max_tokens.ok_or_else(|| {
                CoreError::InvalidRequest("Bedrock Anthropic Messages requires `max_tokens`".to_string())
            })?,
            messages,
            anthropic_beta: extra.remove("anthropic_beta"),
            system,
            stop_sequences,
            temperature,
            top_p,
            top_k,
            tools,
            tool_choice,
            thinking,
            metadata,
            output_config,
            context_management,
        })
    }
}

pub struct BedrockMessagesConfig;

pub const BEDROCK_MESSAGES_CONFIG: BedrockMessagesConfig = BedrockMessagesConfig;

pub fn complete_bedrock_url(
    api_base: Option<&str>,
    model: &str,
    stream: bool,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    let model = model.trim();
    if model.is_empty() {
        return Err(CoreError::InvalidRequest(
            "Bedrock model must not be empty".to_string(),
        ));
    }

    let region = env_lookup(AWS_REGION_NAME)
        .or_else(|| env_lookup(AWS_REGION))
        .unwrap_or_else(|| DEFAULT_BEDROCK_REGION.to_string());
    let endpoint = api_base
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| BEDROCK_RUNTIME_ENDPOINT_TEMPLATE.replace("{region}", &region));
    let model = model
        .bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (byte as char).to_string()
            }
            _ => format!("%{byte:02X}"),
        })
        .collect::<String>();
    let operation = if stream {
        "invoke-with-response-stream"
    } else {
        "invoke"
    };

    Ok(format!(
        "{}/model/{model}/{operation}",
        endpoint.trim_end_matches('/')
    ))
}

impl AnthropicMessagesProviderConfig for BedrockMessagesConfig {
    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        stream: bool,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        complete_bedrock_url(api_base, model, stream, env_lookup)
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
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        Ok(match api_key {
            Some(value) if !value.trim().is_empty() => value.trim().to_string(),
            Some(_) => String::new(),
            None => env_lookup(AWS_BEARER_TOKEN_BEDROCK)
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_default(),
        })
    }

    fn auth_strategy(&self) -> MessagesAuthStrategy {
        MessagesAuthStrategy::BearerOrSigV4
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[("content-type", "application/json")]
    }

    fn transform_request(
        &self,
        request: AnthropicMessagesRequest,
    ) -> CoreResult<AnthropicMessagesRequest> {
        ANTHROPIC_MESSAGES_CONFIG.transform_request(request)
    }

    fn serialize_request(&self, request: AnthropicMessagesRequest) -> CoreResult<Value> {
        serde_json::to_value(BedrockInvokeAnthropicMessagesRequest::try_from(request)?).map_err(
            |err| {
                CoreError::InvalidRequest(format!(
                    "failed to serialize Bedrock Anthropic Messages request: {err}"
                ))
            },
        )
    }

    fn transform_response(
        &self,
        model: &str,
        response: AnthropicMessagesResponse,
    ) -> CoreResult<AnthropicMessagesResponse> {
        if response.model.is_empty() {
            return Ok(AnthropicMessagesResponse {
                model: model.to_string(),
                ..response
            });
        }

        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::bedrock::constants::{
        AWS_REGION, AWS_REGION_NAME, DEFAULT_BEDROCK_REGION,
    };
    use serde_json::{Value, json};

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
    fn auth_uses_an_explicit_bearer_token_or_falls_back_to_sigv4() {
        let env = |key: &str| {
            (key == AWS_BEARER_TOKEN_BEDROCK).then(|| "env-token".to_string())
        };

        assert_eq!(
            BEDROCK_MESSAGES_CONFIG.auth_strategy(),
            MessagesAuthStrategy::BearerOrSigV4
        );
        assert_eq!(
            BEDROCK_MESSAGES_CONFIG
                .resolve_api_key(Some("request-token"), &env)
                .expect("token"),
            "request-token"
        );
        assert_eq!(
            BEDROCK_MESSAGES_CONFIG
                .resolve_api_key(None, &env)
                .expect("token"),
            "env-token"
        );
        assert!(
            BEDROCK_MESSAGES_CONFIG
                .resolve_api_key(Some(" "), &env)
                .expect("token")
                .is_empty()
        );
        assert_eq!(
            BEDROCK_MESSAGES_CONFIG.default_headers(),
            &[("content-type", "application/json")]
        );
    }

    #[test]
    fn request_serialization_allows_only_bedrock_fields() {
        let transformed = BEDROCK_MESSAGES_CONFIG
            .transform_request(request(json!({
                "model": "claude",
                "stream": true,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hello"}],
                "metadata": {"user_id": "retained"},
                "tools": [{"name": "search"}],
                "anthropic_beta": ["fine-grained-tool-streaming-2025-05-14"],
                "output_config": {"effort": "high"},
                "context_management": {"edits": []},
                "service_tier": "auto",
                "container": {"id": "container"},
                "mcp_servers": [{"name": "server"}],
                "output_format": {"type": "json_schema"},
                "speed": "fast",
                "inference_geo": "us",
                "unknown_field": true
            })))
            .expect("transform");
        let value = BEDROCK_MESSAGES_CONFIG
            .serialize_request(transformed)
            .expect("serialize");

        assert_eq!(value["anthropic_version"], BEDROCK_ANTHROPIC_VERSION);
        assert_eq!(value["metadata"], json!({"user_id": "retained"}));
        assert!(value.get("tools").is_some());
        assert!(value.get("anthropic_beta").is_some());
        assert!(value.get("output_config").is_some());
        assert!(value.get("context_management").is_some());
        for field in [
            "model",
            "stream",
            "service_tier",
            "container",
            "mcp_servers",
            "output_format",
            "speed",
            "inference_geo",
            "unknown_field",
        ] {
            assert!(value.get(field).is_none(), "{field} must be omitted");
        }
    }

    #[test]
    fn request_serialization_requires_max_tokens() {
        let err = BEDROCK_MESSAGES_CONFIG
            .serialize_request(request(json!({
                "messages": [{"role": "user", "content": "hello"}]
            })))
            .expect_err("missing max_tokens should be rejected");
        assert!(matches!(err, CoreError::InvalidRequest(_)));
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
        let response = BEDROCK_MESSAGES_CONFIG
            .transform_response("claude", response)
            .expect("response");
        assert_eq!(response.model, "claude");

        let serialized = serde_json::to_value(response).expect("serialize");
        assert!(serialized["stop_reason"].is_null());
        assert!(serialized["stop_sequence"].is_null());
    }
}
