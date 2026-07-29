use crate::error::{CoreError, CoreResult};
use crate::messages::transformation::{AnthropicMessagesProviderConfig, MessagesAuthStrategy};
use crate::messages::types::{AnthropicMessagesRequest, AnthropicMessagesResponse};
use crate::providers::anthropic::messages::transformation::non_empty;
use crate::providers::bedrock::constants::{
    AWS_REGION, AWS_REGION_NAME, BEDROCK_RUNTIME_ENDPOINT_TEMPLATE, DEFAULT_BEDROCK_REGION,
};
use serde_json::Value;

const AWS_DEFAULT_REGION: &str = "AWS_DEFAULT_REGION";
const API_BASE_SCHEME: &str = "https://";
const MODEL_PATH_PREFIX: &str = "/model/";
const INVOKE_PATH: &str = "/invoke";
const STREAM_PATH: &str = "/invoke-with-response-stream";
const ANTHROPIC_VERSION_FIELD: &str = "anthropic_version";
const ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";
const UNSUPPORTED_FIELDS: &[&str] = &[
    "metadata",
    "service_tier",
    "container",
    "mcp_servers",
    "context_management",
    "output_format",
    "output_config",
    "speed",
    "inference_geo",
];

pub struct BedrockMessagesConfig;

pub const BEDROCK_MESSAGES_CONFIG: BedrockMessagesConfig = BedrockMessagesConfig;

fn resolve_region(api_base: Option<&str>, env_lookup: &dyn Fn(&str) -> Option<String>) -> String {
    api_base
        .and_then(bedrock_region_from_api_base)
        .or_else(|| env_lookup(AWS_REGION_NAME))
        .or_else(|| env_lookup(AWS_REGION))
        .or_else(|| env_lookup(AWS_DEFAULT_REGION))
        .unwrap_or_else(|| DEFAULT_BEDROCK_REGION.to_string())
}

fn bedrock_region_from_api_base(api_base: &str) -> Option<String> {
    let host = api_base
        .trim()
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .split('/')
        .next()?
        .split(':')
        .next()?;
    let region = host
        .strip_prefix("bedrock-runtime.")?
        .strip_suffix(".amazonaws.com")?;
    (!region.is_empty()).then(|| region.to_string())
}

fn encode_path_segment(value: &str) -> String {
    value.bytes().fold(String::new(), |mut encoded, byte| {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            encoded.push(byte as char);
        } else {
            encoded.push('%');
            encoded.push_str(&format!("{byte:02X}"));
        }
        encoded
    })
}

fn endpoint_base(api_base: Option<&str>, env_lookup: &dyn Fn(&str) -> Option<String>) -> String {
    non_empty(api_base)
        .map(str::to_string)
        .unwrap_or_else(|| {
            BEDROCK_RUNTIME_ENDPOINT_TEMPLATE.replace("{region}", &resolve_region(None, env_lookup))
        })
        .trim_end_matches('/')
        .to_string()
}

pub fn complete_bedrock_url(
    api_base: Option<&str>,
    model: &str,
    stream: bool,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    let model = non_empty(Some(model))
        .ok_or_else(|| CoreError::InvalidRequest("Bedrock model cannot be empty".to_string()))?;
    let suffix = if stream { STREAM_PATH } else { INVOKE_PATH };
    let base = endpoint_base(api_base, env_lookup);
    let base = if base.starts_with(API_BASE_SCHEME) || base.starts_with("http://") {
        base
    } else {
        format!("{API_BASE_SCHEME}{base}")
    };
    Ok(format!(
        "{base}{MODEL_PATH_PREFIX}{}{suffix}",
        encode_path_segment(model)
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
        api_base: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Option<String> {
        Some(resolve_region(api_base, env_lookup))
    }

    fn resolve_api_key(
        &self,
        _api_key: Option<&str>,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        Ok(String::new())
    }

    fn auth_strategy(&self) -> MessagesAuthStrategy {
        MessagesAuthStrategy::AwsSigV4
    }

    fn transform_request(
        &self,
        mut request: AnthropicMessagesRequest,
    ) -> CoreResult<AnthropicMessagesRequest> {
        request.model.clear();
        request.stream = None;
        request.metadata = None;
        request.service_tier = None;
        request.container = None;
        request.mcp_servers = None;
        request.context_management = None;
        request.output_format = None;
        request.output_config = None;
        request.speed = None;
        request.inference_geo = None;
        request
            .extra
            .retain(|key, _| !UNSUPPORTED_FIELDS.contains(&key.as_str()));
        request.extra.insert(
            ANTHROPIC_VERSION_FIELD.to_string(),
            Value::String(ANTHROPIC_VERSION.to_string()),
        );
        Ok(request)
    }

    fn transform_response(
        &self,
        model: &str,
        mut response: AnthropicMessagesResponse,
    ) -> CoreResult<AnthropicMessagesResponse> {
        if response.model.trim().is_empty() {
            response.model = model.to_string();
        }
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
