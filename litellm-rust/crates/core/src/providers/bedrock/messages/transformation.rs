use crate::error::{CoreError, CoreResult};
use crate::messages::transformation::{
    AnthropicMessagesProviderConfig, MessagesAuthKind, MessagesStreaming,
};
use crate::messages::types::AnthropicMessagesRequest;
use crate::providers::bedrock::common_utils::{
    bedrock_model_id_and_region, resolve_bedrock_region,
};
use crate::providers::bedrock::constants::BEDROCK_RUNTIME_ENDPOINT_TEMPLATE;
use serde_json::{Map, Value};

const BEDROCK_ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";
const BEDROCK_MESSAGES_SUFFIX: &str = "/invoke";
const BEDROCK_STREAM_SUFFIX: &str = "/invoke-with-response-stream";
const CACHE_TTL_5M: &str = "5m";
const CACHE_TTL_1H: &str = "1h";
const CONTEXT_EDIT_COMPACT: &str = "compact_20260112";
const CONTEXT_EDIT_CLEAR_TOOLS: &str = "clear_tool_uses_20250919";
const BETA_COMPACT: &str = "compact-2026-01-12";
const BETA_CONTEXT: &str = "context-management-2025-06-27";
const ALLOWED_FIELDS: &[&str] = &[
    "anthropic_version",
    "max_tokens",
    "messages",
    "anthropic_beta",
    "system",
    "stop_sequences",
    "temperature",
    "top_p",
    "top_k",
    "tools",
    "tool_choice",
    "thinking",
    "metadata",
    "output_config",
    "context_management",
];

pub struct BedrockAnthropicMessagesConfig;

pub const BEDROCK_ANTHROPIC_MESSAGES_CONFIG: BedrockAnthropicMessagesConfig =
    BedrockAnthropicMessagesConfig;

pub fn complete_bedrock_url(
    api_base: Option<&str>,
    model: &str,
    stream: bool,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    let (model_id, model_region) = bedrock_model_id_and_region(model);
    let region = resolve_bedrock_region(model_region.as_deref(), &Map::new(), env_lookup);
    let endpoint = api_base
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| BEDROCK_RUNTIME_ENDPOINT_TEMPLATE.replace("{region}", &region));
    let suffix = if stream {
        BEDROCK_STREAM_SUFFIX
    } else {
        BEDROCK_MESSAGES_SUFFIX
    };
    format!(
        "{}/model/{model_id}{suffix}",
        endpoint.trim_end_matches('/')
    )
}

fn is_claude_4_5(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    ["sonnet", "haiku", "opus"].iter().any(|family| {
        let prefix = format!("claude-{family}");
        model.find(&prefix).is_some_and(|offset| {
            let suffix = &model[offset + prefix.len()..];
            matches!(suffix, "-4-5" | "-4.5" | ".4-5" | ".4.5")
                || suffix.contains("-4-5-")
                || suffix.contains("-4.5-")
                || suffix.contains(".4-5-")
                || suffix.contains(".4.5-")
        })
    })
}

fn sanitize_blocks(value: &mut Value, keep_ttl: bool) {
    match value {
        Value::Array(values) => values
            .iter_mut()
            .for_each(|value| sanitize_blocks(value, keep_ttl)),
        Value::Object(object) => {
            if let Some(cache_control) = object
                .get_mut("cache_control")
                .and_then(Value::as_object_mut)
            {
                cache_control.remove("scope");
                if !keep_ttl
                    || !matches!(
                        cache_control.get("ttl").and_then(Value::as_str),
                        Some(CACHE_TTL_5M | CACHE_TTL_1H)
                    )
                {
                    cache_control.remove("ttl");
                }
            }
            object
                .values_mut()
                .for_each(|value| sanitize_blocks(value, keep_ttl));
        }
        _ => {}
    }
}

fn sanitize_tools(value: &mut Value) {
    let Some(tools) = value.as_array_mut() else {
        return;
    };
    tools.iter_mut().for_each(|tool| {
        if let Some(object) = tool.as_object_mut() {
            object.remove("custom");
        }
    });
}

fn filter_context_management(request: &mut Map<String, Value>) {
    let Some(edits) = request
        .get_mut("context_management")
        .and_then(Value::as_object_mut)
        .and_then(|context| context.get_mut("edits"))
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    edits.retain(|edit| {
        edit.as_object()
            .and_then(|object| object.get("type"))
            .and_then(Value::as_str)
            .is_some_and(|kind| matches!(kind, CONTEXT_EDIT_COMPACT | CONTEXT_EDIT_CLEAR_TOOLS))
    });
    if edits.is_empty() {
        request.remove("context_management");
        return;
    }
    let allowed_edits = edits.clone();
    let betas = request
        .entry("anthropic_beta")
        .or_insert_with(|| Value::Array(Vec::new()));
    let Some(betas) = betas.as_array_mut() else {
        return;
    };
    let has_compact = allowed_edits.iter().any(|edit| {
        edit.get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| kind == CONTEXT_EDIT_COMPACT)
    });
    let has_context = allowed_edits.iter().any(|edit| {
        edit.get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| kind == CONTEXT_EDIT_CLEAR_TOOLS)
    });
    if has_compact && !betas.iter().any(|beta| beta.as_str() == Some(BETA_COMPACT)) {
        betas.push(Value::String(BETA_COMPACT.to_string()));
    }
    if has_context && !betas.iter().any(|beta| beta.as_str() == Some(BETA_CONTEXT)) {
        betas.push(Value::String(BETA_CONTEXT.to_string()));
    }
}

pub fn transform_bedrock_request(
    model: &str,
    request: AnthropicMessagesRequest,
) -> CoreResult<Value> {
    let mut value = serde_json::to_value(request)
        .map_err(|error| CoreError::InvalidRequest(format!("invalid messages request: {error}")))?;
    let object = value.as_object_mut().ok_or_else(|| {
        CoreError::InvalidRequest("messages request must be an object".to_string())
    })?;
    object.remove("model");
    object.remove("stream");
    object
        .entry("anthropic_version")
        .or_insert_with(|| Value::String(BEDROCK_ANTHROPIC_VERSION.to_string()));
    let keep_ttl = is_claude_4_5(model);
    for key in ["system", "messages", "tools"] {
        if let Some(value) = object.get_mut(key) {
            sanitize_blocks(value, keep_ttl);
        }
    }
    if let Some(tools) = object.get_mut("tools") {
        sanitize_tools(tools);
    }
    filter_context_management(object);
    object.retain(|key, _| ALLOWED_FIELDS.contains(&key.as_str()));
    Ok(value)
}

impl AnthropicMessagesProviderConfig for BedrockAnthropicMessagesConfig {
    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        stream: bool,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        Ok(complete_bedrock_url(api_base, model, stream, env_lookup))
    }

    fn auth_kind(
        &self,
        model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<MessagesAuthKind> {
        let (_, model_region) = bedrock_model_id_and_region(model);
        Ok(MessagesAuthKind::AwsSigV4 {
            region: resolve_bedrock_region(model_region.as_deref(), &Map::new(), env_lookup),
        })
    }

    fn streaming(&self) -> MessagesStreaming {
        MessagesStreaming::BedrockEventStream
    }

    fn upstream_body(&self, request: AnthropicMessagesRequest) -> CoreResult<Value> {
        let model = request.model.clone();
        transform_bedrock_request(&model, request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(value: Value) -> AnthropicMessagesRequest {
        serde_json::from_value(value).expect("request")
    }

    #[test]
    fn url_selects_invoke_endpoint_and_region_precedence() {
        let env = |key: &str| match key {
            "AWS_REGION_NAME" => Some("eu-west-1".to_string()),
            "AWS_REGION" => Some("ap-southeast-1".to_string()),
            _ => None,
        };
        assert_eq!(
            complete_bedrock_url(None, "bedrock/us-west-2/claude-test", false, &env),
            "https://bedrock-runtime.us-west-2.amazonaws.com/model/claude-test/invoke"
        );
        assert_eq!(
            complete_bedrock_url(Some("http://localhost:9000/"), "claude-test", true, &env),
            "http://localhost:9000/model/claude-test/invoke-with-response-stream"
        );
        assert_eq!(
            complete_bedrock_url(
                None,
                "arn:aws:bedrock:ap-south-1:123:model/foo",
                false,
                &env
            ),
            "https://bedrock-runtime.ap-south-1.amazonaws.com/model/arn:aws:bedrock:ap-south-1:123:model/foo/invoke"
        );
    }

    #[test]
    fn transform_filters_and_sanitizes_bedrock_request() {
        let input = request(json!({
            "model": "claude-sonnet-4-5-20250929",
            "stream": true,
            "messages": [{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral","scope":"request","ttl":"1h"}}]}],
            "tools": [{"name":"lookup","custom":{"defer_loading":true}}],
            "context_management": {"edits":[{"type":"unsupported"},{"type":"compact_20260112"}]},
            "service_tier": "auto",
            "unknown": true
        }));
        let transformed =
            transform_bedrock_request("claude-sonnet-4-5-20250929", input).expect("transform");
        let output = serde_json::to_value(transformed).expect("json");
        assert_eq!(output["anthropic_version"], BEDROCK_ANTHROPIC_VERSION);
        assert_eq!(
            output["messages"][0]["content"][0]["cache_control"]["ttl"],
            "1h"
        );
        assert!(output["tools"][0].get("custom").is_none());
        assert_eq!(
            output["context_management"]["edits"][0]["type"],
            CONTEXT_EDIT_COMPACT
        );
        assert_eq!(output["anthropic_beta"][0], BETA_COMPACT);
        assert!(output.get("service_tier").is_none());
        assert!(output.get("unknown").is_none());
        assert!(output.get("model").is_none());
        assert!(output.get("stream").is_none());
    }
}
