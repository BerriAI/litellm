//! Python's `AmazonConverseConfig` for `/chat/completions`: `get_optional_params` and
//! `_transform_request` on the way in, `_transform_response` on the way out.

use litellm_auth::AwsParams;
use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_auth_aws::{
    AwsCredentialSource, bedrock_model_id_and_region,
    constants::{AWS_BEARER_TOKEN_BEDROCK, BEDROCK_RUNTIME_ENDPOINT_TEMPLATE, BEDROCK_SERVICE},
    resolve_bedrock_region,
};
use litellm_core_utils::core_helpers::{finish_reason_for, unix_now, usage_from_parts};
use litellm_llms_types::{
    formats::chat_completions::{
        ChatCompletionThinkingBlock, ChatCompletionToolCallChunk,
        ChatCompletionToolCallFunctionChunk, ChatCompletionsChoice, ChatCompletionsChoiceMessage,
        ChatCompletionsResponse, ChatCompletionsUsage, ChatMessage,
    },
    providers::bedrock::{CONVERSE_PATH, ConverseUsage},
};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::{
    Error, ErrorDetail,
    anthropic::chat::{messages::native_redacted_thinking_data, params::apply_sampling_param},
    base_llm::{
        auth::AuthScheme,
        chat::{
            media::{data_url, remote_urls},
            params::{
                ChatRequestContext, RESPONSE_FORMAT_TOOL_NAME, python_dumps, sort_params, truthy,
            },
            streaming::{ChatStream, StreamShape},
            transformation::{
                BaseConfig, Headers, ProviderChatRequestData, ProviderChatResponseData,
                Unsupported, ValidatedEnvironment,
            },
        },
    },
    bedrock::chat::{
        converse_messages::{
            converse_messages, converse_tools, guard_trailing_user_messages,
            neutralize_orphaned_tool_blocks, transform_system_message,
        },
        converse_params::{
            BedrockModel, MappedParams, map_openai_params, output_config_for_response_format,
            supported_params, validate_request_metadata,
        },
        invoke_handler::converse_chat_stream,
    },
};

/// The action a streaming Converse call goes to, in place of [`CONVERSE_PATH`].
const CONVERSE_STREAM_PATH: &str = "/converse-stream";

const AWS_BEDROCK_RUNTIME_ENDPOINT: &str = "aws_bedrock_runtime_endpoint";

/// AWS call configuration a host passes down: consumed for signing and endpoint
/// resolution, never serialized into the Converse body.
const CONFIG_PARAMS: &[&str] = &[
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_region_name",
    "aws_session_name",
    "aws_profile_name",
    "aws_role_name",
    "aws_web_identity_token",
    "aws_sts_endpoint",
    "aws_external_id",
    AWS_BEDROCK_RUNTIME_ENDPOINT,
];

/// `AmazonConverseConfig.__annotations__` plus `top_k`, the tool params and the config
/// blocks: what stays out of `additionalModelRequestFields`.
const INFERENCE_PARAMS: [&str; 6] = [
    "maxTokens",
    "stopSequences",
    "temperature",
    "topP",
    "topK",
    "top_k",
];
const TOOL_PARAMS: [&str; 2] = ["tools", "tool_choice"];
const CONFIG_BLOCKS: [&str; 3] = ["guardrailConfig", "performanceConfig", "serviceTier"];
/// `filter_internal_params`.
const INTERNAL_PARAMS: [&str; 3] = [
    "skip_mcp_handler",
    "mcp_handler_context",
    "_skip_mcp_handler",
];

const FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";
const EFFORT_BETA: &str = "effort-2025-11-24";
const COMPACT_BETA: &str = "compact-2026-01-12";
const COMPACT_EDIT_TYPE: &str = "compact_20260112";

enum ConverseStopReason {
    Value(String),
    Unknown(String),
}

impl ConverseStopReason {
    fn parse(value: Option<String>) -> Option<Self> {
        value.map(|reason| match reason.as_str() {
            "end_turn"
            | "max_tokens"
            | "stop_sequence"
            | "content_filtered"
            | "guardrail_intervened" => Self::Value(reason),
            _ => Self::Unknown(reason),
        })
    }

    fn as_str(&self) -> &str {
        match self {
            Self::Value(value) | Self::Unknown(value) => value,
        }
    }
}

#[derive(Deserialize)]
struct ResponseProjection {
    output: OutputProjection,
    #[serde(default)]
    usage: ConverseUsage,
    #[serde(rename = "stopReason")]
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct OutputProjection {
    message: Option<MessageProjection>,
}

#[derive(Deserialize)]
struct MessageProjection {
    content: Vec<ContentProjection>,
}

#[derive(Deserialize)]
struct ContentProjection {
    #[serde(default)]
    text: Option<String>,
    #[serde(default, rename = "toolUse")]
    tool_use: Option<Value>,
    #[serde(default, rename = "reasoningContent")]
    reasoning_content: Option<Value>,
    #[serde(default, rename = "citationsContent")]
    citations_content: Option<Value>,
}

pub struct AmazonConverseConfig;

pub const BEDROCK_CHAT_COMPLETIONS_CONFIG: AmazonConverseConfig = AmazonConverseConfig;

/// `parse_content_for_reasoning`: a leading `<think>` block split from the answer.
fn parse_content_for_reasoning(text: &str) -> (Option<String>, String) {
    for tag in ["think", "thinking", "budget:thinking"] {
        let Some(rest) = text.strip_prefix(&format!("<{tag}>")) else {
            continue;
        };
        let closes = ["think", "thinking", "budget:thinking"]
            .iter()
            .filter_map(|close| {
                let marker = format!("</{close}>");
                rest.find(&marker).map(|at| (at, marker.len()))
            })
            .min_by_key(|(at, _)| *at);
        if let Some((at, len)) = closes {
            return (Some(rest[..at].to_owned()), rest[at + len..].to_owned());
        }
    }
    (None, text.to_owned())
}

/// `_unwrap_bedrock_properties`: a lone `properties` key gives up its value.
pub fn unwrap_bedrock_properties(text: &str) -> String {
    match serde_json::from_str::<Value>(text) {
        Ok(Value::Object(object)) if object.len() == 1 && object.contains_key("properties") => {
            python_dumps(&object["properties"])
        }
        _ => text.to_owned(),
    }
}

/// `eager_input_streaming_flag` over a chat tool.
fn eager_input_streaming(tool: &Value) -> bool {
    tool.get("eager_input_streaming") == Some(&Value::Bool(true))
        || tool
            .get("function")
            .and_then(|function| function.get("eager_input_streaming"))
            == Some(&Value::Bool(true))
}

/// `_merge_parallel_tool_use_config`.
fn merge_parallel_tool_use_config(additional: &mut Map<String, Value>, config: Map<String, Value>) {
    for (key, value) in config {
        let merged = match (additional.get(&key), &value) {
            (Some(Value::Object(existing)), Value::Object(incoming)) => {
                let mut merged = incoming.clone();
                merged.extend(existing.iter().map(|(k, v)| (k.clone(), v.clone())));
                merged.extend(
                    incoming
                        .iter()
                        .filter(|(k, _)| k.as_str() != "type")
                        .map(|(k, v)| (k.clone(), v.clone())),
                );
                Value::Object(merged)
            }
            _ => value,
        };
        additional.insert(key, merged);
    }
}

impl AmazonConverseConfig {
    fn build_request(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        optional_params: Map<String, Value>,
        context: &ChatRequestContext<'_>,
    ) -> Result<ProviderChatRequestData, Error> {
        let bedrock = BedrockModel::new(model, context);
        let sorted = sort_params(
            optional_params,
            CONFIG_PARAMS,
            context,
            |_| supported_params(&bedrock),
            "bedrock",
            model,
        )?;
        let drop = sorted.drop;
        let MappedParams {
            mut params,
            json_mode,
            response_format_schema,
        } = map_openai_params(&sorted.openai, &bedrock, drop)?;
        for name in &sorted.allowed {
            if let Some(value) = sorted.openai.get(name)
                && !params.contains_key(name)
            {
                params.insert(name.clone(), value.clone());
            }
        }
        params.extend(sorted.provider);

        let messages: Vec<Value> = messages
            .iter()
            .map(|message| {
                serde_json::to_value(message).map(|mut value| {
                    if value.get("role").and_then(Value::as_str) == Some("developer") {
                        value["role"] = json!("system");
                    }
                    value
                })
            })
            .collect::<Result<_, _>>()
            .map_err(|error| Error::InvalidRequest(ErrorDetail::invalid("messages", error)))?;
        let (messages, system) = transform_system_message(messages, &bedrock);
        let guardrail = params.contains_key("guardrailConfig");
        let has_tools = params.get("tools").is_some_and(truthy);
        let mut messages = neutralize_orphaned_tool_blocks(messages, has_tools, guardrail);
        if guardrail {
            messages = guard_trailing_user_messages(messages);
        }

        // `maybe_drop_disabled_thinking`.
        if params
            .get("thinking")
            .and_then(|thinking| thinking.get("type"))
            .and_then(Value::as_str)
            == Some("disabled")
            && bedrock.caps.supports("thinking_always_on")?
        {
            params.remove("thinking");
        }

        // `_prepare_request_params`.
        let normalized = params
            .remove("_output_config_normalized")
            .is_some_and(|flag| truthy(&flag));
        for internal in ["json_mode", "stream", "fake_stream"] {
            params.remove(internal);
        }
        let mut anthropic_output_config = match params.remove("output_config") {
            Some(Value::Object(config)) => Some(config),
            _ => None,
        };
        let mut output_config_format = None;
        if let Some(config) = &mut anthropic_output_config {
            if let Some(Value::Object(format)) = config.remove("format") {
                output_config_format = Some(format);
            }
            if config.is_empty() {
                anthropic_output_config = None;
            }
        }
        let request_metadata = params.remove("requestMetadata").filter(|v| !v.is_null());
        if let Some(metadata) = &request_metadata {
            validate_request_metadata(metadata)?;
        }
        let mut output_config = params.remove("outputConfig").filter(|v| !v.is_null());
        if output_config.is_none()
            && let Some(format) = &output_config_format
            && format.get("type").and_then(Value::as_str) == Some("json_schema")
            && bedrock.is_anthropic()
            && bedrock.supports("supports_native_structured_output")?
        {
            output_config = Some(output_config_for_response_format(
                format.get("schema"),
                format.get("name"),
                format.get("description"),
            ));
        }
        let (mut inference, mut additional): (Map<String, Value>, Map<String, Value>) =
            params.into_iter().partition(|(key, _)| {
                INFERENCE_PARAMS.contains(&key.as_str())
                    || TOOL_PARAMS.contains(&key.as_str())
                    || CONFIG_BLOCKS.contains(&key.as_str())
            });
        if let Some(Value::Object(config)) = additional.remove("_parallel_tool_use_config")
            && bedrock.supports_parallel_tool_use_config()?
        {
            merge_parallel_tool_use_config(&mut additional, config);
        }
        additional.remove("parallel_tool_calls");
        additional.remove("client_metadata");
        let top_k = inference
            .remove("topK")
            .or_else(|| inference.remove("top_k"))
            .filter(|top_k| !top_k.is_null());
        if let Some(top_k) = top_k {
            if bedrock.is_anthropic() {
                apply_sampling_param(
                    &mut additional,
                    &bedrock.caps,
                    drop,
                    "top_k",
                    &top_k,
                    "top_k",
                )?;
            } else if bedrock.base.starts_with("amazon.nova") {
                additional.insert("inferenceConfig".into(), json!({"topK": top_k}));
            }
        }
        for internal in INTERNAL_PARAMS {
            additional.remove(internal);
        }
        if let Some(mut config) = anthropic_output_config {
            if model.contains(":application-inference-profile/") {
                additional.insert("output_config".into(), Value::Object(config));
            } else if bedrock.is_anthropic()
                && !(drop.drops() && !bedrock.supports_effort_param()?)
            {
                if !normalized {
                    bedrock.normalize_output_config_effort(&mut config);
                }
                if let Some(effort) = config.get("effort").filter(|effort| !effort.is_null()) {
                    bedrock.validate_adaptive_effort(effort)?;
                }
                additional.insert("output_config".into(), Value::Object(config));
            }
        }

        // `_process_tools_and_beta`.
        let tools = match inference.remove("tools") {
            Some(Value::Array(tools)) => tools,
            _ => Vec::new(),
        };
        let (tool_blocks, tool_name_reverse_map) = converse_tools(&tools, &bedrock)?;
        let mut betas: Vec<&str> = Vec::new();
        if bedrock.is_anthropic() {
            if tools.iter().any(eager_input_streaming) {
                betas.push(FINE_GRAINED_TOOL_STREAMING_BETA);
            }
            if additional
                .get("output_config")
                .and_then(|config| config.get("effort"))
                .is_some_and(|effort| !effort.is_null())
                && !bedrock.caps.is_adaptive()?
            {
                betas.push(EFFORT_BETA);
            }
            let compact_edits: Option<(Map<String, Value>, Vec<Value>)> = match additional
                .get("context_management")
            {
                Some(Value::Object(management)) => match management.get("edits") {
                    Some(Value::Array(edits)) => Some((
                        management.clone(),
                        edits
                            .iter()
                            .filter(|edit| {
                                edit.get("type").and_then(Value::as_str) == Some(COMPACT_EDIT_TYPE)
                            })
                            .cloned()
                            .collect(),
                    )),
                    _ => None,
                },
                _ => None,
            };
            match compact_edits {
                Some((mut management, edits)) if !edits.is_empty() => {
                    betas.push(COMPACT_BETA);
                    management.insert("edits".into(), Value::Array(edits));
                    additional.insert("context_management".into(), Value::Object(management));
                }
                _ => {
                    additional.remove("context_management");
                }
            }
            if !betas.is_empty() {
                additional.insert("anthropic_beta".into(), json!(betas));
            }
        }
        additional.remove("cache_control_injection_points");

        let mut tool_config = None;
        if !tool_blocks.is_empty() {
            let mut config = Map::from_iter([("tools".to_owned(), Value::Array(tool_blocks))]);
            if let Some(choice) = inference.remove("tool_choice") {
                config.insert("toolChoice".into(), choice);
                if let Some(Value::Object(choice)) = additional.get_mut("tool_choice") {
                    choice.remove("type");
                }
            }
            tool_config = Some(Value::Object(config));
        }
        let config_blocks: Vec<(&str, Option<Value>)> = CONFIG_BLOCKS
            .iter()
            .map(|name| (*name, inference.remove(*name)))
            .collect();

        let mut body = Map::from_iter([(
            "messages".to_owned(),
            Value::Array(converse_messages(&messages, &bedrock)?),
        )]);
        body.insert("inferenceConfig".into(), Value::Object(inference));
        if !additional.is_empty() {
            let thinking = additional.contains_key("thinking");
            body.insert(
                "additionalModelRequestFields".into(),
                Value::Object(additional),
            );
            if thinking {
                body.insert(
                    "additionalModelResponseFieldPaths".into(),
                    json!(["/usage/output_tokens_details"]),
                );
            }
        }
        if !system.is_empty() {
            body.insert("system".into(), Value::Array(system));
        }
        for (name, value) in config_blocks {
            if let Some(value) = value.filter(|value| !value.is_null()) {
                body.insert(name.into(), value);
            }
        }
        if let Some(tool_config) = tool_config {
            body.insert("toolConfig".into(), tool_config);
        }
        if let Some(metadata) = request_metadata {
            body.insert("requestMetadata".into(), metadata);
        }
        if let Some(output_config) = output_config {
            body.insert("outputConfig".into(), output_config);
        }
        Ok(ProviderChatRequestData {
            body: Value::Object(body),
            stream_shape: StreamShape {
                json_mode,
                tool_name_reverse_map,
                response_format_schema,
                ..StreamShape::default()
            },
            headers: Vec::new(),
        })
    }
}

/// `_transform_thinking_blocks`.
fn thinking_blocks(blocks: &[Value]) -> Vec<ChatCompletionThinkingBlock> {
    blocks
        .iter()
        .filter_map(|block| {
            if let Some(text) = block.get("reasoningText") {
                return Some(ChatCompletionThinkingBlock::Thinking {
                    thinking: text.get("text").and_then(Value::as_str).map(str::to_owned),
                    signature: text
                        .get("signature")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    cache_control: None,
                    index: None,
                });
            }
            block
                .get("redactedContent")
                .and_then(Value::as_str)
                .map(|data| ChatCompletionThinkingBlock::RedactedThinking {
                    data: Some(native_redacted_thinking_data(data)),
                    cache_control: None,
                    index: None,
                })
        })
        .collect()
}

fn converse_response(
    model: &str,
    response: ResponseProjection,
    shape: &StreamShape,
) -> ChatCompletionsResponse {
    let mut content = String::new();
    let mut tool_calls: Vec<ChatCompletionToolCallChunk> = Vec::new();
    let mut reasoning_blocks: Option<Vec<Value>> = None;
    let mut citations: Option<Vec<Value>> = None;
    for (index, block) in response
        .output
        .message
        .map(|message| message.content)
        .unwrap_or_default()
        .into_iter()
        .enumerate()
    {
        let mut extracted_reasoning = None;
        if let Some(text) = &block.text {
            let (reasoning, answer) = parse_content_for_reasoning(text);
            extracted_reasoning = reasoning;
            content.push_str(&answer);
        }
        if let Some(tool) = &block.tool_use {
            let name = tool.get("name").and_then(Value::as_str).unwrap_or_default();
            tool_calls.push(ChatCompletionToolCallChunk {
                id: tool
                    .get("toolUseId")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                tool_type: "function".into(),
                function: ChatCompletionToolCallFunctionChunk {
                    name: Some(
                        shape
                            .tool_name_reverse_map
                            .get(name)
                            .cloned()
                            .unwrap_or_else(|| name.to_owned()),
                    ),
                    arguments: python_dumps(tool.get("input").unwrap_or(&Value::Null)),
                    provider_specific_fields: None,
                },
                index: index as i64,
                provider_specific_fields: None,
            });
        }
        if let Some(reasoning) = extracted_reasoning {
            reasoning_blocks
                .get_or_insert_with(Vec::new)
                .push(json!({"reasoningText": {"text": reasoning}}));
        }
        if let Some(reasoning) = block.reasoning_content {
            reasoning_blocks
                .get_or_insert_with(Vec::new)
                .push(reasoning);
        }
        if let Some(citation) = block.citations_content {
            citations.get_or_insert_with(Vec::new).push(citation);
        }
    }
    let mut provider_fields = Map::new();
    if let Some(blocks) = &reasoning_blocks {
        provider_fields.insert(
            "reasoningContentBlocks".into(),
            Value::Array(blocks.clone()),
        );
    }
    if let Some(citations) = citations {
        provider_fields.insert("citationsContent".into(), Value::Array(citations));
    }
    let reasoning_content = reasoning_blocks.as_ref().map(|blocks| {
        blocks
            .iter()
            .filter_map(|block| block.get("reasoningText")?.get("text")?.as_str())
            .collect::<String>()
    });
    let thinking = reasoning_blocks.as_deref().map(thinking_blocks);

    // `_filter_json_mode_tools`.
    let had_tools = !tool_calls.is_empty();
    let json_indices: Vec<usize> = match shape.json_mode {
        true => tool_calls
            .iter()
            .enumerate()
            .filter(|(_, call)| call.function.name.as_deref() == Some(RESPONSE_FORMAT_TOOL_NAME))
            .map(|(index, _)| index)
            .collect(),
        false => Vec::new(),
    };
    let filtered: Vec<ChatCompletionToolCallChunk> = if json_indices.is_empty() {
        tool_calls
    } else {
        let answer = unwrap_bedrock_properties(&tool_calls[json_indices[0]].function.arguments);
        if json_indices.len() == tool_calls.len() {
            content = answer;
            Vec::new()
        } else {
            content = match content.is_empty() {
                true => answer,
                false => format!("{content}{answer}"),
            };
            tool_calls
                .into_iter()
                .enumerate()
                .filter(|(index, _)| !json_indices.contains(index))
                .map(|(_, call)| call)
                .collect()
        }
    };
    let mapped = finish_reason_for(
        ConverseStopReason::parse(response.stop_reason)
            .as_ref()
            .map(ConverseStopReason::as_str)
            .unwrap_or(""),
    )
    .to_string();
    let finish_reason =
        match shape.json_mode && filtered.is_empty() && had_tools && mapped == "tool_calls" {
            true => "stop".to_owned(),
            false => mapped,
        };

    let computed = usage_from_parts(
        response.usage.input_tokens,
        response.usage.output_tokens,
        response.usage.cache_read_input_tokens,
        response.usage.cache_write_input_tokens,
    );
    // Converse reports `totalTokens` and Python passes it straight through. Python does a
    // bare `usage["totalTokens"]` lookup, so a body without the key raises there; the
    // computed total stands in rather than failing the call.
    let usage = ChatCompletionsUsage {
        total_tokens: response.usage.total_tokens.unwrap_or(computed.total_tokens),
        ..computed
    };
    ChatCompletionsResponse {
        created: unix_now(),
        // Converse echoes no model id, so Python reports the requested one.
        model: model.to_string(),
        choices: vec![ChatCompletionsChoice {
            index: 0,
            message: ChatCompletionsChoiceMessage {
                role: "assistant".to_string(),
                // Converse assigns the joined string unconditionally, so an empty answer
                // is `""` here and not `None` as it is on Anthropic.
                content: Some(content),
                tool_calls: (!filtered.is_empty()).then_some(filtered),
                reasoning_content,
                thinking_blocks: thinking,
                provider_specific_fields: (!provider_fields.is_empty()).then_some(provider_fields),
                annotations: None,
            },
            finish_reason,
            logprobs: None,
        }],
        usage,
        extra: Map::new(),
    }
}

impl BaseConfig for AmazonConverseConfig {
    fn secret_names(&self) -> Vec<&'static str> {
        litellm_auth::AwsParams::secret_names()
            .iter()
            .copied()
            .chain([AWS_BEARER_TOKEN_BEDROCK])
            .collect()
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let (model_id, model_region) = bedrock_model_id_and_region(model);
        let region = resolve_bedrock_region(
            model_region.as_deref(),
            &AwsParams::from_optional_params(optional_params),
            env_lookup,
        );
        let endpoint = optional_params
            .get(AWS_BEDROCK_RUNTIME_ENDPOINT)
            .and_then(Value::as_str)
            .or(api_base)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| BEDROCK_RUNTIME_ENDPOINT_TEMPLATE.replace("{region}", &region));
        let endpoint = endpoint.trim_end_matches('/');
        // A host that already built the full Converse URL (LiteLLM's Python
        // path encodes the model id itself) passes it through untouched, the
        // way the Anthropic config leaves a complete `/v1/messages` URL alone.
        if endpoint.ends_with(CONVERSE_PATH) {
            return Ok(endpoint.to_string());
        }
        Ok(format!("{endpoint}/model/{model_id}{CONVERSE_PATH}"))
    }

    fn transform_request(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        optional_params: Map<String, Value>,
        context: &ChatRequestContext<'_>,
    ) -> Result<ProviderChatRequestData, Error> {
        self.build_request(model, messages, optional_params, context)
    }

    fn transform_response(
        &self,
        model: &str,
        response: ProviderChatResponseData,
        shape: &StreamShape,
    ) -> Result<ChatCompletionsResponse, Error> {
        let body = response.body;
        if !body.is_object() {
            return Err(Error::InvalidResponse(
                "converse response is not an object".into(),
            ));
        }
        for field in ["output", "usage"] {
            if body.get(field).is_none() {
                return Err(Error::InvalidResponse(
                    format!("invalid Converse response: missing field `{field}`").into(),
                ));
            }
        }
        let response: ResponseProjection = serde_json::from_value(body).map_err(|error| {
            Error::InvalidResponse(format!("invalid Converse response: {error}").into())
        })?;
        Ok(converse_response(model, response, shape))
    }

    /// Python's `BedrockImageProcessor` downloads every image or file URL a message names.
    fn remote_media(&self, messages: &[ChatMessage]) -> Vec<String> {
        remote_urls(messages, &["user", "assistant", "tool"], true, |url| {
            data_url(url).is_none()
                && !url.contains("base64")
                && (url.contains("http://") || url.contains("https://"))
        })
    }

    /// Python reads `api_key` as the Bedrock bearer token and consults the env only when
    /// the caller passed none, so a caller-supplied empty key falls through to SigV4
    /// without reaching for the environment. An all-whitespace token stays a bearer token
    /// here because Python sends it too: treating it as absent would sign as the host
    /// principal instead, which is the identity swap this branch exists to prevent.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let bearer = match api_key {
            Some(key) => Some(key.to_string()),
            None => env_lookup(AWS_BEARER_TOKEN_BEDROCK),
        }
        .filter(|token| !token.is_empty());
        if let Some(token) = bearer {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Credential {
                    placement: CredentialPlacement::Bearer,
                    secret: SecretValue::new(token),
                },
            });
        }
        let (_, model_region) = bedrock_model_id_and_region(model);
        let params = AwsParams::from_optional_params(optional_params);
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::AwsSigV4 {
                region: resolve_bedrock_region(model_region.as_deref(), &params, env_lookup),
                service: BEDROCK_SERVICE,
                credentials: Box::new(AwsCredentialSource::from_params(&params, env_lookup)),
            },
        })
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[("Content-Type", "application/json")]
    }

    fn model_response_iterator(&self, shape: StreamShape) -> Option<ChatStream> {
        Some(converse_chat_stream(shape))
    }

    /// The same body, sent to the model's `converse-stream` action.
    fn streaming_request(&self, url: String, body: Value) -> Result<(String, Value), Error> {
        let url = match url.strip_suffix(CONVERSE_PATH) {
            Some(base) => format!("{base}{CONVERSE_STREAM_PATH}"),
            None => url,
        };
        Ok((url, body))
    }

    fn config_params(&self) -> &'static [&'static str] {
        CONFIG_PARAMS
    }

    /// Every request is translated; a param or message Python refuses is refused here too.
    fn unsupported_reason(
        &self,
        _messages: &[ChatMessage],
        _optional_params: &Map<String, Value>,
    ) -> Option<Unsupported> {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap as StdHashMap;

    use crate::{
        Error,
        base_llm::{
            auth::AuthScheme,
            chat::{
                params::ChatRequestContext,
                streaming::StreamShape,
                transformation::{BaseConfig, ProviderChatResponseData},
            },
            model_info::ModelInfo,
        },
        bedrock::chat::converse_transformation::BEDROCK_CHAT_COMPLETIONS_CONFIG,
    };
    use litellm_auth::CredentialPlacement;
    use litellm_llms_types::formats::chat_completions::{ChatCompletionsResponse, ChatMessage};
    use rstest::rstest;
    use serde_json::{Map, Value, json};

    struct CostMap(StdHashMap<String, Map<String, Value>>);

    impl ModelInfo for CostMap {
        fn entry(&self, model: &str) -> Option<&Map<String, Value>> {
            self.0.get(model)
        }
        fn declares(&self, key: &str) -> bool {
            self.0.values().any(|entry| entry.contains_key(key))
        }
    }

    fn cost_map() -> CostMap {
        CostMap(StdHashMap::from([(
            "anthropic.claude-sonnet-4-5-v1:0".to_owned(),
            json!({"supports_tool_choice": true, "supports_reasoning": true, "supports_prompt_caching": true})
                .as_object()
                .cloned()
                .unwrap(),
        )]))
    }

    fn messages(value: Value) -> Vec<ChatMessage> {
        serde_json::from_value(value).expect("valid messages")
    }

    fn params(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(map) => map,
            other => panic!("params must be an object, got {other}"),
        }
    }

    fn request(
        msgs: Value,
        opts: Value,
        drop: bool,
    ) -> Result<crate::base_llm::chat::transformation::ProviderChatRequestData, Error> {
        let info = cost_map();
        BEDROCK_CHAT_COMPLETIONS_CONFIG.transform_request(
            "anthropic.claude-sonnet-4-5-v1:0",
            messages(msgs),
            params(opts),
            &ChatRequestContext {
                drop_params: Some(drop),
                model_info: Some(&info),
            },
        )
    }

    fn transform(msgs: Value, opts: Value) -> Value {
        request(msgs, opts, false).expect("request transforms").body
    }

    fn transform_response(body: Value) -> Result<ChatCompletionsResponse, Error> {
        transform_response_with(body, &StreamShape::default())
    }

    fn transform_response_with(
        body: Value,
        shape: &StreamShape,
    ) -> Result<ChatCompletionsResponse, Error> {
        BEDROCK_CHAT_COMPLETIONS_CONFIG.transform_response(
            "anthropic.claude-sonnet-4-5-v1:0",
            ProviderChatResponseData { body },
            shape,
        )
    }

    #[rstest]
    fn builds_the_converse_body_python_builds() {
        let body = transform(
            json!([
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"}
            ]),
            json!({"max_tokens": 128, "temperature": 0.2, "top_p": 0.9, "stop": "END"}),
        );
        assert_eq!(
            body,
            json!({
                "messages": [{"role": "user", "content": [{"text": "hi"}]}],
                "inferenceConfig": {"maxTokens": 128, "temperature": 0.2, "topP": 0.9, "stopSequences": ["END"]},
                "system": [{"text": "be terse"}]
            })
        );
    }

    #[rstest]
    fn tools_and_tool_choice_become_a_tool_config() {
        let body = transform(
            json!([{"role": "user", "content": "weather?"}]),
            json!({
                "max_tokens": 10,
                "tools": [{"type": "function", "function": {"name": "get_weather", "description": "d",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
                "tool_choice": "required"
            }),
        );
        assert_eq!(
            body["toolConfig"],
            json!({"tools": [{"toolSpec": {"inputSchema": {"json": {"type": "object", "properties": {"city": {"type": "string"}}, "required": []}}, "name": "get_weather", "description": "d"}}],
                   "toolChoice": {"any": {}}})
        );
    }

    #[rstest]
    fn tool_choice_none_is_refused_or_dropped() {
        let msgs = json!([{"role": "user", "content": "hi"}]);
        let error = request(msgs.clone(), json!({"tool_choice": "none"}), false)
            .map(|_| ())
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "invalid request: Bedrock doesn't support tool_choice=none. To drop it from the call, set `litellm.drop_params = True."
        );
        let body = request(msgs, json!({"tool_choice": "none"}), true)
            .unwrap()
            .body;
        assert!(body.get("toolConfig").is_none());
    }

    #[rstest]
    fn unsupported_params_are_refused_or_dropped() {
        let msgs = json!([{"role": "user", "content": "hi"}]);
        assert!(request(msgs.clone(), json!({"seed": 3}), false).is_err());
        let body = request(msgs, json!({"seed": 3}), true).unwrap().body;
        assert!(body.get("additionalModelRequestFields").is_none());
        assert_eq!(body["inferenceConfig"], json!({}));
    }

    #[rstest]
    fn reasoning_effort_becomes_thinking_in_additional_fields() {
        let body = transform(
            json!([{"role": "user", "content": "hi"}]),
            json!({"reasoning_effort": "low"}),
        );
        assert_eq!(
            body["additionalModelRequestFields"],
            json!({"thinking": {"type": "enabled", "budget_tokens": 1024}})
        );
        assert_eq!(
            body["additionalModelResponseFieldPaths"],
            json!(["/usage/output_tokens_details"])
        );
        assert_eq!(body["inferenceConfig"]["maxTokens"], json!(1024 + 4096));
    }

    #[rstest]
    fn response_format_becomes_a_json_tool() {
        let data = request(
            json!([{"role": "user", "content": "hi"}]),
            json!({"response_format": {"type": "json_schema", "json_schema": {"name": "x",
                "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}}}),
            false,
        )
        .unwrap();
        assert_eq!(
            data.body["toolConfig"]["toolChoice"],
            json!({"tool": {"name": "json_tool_call"}})
        );
        assert!(data.stream_shape.json_mode);
        let response = transform_response_with(
            json!({"output": {"message": {"content": [
                {"toolUse": {"toolUseId": "t", "name": "json_tool_call", "input": {"properties": {"a": "b"}}}}
            ]}}, "stopReason": "tool_use", "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}),
            &data.stream_shape,
        )
        .unwrap();
        assert_eq!(
            response.choices[0].message.content.as_deref(),
            Some("{\"a\": \"b\"}")
        );
        assert_eq!(response.choices[0].finish_reason, "stop");
    }

    #[rstest]
    fn orphaned_tool_calls_are_sent_as_text() {
        let body = transform(
            json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c", "content": "ok"}
            ]),
            json!({}),
        );
        assert_eq!(
            body["messages"][1]["content"][0]["text"],
            "[tool call c: f({})]"
        );
    }

    #[rstest]
    fn tool_use_and_reasoning_come_back_as_openai_fields() {
        let shape = StreamShape {
            tool_name_reverse_map: StdHashMap::from([("get_weather".into(), "get.weather".into())]),
            ..StreamShape::default()
        };
        let response = transform_response_with(
            json!({"output": {"message": {"content": [
                {"reasoningContent": {"reasoningText": {"text": "hmm", "signature": "sig"}}},
                {"text": "checking"},
                {"toolUse": {"toolUseId": "t1", "name": "get_weather", "input": {"city": "Paris"}}}
            ]}}, "stopReason": "tool_use", "usage": {"inputTokens": 1, "outputTokens": 1}}),
            &shape,
        )
        .unwrap();
        let message = serde_json::to_value(&response.choices[0].message).unwrap();
        assert_eq!(
            message["tool_calls"],
            json!([{"id": "t1", "type": "function", "function": {"name": "get.weather", "arguments": "{\"city\": \"Paris\"}"}, "index": 2}])
        );
        assert_eq!(message["reasoning_content"], "hmm");
        assert_eq!(
            message["thinking_blocks"],
            json!([{"type": "thinking", "thinking": "hmm", "signature": "sig"}])
        );
        assert_eq!(response.choices[0].finish_reason, "tool_calls");
    }

    #[rstest]
    #[case("<think>a</think>b", Some("a"), "b")]
    #[case("plain", None, "plain")]
    fn leading_think_tags_are_reasoning(
        #[case] text: &str,
        #[case] reasoning: Option<&str>,
        #[case] answer: &str,
    ) {
        let (found, rest) = super::parse_content_for_reasoning(text);
        assert_eq!(found.as_deref(), reasoning);
        assert_eq!(rest, answer);
    }

    #[test]
    fn builds_the_converse_url_from_the_region_in_the_model_id() {
        let config = &BEDROCK_CHAT_COMPLETIONS_CONFIG;
        assert_eq!(
            config
                .get_complete_url(None, "us-east-1/anthropic.claude-v2", &Map::new(), &|_| {
                    None
                })
                .expect("url builds"),
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/converse"
        );
    }

    #[test]
    fn falls_back_to_the_region_env_then_the_default_region() {
        let config = &BEDROCK_CHAT_COMPLETIONS_CONFIG;
        let with_env = |key: &str| (key == "AWS_REGION_NAME").then(|| "eu-west-1".to_string());
        assert_eq!(
            config
                .get_complete_url(None, "anthropic.claude-v2", &Map::new(), &with_env)
                .expect("url builds"),
            "https://bedrock-runtime.eu-west-1.amazonaws.com/model/anthropic.claude-v2/converse"
        );
        assert_eq!(
            config
                .get_complete_url(None, "anthropic.claude-v2", &Map::new(), &|_| None)
                .expect("url builds"),
            "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-v2/converse"
        );
    }

    #[test]
    fn prefers_an_explicit_runtime_endpoint_over_the_api_base() {
        let config = &BEDROCK_CHAT_COMPLETIONS_CONFIG;
        let overrides = params(json!({"aws_bedrock_runtime_endpoint": "https://vpce.internal/"}));
        assert_eq!(
            config
                .get_complete_url(
                    Some("https://ignored.example"),
                    "anthropic.claude-v2",
                    &overrides,
                    &|_| None
                )
                .expect("url builds"),
            "https://vpce.internal/model/anthropic.claude-v2/converse"
        );
    }

    /// The bearer token a config named, or `None` for a SigV4 scheme in the given region.
    fn bearer_or_region(auth: AuthScheme) -> Result<String, String> {
        match auth {
            AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                secret,
            } => Ok(secret.expose().to_string()),
            AuthScheme::AwsSigV4 {
                region,
                service: "bedrock",
                ..
            } => Err(region),
            other => panic!("unexpected auth {other:?}"),
        }
    }

    #[test]
    fn signs_with_sigv4_in_the_resolved_region() {
        let validated = BEDROCK_CHAT_COMPLETIONS_CONFIG
            .validate_environment(
                Vec::new(),
                None,
                "eu-central-1/anthropic.claude-v2",
                &Map::new(),
                &|_| None,
            )
            .expect("auth resolves");
        assert_eq!(
            bearer_or_region(validated.auth),
            Err("eu-central-1".to_string())
        );
    }

    #[test]
    fn a_bearer_token_outranks_sigv4_the_way_python_resolves_it() {
        // Python's get_request_headers reads `api_key` as the Bedrock bearer token
        // and only falls back to the env when the caller passed none, so each case
        // pins one of its precedence rules. Signing as the host principal when a
        // bearer identity is configured would cross an account and quota boundary.
        let bedrock_env =
            |key: &str| (key == "AWS_BEARER_TOKEN_BEDROCK").then(|| "from-env".to_string());
        let no_env = |_: &str| None;
        let resolve = |api_key, env: &dyn Fn(&str) -> Option<String>| {
            bearer_or_region(
                BEDROCK_CHAT_COMPLETIONS_CONFIG
                    .validate_environment(
                        Vec::new(),
                        api_key,
                        "eu-central-1/anthropic.claude-v2",
                        &Map::new(),
                        env,
                    )
                    .expect("auth resolves")
                    .auth,
            )
        };
        let bearer = |token: &str| Ok(token.to_string());
        let sigv4 = Err("eu-central-1".to_string());

        // A caller-supplied key is the bearer token, and outranks the env.
        assert_eq!(
            resolve(Some("bedrock-api-key"), &bedrock_env),
            bearer("bedrock-api-key")
        );
        // No key, so the env supplies it.
        assert_eq!(resolve(None, &bedrock_env), bearer("from-env"));
        // An empty key is not a bearer token, and deliberately does NOT reach for
        // the env, which is what Python's `is not None` check does.
        assert_eq!(resolve(Some(""), &bedrock_env), sigv4);
        // Whitespace is truthy in Python, so it stays a bearer token rather than
        // silently becoming a host-credentialed SigV4 request.
        assert_eq!(resolve(Some("  "), &no_env), bearer("  "));
        // Neither present, so SigV4 as before.
        assert_eq!(resolve(None, &no_env), sigv4);
    }

    #[test]
    fn normalizes_a_converse_response_into_openai_shape() {
        let response = transform_response(json!({
            "output": {"message": {"role": "assistant", "content": [
                {"text": "hello"}, {"text": " there"}
            ]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}
        }))
        .expect("response transforms");

        assert_eq!(response.model, "anthropic.claude-sonnet-4-5-v1:0");
        assert_eq!(
            response.choices[0].message.content.as_deref(),
            Some("hello there")
        );
        assert_eq!(response.choices[0].finish_reason, "stop");
        assert_eq!(response.usage.prompt_tokens, 11);
        assert_eq!(response.usage.completion_tokens, 4);
        assert_eq!(response.usage.total_tokens, 15);
    }

    #[test]
    fn maps_converse_stop_reasons_python_maps() {
        for (provider_reason, expected) in [
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("max_tokens", "length"),
            ("guardrail_intervened", "content_filter"),
            // Converse emits this one, and Python's `_FINISH_REASON_MAP` carries
            // it. Folding it into `stop` reports a filtered completion as a normal
            // one to anything keying on the finish reason.
            ("content_filtered", "content_filter"),
            ("content_filter", "content_filter"),
        ] {
            let response = transform_response(json!({
                "output": {"message": {"content": [{"text": "x"}]}},
                "stopReason": provider_reason,
                "usage": {"inputTokens": 1, "outputTokens": 1}
            }))
            .expect("response transforms");
            assert_eq!(
                response.choices[0].finish_reason, expected,
                "stopReason {provider_reason}"
            );
        }
    }

    #[test]
    fn reports_an_empty_converse_answer_as_an_empty_string_not_null() {
        // Converse assigns the joined text unconditionally
        // (`chat_completion_message["content"] = content_str`), unlike Anthropic's
        // `merged_text or None`, so an empty answer is `""` on both paths. A caller
        // calling `.strip()` on it would break on the Rust path alone. Reachable
        // through a filtered or guardrail-intervened response.
        for content in [json!([]), json!([{"text": ""}])] {
            let response = transform_response(json!({
                "output": {"message": {"content": content}},
                "stopReason": "content_filtered",
                "usage": {"inputTokens": 1, "outputTokens": 0}
            }))
            .expect("response transforms");
            assert_eq!(response.choices[0].message.content, Some(String::new()));
        }
    }

    #[test]
    fn reports_the_total_tokens_converse_sent_rather_than_recomputing_them() {
        // Python reads `usage["totalTokens"]` straight through here, where Anthropic
        // has no such field and adds the two counts instead. The two agree while the
        // gate declines every cache_control request, so this is what keeps them
        // agreeing if that ever widens.
        let response = transform_response(json!({
            "output": {"message": {"content": [{"text": "x"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 4, "cacheReadInputTokens": 7, "totalTokens": 14}
        }))
        .expect("response transforms");
        assert_eq!(
            response.usage.total_tokens, 14,
            "provider total was recomputed"
        );
        assert_eq!(response.usage.prompt_tokens, 17);
        assert_eq!(response.usage.completion_tokens, 4);
    }

    #[test]
    fn falls_back_to_the_computed_total_when_converse_omits_it() {
        // Python raises a KeyError on a body with no `totalTokens`. Reporting a zero
        // instead would be a worse divergence than the one above, so the computed
        // total stands in.
        let response = transform_response(json!({
            "output": {"message": {"content": [{"text": "x"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 4}
        }))
        .expect("response transforms");
        assert_eq!(response.usage.total_tokens, 14);
    }

    #[test]
    fn folds_converse_cache_tokens_into_prompt_tokens() {
        let response = transform_response(json!({
            "output": {"message": {"content": [{"text": "x"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 2,
                "cacheReadInputTokens": 5,
                "cacheWriteInputTokens": 3
            }
        }))
        .expect("response transforms");
        assert_eq!(response.usage.prompt_tokens, 18);
        assert_eq!(response.usage.prompt_tokens_details.cached_tokens, 5);
        assert_eq!(
            response.usage.prompt_tokens_details.cache_creation_tokens,
            3
        );
        assert_eq!(response.usage.prompt_tokens_details.text_tokens, 10);
    }

    #[test]
    fn errors_on_a_response_missing_required_fields() {
        assert_eq!(
            transform_response(json!("nope")).expect_err("not an object"),
            Error::InvalidResponse("converse response is not an object".to_string().into())
        );
        assert_eq!(
            transform_response(json!({"usage": {}})).expect_err("no output"),
            Error::InvalidResponse("invalid Converse response: missing field `output`".into())
        );
        assert_eq!(
            transform_response(json!({"output": {"message": {"content": []}}}))
                .expect_err("no usage"),
            Error::InvalidResponse("invalid Converse response: missing field `usage`".into())
        );
    }

    #[test]
    fn rejects_malformed_text_and_token_counts() {
        for response in [
            json!({"output": {"message": {"content": [{"text": 123}]}}, "usage": {"inputTokens": 1, "outputTokens": 1}}),
            json!({"output": {"message": {"content": [{"text": "x"}]}}, "usage": {"inputTokens": "1", "outputTokens": 1}}),
        ] {
            assert!(matches!(
                transform_response(response),
                Err(Error::InvalidResponse(message)) if message.to_string().starts_with("invalid Converse response:")
            ));
        }
    }

    #[test]
    fn accepts_aws_call_configuration_without_serializing_it() {
        let call_config = json!({
            "max_tokens": 16,
            "aws_access_key_id": "AKIA",
            "aws_secret_access_key": "secret",
            "aws_session_token": "token",
            "aws_region_name": "us-east-1",
            "aws_profile_name": "litellm-stage",
            "aws_role_name": "role",
            "aws_session_name": "session",
            "aws_web_identity_token": "wit",
            "aws_sts_endpoint": "https://sts.example",
            "aws_external_id": "ext",
            "aws_bedrock_runtime_endpoint": "https://vpce.internal"
        });
        let body = transform(json!([{"role": "user", "content": "hi"}]), call_config);
        assert_eq!(
            body,
            json!({
                "messages": [{"role": "user", "content": [{"text": "hi"}]}],
                "inferenceConfig": {"maxTokens": 16}
            }),
            "aws call configuration must not reach the Converse body"
        );
    }

    #[test]
    fn leaves_a_complete_converse_url_untouched() {
        let config = &BEDROCK_CHAT_COMPLETIONS_CONFIG;
        let already_built = "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-v2%3A0/converse";
        assert_eq!(
            config
                .get_complete_url(
                    Some(already_built),
                    "anthropic.claude-v2",
                    &Map::new(),
                    &|_| None
                )
                .expect("url builds"),
            already_built,
            "a host that encoded the model id itself must not have it re-derived"
        );
    }

    #[rstest]
    #[case::full_static_pair(
        json!({"aws_access_key_id": "AKIAHOST", "aws_secret_access_key": "hostsecret", "aws_session_token": "hosttoken"}),
        Some(("AKIAHOST", "hostsecret", Some("hosttoken")))
    )]
    #[case::pair_without_session_token(
        json!({"aws_access_key_id": "AKIAHOST", "aws_secret_access_key": "hostsecret"}),
        Some(("AKIAHOST", "hostsecret", None))
    )]
    #[case::key_id_alone(json!({"aws_access_key_id": "AKIA"}), None)]
    #[case::blank_key_id(json!({"aws_access_key_id": "  ", "aws_secret_access_key": "s"}), None)]
    #[case::nothing(json!({}), None)]
    fn host_supplied_credentials_need_a_full_static_pair(
        #[case] optional_params: Value,
        #[case] expected: Option<(&str, &str, Option<&str>)>,
    ) {
        use litellm_auth::AwsParams;
        use litellm_auth_aws::host_supplied_credentials;

        let credentials =
            host_supplied_credentials(&AwsParams::from_optional_params(&params(optional_params)));

        assert_eq!(
            credentials.as_ref().map(|credentials| (
                credentials.access_key_id(),
                credentials.secret_access_key(),
                credentials.session_token(),
            )),
            expected
        );
    }
}
