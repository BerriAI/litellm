use std::convert::Infallible;

use crate::{
    ErrorDetail,
    anthropic::{
        common_utils::supports_effort_param,
        messages::{
            handler::shape_anthropic_messages_request,
            transformation::{provider_feature_betas, transform_messages_request},
        },
    },
    base_llm::messages::{
        context::{MessagesModelCapabilities, MessagesTransformContext, ThinkingBudgets},
        normalization::normalize_system_role_messages,
    },
};
use futures_util::StreamExt;
use litellm_auth::AwsParams;
use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_auth_aws::{
    AwsCredentialSource, bedrock_model_id_and_region,
    constants::{
        AWS_BEARER_TOKEN_BEDROCK, AWS_BEDROCK_RUNTIME_ENDPOINT, AWS_DEFAULT_REGION, AWS_REGION,
        AWS_REGION_NAME, BEDROCK_RUNTIME_ENDPOINT_TEMPLATE, BEDROCK_SERVICE,
    },
    resolve_bedrock_region,
};
use litellm_http::request::header_values;
use litellm_llms_types::formats::messages::{
    MessagesOptionalParams, MessagesRequest,
    streaming::{MessagesStreamEvent, MessagesStreamUsage},
};
use litellm_llms_types::providers::{
    anthropic::{AnthropicBeta, BETA_HEADER, BetaProvider, BetaSet},
    bedrock::{ANTHROPIC_VERSION, INVOCATION_METRICS_KEY, INVOKE_PATH, INVOKE_STREAM_PATH},
};
use litellm_router_types::LitellmParams;
use serde_json::{Map, Value, json};

use crate::{
    Error,
    base_llm::{
        auth::AuthScheme,
        base_model_iterator::{StreamError, StreamTransformer, transform_stream},
        messages::{
            streaming::{ByteStream, EventStream, StreamDecoder},
            transformation::{BaseMessagesConfig, Headers, ValidatedEnvironment},
        },
    },
    bedrock::chat::invoke_handler::{decode_invoke_anthropic_chunk, invoke_chunk_stream},
};

const METRICS_USAGE_KEYS: [(&str, &str); 4] = [
    ("input_tokens", "inputTokenCount"),
    ("output_tokens", "outputTokenCount"),
    ("cache_read_input_tokens", "cacheReadInputTokenCount"),
    ("cache_creation_input_tokens", "cacheWriteInputTokenCount"),
];

const INVOKE_MODEL_PREFIX: &str = "invoke/";

const SECRET_NAMES: &[&str] = &[
    AWS_BEARER_TOKEN_BEDROCK,
    AWS_BEDROCK_RUNTIME_ENDPOINT,
    AWS_REGION_NAME,
    AWS_REGION,
    AWS_DEFAULT_REGION,
];

/// Python's `BedrockInvokeAnthropicMessagesRequest`. Any other field is a Bedrock 400.
const ALLOWED_TOP_LEVEL_FIELDS: [&str; 16] = [
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
    "safeguards",
    "context_management",
];

/// `clear_thinking_20251015` is absent on purpose, it becomes a `thinking` config instead.
const SUPPORTED_CONTEXT_EDITS: [&str; 2] = ["compact_20260112", "clear_tool_uses_20250919"];
const CLEAR_THINKING_EDIT: &str = "clear_thinking_20251015";
const MIN_THINKING_BUDGET_TOKENS: u64 = 1024;
const LEGACY_THINKING_MODEL_PATTERNS: [&str; 12] = [
    "opus-4.5", "opus_4.5", "opus-4-5", "opus_4_5", "opus-4.1", "opus_4.1", "opus-4-1", "opus_4_1",
    "opus-4", "opus_4", "sonnet-4", "sonnet_4",
];
const ADAPTIVE_THINKING_KEYS: [&str; 3] = ["type", "display", "block_binding"];
const WEB_SEARCH_TOOL_PREFIX: &str = "web_search_";
const WEB_SEARCH_GUIDANCE: &str = "To use web search with this model, enable LiteLLM's web search interception so the proxy executes the search instead: https://docs.litellm.ai/docs/integrations/websearch_interception. Alternatively, remove the web_search tool from the request.";
const UNNAMED_TOOL_PREFIX: &str = "litellm_unnamed_tool_";

pub struct AmazonAnthropicClaudeMessagesConfig;

pub const BEDROCK_ANTHROPIC_MESSAGES_CONFIG: AmazonAnthropicClaudeMessagesConfig =
    AmazonAnthropicClaudeMessagesConfig;

fn bearer_token(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    match api_key {
        Some(key) => Some(key.to_string()),
        None => env_lookup(AWS_BEARER_TOKEN_BEDROCK),
    }
    .filter(|token| !token.is_empty())
}

fn invoke_url(
    api_base: Option<&str>,
    model: &str,
    aws: &AwsParams,
    stream: bool,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    let (model_id, model_region) =
        bedrock_model_id_and_region(model.strip_prefix(INVOKE_MODEL_PREFIX).unwrap_or(model));
    let region = resolve_bedrock_region(model_region.as_deref(), aws, env_lookup);
    let path = if stream {
        INVOKE_STREAM_PATH
    } else {
        INVOKE_PATH
    };
    let configured = |value: Option<&str>| {
        value
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    };
    let endpoint = configured(api_base)
        .or_else(|| configured(aws.aws_bedrock_runtime_endpoint.as_deref()))
        .or_else(|| env_lookup(AWS_BEDROCK_RUNTIME_ENDPOINT))
        .unwrap_or_else(|| BEDROCK_RUNTIME_ENDPOINT_TEMPLATE.replace("{region}", &region));
    format!("{}/model/{model_id}/{path}", endpoint.trim_end_matches('/'))
}

impl BaseMessagesConfig for AmazonAnthropicClaudeMessagesConfig {
    fn shape_request(
        &self,
        request: MessagesRequest,
        reasoning_auto_summary: bool,
    ) -> Result<MessagesRequest, Error> {
        shape_anthropic_messages_request(request, reasoning_auto_summary)
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        litellm_params: &LitellmParams,
        stream: bool,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(invoke_url(
            api_base,
            model,
            &litellm_params.aws,
            stream,
            env_lookup,
        ))
    }

    /// Python's `transform_anthropic_messages_request`: the shared Anthropic shaping, then the
    /// InvokeModel body rules. `model` and `stream` stay on the typed request (they pick the
    /// URL) and leave in `wire_body`, as does every field outside the allowlist.
    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        let capabilities = &context.thinking.capabilities;
        // InvokeModel has no fast mode, and Python drops `speed` with the allowlist rather than
        // rejecting it the way the shared Anthropic stage does for models without `supports_speed`.
        let request = MessagesRequest {
            params: MessagesOptionalParams {
                speed: None,
                ..request.params
            },
            ..request
        };
        let request = normalize_system_role_messages(
            transform_messages_request(request, context)?,
            capabilities.supports_mid_conversation_system,
        );
        let model = request.model.clone();
        let Value::Object(body) = serde_json::to_value(request).map_err(invalid_request)? else {
            return Err(Error::InvalidType {
                expected: "a JSON object",
                actual: "another JSON value",
            });
        };
        let eager_input_streaming = uses_eager_input_streaming(&body);
        let body = invoke_body(body, &model, context)?;
        let shaped: MessagesRequest =
            serde_json::from_value(Value::Object(body)).map_err(invalid_request)?;
        let betas = feature_betas(&shaped, eager_input_streaming);
        Ok(with_body_betas(shaped, betas))
    }

    fn secret_names(&self) -> &'static [&'static str] {
        SECRET_NAMES
    }

    /// Python reads `api_key` as the Bedrock bearer token and consults the env only when the
    /// caller passed none. Without one the request is signed with SigV4.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        model: &str,
        litellm_params: &LitellmParams,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        if let Some(token) = bearer_token(api_key, env_lookup) {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Credential {
                    placement: CredentialPlacement::Bearer,
                    secret: SecretValue::new(token),
                },
            });
        }
        let (_, model_region) =
            bedrock_model_id_and_region(model.strip_prefix(INVOKE_MODEL_PREFIX).unwrap_or(model));
        let params = &litellm_params.aws;
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::AwsSigV4 {
                region: resolve_bedrock_region(model_region.as_deref(), params, env_lookup),
                service: BEDROCK_SERVICE,
                credentials: Box::new(AwsCredentialSource::from_params(params, env_lookup)),
            },
        })
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[("content-type", "application/json")]
    }

    /// The caller's `anthropic-beta` values join the feature betas in the body's
    /// `anthropic_beta`, which is where InvokeModel reads them. Values Bedrock has no mapping
    /// for are dropped, since forwarding them is a 400.
    fn request_with_headers(&self, request: MessagesRequest, headers: &Headers) -> MessagesRequest {
        let caller = caller_betas(headers);
        if caller.is_empty() {
            return request;
        }
        let body = body_betas(&request);
        with_body_betas(request, body.union(caller))
    }

    fn wire_body(&self, body: Value) -> Value {
        match body {
            Value::Object(fields) => Value::Object(
                fields
                    .into_iter()
                    .filter(|(name, _)| ALLOWED_TOP_LEVEL_FIELDS.contains(&name.as_str()))
                    .collect(),
            ),
            other => other,
        }
    }

    fn stream_decoder(&self) -> Option<StreamDecoder> {
        Some(bedrock_anthropic_messages_event_stream)
    }
}

fn invalid_request(error: serde_json::Error) -> Error {
    Error::InvalidRequest(ErrorDetail::invalid(
        "Bedrock invoke messages request",
        error,
    ))
}

/// The InvokeModel body rules of Python's transformation, in its order.
fn invoke_body(
    mut body: Map<String, Value>,
    model: &str,
    context: &MessagesTransformContext,
) -> Result<Map<String, Value>, Error> {
    let capabilities = &context.thinking.capabilities;
    reject_web_search_tools(&body, model)?;
    body.entry("anthropic_version")
        .or_insert_with(|| json!(ANTHROPIC_VERSION));
    ensure_thinking_for_clear_thinking(&mut body, model, context);
    sanitize_cache_controls(&mut body, capabilities.supports_prompt_cache_ttl);
    apply_structured_output(&mut body, capabilities.supports_native_structured_output);
    strip_unsupported_output_config(&mut body, capabilities);
    normalize_tools(&mut body);
    filter_context_management(&mut body);
    let unsupported_effort = body
        .get("output_config")
        .and_then(Value::as_object)
        .is_some_and(|config| config.keys().any(|key| key != "format"))
        && !supports_effort_param(capabilities);
    if context.drop_params && unsupported_effort {
        body.remove("output_config");
    }
    Ok(body)
}

fn tools(body: &Map<String, Value>) -> impl Iterator<Item = &Map<String, Value>> {
    body.get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
}

fn tools_mut(body: &mut Map<String, Value>) -> impl Iterator<Item = &mut Map<String, Value>> {
    body.get_mut("tools")
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object_mut)
}

fn uses_eager_input_streaming(body: &Map<String, Value>) -> bool {
    tools(body).any(|tool| tool.get("eager_input_streaming") == Some(&Value::Bool(true)))
}

/// Bedrock cannot run Anthropic's server-side `web_search_*` tool and answers an opaque 400,
/// so the request fails before it is sent with an error that names the fix.
fn reject_web_search_tools(body: &Map<String, Value>, model: &str) -> Result<(), Error> {
    let Some(tool) = tools(body)
        .filter_map(|tool| tool.get("type")?.as_str())
        .find(|kind| kind.starts_with(WEB_SEARCH_TOOL_PREFIX))
    else {
        return Ok(());
    };
    Err(Error::InvalidRequest(ErrorDetail::UnsupportedTool {
        provider: "Bedrock",
        tool: tool.to_string(),
        model: model.to_string(),
        guidance: WEB_SEARCH_GUIDANCE,
    }))
}

fn supports_extended_thinking(model: &str, capabilities: &MessagesModelCapabilities) -> bool {
    let model = model.to_ascii_lowercase();
    capabilities.supports_adaptive_thinking
        || LEGACY_THINKING_MODEL_PATTERNS
            .iter()
            .any(|pattern| model.contains(pattern))
}

fn effort_from_budget(budget: u64, budgets: &ThinkingBudgets) -> &'static str {
    match budget {
        budget if budget >= budgets.xhigh => "xhigh",
        budget if budget >= budgets.high => "high",
        budget if budget >= budgets.medium => "medium",
        _ => "low",
    }
}

fn inject_adaptive_thinking(
    body: &mut Map<String, Value>,
    budget: u64,
    context: &MessagesTransformContext,
) {
    let mut output_config = match body.remove("output_config") {
        Some(Value::Object(config)) => config,
        _ => Map::new(),
    };
    output_config
        .entry("effort")
        .or_insert_with(|| json!(effort_from_budget(budget, &context.thinking.budgets)));
    let carried: Map<String, Value> = match body.remove("thinking") {
        Some(Value::Object(thinking)) => thinking
            .into_iter()
            .filter(|(key, _)| {
                !matches!(key.as_str(), "type" | "budget_tokens")
                    && (!context.drop_params || ADAPTIVE_THINKING_KEYS.contains(&key.as_str()))
            })
            .collect(),
        _ => Map::new(),
    };
    body.insert("output_config".into(), Value::Object(output_config));
    body.insert(
        "thinking".into(),
        Value::Object(
            carried
                .into_iter()
                .chain([("type".to_string(), json!("adaptive"))])
                .collect(),
        ),
    );
}

/// Bedrock rejects a `clear_thinking_20251015` edit unless thinking is on, and Claude Code
/// sends that edit without a `thinking` field, so a minimal thinking config is added: adaptive
/// with an effort for adaptive models, a minimum budget for the legacy ones.
fn ensure_thinking_for_clear_thinking(
    body: &mut Map<String, Value>,
    model: &str,
    context: &MessagesTransformContext,
) {
    let capabilities = &context.thinking.capabilities;
    let clears_thinking = body
        .get("context_management")
        .and_then(|management| management.get("edits"))
        .and_then(Value::as_array)
        .is_some_and(|edits| {
            edits
                .iter()
                .any(|edit| edit.get("type").and_then(Value::as_str) == Some(CLEAR_THINKING_EDIT))
        });
    if !clears_thinking || !supports_extended_thinking(model, capabilities) {
        return;
    }
    let adaptive_model = capabilities.supports_adaptive_thinking;
    let thinking = body.get("thinking").and_then(Value::as_object);
    match thinking.and_then(|thinking| thinking.get("type")?.as_str()) {
        Some("adaptive") => return,
        Some("enabled") if !adaptive_model => return,
        Some("enabled") => {
            let budget = thinking
                .and_then(|thinking| thinking.get("budget_tokens")?.as_u64())
                .unwrap_or(MIN_THINKING_BUDGET_TOKENS);
            inject_adaptive_thinking(body, budget, context);
            return;
        }
        _ => {}
    }
    if body
        .get("max_tokens")
        .and_then(Value::as_u64)
        .is_some_and(|max_tokens| max_tokens <= MIN_THINKING_BUDGET_TOKENS)
    {
        return;
    }
    if adaptive_model {
        inject_adaptive_thinking(body, MIN_THINKING_BUDGET_TOKENS, context);
        return;
    }
    body.insert(
        "thinking".into(),
        json!({"type": "enabled", "budget_tokens": MIN_THINKING_BUDGET_TOKENS}),
    );
}

/// Bedrock takes only `type` and, on models that support it, a "5m" or "1h" `ttl` in a
/// `cache_control`. `scope` always goes.
fn sanitize_cache_control(block: &mut Map<String, Value>, supports_ttl: bool) {
    let Some(Value::Object(cache_control)) = block.get_mut("cache_control") else {
        return;
    };
    cache_control.remove("scope");
    let keeps_ttl = supports_ttl
        && cache_control
            .get("ttl")
            .and_then(Value::as_str)
            .is_some_and(|ttl| matches!(ttl, "5m" | "1h"));
    if !keeps_ttl {
        cache_control.remove("ttl");
    }
}

fn sanitize_cache_controls(body: &mut Map<String, Value>, supports_ttl: bool) {
    tools_mut(body).for_each(|tool| sanitize_cache_control(tool, supports_ttl));
    body.iter_mut()
        .flat_map(|(name, value)| match (name.as_str(), value) {
            ("system", Value::Array(blocks)) => blocks.iter_mut().collect(),
            ("messages", Value::Array(messages)) => messages
                .iter_mut()
                .filter_map(|message| message.get_mut("content")?.as_array_mut())
                .flatten()
                .collect(),
            _ => Vec::new(),
        })
        .filter_map(Value::as_object_mut)
        .for_each(|block| sanitize_cache_control(block, supports_ttl));
}

/// Python's `apply_bedrock_invoke_structured_output`: the legacy `output_format` wins over
/// `output_config.format`. Models with native structured output get it as
/// `output_config.format`; for the rest the schema is appended to the last user turn as text.
fn apply_structured_output(body: &mut Map<String, Value>, native: bool) {
    let legacy = body.remove("output_format").filter(Value::is_object);
    let nested = match body.get_mut("output_config") {
        Some(Value::Object(config)) => {
            let format = config.remove("format");
            if config.is_empty() {
                body.remove("output_config");
            }
            format
        }
        _ => None,
    };
    let Some(format) = legacy.or(nested.filter(Value::is_object)) else {
        return;
    };
    if native {
        match body.get_mut("output_config") {
            Some(Value::Object(config)) => {
                config.insert("format".into(), format);
            }
            _ => {
                body.insert("output_config".into(), json!({"format": format}));
            }
        }
        return;
    }
    inline_schema(body, &format);
}

fn inline_schema(body: &mut Map<String, Value>, format: &Value) {
    let Some(schema) = format.get("schema").filter(|schema| match schema {
        Value::Null => false,
        Value::Object(fields) => !fields.is_empty(),
        _ => true,
    }) else {
        return;
    };
    let schema_block = json!({"type": "text", "text": python_json_dumps(schema)});
    let Some(last_user) = body
        .get_mut("messages")
        .and_then(Value::as_array_mut)
        .and_then(|messages| {
            messages
                .iter_mut()
                .rev()
                .find(|message| message.get("role").and_then(Value::as_str) == Some("user"))
        })
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    let content = match last_user.remove("content") {
        Some(Value::String(text)) => json!([{"type": "text", "text": text}, schema_block]),
        Some(Value::Array(blocks)) => {
            Value::Array(blocks.into_iter().chain([schema_block]).collect())
        }
        None => json!([schema_block]),
        Some(other) => other,
    };
    last_user.insert("content".into(), content);
}

/// `json.dumps` spacing (`", "` and `": "`), so the inlined schema text matches Python's.
fn python_json_dumps(value: &Value) -> String {
    match value {
        Value::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(python_json_dumps)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(fields) => format!(
            "{{{}}}",
            fields
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    Value::from(key.as_str()),
                    python_json_dumps(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        scalar => scalar.to_string(),
    }
}

/// Python's `strip_unsupported_bedrock_invoke_output_config_keys`: `format` always survives,
/// effort-bearing keys only on models that accept `output_config` or an effort tier.
fn strip_unsupported_output_config(
    body: &mut Map<String, Value>,
    capabilities: &MessagesModelCapabilities,
) {
    let Some(Value::Object(config)) = body.get("output_config") else {
        return;
    };
    if config.keys().all(|key| key == "format")
        || capabilities.supports_output_config
        || supports_effort_param(capabilities)
    {
        return;
    }
    match config.get("format").cloned() {
        Some(format) => body.insert("output_config".into(), json!({"format": format})),
        None => body.remove("output_config"),
    };
}

/// Python's `normalize_json_schema_custom_types_to_object`, iterative like the original.
fn custom_schema_types_to_object(schema: &mut Value) {
    let mut stack = vec![schema];
    while let Some(Value::Object(node)) = stack.pop() {
        if node.get("type").and_then(Value::as_str) == Some("custom") {
            node.insert("type".into(), json!("object"));
        }
        for (key, child) in node.iter_mut() {
            match (key.as_str(), child) {
                ("items" | "additionalProperties", child @ Value::Object(_)) => stack.push(child),
                ("properties", Value::Object(properties)) => {
                    stack.extend(properties.values_mut().filter(|sub| sub.is_object()));
                }
                ("allOf" | "anyOf" | "oneOf", Value::Array(subs)) => {
                    stack.extend(subs.iter_mut().filter(|sub| sub.is_object()));
                }
                _ => {}
            }
        }
    }
}

/// Bedrock's tool validation: `custom` is rejected (its boolean `defer_loading` is hoisted
/// first), `input_schema` may not use `type: "custom"`, every tool needs a `name`, and
/// `eager_input_streaming` is carried by the fine-grained streaming beta instead.
fn normalize_tools(body: &mut Map<String, Value>) {
    for (index, tool) in tools_mut(body).enumerate() {
        if let Some(Value::Object(custom)) = tool.remove("custom")
            && !tool.contains_key("defer_loading")
            && let Some(deferred @ Value::Bool(_)) = custom.get("defer_loading")
        {
            tool.insert("defer_loading".into(), deferred.clone());
        }
        if let Some(schema) = tool.get_mut("input_schema") {
            custom_schema_types_to_object(schema);
        }
        let named = match tool.get("name") {
            None | Some(Value::Null) => false,
            Some(Value::String(name)) => !name.trim().is_empty(),
            Some(_) => true,
        };
        if !named {
            tool.insert(
                "name".into(),
                json!(format!("{UNNAMED_TOOL_PREFIX}{index}")),
            );
        }
        tool.remove("eager_input_streaming");
    }
}

/// Keeps the edits InvokeModel accepts; drops `context_management` when none remain.
fn filter_context_management(body: &mut Map<String, Value>) {
    let Some(Value::Object(management)) = body.get_mut("context_management") else {
        return;
    };
    let retained: Vec<Value> = management
        .get("edits")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|edit| {
            edit.get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| SUPPORTED_CONTEXT_EDITS.contains(&kind))
        })
        .cloned()
        .collect();
    if retained.is_empty() {
        body.remove("context_management");
        return;
    }
    management.insert("edits".into(), Value::Array(retained));
}

/// The betas the request's own features need, under Bedrock's beta policy.
fn feature_betas(request: &MessagesRequest, eager_input_streaming: bool) -> BetaSet {
    let extra = &request.params.extra;
    let added: BetaSet = [
        eager_input_streaming.then_some(AnthropicBeta::FineGrainedToolStreaming20250514),
        extra
            .get("safeguards")
            .is_some_and(|value| !value.is_null())
            .then_some(AnthropicBeta::DangerousToolUse20260903),
    ]
    .into_iter()
    .flatten()
    .filter_map(|beta| beta.on(BetaProvider::Bedrock))
    .collect();
    provider_feature_betas(request, BetaProvider::Bedrock).union(added)
}

/// Python's `get_anthropic_beta_from_headers`: a JSON array or comma-separated values, every
/// casing of the header, filtered through Bedrock's beta policy.
fn caller_betas(headers: &Headers) -> BetaSet {
    header_values(headers, BETA_HEADER)
        .flat_map(|value| {
            let value = value.trim();
            match serde_json::from_str::<Vec<Value>>(value) {
                Ok(items) if value.starts_with('[') => items
                    .iter()
                    .map(|item| match item {
                        Value::String(text) => text.trim().to_string(),
                        other => other.to_string(),
                    })
                    .collect(),
                _ => value
                    .split(',')
                    .map(|piece| piece.trim().to_string())
                    .collect::<Vec<_>>(),
            }
        })
        .filter(|beta| !beta.is_empty())
        .filter_map(|beta| {
            beta.parse::<AnthropicBeta>()
                .ok()?
                .on(BetaProvider::Bedrock)
        })
        .collect()
}

fn body_betas(request: &MessagesRequest) -> BetaSet {
    request
        .params
        .extra
        .get("anthropic_beta")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter_map(|beta| beta.parse::<AnthropicBeta>().ok())
        .collect()
}

fn with_body_betas(request: MessagesRequest, betas: BetaSet) -> MessagesRequest {
    if betas.is_empty() {
        return request;
    }
    let mut params = request.params;
    params.extra.insert(
        "anthropic_beta".into(),
        Value::Array(betas.iter().map(|beta| json!(beta.as_str())).collect()),
    );
    MessagesRequest { params, ..request }
}

fn with_invocation_usage(chunk: Value) -> Value {
    match chunk {
        Value::Object(fields) => Value::Object(with_metrics_usage(fields)),
        other => other,
    }
}

fn with_metrics_usage(mut fields: Map<String, Value>) -> Map<String, Value> {
    let Some(Value::Object(metrics)) = fields.remove(INVOCATION_METRICS_KEY) else {
        return fields;
    };
    if metrics.is_empty() {
        return fields;
    }
    let preserved = match fields.remove("usage") {
        Some(Value::Object(usage)) => usage,
        _ => Map::new(),
    };
    let usage: Map<String, Value> = METRICS_USAGE_KEYS
        .iter()
        .filter_map(|(anthropic, metric)| {
            Some((anthropic.to_string(), metrics.get(*metric)?.clone()))
        })
        .chain(preserved)
        .collect();
    fields.insert("usage".to_string(), Value::Object(usage));
    fields
}

pub fn bedrock_anthropic_messages_event_stream(bytes: ByteStream) -> EventStream {
    let events = invoke_chunk_stream(bytes)
        .map(|chunk| decode_invoke_anthropic_chunk(with_invocation_usage(chunk?)));
    Box::pin(
        transform_stream(events, MessageStopUsagePromoter::default())
            .map(|item| item.map_err(StreamError::into_decode)),
    )
}

#[derive(Default)]
pub struct MessageStopUsagePromoter {
    pending_delta: Option<MessagesStreamEvent>,
    start_usage: Option<MessagesStreamUsage>,
}

fn promoted_usage(
    delta: Option<MessagesStreamUsage>,
    stop: Option<&MessagesStreamUsage>,
    start: Option<&MessagesStreamUsage>,
) -> Option<MessagesStreamUsage> {
    let delta = delta.unwrap_or_default();
    let merged = MessagesStreamUsage {
        input_tokens: stop
            .and_then(|stop| stop.input_tokens)
            .or(delta.input_tokens),
        cache_creation_input_tokens: stop
            .and_then(|stop| stop.cache_creation_input_tokens)
            .or(delta.cache_creation_input_tokens)
            .or_else(|| start.and_then(|start| start.cache_creation_input_tokens)),
        cache_read_input_tokens: stop
            .and_then(|stop| stop.cache_read_input_tokens)
            .or(delta.cache_read_input_tokens)
            .or_else(|| start.and_then(|start| start.cache_read_input_tokens)),
        extra: delta
            .extra
            .into_iter()
            .chain(
                start
                    .and_then(|start| start.extra.get_key_value("cache_creation"))
                    .map(|(key, value)| (key.clone(), value.clone())),
            )
            .fold(Map::new(), |mut extra, (key, value)| {
                extra.entry(key).or_insert(value);
                extra
            }),
        ..delta
    };
    (merged != MessagesStreamUsage::default()).then_some(merged)
}

fn promoted(
    event: MessagesStreamEvent,
    stop: Option<&MessagesStreamUsage>,
    start: Option<&MessagesStreamUsage>,
) -> MessagesStreamEvent {
    match event {
        MessagesStreamEvent::MessageDelta {
            delta,
            usage,
            context_management,
        } => MessagesStreamEvent::MessageDelta {
            delta,
            usage: promoted_usage(usage, stop, start),
            context_management,
        },
        other => other,
    }
}

impl StreamTransformer for MessageStopUsagePromoter {
    type Input = MessagesStreamEvent;
    type Output = MessagesStreamEvent;
    type Error = Infallible;

    fn transform(
        &mut self,
        input: MessagesStreamEvent,
    ) -> Result<Vec<MessagesStreamEvent>, Infallible> {
        let pending = self.pending_delta.take();
        match input {
            MessagesStreamEvent::MessageDelta { .. } => {
                self.pending_delta = Some(input);
                Ok(pending.into_iter().collect())
            }
            MessagesStreamEvent::MessageStop { usage } => Ok(pending
                .map(|delta| promoted(delta, usage.as_ref(), self.start_usage.as_ref()))
                .into_iter()
                .chain([MessagesStreamEvent::MessageStop { usage }])
                .collect()),
            MessagesStreamEvent::MessageStart { message } => {
                self.start_usage = Some(message.usage.clone());
                Ok(pending
                    .into_iter()
                    .chain([MessagesStreamEvent::MessageStart { message }])
                    .collect())
            }
            other => Ok(pending.into_iter().chain([other]).collect()),
        }
    }

    fn finish(&mut self) -> Result<Vec<MessagesStreamEvent>, Infallible> {
        Ok(self
            .pending_delta
            .take()
            .map(|delta| promoted(delta, None, self.start_usage.as_ref()))
            .into_iter()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use aws_smithy_eventstream::frame::write_message_to;
    use aws_smithy_types::event_stream::{Header, HeaderValue, Message};
    use base64::{Engine, engine::general_purpose::STANDARD};
    use bytes::Bytes;
    use futures_util::TryStreamExt;
    use rstest::rstest;
    use serde_json::json;

    use litellm_auth_aws::constants::DEFAULT_BEDROCK_REGION;

    use super::*;
    use crate::base_llm::messages::streaming;

    fn event(value: Value) -> MessagesStreamEvent {
        serde_json::from_value(value).unwrap()
    }

    fn message_start(usage: Value) -> MessagesStreamEvent {
        event(json!({
            "type": "message_start",
            "message": {
                "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
                "content": [], "stop_reason": null, "stop_sequence": null, "usage": usage
            }
        }))
    }

    fn message_delta(usage: Value) -> MessagesStreamEvent {
        event(json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": usage
        }))
    }

    fn message_stop(usage: Option<Value>) -> MessagesStreamEvent {
        match usage {
            Some(usage) => event(json!({"type": "message_stop", "usage": usage})),
            None => event(json!({"type": "message_stop"})),
        }
    }

    fn promote(events: Vec<MessagesStreamEvent>) -> Vec<MessagesStreamEvent> {
        let mut promoter = MessageStopUsagePromoter::default();
        let mut output: Vec<_> = events
            .into_iter()
            .flat_map(|event| promoter.transform(event).unwrap())
            .collect();
        output.extend(promoter.finish().unwrap());
        output
    }

    #[rstest]
    #[case::cache_fields_on_message_stop(
        json!({"input_tokens": 10, "output_tokens": 0}),
        json!({"output_tokens": 5}),
        Some(json!({"input_tokens": 3, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20})),
        json!({"input_tokens": 3, "output_tokens": 5, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20}),
    )]
    #[case::cache_only_on_message_start(
        json!({"input_tokens": 10, "output_tokens": 0, "cache_read_input_tokens": 80, "cache_creation_input_tokens": 4, "cache_creation": {"ephemeral_5m_input_tokens": 4}}),
        json!({"output_tokens": 5}),
        Some(json!({"input_tokens": 10})),
        json!({"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 80, "cache_creation_input_tokens": 4, "cache_creation": {"ephemeral_5m_input_tokens": 4}}),
    )]
    #[case::message_stop_wins_over_message_start(
        json!({"input_tokens": 10, "cache_read_input_tokens": 80}),
        json!({"output_tokens": 5}),
        Some(json!({"cache_read_input_tokens": 100})),
        json!({"output_tokens": 5, "cache_read_input_tokens": 100}),
    )]
    #[case::delta_cache_fields_are_kept(
        json!({"input_tokens": 10, "cache_read_input_tokens": 80}),
        json!({"output_tokens": 5, "cache_read_input_tokens": 7}),
        None,
        json!({"output_tokens": 5, "cache_read_input_tokens": 7}),
    )]
    fn message_delta_usage_is_completed_from_stop_then_start(
        #[case] start: Value,
        #[case] delta: Value,
        #[case] stop: Option<Value>,
        #[case] expected: Value,
    ) {
        let output = promote(vec![
            message_start(start),
            message_delta(delta),
            message_stop(stop.clone()),
        ]);

        assert_eq!(output.len(), 3);
        assert_eq!(output[1], message_delta(expected));
        assert_eq!(output[2], message_stop(stop));
    }

    #[test]
    fn a_delta_is_flushed_with_start_usage_when_the_stream_ends_without_a_stop() {
        let output = promote(vec![
            message_start(json!({"input_tokens": 10, "cache_read_input_tokens": 80})),
            message_delta(json!({"output_tokens": 5})),
        ]);

        assert_eq!(
            output[1],
            message_delta(json!({"output_tokens": 5, "cache_read_input_tokens": 80}))
        );
    }

    #[test]
    fn events_around_the_delta_keep_their_order() {
        let ping = event(json!({"type": "ping"}));
        let output = promote(vec![
            message_delta(json!({"output_tokens": 5})),
            ping.clone(),
            message_stop(None),
        ]);

        assert_eq!(
            output,
            vec![
                message_delta(json!({"output_tokens": 5})),
                ping,
                message_stop(None)
            ]
        );
    }

    #[rstest]
    #[case::metrics_fill_missing_usage(
        json!({"type": "message_stop", "amazon-bedrock-invocationMetrics": {"inputTokenCount": 3, "outputTokenCount": 9}}),
        json!({"type": "message_stop", "usage": {"input_tokens": 3, "output_tokens": 9}}),
    )]
    #[case::the_chunks_own_usage_wins(
        json!({"type": "message_stop", "usage": {"input_tokens": 1}, "amazon-bedrock-invocationMetrics": {"inputTokenCount": 3, "cacheReadInputTokenCount": 40}}),
        json!({"type": "message_stop", "usage": {"cache_read_input_tokens": 40, "input_tokens": 1}}),
    )]
    #[case::no_metrics_leaves_the_chunk(
        json!({"type": "message_stop"}),
        json!({"type": "message_stop"}),
    )]
    #[case::empty_metrics_are_dropped(
        json!({"type": "message_stop", "amazon-bedrock-invocationMetrics": {}}),
        json!({"type": "message_stop"}),
    )]
    fn invocation_metrics_become_anthropic_usage(#[case] chunk: Value, #[case] expected: Value) {
        assert_eq!(with_invocation_usage(chunk), expected);
    }

    fn aws_frame(chunk: &Value) -> Vec<u8> {
        let payload = json!({"bytes": STANDARD.encode(chunk.to_string())});
        let message = Message::new(Bytes::from(serde_json::to_vec(&payload).unwrap())).add_header(
            Header::new(":event-type", HeaderValue::String("chunk".into())),
        );
        let mut wire = Vec::new();
        write_message_to(&message, &mut wire).unwrap();
        wire
    }

    #[tokio::test]
    async fn bedrock_stream_yields_the_sse_an_anthropic_client_reads() {
        let chunks = [
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}),
            json!({"type": "message_stop", "amazon-bedrock-invocationMetrics": {"inputTokenCount": 3, "cacheReadInputTokenCount": 40}}),
        ];
        let wire: Vec<u8> = chunks.iter().flat_map(aws_frame).collect();
        let bytes: ByteStream = futures_util::stream::iter(
            wire.chunks(7)
                .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
                .collect::<Vec<_>>(),
        )
        .boxed();

        let sse = bedrock_anthropic_messages_event_stream(bytes)
            .map_ok(|event| streaming::encode(&event).unwrap())
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .concat();

        let expected: Vec<u8> = [
            message_delta(
                json!({"output_tokens": 5, "cache_read_input_tokens": 40, "input_tokens": 3}),
            ),
            message_stop(Some(
                json!({"input_tokens": 3, "cache_read_input_tokens": 40}),
            )),
        ]
        .iter()
        .flat_map(|event| streaming::encode(event).unwrap())
        .collect();
        assert_eq!(sse, expected);
    }

    fn in_region(region: &str) -> LitellmParams {
        LitellmParams {
            aws: AwsParams {
                aws_region_name: Some(region.into()),
                ..AwsParams::default()
            },
            ..LitellmParams::default()
        }
    }

    fn with_runtime_endpoint(endpoint: &str) -> LitellmParams {
        LitellmParams {
            aws: AwsParams {
                aws_bedrock_runtime_endpoint: Some(endpoint.into()),
                ..in_region("us-west-2").aws
            },
            ..LitellmParams::default()
        }
    }

    #[rstest]
    #[case::invoke_in_the_params_region(
        None,
        in_region("us-west-2"),
        false,
        None,
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3/invoke"
    )]
    #[case::stream_in_the_params_region(
        None,
        in_region("us-west-2"),
        true,
        None,
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3/invoke-with-response-stream"
    )]
    #[case::api_base_outranks_the_params_endpoint(
        Some("https://base.test/"),
        with_runtime_endpoint("https://params.test"),
        false,
        None,
        "https://base.test/model/anthropic.claude-3/invoke"
    )]
    #[case::a_blank_api_base_does_not_hide_the_params_endpoint(
        Some("  "),
        with_runtime_endpoint("https://params.test"),
        false,
        Some("https://env.test"),
        "https://params.test/model/anthropic.claude-3/invoke"
    )]
    #[case::params_endpoint_outranks_the_environment(
        None,
        with_runtime_endpoint("https://params.test"),
        false,
        Some("https://env.test"),
        "https://params.test/model/anthropic.claude-3/invoke"
    )]
    #[case::environment_outranks_the_region_template(
        None,
        in_region("us-west-2"),
        false,
        Some("https://env.test"),
        "https://env.test/model/anthropic.claude-3/invoke"
    )]
    fn url_follows_python_endpoint_precedence_and_the_stream_path(
        #[case] api_base: Option<&str>,
        #[case] litellm_params: LitellmParams,
        #[case] stream: bool,
        #[case] env_endpoint: Option<&str>,
        #[case] expected: &str,
    ) {
        let env = |name: &str| {
            (name == AWS_BEDROCK_RUNTIME_ENDPOINT)
                .then_some(env_endpoint)
                .flatten()
                .map(str::to_string)
        };
        assert_eq!(
            AmazonAnthropicClaudeMessagesConfig
                .get_complete_url(
                    api_base,
                    "anthropic.claude-3",
                    &litellm_params,
                    stream,
                    &env
                )
                .unwrap(),
            expected
        );
    }

    #[test]
    fn sigv4_scope_and_credentials_come_from_the_litellm_params() {
        let litellm_params = LitellmParams {
            aws: AwsParams {
                aws_access_key_id: Some("AKIAPARAMS".into()),
                aws_secret_access_key: Some("params-secret".into()),
                ..in_region("eu-central-1").aws
            },
            ..LitellmParams::default()
        };
        let validated = AmazonAnthropicClaudeMessagesConfig
            .validate_environment(
                Vec::new(),
                None,
                "anthropic.claude-3",
                &litellm_params,
                &|_| None,
            )
            .unwrap();
        let AuthScheme::AwsSigV4 {
            region,
            credentials,
            ..
        } = validated.auth
        else {
            panic!("expected SigV4, got {:?}", validated.auth);
        };
        let AwsCredentialSource::HostSupplied(credentials) = *credentials else {
            panic!("expected the params' static keys, got {credentials:?}");
        };
        assert_eq!(
            (region.as_str(), credentials.access_key_id()),
            ("eu-central-1", "AKIAPARAMS")
        );
    }

    #[rstest]
    #[case::an_explicit_key_is_a_bearer_token(Some("token"), None, Some("token"))]
    #[case::the_env_token_is_a_bearer_token(None, Some("env-token"), Some("env-token"))]
    #[case::no_token_signs_with_sigv4(None, None, None)]
    fn requests_sign_only_without_a_bearer_token(
        #[case] api_key: Option<&str>,
        #[case] env_token: Option<&str>,
        #[case] expected_bearer: Option<&str>,
    ) {
        let env = |name: &str| {
            (name == AWS_BEARER_TOKEN_BEDROCK)
                .then(|| env_token.map(str::to_string))
                .flatten()
        };
        let validated = AmazonAnthropicClaudeMessagesConfig
            .validate_environment(
                vec![("authorization".into(), "Bearer forwarded".into())],
                api_key,
                "anthropic.claude-3",
                &LitellmParams::default(),
                &env,
            )
            .unwrap();
        match (validated.auth, expected_bearer) {
            (
                AuthScheme::Credential {
                    placement: CredentialPlacement::Bearer,
                    secret,
                },
                Some(expected),
            ) => assert_eq!(secret.expose(), expected),
            (
                AuthScheme::AwsSigV4 {
                    region, service, ..
                },
                None,
            ) => {
                assert_eq!(
                    (region.as_str(), service),
                    (DEFAULT_BEDROCK_REGION, BEDROCK_SERVICE)
                );
            }
            (other, _) => panic!("unexpected auth {other:?}"),
        }
    }

    use crate::base_llm::messages::context::{SupportedEffortTiers, ThinkingContext};

    const OPUS: &str = "us.anthropic.claude-opus-5-5";

    /// What Python reads from the model map for Bedrock's Opus 5.5 that the shaping consults,
    /// with sampling params left on so the shared Anthropic stage passes them through as
    /// Python's messages path does.
    fn opus_capabilities() -> MessagesModelCapabilities {
        MessagesModelCapabilities {
            supports_reasoning: true,
            supports_adaptive_thinking: true,
            thinking_always_on: true,
            supports_output_config: true,
            supports_mid_conversation_system: true,
            supports_prompt_cache_ttl: true,
            effort_tiers: SupportedEffortTiers {
                low: true,
                medium: true,
                high: true,
                xhigh: true,
                max: true,
                ..SupportedEffortTiers::default()
            },
            ..MessagesModelCapabilities::default()
        }
    }

    fn context(capabilities: MessagesModelCapabilities) -> MessagesTransformContext {
        MessagesTransformContext {
            thinking: ThinkingContext {
                capabilities,
                budgets: ThinkingBudgets::default(),
            },
            drop_params: false,
        }
    }

    fn wire(
        model: &str,
        fields: Value,
        headers: &[(&str, &str)],
        capabilities: MessagesModelCapabilities,
    ) -> Result<Value, Error> {
        let Value::Object(fields) = fields else {
            unreachable!("cases are objects")
        };
        let request: MessagesRequest = serde_json::from_value(Value::Object(
            [("model".to_string(), json!(model))]
                .into_iter()
                .chain(fields)
                .collect(),
        ))
        .unwrap();
        let headers: Headers = headers
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        let config = AmazonAnthropicClaudeMessagesConfig;
        let transformed =
            config.transform_anthropic_messages_request(request, &context(capabilities))?;
        let transformed = config.request_with_headers(transformed, &headers);
        Ok(config.wire_body(serde_json::to_value(transformed).unwrap()))
    }

    // Expected bodies are Python's `AmazonAnthropicClaudeMessagesConfig().
    // transform_anthropic_messages_request` output for the same input and model.
    #[rstest]
    #[case::plain(
        json!({"max_tokens": 1024, "stream": true, "temperature": 0.5, "metadata": {"user_id": "u1"},
               "messages": [{"role": "user", "content": "Hello"}]}),
        &[],
        json!({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 1024, "temperature": 0.5,
               "metadata": {"user_id": "u1"}, "messages": [{"role": "user", "content": "Hello"}]}),
    )]
    #[case::tools(
        json!({"max_tokens": 1024, "messages": [{"role": "user", "content": "weather?"}], "tool_choice": {"type": "auto"},
               "tools": [
                   {"name": "get_weather", "description": "w", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}, "eager_input_streaming": true},
                   {"input_schema": {"type": "custom", "properties": {"q": {"type": "custom"}}}, "custom": {"defer_loading": true}}
               ]}),
        &[],
        json!({"anthropic_beta": ["fine-grained-tool-streaming-2025-05-14"], "anthropic_version": "bedrock-2023-05-31",
               "max_tokens": 1024, "messages": [{"role": "user", "content": "weather?"}], "tool_choice": {"type": "auto"},
               "tools": [
                   {"name": "get_weather", "description": "w", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}},
                   {"name": "litellm_unnamed_tool_1", "defer_loading": true, "input_schema": {"type": "object", "properties": {"q": {"type": "object"}}}}
               ]}),
    )]
    #[case::thinking_and_cache_control(
        json!({"max_tokens": 4096, "thinking": {"type": "enabled", "budget_tokens": 2048},
               "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral", "scope": "global", "ttl": "1h"}}],
               "messages": [{"role": "user", "content": "Hello"}]}),
        &[],
        json!({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 4096, "messages": [{"role": "user", "content": "Hello"}],
               "output_config": {"effort": "medium"}, "thinking": {"type": "adaptive"},
               "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}),
    )]
    #[case::anthropic_beta_header(
        json!({"max_tokens": 1024, "speed": "fast", "mcp_servers": [{"type": "url", "url": "https://x"}],
               "messages": [{"role": "user", "content": "Hello"}]}),
        &[("anthropic-beta", "context-1m-2025-08-07,claude-code-20250219,unknown-beta-2099-01-01,interleaved-thinking-2025-05-14")],
        json!({"anthropic_beta": ["context-1m-2025-08-07"], "anthropic_version": "bedrock-2023-05-31", "max_tokens": 1024,
               "messages": [{"role": "user", "content": "Hello"}]}),
    )]
    #[case::clear_thinking_becomes_adaptive_thinking(
        json!({"max_tokens": 4096, "context_management": {"edits": [{"type": "clear_thinking_20251015"}, {"type": "clear_tool_uses_20250919"}]},
               "messages": [{"role": "user", "content": "Hello"}]}),
        &[],
        json!({"anthropic_beta": ["context-management-2025-06-27"], "anthropic_version": "bedrock-2023-05-31",
               "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}, "max_tokens": 4096,
               "messages": [{"role": "user", "content": "Hello"}], "output_config": {"effort": "low"}, "thinking": {"type": "adaptive"}}),
    )]
    #[case::safeguards_and_effort(
        json!({"max_tokens": 1024, "safeguards": [{"type": "dangerous_tool_use"}], "output_config": {"effort": "high"},
               "messages": [{"role": "user", "content": "Hello"}]}),
        &[],
        json!({"anthropic_beta": ["dangerous-tool-use-2026-09-03"], "anthropic_version": "bedrock-2023-05-31", "max_tokens": 1024,
               "messages": [{"role": "user", "content": "Hello"}], "output_config": {"effort": "high"},
               "safeguards": [{"type": "dangerous_tool_use"}]}),
    )]
    fn invoke_body_matches_python(
        #[case] fields: Value,
        #[case] headers: &[(&str, &str)],
        #[case] expected: Value,
    ) {
        assert_eq!(
            wire(OPUS, fields, headers, opus_capabilities()),
            Ok(expected)
        );
    }

    #[rstest]
    #[case::comma_separated("context-1m-2025-08-07, unknown-beta")]
    #[case::json_array(r#"["context-1m-2025-08-07", "unknown-beta"]"#)]
    fn caller_betas_parse_both_header_spellings(#[case] header: &str) {
        let betas = caller_betas(&vec![("Anthropic-Beta".into(), header.into())]);
        assert_eq!(betas.to_string(), "context-1m-2025-08-07");
    }

    #[rstest]
    fn body_betas_from_the_client_join_the_header_betas() {
        let body = wire(
            OPUS,
            json!({"max_tokens": 16, "anthropic_beta": ["context-management-2025-06-27"], "messages": [{"role": "user", "content": "hi"}]}),
            &[("anthropic-beta", "context-1m-2025-08-07")],
            opus_capabilities(),
        )
        .unwrap();
        assert_eq!(
            body["anthropic_beta"],
            json!(["context-1m-2025-08-07", "context-management-2025-06-27"])
        );
    }

    #[rstest]
    fn a_server_side_web_search_tool_fails_before_the_call() {
        let error = wire(
            OPUS,
            json!({"max_tokens": 16, "messages": [{"role": "user", "content": "hi"}],
                   "tools": [{"type": "web_search_20250305", "name": "web_search"}]}),
            &[],
            opus_capabilities(),
        )
        .unwrap_err();
        assert!(matches!(
            &error,
            Error::InvalidRequest(ErrorDetail::UnsupportedTool { tool, .. }) if tool == "web_search_20250305"
        ));
    }

    #[rstest]
    #[case::ttl_supported(true, json!({"type": "ephemeral", "ttl": "5m"}))]
    #[case::ttl_unsupported(false, json!({"type": "ephemeral"}))]
    fn cache_control_keeps_ttl_only_where_bedrock_accepts_it(
        #[case] supports_prompt_cache_ttl: bool,
        #[case] expected: Value,
    ) {
        let body = wire(
            OPUS,
            json!({"max_tokens": 16, "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "5m", "scope": "global"}}
            ]}]}),
            &[],
            MessagesModelCapabilities {
                supports_prompt_cache_ttl,
                ..opus_capabilities()
            },
        )
        .unwrap();
        assert_eq!(body["messages"][0]["content"][0]["cache_control"], expected);
    }

    #[rstest]
    #[case::native(
        true,
        json!({"format": {"type": "json_schema", "schema": {"type": "object"}}}),
        json!([{"role": "user", "content": "hi"}]),
    )]
    #[case::inlined(
        false,
        Value::Null,
        json!([{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "text", "text": "{\"type\": \"object\"}"}]}]),
    )]
    fn structured_output_is_native_or_inlined_into_the_last_user_turn(
        #[case] supports_native_structured_output: bool,
        #[case] output_config: Value,
        #[case] messages: Value,
    ) {
        let body = wire(
            OPUS,
            json!({"max_tokens": 16, "messages": [{"role": "user", "content": "hi"}],
                   "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}}}),
            &[],
            MessagesModelCapabilities {
                supports_native_structured_output,
                ..opus_capabilities()
            },
        )
        .unwrap();
        assert_eq!(
            (
                body.get("output_config").cloned().unwrap_or(Value::Null),
                &body["messages"]
            ),
            (output_config, &messages)
        );
    }

    #[rstest]
    #[case::legacy_model_gets_a_budget(
        "anthropic.claude-sonnet-4-20250514-v1:0",
        json!({"type": "enabled", "budget_tokens": 1024})
    )]
    #[case::model_without_extended_thinking_is_left_alone("anthropic.claude-3-haiku", Value::Null)]
    fn clear_thinking_on_non_adaptive_models(#[case] model: &str, #[case] expected: Value) {
        let body = wire(
            model,
            json!({"max_tokens": 4096, "context_management": {"edits": [{"type": "clear_thinking_20251015"}]},
                   "messages": [{"role": "user", "content": "hi"}]}),
            &[],
            MessagesModelCapabilities::default(),
        )
        .unwrap();
        assert_eq!(
            body.get("thinking").cloned().unwrap_or(Value::Null),
            expected
        );
        assert_eq!(body.get("context_management"), None);
    }

    #[rstest]
    fn effort_keys_go_when_the_model_declares_no_output_config() {
        let body = wire(
            "anthropic.claude-3-haiku",
            json!({"max_tokens": 16, "output_config": {"effort": "high"}, "messages": [{"role": "user", "content": "hi"}]}),
            &[],
            MessagesModelCapabilities::default(),
        )
        .unwrap();
        assert_eq!(body.get("output_config"), None);
    }

    #[tokio::test]
    async fn a_full_bedrock_stream_becomes_anthropic_sse() {
        let chunks = [
            json!({"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": "m", "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 3, "output_tokens": 1}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}}),
            json!({"type": "message_stop", "amazon-bedrock-invocationMetrics": {"inputTokenCount": 3, "outputTokenCount": 2}}),
        ];
        let wire: Vec<u8> = chunks.iter().flat_map(aws_frame).collect();
        let bytes: ByteStream = futures_util::stream::iter(
            wire.chunks(5)
                .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
                .collect::<Vec<_>>(),
        )
        .boxed();
        let events = bedrock_anthropic_messages_event_stream(bytes)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let types: Vec<Value> = events
            .iter()
            .map(|event| serde_json::to_value(event).unwrap()["type"].clone())
            .collect();
        assert_eq!(
            Value::Array(types),
            json!([
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ])
        );
        let sse = String::from_utf8(
            events
                .iter()
                .flat_map(|event| streaming::encode(event).unwrap())
                .collect(),
        )
        .unwrap();
        assert!(sse.starts_with("event: message_start\ndata: "));
        assert!(sse.contains("\"text\":\"lo\""));
    }
}
