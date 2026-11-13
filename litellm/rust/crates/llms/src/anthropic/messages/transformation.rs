use litellm_auth::CredentialPlacement;
use litellm_host::interceptors::{RetriedRequest, RetryReason};
use litellm_http::request::{with_header, without_headers};
use litellm_llms_types::{
    formats::messages::{
        ContentBlock, ContentBlockType, ContextEdit, ContextManagement, Message, MessageContent,
        MessagesOptionalParams, MessagesRequest, Speed, ThinkingConfig, ThinkingDisplay,
    },
    providers::anthropic::{AnthropicBeta, BETA_HEADER, BetaProvider, BetaSet, DEFAULT_HEADERS},
    recognized::{Recognized, map_known_runs},
};
use litellm_router_types::LitellmParams;
use serde_json::{Map, Value, json};

use super::{
    handler::shape_anthropic_messages_request,
    scalar::{is_one, python_str, python_value},
    thinking::translate_thinking,
};
use crate::anthropic::beta_headers::accepted_or_unlisted;
use crate::base_llm::messages::{
    context::{MessagesShapeOptions, MessagesTransformContext, UnparsedMessages},
    tools::ToolCustomField,
};
use crate::{
    Error,
    anthropic::common_utils::{
        ANTHROPIC_API_BASE_ENV, ANTHROPIC_API_KEY_ENV, ANTHROPIC_AUTH_TOKEN_ENV,
        ANTHROPIC_BASE_URL_ENV, OauthHandling,
        anthropic_request_errors_on_thinking_binding_mismatch, complete_anthropic_url,
        existing_betas, get_auth_header, has_advisor_tool, has_anthropic_credential,
        is_anthropic_invalid_thinking_block_error, is_anthropic_thinking_binding_mismatch_error,
        is_claude_strict_tools_rejection, is_tool_search_used, map_messages, merge_beta_headers,
        minted_oauth_environment, optionally_handle_anthropic_oauth,
        requires_native_compaction_beta, strip_advisor_blocks, strip_encrypted_reasoning_blocks,
        strip_encrypted_reasoning_blocks_restoring_redacted_data, strip_raw_advisor_blocks,
        strip_raw_encrypted_reasoning_blocks,
        strip_thinking_blocks_from_anthropic_messages_request, tools_without_strict,
    },
    base_llm::{
        auth::AuthScheme,
        messages::transformation::{
            BaseMessagesConfig, FederatedEnvironment, Headers, ValidatedEnvironment,
        },
    },
};

pub struct AnthropicMessagesConfig;

pub const ANTHROPIC_MESSAGES_CONFIG: AnthropicMessagesConfig = AnthropicMessagesConfig;

impl BaseMessagesConfig for AnthropicMessagesConfig {
    fn shape_request(
        &self,
        request: MessagesRequest,
        options: MessagesShapeOptions,
    ) -> Result<MessagesRequest, Error> {
        shape_anthropic_messages_request(request, options)
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _litellm_params: &LitellmParams,
        _stream: bool,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(complete_anthropic_url(api_base, env_lookup))
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        transform_messages_request(request, &context.keeping_display_updates())
    }

    fn tool_custom_field(&self, model: &str) -> ToolCustomField {
        ToolCustomField::Rewrite {
            hoist_defer_loading: is_claude_model(model),
        }
    }

    fn secret_names(&self) -> &'static [&'static str] {
        &[
            ANTHROPIC_API_KEY_ENV,
            ANTHROPIC_AUTH_TOKEN_ENV,
            ANTHROPIC_API_BASE_ENV,
            ANTHROPIC_BASE_URL_ENV,
        ]
    }

    /// Python's `validate_anthropic_messages_environment` up to the beta merge, which
    /// `request_headers` does once the request is final.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        _optional_params: &MessagesOptionalParams,
        _litellm_params: &LitellmParams,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        match anthropic_environment(headers, api_key, env_lookup) {
            FederatedEnvironment::Validated(validated) => Ok(validated),
            FederatedEnvironment::Unauthenticated { missing, .. } => Err(missing),
        }
    }

    fn validate_federated_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        _optional_params: &MessagesOptionalParams,
        _litellm_params: &LitellmParams,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<FederatedEnvironment, Error> {
        Ok(anthropic_environment(headers, api_key, env_lookup))
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_HEADERS
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        merge_beta_headers(
            headers,
            provider_feature_betas(request, BetaProvider::Anthropic),
        )
    }

    fn retry_after_http_error(
        &self,
        status: u16,
        error_text: &str,
        request_body: &Value,
        context: &MessagesTransformContext,
    ) -> Option<RetriedRequest> {
        retry_anthropic_messages_after_http_error(status, error_text, request_body, context)
    }

    fn headers_for_url(&self, headers: Headers, url: &str) -> Headers {
        if !is_foundry_url(url) {
            return headers;
        }
        let betas = existing_betas(&headers);
        if betas.is_empty() {
            return headers;
        }
        let kept = accepted_or_unlisted(betas, BetaProvider::AzureAi);
        if kept.is_empty() {
            return without_headers(headers, &[BETA_HEADER]);
        }
        with_header(headers, BETA_HEADER, kept.to_string())
    }
}

const OUTPUT_CONFIG: &str = "output_config";

const FOUNDRY_HOST_SUFFIXES: [&str; 3] = [
    "services.ai.azure.com",
    "openai.azure.com",
    "cognitiveservices.azure.com",
];

fn is_foundry_url(url: &str) -> bool {
    let absolute = if url.contains("://") {
        url.to_string()
    } else {
        format!("https://{url}")
    };
    url::Url::parse(&absolute)
        .ok()
        .and_then(|parsed| parsed.host_str().map(str::to_ascii_lowercase))
        .is_some_and(|host| {
            FOUNDRY_HOST_SUFFIXES
                .iter()
                .any(|suffix| host == *suffix || host.ends_with(&format!(".{suffix}")))
        })
}

pub(crate) fn is_claude_model(model: &str) -> bool {
    model.to_lowercase().contains("claude")
}

pub fn should_retry_anthropic_messages_on_http_error(
    status: u16,
    error_text: &str,
    request_body: &Value,
    context: &MessagesTransformContext,
) -> bool {
    if status != 400 {
        return false;
    }
    let drops_rejected_strict_tools = context.drops_unsupported_params
        && is_claude_strict_tools_rejection(status, error_text)
        && tools_without_strict(request_body.get("tools")).is_some();
    if drops_rejected_strict_tools {
        return true;
    }
    let caller_asked_for_this_error =
        anthropic_request_errors_on_thinking_binding_mismatch(request_body)
            && is_anthropic_thinking_binding_mismatch_error(error_text);
    !caller_asked_for_this_error && is_anthropic_invalid_thinking_block_error(error_text)
}

pub fn transform_anthropic_messages_request_on_http_error(
    status: u16,
    error_text: &str,
    request_body: Value,
) -> RetriedRequest {
    let mut reasons = Vec::new();
    let mut body = if status == 400 && is_anthropic_invalid_thinking_block_error(error_text) {
        reasons.push(RetryReason::ThinkingSignature);
        strip_thinking_blocks_from_anthropic_messages_request(request_body)
    } else {
        request_body
    };
    let without_strict = if is_claude_strict_tools_rejection(status, error_text) {
        tools_without_strict(body.get("tools"))
    } else {
        None
    };
    if let (Some(without_strict), Value::Object(fields)) = (without_strict, &mut body) {
        reasons.push(RetryReason::StrictTools {
            dropped: without_strict.dropped_count,
        });
        fields.insert("tools".to_owned(), Value::Array(without_strict.tools));
    }
    RetriedRequest { reasons, body }
}

pub fn retry_anthropic_messages_after_http_error(
    status: u16,
    error_text: &str,
    request_body: &Value,
    context: &MessagesTransformContext,
) -> Option<RetriedRequest> {
    should_retry_anthropic_messages_on_http_error(status, error_text, request_body, context).then(
        || {
            transform_anthropic_messages_request_on_http_error(
                status,
                error_text,
                request_body.clone(),
            )
        },
    )
}

fn anthropic_environment(
    headers: Headers,
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> FederatedEnvironment {
    let headers = match optionally_handle_anthropic_oauth(headers, api_key) {
        OauthHandling::Bearer { headers, token } => {
            return FederatedEnvironment::Validated(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Credential {
                    placement: CredentialPlacement::Bearer,
                    secret: token,
                },
            });
        }
        OauthHandling::Untouched(headers) => headers,
    };
    if has_anthropic_credential(&headers) {
        return FederatedEnvironment::Validated(ValidatedEnvironment {
            headers,
            auth: AuthScheme::Forwarded,
        });
    }
    match get_auth_header(api_key, env_lookup) {
        Some(auth) => FederatedEnvironment::Validated(ValidatedEnvironment { headers, auth }),
        None => FederatedEnvironment::Unauthenticated {
            headers,
            missing: Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "Anthropic",
                environment_variable: ANTHROPIC_API_KEY_ENV,
            }),
            federate: minted_oauth_environment,
        },
    }
}

pub(crate) fn transform_messages_request(
    request: MessagesRequest,
    context: &MessagesTransformContext,
) -> Result<MessagesRequest, Error> {
    if request.params.max_tokens.is_none() {
        return Err(Error::MissingField("max_tokens"));
    }
    let request = without_unstructured_output_format(request, context);
    let request = drop_unsupported_params(request, context)?;
    let request = translate_thinking(request, &context.thinking)?;
    let context_management = request
        .params
        .context_management
        .clone()
        .map(map_openai_context_management_to_anthropic);
    let reads_unparsed = context.unparsed.read();
    let messages = if has_advisor_tool(request.params.tools.as_deref())
        || continues_message_thread(&request.params)
    {
        request.messages
    } else if reads_unparsed {
        map_messages(
            request.messages,
            strip_advisor_blocks,
            strip_raw_advisor_blocks,
        )
    } else {
        map_known_runs(request.messages, strip_advisor_blocks)
    };
    let messages = if context.drops_unsupported_params {
        drop_non_string_tool_result_texts(messages)
    } else {
        messages
    };
    Ok(MessagesRequest {
        messages: if reads_unparsed {
            map_messages(
                messages,
                strip_encrypted_reasoning_blocks_restoring_redacted_data,
                strip_raw_encrypted_reasoning_blocks,
            )
        } else {
            map_known_runs(messages, strip_encrypted_reasoning_blocks)
        },
        params: MessagesOptionalParams {
            context_management,
            extra: without_null_safeguards(request.params.extra),
            ..request.params
        },
        ..request
    })
}

fn without_unstructured_output_format(
    request: MessagesRequest,
    context: &MessagesTransformContext,
) -> MessagesRequest {
    if !context.drops_unsupported_params
        || request
            .params
            .output_format
            .as_ref()
            .is_none_or(Value::is_object)
    {
        return request;
    }
    MessagesRequest {
        params: MessagesOptionalParams {
            output_format: None,
            ..request.params
        },
        ..request
    }
}

fn is_python_str(value: &Value) -> bool {
    match value {
        Value::String(text) => {
            litellm_host::raw_json::decode(text).is_none_or(|raw| raw.starts_with('"'))
        }
        _ => false,
    }
}

fn is_non_string_text(part: &Value) -> bool {
    part.get("type").and_then(Value::as_str) == Some("text")
        && part.get("text").is_some_and(|text| !is_python_str(text))
}

fn without_non_string_texts(content: &mut Value) {
    if let Value::Array(parts) = content {
        parts.retain(|part| !is_non_string_text(part));
    }
}

fn drop_non_string_tool_result_texts(
    messages: Vec<Recognized<Message>>,
) -> Vec<Recognized<Message>> {
    messages
        .into_iter()
        .map(|message| match message {
            Recognized::Known(mut message) => {
                if message.role == "user"
                    && let MessageContent::Blocks(blocks) = &mut message.content
                {
                    blocks
                        .iter_mut()
                        .filter(|block| block.is_type(ContentBlockType::ToolResult))
                        .filter_map(|block| block.content.as_mut())
                        .for_each(without_non_string_texts);
                }
                Recognized::Known(message)
            }
            Recognized::Unrecognized(mut message) => {
                if message.get("role").and_then(Value::as_str) == Some("user")
                    && let Some(Value::Array(blocks)) = message.get_mut("content")
                {
                    blocks
                        .iter_mut()
                        .filter(|block| {
                            block.get("type").and_then(Value::as_str) == Some("tool_result")
                        })
                        .filter_map(|block| block.get_mut("content"))
                        .for_each(without_non_string_texts);
                }
                Recognized::Unrecognized(message)
            }
        })
        .collect()
}

fn without_null_safeguards(extra: Map<String, Value>) -> Map<String, Value> {
    extra
        .into_iter()
        .filter(|(key, value)| key != "safeguards" || !value.is_null())
        .collect()
}

pub(crate) fn update_headers_with_anthropic_beta(
    headers: Headers,
    request: &MessagesRequest,
) -> Headers {
    merge_beta_headers(headers, feature_betas(request, UnparsedMessages::Skipped))
}

/// `unparsed` says whether a beta trigger in a message that did not parse counts, as it does
/// for Python, which reads any mapping.
pub(crate) fn update_headers_with_safeguards_beta(
    headers: Headers,
    request: &MessagesRequest,
    provider: BetaProvider,
    unparsed: UnparsedMessages,
) -> Headers {
    merge_beta_headers(
        headers,
        feature_betas(request, unparsed)
            .union(on_provider(mapped_host_betas(&request.params), provider)),
    )
}

/// The betas a request needs on a host with a column in the beta map, filtered and renamed
/// through that column.
pub(crate) fn provider_feature_betas(request: &MessagesRequest, provider: BetaProvider) -> BetaSet {
    on_provider(
        feature_betas(request, UnparsedMessages::Skipped).union(mapped_host_betas(&request.params)),
        provider,
    )
}

fn on_provider(betas: BetaSet, provider: BetaProvider) -> BetaSet {
    betas.iter().filter_map(|beta| beta.on(provider)).collect()
}

fn mapped_host_betas(params: &MessagesOptionalParams) -> BetaSet {
    carries_safeguards(params)
        .then_some(AnthropicBeta::DangerousToolUse20260903)
        .into_iter()
        .collect()
}

fn continues_message_thread(params: &MessagesOptionalParams) -> bool {
    let Some(Value::Object(thread)) = params.extra.get("thread") else {
        return false;
    };
    thread.get("type").and_then(Value::as_str) == Some("continue")
        && thread
            .get("previous_message_id")
            .is_none_or(Value::is_string)
}

fn is_threaded(params: &MessagesOptionalParams) -> bool {
    params
        .extra
        .get("thread")
        .is_some_and(|thread| !thread.is_null())
}

fn feature_betas(request: &MessagesRequest, unparsed: UnparsedMessages) -> BetaSet {
    let params = &request.params;
    let tools = params.tools.as_deref();
    let scanned_messages: &[Recognized<Message>] = if continues_message_thread(params) {
        &[]
    } else {
        &request.messages
    };
    [
        requires_native_compaction_beta(params.compaction.as_ref(), scanned_messages, unparsed)
            .then_some(AnthropicBeta::Compact20260904),
        uses_structured_output(params).then_some(AnthropicBeta::StructuredOutputs20251113),
        (params.speed == Some(Recognized::Known(Speed::Fast)))
            .then_some(AnthropicBeta::FastMode20260201),
        messages_carry_output_config(scanned_messages, unparsed)
            .then_some(AnthropicBeta::PerTurnControl20260701),
        is_threaded(params).then_some(AnthropicBeta::MessageThreads20260812),
        uses_thinking_display_updates(params)
            .then_some(AnthropicBeta::ThinkingDisplayUpdates20260818),
        uses_thinking_block_binding(params)
            .then_some(AnthropicBeta::ThinkingBindingControls20260801),
        messages_carry_tool_changes(scanned_messages, unparsed)
            .then_some(AnthropicBeta::MidConversationToolChanges20260701),
        has_advisor_tool(tools).then_some(AnthropicBeta::AdvisorTool20260301),
        is_tool_search_used(tools).then_some(AnthropicBeta::AdvancedToolUse20251120),
    ]
    .into_iter()
    .flatten()
    .chain(context_management_betas(params.context_management.as_ref()))
    .collect()
}

fn is_compact_edit(edit: &Recognized<ContextEdit>) -> bool {
    matches!(edit, Recognized::Known(ContextEdit::Compact { .. }))
}

fn context_management_betas(
    context_management: Option<&Recognized<ContextManagement>>,
) -> impl Iterator<Item = AnthropicBeta> {
    let edits = context_management
        .and_then(Recognized::known)
        .and_then(|context_management| context_management.edits.as_deref())
        .unwrap_or_default();
    let compact = edits.iter().any(is_compact_edit);
    let other = edits.iter().any(|edit| !is_compact_edit(edit));
    compact
        .then_some(AnthropicBeta::Compact20260112)
        .into_iter()
        .chain(other.then_some(AnthropicBeta::ContextManagement20250627))
}

fn uses_structured_output(params: &MessagesOptionalParams) -> bool {
    params.output_format.is_some()
        || params
            .output_config
            .as_ref()
            .and_then(Recognized::known)
            .is_some_and(|config| config.format.is_some())
}

fn carries_safeguards(params: &MessagesOptionalParams) -> bool {
    params
        .extra
        .get("safeguards")
        .is_some_and(|value| !value.is_null())
}

/// Python's `messages_carry_output_config` reads any mapping, so where unparsed messages are read
/// a message that did not parse, such as a system turn sent without content, counts too.
fn messages_carry_output_config(
    messages: &[Recognized<Message>],
    unparsed: UnparsedMessages,
) -> bool {
    messages.iter().any(|message| match message {
        Recognized::Known(message) => message.extra.contains_key(OUTPUT_CONFIG),
        Recognized::Unrecognized(Value::Object(fields)) => {
            unparsed.read() && fields.contains_key(OUTPUT_CONFIG)
        }
        Recognized::Unrecognized(_) => false,
    })
}

fn uses_thinking_display_updates(params: &MessagesOptionalParams) -> bool {
    let display = match params.thinking.as_ref().and_then(Recognized::known) {
        Some(ThinkingConfig::Enabled(thinking)) => thinking.display.as_ref(),
        Some(ThinkingConfig::Adaptive(thinking)) => thinking.display.as_ref(),
        _ => None,
    };
    display.and_then(Recognized::known) == Some(&ThinkingDisplay::Updates)
}

/// `AnthropicModelInfo.thinking_betas`: any thinking object carrying `block_binding`.
fn uses_thinking_block_binding(params: &MessagesOptionalParams) -> bool {
    match params.thinking.as_ref() {
        Some(Recognized::Known(ThinkingConfig::Enabled(thinking))) => {
            thinking.extra.contains_key("block_binding")
        }
        Some(Recognized::Known(ThinkingConfig::Adaptive(thinking))) => {
            thinking.extra.contains_key("block_binding")
        }
        Some(Recognized::Known(ThinkingConfig::Disabled(thinking))) => {
            thinking.extra.contains_key("block_binding")
        }
        Some(Recognized::Unrecognized(Value::Object(thinking))) => {
            thinking.contains_key("block_binding")
        }
        Some(Recognized::Unrecognized(_)) | None => false,
    }
}

fn is_tool_reference_change(block: &ContentBlock) -> bool {
    matches!(
        &block.block_type,
        Some(ContentBlockType::Other(block_type))
            if block_type == "tool_addition" || block_type == "tool_removal"
    ) && block
        .extra
        .get("tool")
        .and_then(|tool| tool.get("type"))
        .and_then(Value::as_str)
        == Some("tool_reference")
}

fn is_raw_tool_reference_change(block: &Value) -> bool {
    block
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|block_type| block_type == "tool_addition" || block_type == "tool_removal")
        && block
            .get("tool")
            .and_then(|tool| tool.get("type"))
            .and_then(Value::as_str)
            == Some("tool_reference")
}

/// `AnthropicModelInfo.is_mid_conversation_tool_change_used` reads any mapping, so where unparsed
/// messages are read a system message that did not parse is scanned through its raw content too.
fn messages_carry_tool_changes(
    messages: &[Recognized<Message>],
    unparsed: UnparsedMessages,
) -> bool {
    messages.iter().any(|message| match message {
        Recognized::Known(message) => {
            message.role == "system" && message.blocks().iter().any(is_tool_reference_change)
        }
        Recognized::Unrecognized(Value::Object(fields)) => {
            unparsed.read()
                && fields.get("role").and_then(Value::as_str) == Some("system")
                && matches!(
                    fields.get("content"),
                    Some(Value::Array(blocks)) if blocks.iter().any(is_raw_tool_reference_change)
                )
        }
        Recognized::Unrecognized(_) => false,
    })
}

fn unsupported_param(model: &str, param: &str, value: &str, hint: &str) -> Error {
    Error::InvalidRequest(crate::ErrorDetail::UnsupportedParameter {
        model: model.into(),
        param: param.into(),
        value: value.into(),
        hint: hint.into(),
    })
}

/// The speed and sampling gates of Python's `get_requested_anthropic_messages_optional_param`.
pub fn drop_unsupported_params(
    request: MessagesRequest,
    context: &MessagesTransformContext,
) -> Result<MessagesRequest, Error> {
    let capabilities = &context.thinking.capabilities;
    let model = request.model.clone();
    let reject = |param: &str, value: String, hint: &str| -> Result<(), Error> {
        if context.drop_params {
            return Ok(());
        }
        Err(unsupported_param(&model, param, &value, hint))
    };
    let params = request.params;
    let speed = match &params.speed {
        Some(speed) if !capabilities.supports_speed => {
            reject("speed", speed_repr(speed), "")?;
            if context.logs_dropped_params {
                litellm_tracing::warn!(
                    "Dropping unsupported `speed` for model={model} (drop_params=True). Fast mode is only supported on select Opus models."
                );
            }
            None
        }
        _ => params.speed.clone(),
    };
    if capabilities.supports_sampling_params {
        return Ok(MessagesRequest {
            params: MessagesOptionalParams { speed, ..params },
            ..request
        });
    }
    let temperature = match params.temperature {
        Some(temperature) if !is_one(&temperature) => {
            reject(
                "temperature",
                python_str(&temperature),
                "Only temperature=1 is supported. ",
            )?;
            None
        }
        temperature => temperature,
    };
    if let Some(top_p) = &params.top_p {
        reject("top_p", python_str(top_p), "")?;
    }
    if let Some(top_k) = &params.top_k {
        reject("top_k", python_str(top_k), "")?;
    }
    Ok(MessagesRequest {
        params: MessagesOptionalParams {
            speed,
            temperature,
            top_p: None,
            top_k: None,
            ..params
        },
        ..request
    })
}

fn speed_repr(speed: &Recognized<Speed>) -> String {
    match speed {
        Recognized::Known(speed) => format!("'{}'", <&'static str>::from(*speed)),
        Recognized::Unrecognized(other) => litellm_python_compat::repr::repr(&python_value(other)),
    }
}

fn compact_edit_from_openai(entry: &Map<String, Value>) -> Option<ContextEdit> {
    if entry.get("type").and_then(Value::as_str) != Some("compaction") {
        return None;
    }
    let trigger = entry
        .get("compact_threshold")
        .and_then(Value::as_f64)
        .map(|threshold| json!({"type": "input_tokens", "value": threshold as i64}));
    let passthrough = entry
        .iter()
        .filter(|(key, _)| !matches!(key.as_str(), "type" | "compact_threshold"))
        .map(|(key, value)| (key.clone(), value.clone()));
    Some(ContextEdit::Compact {
        extra: trigger
            .map(|trigger| ("trigger".to_string(), trigger))
            .into_iter()
            .chain(passthrough)
            .collect(),
    })
}

/// An OpenAI-style `context_management` list becomes Anthropic `edits` when it holds
/// compaction entries. Anything else, native edits included, is sent as it came.
pub fn map_openai_context_management_to_anthropic(
    context_management: Recognized<ContextManagement>,
) -> Recognized<ContextManagement> {
    let Recognized::Unrecognized(Value::Array(entries)) = &context_management else {
        return context_management;
    };
    let edits: Vec<Recognized<ContextEdit>> = entries
        .iter()
        .filter_map(Value::as_object)
        .filter_map(compact_edit_from_openai)
        .map(Recognized::Known)
        .collect();
    if edits.is_empty() {
        return context_management;
    }
    Recognized::Known(ContextManagement {
        edits: Some(edits),
        extra: Map::new(),
    })
}

#[cfg(test)]
mod tests {
    use crate::base_llm::messages::context::{
        MessagesModelCapabilities, ThinkingBudgets, ThinkingContext,
    };
    use std::process::Command;

    use rstest::{fixture, rstest};

    use super::*;
    use crate::anthropic::common_utils::ENCRYPTED_REASONING_SIGNATURE_PREFIX;
    use litellm_http::request::header_values;

    type Env = &'static [(&'static str, &'static str)];

    const OAUTH_TOKEN: &str = "sk-ant-oat01-token";
    const OAUTH_BEARER: &str = "Bearer sk-ant-oat01-token";
    const OAUTH_BETA: &str = "oauth-2025-04-20";
    const BROWSER_ACCESS: (&str, &str) = ("anthropic-dangerous-direct-browser-access", "true");
    const LOW_BUDGET_ENV: &str = "DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET";
    const PROCESS_ENV_PROBE: &str = "LITELLM_MESSAGES_TRANSFORM_CONTEXT_PROBE";

    fn merged(base: Value, fields: Value) -> Value {
        Value::Object(
            base.as_object()
                .unwrap()
                .clone()
                .into_iter()
                .chain(fields.as_object().unwrap().clone())
                .collect(),
        )
    }

    fn body(fields: Value) -> Value {
        merged(
            json!({
                "model": "claude",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}]
            }),
            fields,
        )
    }

    fn request(fields: Value) -> MessagesRequest {
        serde_json::from_value(body(fields)).unwrap()
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn env(vars: Env) -> impl Fn(&str) -> Option<String> {
        move |name| {
            vars.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    fn headers(pairs: &[(&str, &str)]) -> Headers {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    fn transform(
        fields: Value,
        capabilities: MessagesModelCapabilities,
        drop_params: bool,
    ) -> Result<Value, Error> {
        ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(
                request(fields),
                &MessagesTransformContext::with_lookup(capabilities, drop_params, &no_env),
            )
            .map(|transformed| serde_json::to_value(transformed).unwrap())
    }

    fn advisor_history() -> Value {
        json!([
            {"role": "user", "content": "Build a worker pool."},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me consult the advisor."},
                {"type": "server_tool_use", "id": "srvtoolu_abc123", "name": "advisor", "input": {}},
                {"type": "advisor_tool_result", "tool_use_id": "srvtoolu_abc123", "content": {"type": "advisor_result", "text": "Use channels."}},
                {"type": "text", "text": "Here is the implementation."}
            ]}
        ])
    }

    #[fixture]
    fn unmapped() -> MessagesModelCapabilities {
        MessagesModelCapabilities::default()
    }

    #[fixture]
    fn sampling_removed() -> MessagesModelCapabilities {
        MessagesModelCapabilities {
            supports_sampling_params: false,
            ..Default::default()
        }
    }

    #[fixture]
    fn fast_mode() -> MessagesModelCapabilities {
        MessagesModelCapabilities {
            supports_speed: true,
            ..Default::default()
        }
    }

    #[rstest]
    #[case::alone(json!({"max_tokens": null}))]
    #[case::ahead_of_the_param_gate(json!({"max_tokens": null, "speed": "fast"}))]
    fn missing_max_tokens_is_rejected(#[case] fields: Value, unmapped: MessagesModelCapabilities) {
        assert_eq!(
            transform(fields, unmapped, false),
            Err(Error::MissingField("max_tokens"))
        );
    }

    #[rstest]
    #[case::sampling_params_on_a_sampling_model(
        unmapped(),
        false,
        json!({"temperature": 0.3, "top_p": 0.9, "top_k": 40})
    )]
    #[case::sampling_params_on_a_sampling_model_under_drop_params(
        unmapped(),
        true,
        json!({"temperature": 0.3, "top_p": 0.9, "top_k": 40})
    )]
    #[case::unit_temperature_on_a_sampling_removed_model(
        sampling_removed(),
        false,
        json!({"temperature": 1.0})
    )]
    #[case::unit_temperature_on_a_sampling_removed_model_under_drop_params(
        sampling_removed(),
        true,
        json!({"temperature": 1.0})
    )]
    #[case::speed_on_a_fast_mode_model(fast_mode(), false, json!({"speed": "fast"}))]
    #[case::speed_on_a_fast_mode_model_under_drop_params(fast_mode(), true, json!({"speed": "fast"}))]
    #[case::native_context_management_edits(unmapped(), false, json!({"context_management": {"edits": [{
        "type": "clear_tool_uses_20250919",
        "trigger": {"type": "input_tokens", "value": 30000},
        "keep": {"type": "tool_uses", "value": 3},
        "clear_at_least": {"type": "input_tokens", "value": 5000},
        "exclude_tools": ["web_search"],
        "clear_tool_inputs": false
    }]}}))]
    #[case::first_party_billing_header_system_block(unmapped(), false, json!({"system": [
        {"type": "text", "text": "x-anthropic-billing-header: cc_version=1"},
        {"type": "text", "text": "real system prompt"}
    ]}))]
    #[case::anthropic_signed_reasoning_history(unmapped(), false, json!({"messages": [
        {"role": "user", "content": "Solve it."},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "plan", "signature": "EqQBCkYIAxgCIkA_anthropic_signed"},
            {"type": "redacted_thinking", "data": "EmwKAhgBEgy_anthropic_minted"},
            {"type": "text", "text": "The answer."}
        ]}
    ]}))]
    #[case::advisor_history_alongside_the_advisor_tool(unmapped(), false, json!({
        "messages": advisor_history(),
        "tools": [{"type": "advisor_20260301", "name": "advisor"}]
    }))]
    fn request_is_forwarded_unchanged(
        #[case] capabilities: MessagesModelCapabilities,
        #[case] drop_params: bool,
        #[case] fields: Value,
    ) {
        assert_eq!(
            transform(fields.clone(), capabilities, drop_params),
            Ok(body(fields))
        );
    }

    #[rstest]
    #[case::temperature(sampling_removed(), json!({"temperature": 0.3}), json!({}))]
    #[case::top_p(sampling_removed(), json!({"top_p": 0.9}), json!({}))]
    #[case::top_k(sampling_removed(), json!({"top_k": 40}), json!({}))]
    #[case::every_sampling_param_keeping_the_rest(
        sampling_removed(),
        json!({"temperature": 0.3, "top_p": 0.9, "top_k": 40, "stream": true}),
        json!({"stream": true})
    )]
    #[case::speed_on_a_sampling_model(
        unmapped(),
        json!({"speed": "fast", "temperature": 0.5}),
        json!({"temperature": 0.5})
    )]
    #[case::speed_on_a_sampling_removed_model(
        sampling_removed(),
        json!({"speed": "fast", "temperature": 1.0}),
        json!({"temperature": 1.0})
    )]
    fn removed_params_are_dropped_under_drop_params(
        #[case] capabilities: MessagesModelCapabilities,
        #[case] fields: Value,
        #[case] expected: Value,
    ) {
        assert_eq!(transform(fields, capabilities, true), Ok(body(expected)));
    }

    #[rstest]
    #[case::temperature(
        sampling_removed(),
        json!({"temperature": 0.3}),
        "claude does not support temperature=0.3. Only temperature=1 is supported. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::temperature_just_below_one(
        sampling_removed(),
        json!({"temperature": 0.99}),
        "claude does not support temperature=0.99. Only temperature=1 is supported. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::whole_number_temperature_keeps_its_decimal(
        sampling_removed(),
        json!({"temperature": 2.0}),
        "claude does not support temperature=2.0. Only temperature=1 is supported. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::top_p(
        sampling_removed(),
        json!({"top_p": 0.9}),
        "claude does not support top_p=0.9. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::top_k(
        sampling_removed(),
        json!({"top_k": 5}),
        "claude does not support top_k=5. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::top_k_next_to_unit_temperature(
        sampling_removed(),
        json!({"temperature": 1.0, "top_k": 5}),
        "claude does not support top_k=5. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::temperature_ahead_of_top_k(
        sampling_removed(),
        json!({"temperature": 0.5, "top_k": 5}),
        "claude does not support temperature=0.5. Only temperature=1 is supported. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::top_p_ahead_of_top_k(
        sampling_removed(),
        json!({"top_p": 0.9, "top_k": 5}),
        "claude does not support top_p=0.9. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::speed(
        unmapped(),
        json!({"speed": "fast"}),
        "claude does not support speed='fast'. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::integer_temperature_keeps_no_decimal(
        sampling_removed(),
        json!({"temperature": 0}),
        "claude does not support temperature=0. Only temperature=1 is supported. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::integer_top_p(
        sampling_removed(),
        json!({"top_p": 1}),
        "claude does not support top_p=1. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::non_string_speed_is_reprd(
        unmapped(),
        json!({"speed": 2}),
        "claude does not support speed=2. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    #[case::speed_ahead_of_sampling_params(
        sampling_removed(),
        json!({"speed": "fast", "temperature": 0.5}),
        "claude does not support speed='fast'. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    fn removed_params_are_rejected_without_drop_params(
        #[case] capabilities: MessagesModelCapabilities,
        #[case] fields: Value,
        #[case] message: &str,
    ) {
        let Error::InvalidRequest(detail) = transform(fields, capabilities, false).unwrap_err()
        else {
            panic!("expected an invalid request");
        };
        assert_eq!(detail.to_string(), message);
    }

    #[rstest]
    #[case::string_dropped(json!("json"), true, None)]
    #[case::string_kept_without_drop_params(json!("json"), false, Some(json!("json")))]
    #[case::object_kept(json!({"type": "json_schema"}), true, Some(json!({"type": "json_schema"})))]
    fn a_non_object_output_format_is_dropped_under_drop_params(
        #[case] output_format: Value,
        #[case] drops_unsupported_params: bool,
        #[case] expected: Option<Value>,
    ) {
        let context = MessagesTransformContext {
            drops_unsupported_params,
            ..MessagesTransformContext::with_lookup(unmapped(), false, &no_env)
        };
        let transformed = ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(
                request(json!({"output_format": output_format})),
                &context,
            )
            .unwrap();
        assert_eq!(transformed.params.output_format, expected);
    }

    #[rstest]
    #[case::compaction_threshold(
        json!([{"type": "compaction", "compact_threshold": 200000}]),
        Some(json!({"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 200000}}]}))
    )]
    #[case::other_keys_pass_through(
        json!([{"type": "compaction", "compact_threshold": 150000, "instructions": "Focus on preserving code snippets"}]),
        Some(json!({"edits": [{
            "type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": 150000},
            "instructions": "Focus on preserving code snippets"
        }]}))
    )]
    #[case::float_threshold_is_truncated(
        json!([{"type": "compaction", "compact_threshold": 150000.9}]),
        Some(json!({"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150000}}]}))
    )]
    #[case::compaction_without_threshold(
        json!([{"type": "compaction"}]),
        Some(json!({"edits": [{"type": "compact_20260112"}]}))
    )]
    #[case::non_numeric_threshold_is_dropped(
        json!([{"type": "compaction", "compact_threshold": "150000"}]),
        Some(json!({"edits": [{"type": "compact_20260112"}]}))
    )]
    #[case::non_object_entries_are_skipped(
        json!([42, "compaction", null, [], {"type": "compaction", "compact_threshold": 1000}]),
        Some(json!({"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 1000}}]}))
    )]
    #[case::only_compaction_entries_are_mapped_in_order(
        json!([
            {"type": "compaction", "compact_threshold": 1000},
            {"type": "other", "compact_threshold": 5},
            {"type": "compaction", "instructions": "second"}
        ]),
        Some(json!({"edits": [
            {"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 1000}},
            {"type": "compact_20260112", "instructions": "second"}
        ]}))
    )]
    #[case::list_without_compaction(json!([{"type": "other"}]), None)]
    #[case::empty_list(json!([]), None)]
    #[case::anthropic_edits_pass_through(
        json!({"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150000}}]}),
        None
    )]
    #[case::object_without_edits(json!({"type": "compaction"}), None)]
    #[case::scalar(json!("compaction"), None)]
    fn openai_context_management_maps_to_anthropic_edits(
        #[case] context_management: Value,
        #[case] mapped: Option<Value>,
    ) {
        let parsed: Recognized<ContextManagement> =
            serde_json::from_value(context_management.clone()).unwrap();
        assert_eq!(
            serde_json::to_value(map_openai_context_management_to_anthropic(parsed)).unwrap(),
            mapped.unwrap_or(context_management)
        );
    }

    #[rstest]
    #[case::openai_list_is_mapped(
        json!([{"type": "compaction", "compact_threshold": 200000}]),
        json!({"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 200000}}]})
    )]
    #[case::unmappable_list_is_kept(json!([{"type": "other"}]), json!([{"type": "other"}]))]
    #[case::unmappable_object_is_kept(json!({"type": "other"}), json!({"type": "other"}))]
    fn context_management_reaches_the_wire(
        #[case] context_management: Value,
        #[case] expected: Value,
        unmapped: MessagesModelCapabilities,
    ) {
        assert_eq!(
            transform(
                json!({"context_management": context_management}),
                unmapped,
                false
            ),
            Ok(body(json!({"context_management": expected})))
        );
    }

    #[rstest]
    #[case::without_tools(json!({}))]
    #[case::with_only_other_tools(json!({"tools": [{"name": "get_weather", "input_schema": {"type": "object"}}]}))]
    fn advisor_history_is_stripped_without_the_advisor_tool(
        #[case] tools: Value,
        unmapped: MessagesModelCapabilities,
    ) {
        let stripped = json!([
            {"role": "user", "content": "Build a worker pool."},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me consult the advisor."},
                {"type": "text", "text": "Here is the implementation."}
            ]}
        ]);
        assert_eq!(
            transform(
                merged(tools.clone(), json!({"messages": advisor_history()})),
                unmapped,
                false
            ),
            Ok(body(merged(tools, json!({"messages": stripped}))))
        );
    }

    #[rstest]
    #[case::continue_turn(json!({"type": "continue", "previous_message_id": "msg_1"}), false)]
    #[case::continue_without_a_previous_message_id(json!({"type": "continue"}), false)]
    #[case::continue_with_extra_keys(json!({"type": "continue", "previous_message_id": "msg_1", "note": 1}), false)]
    #[case::create_turn(json!({"type": "create"}), true)]
    #[case::continue_with_a_null_previous_message_id(json!({"type": "continue", "previous_message_id": null}), true)]
    #[case::continue_with_a_numeric_previous_message_id(json!({"type": "continue", "previous_message_id": 1}), true)]
    #[case::not_an_object(json!("continue"), true)]
    fn advisor_history_is_kept_on_a_thread_continue_turn(
        #[case] thread: Value,
        #[case] stripped: bool,
        unmapped: MessagesModelCapabilities,
    ) {
        let messages = if stripped {
            json!([
                {"role": "user", "content": "Build a worker pool."},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me consult the advisor."},
                    {"type": "text", "text": "Here is the implementation."}
                ]}
            ])
        } else {
            advisor_history()
        };
        assert_eq!(
            transform(
                json!({"messages": advisor_history(), "thread": thread.clone()}),
                unmapped,
                false
            ),
            Ok(body(json!({"messages": messages, "thread": thread})))
        );
    }

    #[rstest]
    fn bridge_minted_reasoning_is_stripped_from_the_wire(unmapped: MessagesModelCapabilities) {
        let messages = json!([
            {"role": "user", "content": "Solve it."},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "plan", "signature": format!("{ENCRYPTED_REASONING_SIGNATURE_PREFIX}gAAAA_1")},
                {"type": "redacted_thinking", "data": format!("{ENCRYPTED_REASONING_SIGNATURE_PREFIX}gAAAA_2")},
                {"type": "text", "text": "The answer."}
            ]},
            {"role": "user", "content": "And the next one?"}
        ]);
        assert_eq!(
            transform(json!({"messages": messages}), unmapped, false),
            Ok(body(json!({"messages": [
                {"role": "user", "content": "Solve it."},
                {"role": "assistant", "content": [{"type": "text", "text": "The answer."}]},
                {"role": "user", "content": "And the next one?"}
            ]})))
        );
    }

    #[test]
    fn thinking_is_translated_with_the_context_budgets() {
        let context = MessagesTransformContext::with_lookup(
            MessagesModelCapabilities {
                supports_reasoning: true,
                ..Default::default()
            },
            false,
            &env(&[(LOW_BUDGET_ENV, "2000")]),
        );
        let transformed = ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(
                request(json!({"max_tokens": 4096, "reasoning_effort": "low"})),
                &context,
            )
            .map(|transformed| serde_json::to_value(transformed).unwrap());
        assert_eq!(
            transformed,
            Ok(body(json!({
                "max_tokens": 4096,
                "thinking": {"type": "enabled", "budget_tokens": 2000}
            })))
        );
    }

    #[test]
    fn new_reads_thinking_budgets_from_the_process_environment() {
        if std::env::var_os(PROCESS_ENV_PROBE).is_some() {
            assert_eq!(
                MessagesTransformContext::new(sampling_removed(), true),
                MessagesTransformContext {
                    thinking: ThinkingContext {
                        capabilities: sampling_removed(),
                        budgets: ThinkingBudgets {
                            low: 2000,
                            ..ThinkingBudgets::default()
                        },
                        drop_params: true,
                        keeps_display_updates: false,
                    },
                    drop_params: true,
                    drops_unsupported_params: false,
                    unparsed: UnparsedMessages::Skipped,
                    logs_dropped_params: false,
                }
            );
            return;
        }
        let (_, test_path) = concat!(
            module_path!(),
            "::new_reads_thinking_budgets_from_the_process_environment"
        )
        .split_once("::")
        .unwrap();
        let other_tiers = ["MINIMAL", "MEDIUM", "HIGH", "XHIGH", "MAX"]
            .map(|tier| format!("DEFAULT_REASONING_EFFORT_{tier}_THINKING_BUDGET"));
        let output = other_tiers
            .iter()
            .fold(
                Command::new(std::env::current_exe().unwrap()),
                |mut command, name| {
                    command.env_remove(name);
                    command
                },
            )
            .args([test_path, "--exact"])
            .env(PROCESS_ENV_PROBE, "1")
            .env(LOW_BUDGET_ENV, "2000")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "{stdout}{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[rstest]
    #[case::public_endpoint(None, &[], "https://api.anthropic.com/v1/messages")]
    #[case::base_url_env(
        None,
        &[(ANTHROPIC_BASE_URL_ENV, "https://custom.example.com")],
        "https://custom.example.com/v1/messages"
    )]
    #[case::custom_base(Some("https://proxy.internal"), &[], "https://proxy.internal/v1/messages")]
    #[case::trailing_slash(Some("https://proxy.internal/"), &[], "https://proxy.internal/v1/messages")]
    #[case::complete_endpoint(
        Some("https://proxy.internal/v1/messages"),
        &[],
        "https://proxy.internal/v1/messages"
    )]
    #[case::complete_endpoint_with_trailing_slash(
        Some("https://proxy.internal/v1/messages/"),
        &[],
        "https://proxy.internal/v1/messages"
    )]
    fn complete_url_ends_in_the_messages_path(
        #[case] api_base: Option<&str>,
        #[case] vars: Env,
        #[case] expected: &str,
    ) {
        assert_eq!(
            ANTHROPIC_MESSAGES_CONFIG.get_complete_url(
                api_base,
                "claude",
                &LitellmParams::default(),
                false,
                &env(vars),
            ),
            Ok(expected.to_string())
        );
    }

    fn betas(values: &[&str]) -> BetaSet {
        values.join(",").parse().unwrap()
    }

    fn validated(
        forwarded: &[(&str, &str)],
        api_key: Option<&str>,
        vars: Env,
    ) -> Result<ValidatedEnvironment, Error> {
        ANTHROPIC_MESSAGES_CONFIG.validate_environment(
            headers(forwarded),
            api_key,
            "claude",
            &MessagesOptionalParams::default(),
            &LitellmParams::default(),
            &env(vars),
        )
    }

    fn credential(auth: &AuthScheme) -> Option<(&'static str, &str)> {
        match auth {
            AuthScheme::Credential { placement, secret } => {
                Some((placement.header_name(), secret.expose()))
            }
            AuthScheme::Forwarded => None,
            other => panic!("unexpected auth scheme {other:?}"),
        }
    }

    #[rstest]
    #[case::forwarded_oauth_bearer(
        &[("anthropic-version", "2023-06-01"), ("X-Api-Key", "sk-caller"), ("Authorization", OAUTH_BEARER)],
        Some("sk-deployment"),
        &[("ANTHROPIC_API_KEY", "sk-env")],
        &[("anthropic-version", "2023-06-01"), ("anthropic-beta", OAUTH_BETA), BROWSER_ACCESS],
        Some(("Authorization", OAUTH_TOKEN)),
    )]
    #[case::oauth_api_key(
        &[("x-api-key", OAUTH_TOKEN), ("anthropic-beta", "web-search-2025-03-05")],
        Some(OAUTH_TOKEN),
        &[],
        &[("anthropic-beta", "oauth-2025-04-20,web-search-2025-03-05"), BROWSER_ACCESS],
        Some(("Authorization", OAUTH_TOKEN)),
    )]
    #[case::forwarded_x_api_key_is_kept_over_the_deployment_key(
        &[("X-API-KEY", "caller-key")],
        Some("sk-other"),
        &[("ANTHROPIC_API_KEY", "sk-env")],
        &[("X-API-KEY", "caller-key")],
        None,
    )]
    #[case::forwarded_non_oauth_bearer_is_kept(
        &[("Authorization", "Bearer some-proxy-token")],
        Some("sk-ant-api03-regular"),
        &[],
        &[("Authorization", "Bearer some-proxy-token")],
        None,
    )]
    #[case::oauth_token_without_the_bearer_scheme_is_kept(
        &[("authorization", OAUTH_TOKEN)],
        None,
        &[],
        &[("authorization", OAUTH_TOKEN)],
        None,
    )]
    #[case::api_key_param(
        &[("anthropic-beta", "web-search-2025-03-05")],
        Some("sk-param"),
        &[("ANTHROPIC_API_KEY", "sk-env"), ("ANTHROPIC_AUTH_TOKEN", "env-token")],
        &[("anthropic-beta", "web-search-2025-03-05")],
        Some(("x-api-key", "sk-param")),
    )]
    #[case::env_key_when_the_param_is_blank(
        &[],
        Some("  "),
        &[("ANTHROPIC_API_KEY", "sk-env"), ("ANTHROPIC_AUTH_TOKEN", "env-token")],
        &[],
        Some(("x-api-key", "sk-env")),
    )]
    #[case::auth_token_when_no_key_is_set(
        &[],
        None,
        &[("ANTHROPIC_API_KEY", " \t"), ("ANTHROPIC_AUTH_TOKEN", "env-token")],
        &[],
        Some(("Authorization", "env-token")),
    )]
    #[case::oauth_env_key_as_a_bearer(
        &[],
        None,
        &[("ANTHROPIC_API_KEY", "sk-ant-oat01-env")],
        &[],
        Some(("Authorization", "sk-ant-oat01-env")),
    )]
    fn validate_environment_shapes_the_headers_and_names_the_credential(
        #[case] forwarded: &[(&str, &str)],
        #[case] api_key: Option<&str>,
        #[case] vars: Env,
        #[case] expected_headers: &[(&str, &str)],
        #[case] expected_credential: Option<(&str, &str)>,
    ) {
        let environment = validated(forwarded, api_key, vars).unwrap();
        assert_eq!(environment.headers, headers(expected_headers));
        assert_eq!(credential(&environment.auth), expected_credential);
    }

    #[rstest]
    #[case::no_credentials(&[], None, &[])]
    #[case::empty_api_key(&[], Some(""), &[])]
    #[case::whitespace_only_env_values(&[], None, &[("ANTHROPIC_API_KEY", "  "), ("ANTHROPIC_AUTH_TOKEN", " \t")])]
    #[case::unrelated_forwarded_headers(&[("anthropic-beta", "web-search-2025-03-05")], None, &[])]
    fn missing_credentials_are_an_auth_error(
        #[case] forwarded: &[(&str, &str)],
        #[case] api_key: Option<&str>,
        #[case] vars: Env,
    ) {
        assert!(matches!(
            validated(forwarded, api_key, vars),
            Err(Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "Anthropic",
                environment_variable: "ANTHROPIC_API_KEY",
            }))
        ));
    }

    #[rstest]
    #[case::no_features(json!({}), &[])]
    #[case::output_format(json!({"output_format": {"type": "json_schema"}}), &["structured-outputs-2025-11-13"])]
    #[case::null_output_format(json!({"output_format": null}), &[])]
    #[case::output_config_format(
        json!({"output_config": {"format": {"type": "json_schema"}, "effort": "xhigh"}}),
        &["structured-outputs-2025-11-13"]
    )]
    #[case::null_output_config_format(json!({"output_config": {"format": null}}), &[])]
    #[case::top_level_output_config_without_format(json!({"output_config": {"effort": "high"}}), &[])]
    #[case::fast_speed(json!({"speed": "fast"}), &["fast-mode-2026-02-01"])]
    #[case::standard_speed(json!({"speed": "standard"}), &[])]
    #[case::unknown_speed(json!({"speed": "turbo"}), &[])]
    #[case::compaction_param(json!({"compaction": {"enabled": true}}), &["compact-2026-09-04"])]
    #[case::empty_compaction_param(json!({"compaction": {}}), &["compact-2026-09-04"])]
    #[case::signed_compaction_block_in_history(
        json!({"messages": [
            {"role": "assistant", "content": [{"type": "compaction", "content": "summary", "signature": "sig"}]},
            {"role": "user", "content": "Continue"},
        ]}),
        &["compact-2026-09-04"]
    )]
    #[case::unsigned_compaction_block_in_history(
        json!({"messages": [
            {"role": "assistant", "content": [{"type": "compaction", "content": "summary", "signature": ""}]},
            {"role": "user", "content": "Continue"},
        ]}),
        &[]
    )]
    #[case::advisor_tool(
        json!({"tools": [{"type": "advisor_20260301", "name": "advisor", "model": "claude-opus-4-6"}]}),
        &["advisor-tool-2026-03-01"]
    )]
    #[case::no_tools(json!({"tools": []}), &[])]
    #[case::regex_tool_search(
        json!({"tools": [{"type": "tool_search_tool_regex_20251119"}]}),
        &["advanced-tool-use-2025-11-20"]
    )]
    #[case::bm25_tool_search(
        json!({"tools": [{"type": "tool_search_tool_bm25_20251119"}]}),
        &["advanced-tool-use-2025-11-20"]
    )]
    #[case::unrelated_server_tool(json!({"tools": [{"type": "web_search_20250305", "name": "web_search"}]}), &[])]
    #[case::only_compact_edits(
        json!({"context_management": {"edits": [{"type": "compact_20260112"}]}}),
        &["compact-2026-01-12"]
    )]
    #[case::only_other_edits(
        json!({"context_management": {"edits": [{"type": "clear_tool_uses_20250919", "keep": {"type": "tool_uses", "value": 3}}]}}),
        &["context-management-2025-06-27"]
    )]
    #[case::compact_and_other_edits(
        json!({"context_management": {"edits": [{"type": "compact_20260112"}, {"type": "clear_tool_uses_20250919"}]}}),
        &["compact-2026-01-12", "context-management-2025-06-27"]
    )]
    #[case::edit_without_a_type(json!({"context_management": {"edits": [{}]}}), &["context-management-2025-06-27"])]
    #[case::unknown_edit_type(json!({"context_management": {"edits": [{"type": "future"}]}}), &["context-management-2025-06-27"])]
    #[case::empty_edits(json!({"context_management": {"edits": []}}), &[])]
    #[case::context_management_without_edits(json!({"context_management": {}}), &[])]
    #[case::unmapped_openai_context_management(json!({"context_management": [{"type": "other"}]}), &[])]
    #[case::per_message_output_config(
        json!({"messages": [{"role": "user", "content": "hi", "output_config": {"effort": "low"}}]}),
        &["per-turn-control-2026-07-01"]
    )]
    #[case::per_message_null_output_config(
        json!({"messages": [{"role": "user", "content": "hi", "output_config": null}]}),
        &["per-turn-control-2026-07-01"]
    )]
    #[case::contentless_system_output_config(
        json!({"messages": [
            {"role": "", "content": "Hello"},
            {"role": "system", "output_config": {"effort": "high"}},
        ]}),
        &["per-turn-control-2026-07-01"]
    )]
    #[case::unparsed_message_output_config(
        json!({"messages": [{"role": "user", "content": 5, "output_config": {"effort": "low"}}]}),
        &["per-turn-control-2026-07-01"]
    )]
    #[case::unparsed_signed_compaction_block_in_history(
        json!({"messages": [
            {"role": "assistant", "content": [{"type": "text", "text": 5}, {"type": "compaction", "content": "summary", "signature": "sig"}]},
            {"role": "user", "content": "Continue"},
        ]}),
        &["compact-2026-09-04"]
    )]
    #[case::unparsed_unsigned_compaction_block_in_history(
        json!({"messages": [
            {"role": "assistant", "content": [{"type": "text", "text": 5}, {"type": "compaction", "content": "summary", "signature": ""}]},
        ]}),
        &[]
    )]
    #[case::unparsed_system_tool_addition(
        json!({"messages": [
            {"role": "user", "content": "Use the ping tool"},
            {"role": "system", "content": [{"type": "text", "text": 5}, {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "ping"}}]},
        ]}),
        &["mid-conversation-tool-changes-2026-07-01"]
    )]
    #[case::unparsed_user_tool_addition(
        json!({"messages": [
            {"role": "user", "content": [{"type": "text", "text": 5}, {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "ping"}}]},
        ]}),
        &[]
    )]
    #[case::safeguards(
        json!({"safeguards": [{"type": "dangerous_tool_use", "classifier_context": {"v": 1, "permission_mode": "auto"}}]}),
        &[]
    )]
    #[case::adaptive_display_updates(
        json!({"thinking": {"type": "adaptive", "display": "updates"}}),
        &["thinking-display-updates-2026-08-18"]
    )]
    #[case::enabled_display_updates(
        json!({"thinking": {"type": "enabled", "budget_tokens": 2048, "display": "updates"}}),
        &["thinking-display-updates-2026-08-18"]
    )]
    #[case::summarized_display(json!({"thinking": {"type": "adaptive", "display": "summarized"}}), &[])]
    #[case::disabled_thinking(json!({"thinking": {"type": "disabled"}}), &[])]
    #[case::system_tool_addition(
        json!({"messages": [
            {"role": "user", "content": "Use the ping tool"},
            {"role": "system", "content": [{"type": "tool_addition", "tool": {"type": "tool_reference", "name": "ping"}}]},
        ]}),
        &["mid-conversation-tool-changes-2026-07-01"]
    )]
    #[case::system_tool_removal(
        json!({"messages": [
            {"role": "system", "content": [{"type": "tool_removal", "tool": {"type": "tool_reference", "name": "ping"}}]},
        ]}),
        &["mid-conversation-tool-changes-2026-07-01"]
    )]
    #[case::user_tool_addition(
        json!({"messages": [
            {"role": "user", "content": [{"type": "tool_addition", "tool": {"type": "tool_reference", "name": "ping"}}]},
        ]}),
        &[]
    )]
    #[case::tool_addition_without_a_tool_reference(
        json!({"messages": [
            {"role": "system", "content": [{"type": "tool_addition", "tool": {"type": "tool_definition"}}]},
        ]}),
        &[]
    )]
    #[case::thread_create(json!({"thread": {"type": "create"}}), &["message-threads-2026-08-12"])]
    #[case::thread_continue(
        json!({"thread": {"type": "continue", "previous_message_id": "msg_1"}}),
        &["message-threads-2026-08-12"]
    )]
    #[case::thread_of_any_shape(json!({"thread": "opaque"}), &["message-threads-2026-08-12"])]
    #[case::null_thread(json!({"thread": null}), &[])]
    #[case::thread_create_still_scans_history(
        json!({
            "thread": {"type": "create"},
            "messages": [
                {"role": "assistant", "content": [{"type": "compaction", "content": "summary", "signature": "sig"}]},
                {"role": "user", "content": "hi", "output_config": {"effort": "low"}},
                {"role": "system", "content": [{"type": "tool_addition", "tool": {"type": "tool_reference", "name": "ping"}}]},
            ],
        }),
        &[
            "compact-2026-09-04",
            "message-threads-2026-08-12",
            "mid-conversation-tool-changes-2026-07-01",
            "per-turn-control-2026-07-01",
        ]
    )]
    #[case::thread_continue_skips_history(
        json!({
            "thread": {"type": "continue", "previous_message_id": "msg_1"},
            "messages": [
                {"role": "assistant", "content": [{"type": "compaction", "content": "summary", "signature": "sig"}]},
                {"role": "user", "content": "hi", "output_config": {"effort": "low"}},
                {"role": "system", "content": [{"type": "tool_addition", "tool": {"type": "tool_reference", "name": "ping"}}]},
            ],
        }),
        &["message-threads-2026-08-12"]
    )]
    #[case::thread_continue_keeps_param_betas(
        json!({
            "thread": {"type": "continue", "previous_message_id": "msg_1"},
            "compaction": {},
            "speed": "fast",
        }),
        &["compact-2026-09-04", "fast-mode-2026-02-01", "message-threads-2026-08-12"]
    )]
    #[case::malformed_continue_scans_history(
        json!({
            "thread": {"type": "continue", "previous_message_id": 7},
            "messages": [{"role": "user", "content": "hi", "output_config": {"effort": "low"}}],
        }),
        &["message-threads-2026-08-12", "per-turn-control-2026-07-01"]
    )]
    fn feature_betas_follow_the_request(#[case] fields: Value, #[case] expected: &[&str]) {
        assert_eq!(
            feature_betas(&request(fields), UnparsedMessages::Read),
            betas(expected)
        );
    }

    #[rstest]
    #[case::contentless_system_output_config(json!({"messages": [
        {"role": "", "content": "Hello"},
        {"role": "system", "output_config": {"effort": "high"}},
    ]}))]
    #[case::unparsed_message_output_config(
        json!({"messages": [{"role": "user", "content": 5, "output_config": {"effort": "low"}}]})
    )]
    #[case::unparsed_signed_compaction_block_in_history(json!({"messages": [
        {"role": "assistant", "content": [{"type": "text", "text": 5}, {"type": "compaction", "content": "summary", "signature": "sig"}]},
        {"role": "user", "content": "Continue"},
    ]}))]
    #[case::unparsed_system_tool_addition(json!({"messages": [
        {"role": "user", "content": "Use the ping tool"},
        {"role": "system", "content": [{"type": "text", "text": 5}, {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "ping"}}]},
    ]}))]
    fn a_beta_trigger_in_an_unparsed_message_counts_only_where_unparsed_messages_are_read(
        #[case] fields: Value,
    ) {
        let request = request(fields);
        assert_eq!(
            feature_betas(&request, UnparsedMessages::Skipped),
            BetaSet::default()
        );
        assert_eq!(
            provider_feature_betas(&request, BetaProvider::Anthropic),
            BetaSet::default()
        );
        assert_ne!(
            feature_betas(&request, UnparsedMessages::Read),
            BetaSet::default()
        );
    }

    #[rstest]
    #[case::anthropic(BetaProvider::Anthropic, json!([{"type": "dangerous_tool_use"}]), &["dangerous-tool-use-2026-09-03"])]
    #[case::vertex_ai(BetaProvider::VertexAi, json!([{"type": "dangerous_tool_use"}]), &["dangerous-tool-use-2026-09-03"])]
    #[case::empty_list(BetaProvider::Anthropic, json!([]), &["dangerous-tool-use-2026-09-03"])]
    #[case::not_a_list(BetaProvider::Anthropic, json!(1), &["dangerous-tool-use-2026-09-03"])]
    #[case::null(BetaProvider::Anthropic, Value::Null, &[])]
    #[case::bedrock_converse_has_a_null_cell(BetaProvider::BedrockConverse, json!([{"type": "dangerous_tool_use"}]), &[])]
    fn safeguards_add_the_dangerous_tool_use_beta_where_the_beta_map_has_a_cell(
        #[case] provider: BetaProvider,
        #[case] safeguards: Value,
        #[case] expected: &[&str],
    ) {
        assert_eq!(
            provider_feature_betas(&request(json!({"safeguards": safeguards})), provider),
            betas(expected)
        );
    }

    #[rstest]
    #[case::vertex_ai(
        BetaProvider::VertexAi,
        &["mid-conversation-tool-changes-2026-07-01", "thinking-display-updates-2026-08-18"]
    )]
    #[case::azure_ai(
        BetaProvider::AzureAi,
        &["mid-conversation-tool-changes-2026-07-01", "thinking-display-updates-2026-08-18"]
    )]
    #[case::bedrock_converse(BetaProvider::BedrockConverse, &["thinking-display-updates-2026-08-18"])]
    #[case::databricks(BetaProvider::Databricks, &["thinking-display-updates-2026-08-18"])]
    fn tool_change_and_display_betas_reach_the_providers_that_accept_them(
        #[case] provider: BetaProvider,
        #[case] expected: &[&str],
    ) {
        let fields = json!({
            "thinking": {"type": "adaptive", "display": "updates"},
            "messages": [
                {"role": "system", "content": [{"type": "tool_addition", "tool": {"type": "tool_reference", "name": "ping"}}]},
            ],
        });
        assert_eq!(
            provider_feature_betas(&request(fields), provider),
            betas(expected)
        );
    }

    #[rstest]
    fn hosts_without_a_beta_map_column_get_safeguards_without_a_derived_beta() {
        let client = headers(&[("anthropic-beta", "claude-code-20250219")]);
        assert_eq!(
            update_headers_with_anthropic_beta(
                client.clone(),
                &request(json!({"safeguards": [{"type": "dangerous_tool_use"}]}))
            ),
            client
        );
    }

    #[rstest]
    #[case::no_betas(&[("x-api-key", "k"), ("anthropic-version", "2023-06-01")], json!({}), &[("x-api-key", "k"), ("anthropic-version", "2023-06-01")])]
    #[case::blank_beta_header(&[("Anthropic-Beta", " , "), ("x-api-key", "k")], json!({}), &[("Anthropic-Beta", " , "), ("x-api-key", "k")])]
    #[case::feature_beta_is_appended(
        &[("x-api-key", "k")],
        json!({"speed": "fast"}),
        &[("x-api-key", "k"), ("anthropic-beta", "fast-mode-2026-02-01")],
    )]
    #[case::existing_betas_are_normalized_without_features(
        &[("Anthropic-Beta", "web-search-2025-03-05, interleaved-thinking-2025-05-14 ,web-search-2025-03-05"), ("x-api-key", "k")],
        json!({}),
        &[("x-api-key", "k"), ("anthropic-beta", "interleaved-thinking-2025-05-14,web-search-2025-03-05")],
    )]
    #[case::existing_advisor_beta_is_kept_without_an_advisor_tool(
        &[("anthropic-beta", "advisor-tool-2026-03-01")],
        json!({"tools": []}),
        &[("anthropic-beta", "advisor-tool-2026-03-01")],
    )]
    #[case::feature_already_sent_is_not_duplicated(
        &[("anthropic-beta", "fast-mode-2026-02-01")],
        json!({"speed": "fast"}),
        &[("anthropic-beta", "fast-mode-2026-02-01")],
    )]
    #[case::differently_cased_beta_header_is_replaced_by_one_sorted_header(
        &[("Anthropic-Beta", "interleaved-thinking-2025-05-14")],
        json!({"messages": [{"role": "system", "content": "env", "output_config": {"effort": "low"}}]}),
        &[("anthropic-beta", "interleaved-thinking-2025-05-14,per-turn-control-2026-07-01")],
    )]
    #[case::every_beta_header_casing_is_unioned_into_one_header(
        &[("anthropic-beta", "interleaved-thinking-2025-05-14"), ("Anthropic-Beta", "web-search-2025-03-05")],
        json!({"speed": "fast"}),
        &[("anthropic-beta", "fast-mode-2026-02-01,interleaved-thinking-2025-05-14,web-search-2025-03-05")],
    )]
    #[case::unknown_client_betas_survive_alongside_the_added_one(
        &[("anthropic-beta", "claude-code-20250219,interleaved-thinking-2025-05-14,context-management-2025-06-27,per-turn-control-2026-07-01,effort-2025-11-24")],
        json!({"messages": [{"role": "user", "content": "hi", "output_config": {"effort": "low"}}]}),
        &[("anthropic-beta", "claude-code-20250219,context-management-2025-06-27,effort-2025-11-24,interleaved-thinking-2025-05-14,per-turn-control-2026-07-01")],
    )]
    #[case::safeguards_add_the_dangerous_tool_use_beta_to_the_client_betas(
        &[("anthropic-beta", "claude-code-20250219,interleaved-thinking-2025-05-14")],
        json!({"safeguards": [{"type": "dangerous_tool_use"}]}),
        &[("anthropic-beta", "claude-code-20250219,dangerous-tool-use-2026-09-03,interleaved-thinking-2025-05-14")],
    )]
    fn request_headers_merge_the_feature_betas(
        #[case] input: &[(&str, &str)],
        #[case] fields: Value,
        #[case] expected: &[(&str, &str)],
    ) {
        assert_eq!(
            ANTHROPIC_MESSAGES_CONFIG.request_headers(headers(input), &request(fields)),
            headers(expected)
        );
    }

    #[test]
    fn every_feature_merges_with_the_oauth_beta_sorted() {
        let environment = validated(&[], Some(OAUTH_TOKEN), &[]).unwrap();
        let all_features = request(json!({
            "compaction": {"enabled": true},
            "output_format": {"type": "json_schema"},
            "speed": "fast",
            "tools": [{"type": "advisor_20260301"}, {"type": "tool_search_tool_bm25_20251119"}],
            "context_management": {"edits": [{"type": "compact_20260112"}, {"type": "clear_thinking_20251015"}]},
            "messages": [{"role": "user", "content": "hi", "output_config": {"effort": "low"}}],
        }));
        assert_eq!(
            ANTHROPIC_MESSAGES_CONFIG.request_headers(environment.headers, &all_features),
            headers(&[
                BROWSER_ACCESS,
                (
                    "anthropic-beta",
                    "advanced-tool-use-2025-11-20,advisor-tool-2026-03-01,compact-2026-01-12,compact-2026-09-04,context-management-2025-06-27,fast-mode-2026-02-01,oauth-2025-04-20,per-turn-control-2026-07-01,structured-outputs-2025-11-13"
                ),
            ])
        );
        assert_eq!(
            credential(&environment.auth),
            Some(("Authorization", OAUTH_TOKEN))
        );
    }

    #[test]
    fn default_headers_match_anthropic() {
        assert_eq!(
            ANTHROPIC_MESSAGES_CONFIG.default_headers(),
            &[
                ("anthropic-version", "2023-06-01"),
                ("content-type", "application/json"),
            ]
        );
    }

    #[test]
    fn secret_names_cover_every_credential_and_base_lookup() {
        let requested = std::cell::RefCell::new(Vec::<String>::new());
        let record = |name: &str| -> Option<String> {
            requested.borrow_mut().push(name.to_string());
            None
        };
        let _ = ANTHROPIC_MESSAGES_CONFIG.validate_environment(
            Vec::new(),
            None,
            "claude",
            &MessagesOptionalParams::default(),
            &LitellmParams::default(),
            &record,
        );
        let _ = ANTHROPIC_MESSAGES_CONFIG.get_complete_url(
            None,
            "claude",
            &LitellmParams::default(),
            false,
            &record,
        );
        let requested = requested.into_inner();
        assert!(!requested.is_empty());
        let undeclared: Vec<&String> = requested
            .iter()
            .filter(|name| {
                !ANTHROPIC_MESSAGES_CONFIG
                    .secret_names()
                    .contains(&name.as_str())
            })
            .collect();
        assert_eq!(undeclared, Vec::<&String>::new());
    }

    const FOUNDRY: &str = "https://team.services.ai.azure.com/anthropic/v1/messages";

    #[rstest]
    #[case::drops_a_beta_foundry_lacks(
        FOUNDRY,
        "advisor-tool-2026-03-01,context-1m-2025-08-07",
        Some("context-1m-2025-08-07")
    )]
    #[case::forwards_unknown_betas_sorted(
        FOUNDRY,
        "zzz,files-api-2025-04-14,aaa",
        Some("aaa,files-api-2025-04-14,zzz")
    )]
    #[case::removes_the_header_when_nothing_is_left(FOUNDRY, "advisor-tool-2026-03-01", None)]
    #[case::cognitive_services_host(
        "team.cognitiveservices.azure.com/anthropic",
        "advisor-tool-2026-03-01",
        None
    )]
    #[case::anthropic_host_is_untouched(
        "https://api.anthropic.com/v1/messages",
        "advisor-tool-2026-03-01",
        Some("advisor-tool-2026-03-01")
    )]
    #[case::lookalike_host_is_untouched(
        "https://notservices.ai.azure.com.example/v1/messages",
        "advisor-tool-2026-03-01",
        Some("advisor-tool-2026-03-01")
    )]
    fn foundry_urls_filter_betas_through_the_azure_ai_row(
        #[case] url: &str,
        #[case] betas: &str,
        #[case] expected: Option<&str>,
    ) {
        let filtered = ANTHROPIC_MESSAGES_CONFIG
            .headers_for_url(headers(&[("x-a", "1"), (BETA_HEADER, betas)]), url);
        assert_eq!(
            header_values(&filtered, BETA_HEADER).collect::<Vec<_>>(),
            expected.into_iter().collect::<Vec<_>>()
        );
        assert_eq!(header_values(&filtered, "x-a").collect::<Vec<_>>(), ["1"]);
    }

    #[rstest]
    fn foundry_url_without_betas_keeps_the_headers() {
        let sent = headers(&[("x-a", "1")]);
        assert_eq!(
            ANTHROPIC_MESSAGES_CONFIG.headers_for_url(sent.clone(), FOUNDRY),
            sent
        );
    }

    #[test]
    fn the_http_error_retry_matches_the_cases_shared_with_python() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/anthropic_messages_http_error_retry.json"
        )))
        .unwrap();
        let table = &fixture["http_error_retry"];
        let cases = table["cases"].as_array().unwrap();
        assert!(!cases.is_empty());
        let mismatched: Vec<&str> = cases
            .iter()
            .filter(|case| {
                let mut body = table["request"].clone();
                body["thinking"] = case["thinking"].clone();
                let status = u16::try_from(case["status"].as_u64().unwrap()).unwrap();
                let error_text = case["error_text"].as_str().unwrap();
                let retried = should_retry_anthropic_messages_on_http_error(
                    status,
                    error_text,
                    &body,
                    &MessagesTransformContext::default(),
                );
                let stripped = case["stripped"].as_bool().unwrap();
                let transformed = transform_anthropic_messages_request_on_http_error(
                    status,
                    error_text,
                    body.clone(),
                );
                let expected = if stripped {
                    RetriedRequest {
                        reasons: vec![RetryReason::ThinkingSignature],
                        body: table["stripped_request"].clone(),
                    }
                } else {
                    RetriedRequest {
                        reasons: Vec::new(),
                        body,
                    }
                };
                retried != case["retried"].as_bool().unwrap() || transformed != expected
            })
            .map(|case| case["name"].as_str().unwrap())
            .collect();
        assert_eq!(mismatched, Vec::<&str>::new());
    }

    #[test]
    fn the_strict_tools_retry_matches_the_cases_shared_with_python() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/anthropic_messages_http_error_retry.json"
        )))
        .unwrap();
        let replay = &fixture["http_error_retry"];
        let table = &fixture["strict_tools_retry"];
        let cases = table["cases"].as_array().unwrap();
        assert!(!cases.is_empty());
        let mismatched: Vec<&str> = cases
            .iter()
            .filter(|case| {
                let mut body = replay["request"].clone();
                body["thinking"] = case["thinking"].clone();
                body["tools"] = table["tools"].clone();
                let status = u16::try_from(case["status"].as_u64().unwrap()).unwrap();
                let error_text = case["error_text"].as_str().unwrap();
                let context = MessagesTransformContext {
                    drops_unsupported_params: case["drops_unsupported_params"].as_bool().unwrap(),
                    ..MessagesTransformContext::default()
                };
                let mut reasons = Vec::new();
                let mut expected = if case["stripped"].as_bool().unwrap() {
                    reasons.push(RetryReason::ThinkingSignature);
                    replay["stripped_request"].clone()
                } else {
                    body.clone()
                };
                expected["tools"] = if case["strict_dropped"].as_bool().unwrap() {
                    reasons.push(RetryReason::StrictTools {
                        dropped: usize::try_from(table["dropped"].as_u64().unwrap()).unwrap(),
                    });
                    table["tools_without_strict"].clone()
                } else {
                    table["tools"].clone()
                };
                let retried = should_retry_anthropic_messages_on_http_error(
                    status, error_text, &body, &context,
                );
                let transformed =
                    transform_anthropic_messages_request_on_http_error(status, error_text, body);
                retried != case["retried"].as_bool().unwrap()
                    || transformed
                        != RetriedRequest {
                            reasons,
                            body: expected,
                        }
            })
            .map(|case| case["name"].as_str().unwrap())
            .collect();
        assert_eq!(mismatched, Vec::<&str>::new());
    }
}
