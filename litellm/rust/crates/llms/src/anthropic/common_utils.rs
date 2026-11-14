use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use crate::base_llm::messages::context::{MessagesModelCapabilities, UnparsedMessages};
use fancy_regex::Regex;
use litellm_auth::CredentialPlacement;
use litellm_secret_value::SecretValue;

use litellm_core_utils::settings::resolve_non_empty;
use litellm_http::request::{
    has_header, header_value, header_values, with_header, without_headers,
};
use litellm_llms_types::{
    formats::messages::{
        ContentBlock, ContentBlockType, EffortLevel, Message, MessageContent, MessagesTool,
        SystemPrompt,
    },
    providers::anthropic::{
        API_BASE, API_KEY_HEADER, AnthropicBeta, BETA_HEADER, BetaSet,
        DIRECT_BROWSER_ACCESS_HEADER, MESSAGES_PATH,
    },
    recognized::{Recognized, map_known_runs, python_truthy},
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    anthropic::ANTHROPIC_OAUTH_TOKEN_PREFIX,
    base_llm::auth::{AuthScheme, Headers, ValidatedEnvironment},
};

pub const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
pub const ANTHROPIC_AUTH_TOKEN_ENV: &str = "ANTHROPIC_AUTH_TOKEN";
pub const ENCRYPTED_REASONING_SIGNATURE_PREFIX: &str = "litellm_encrypted_reasoning:";
const THOUGHT_SIGNATURE_SEPARATOR: &str = "__thought__";
pub const ANTHROPIC_API_BASE_ENV: &str = "ANTHROPIC_API_BASE";
pub const ANTHROPIC_BASE_URL_ENV: &str = "ANTHROPIC_BASE_URL";
pub const API_KEY_PLACEMENT: CredentialPlacement = CredentialPlacement::Header(API_KEY_HEADER);
const AUTHORIZATION: &str = CredentialPlacement::Bearer.header_name();

pub fn supports_effort_tier(capabilities: &MessagesModelCapabilities, level: EffortLevel) -> bool {
    match level {
        EffortLevel::Low => capabilities.effort_tiers.low,
        EffortLevel::Medium => capabilities.effort_tiers.medium,
        EffortLevel::High => capabilities.effort_tiers.high,
        EffortLevel::Xhigh => capabilities.effort_tiers.xhigh,
        EffortLevel::Max => capabilities.effort_tiers.max,
    }
}

pub fn supports_effort_param(capabilities: &MessagesModelCapabilities) -> bool {
    capabilities.supports_output_config || capabilities.effort_tiers.any()
}

pub fn accepts_effort(capabilities: &MessagesModelCapabilities, level: EffortLevel) -> bool {
    match level {
        EffortLevel::Max => {
            capabilities.supports_adaptive_thinking || capabilities.effort_tiers.max
        }
        EffortLevel::Xhigh => capabilities.effort_tiers.xhigh,
        EffortLevel::Low | EffortLevel::Medium | EffortLevel::High => true,
    }
}

/// An Anthropic OAuth access token, which authenticates as a bearer instead of an `x-api-key`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OauthToken<'a>(&'a str);

impl<'a> OauthToken<'a> {
    /// The raw token, as a caller passes it in `api_key`.
    pub fn parse(value: &'a str) -> Option<Self> {
        value
            .starts_with(ANTHROPIC_OAUTH_TOKEN_PREFIX)
            .then_some(Self(value))
    }

    /// A configured key, which users paste either raw or already prefixed with `Bearer `.
    pub fn parse_key(value: &'a str) -> Option<Self> {
        Self::parse(value.strip_prefix("Bearer ").unwrap_or(value))
    }

    pub fn as_str(self) -> &'a str {
        self.0
    }

    pub fn into_auth(self) -> AuthScheme {
        oauth_bearer(SecretValue::new(self.0))
    }
}

fn oauth_bearer(secret: SecretValue) -> AuthScheme {
    AuthScheme::Credential {
        placement: CredentialPlacement::Bearer,
        secret,
    }
}

pub fn minted_oauth_environment(headers: Headers, token: SecretValue) -> ValidatedEnvironment {
    ValidatedEnvironment {
        headers: with_oauth_companions(headers, &[], BrowserAccess::Omitted),
        auth: oauth_bearer(token),
    }
}

/// Python's `AnthropicModelInfo.get_api_key`: the param, else `ANTHROPIC_API_KEY`.
pub fn get_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    nonempty_secret(api_key.map(str::to_owned))
        .or_else(|| nonempty_secret(env_lookup(ANTHROPIC_API_KEY_ENV)))
}

pub fn get_auth_token(env_lookup: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    nonempty_secret(env_lookup(ANTHROPIC_AUTH_TOKEN_ENV))
}

/// Python's `normalize_nonempty_secret_str`: `str.strip()` also strips the ASCII separators.
fn nonempty_secret(value: Option<String>) -> Option<String> {
    value
        .map(|value| {
            value
                .trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
                .to_owned()
        })
        .filter(|value| !value.is_empty())
}

/// Python's `AnthropicModelInfo.get_auth_header`, naming the credential instead of building
/// the header: the key goes in `x-api-key` unless it is an OAuth token, and without a key
/// `ANTHROPIC_AUTH_TOKEN` is sent as a bearer.
pub fn get_auth_header(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<AuthScheme> {
    if let Some(key) = get_api_key(api_key, env_lookup) {
        return Some(match OauthToken::parse_key(&key) {
            Some(token) => token.into_auth(),
            None => AuthScheme::Credential {
                placement: API_KEY_PLACEMENT,
                secret: SecretValue::new(key),
            },
        });
    }
    get_auth_token(env_lookup).map(|token| AuthScheme::Credential {
        placement: CredentialPlacement::Bearer,
        secret: SecretValue::new(token),
    })
}

pub fn resolve_anthropic_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, litellm_auth::Error> {
    get_api_key(api_key, env_lookup).ok_or(litellm_auth::Error::MissingApiKey {
        provider: "Anthropic",
        environment_variable: ANTHROPIC_API_KEY_ENV,
    })
}

/// Whether the caller already forwarded an Anthropic credential, in either header.
pub fn has_anthropic_credential(headers: &[(String, String)]) -> bool {
    has_header(headers, API_KEY_HEADER) || has_header(headers, AUTHORIZATION)
}

pub fn resolve_anthropic_api_base(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    resolve_non_empty(
        api_base,
        env_lookup,
        &[ANTHROPIC_API_BASE_ENV, ANTHROPIC_BASE_URL_ENV],
    )
    .unwrap_or_else(|| API_BASE.to_string())
}

pub fn complete_anthropic_url(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    let api_base = resolve_anthropic_api_base(api_base, env_lookup);

    ensure_api_route(&api_base, MESSAGES_PATH)
}

/// Python's `urlsplit(url).path`, for the URLs an `api_base` holds.
fn url_path(url: &str) -> &str {
    let before_fragment = url.split('#').next().unwrap_or_default();
    let before_query = before_fragment.split('?').next().unwrap_or_default();
    match before_query.split_once("://") {
        Some((_, rest)) => rest.find('/').map_or("", |slash| &rest[slash..]),
        None => before_query,
    }
}

/// Python's `_strip_trailing_slashes_keeping_the_scheme`.
pub(crate) fn strip_trailing_slashes(api_base: &str) -> String {
    let stripped = api_base.trim_end_matches('/');
    if stripped.ends_with(':') && !stripped.contains('/') && !stripped.contains('?') {
        return api_base.to_string();
    }
    match stripped.split_once('?') {
        Some((before, query)) if url_path(stripped).ends_with('/') => {
            format!("{}?{query}", before.trim_end_matches('/'))
        }
        _ => stripped.to_string(),
    }
}

/// Python's `ensure_api_route`: an `api_base` whose path already ends in the route is kept as
/// it is, query string included; anything else gets the route appended to the whole string.
fn ensure_api_route(api_base: &str, route: &str) -> String {
    let base = strip_trailing_slashes(api_base);
    let route = route.trim_start_matches('/');
    if url_path(&base).ends_with(&format!("/{route}")) {
        return base;
    }
    format!("{}/{route}", strip_trailing_slashes(&base))
}

pub fn existing_betas(headers: &[(String, String)]) -> BetaSet {
    header_values(headers, BETA_HEADER)
        .flat_map(|value| {
            value
                .parse::<BetaSet>()
                .unwrap_or_else(|never| match never {})
        })
        .collect()
}

/// Python's `_merge_beta_headers`, over every casing of the header at once: the union of what
/// the caller sent and `added` replaces the header, sorted and deduplicated. Headers without
/// any beta value stay as they are.
pub fn merge_beta_headers(headers: Headers, added: BetaSet) -> Headers {
    let merged = existing_betas(&headers).union(added);
    if merged.is_empty() {
        return headers;
    }
    with_header(headers, BETA_HEADER, merged.to_string())
}

/// The outcome of Python's `optionally_handle_anthropic_oauth`.
#[derive(Clone, Debug, PartialEq)]
pub enum OauthHandling {
    /// An OAuth token is the whole credential. The headers carry its companions and no
    /// longer any `x-api-key` or `authorization`, so the bearer is applied on top.
    Bearer {
        headers: Headers,
        token: SecretValue,
    },
    Untouched(Headers),
}

/// The OAuth token a caller forwarded as `Authorization: Bearer sk-ant-oat…`.
pub fn forwarded_oauth_bearer(headers: &[(String, String)]) -> Option<OauthToken<'_>> {
    header_value(headers, AUTHORIZATION)
        .and_then(|value| value.strip_prefix("Bearer "))
        .and_then(OauthToken::parse)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BrowserAccess {
    Granted,
    Omitted,
}

fn with_oauth_companions(headers: Headers, dropped: &[&str], browser: BrowserAccess) -> Headers {
    let headers = merge_beta_headers(
        without_headers(headers, dropped),
        BetaSet::from_iter([AnthropicBeta::Oauth20250420]),
    );
    match browser {
        BrowserAccess::Granted => headers
            .into_iter()
            .chain([(DIRECT_BROWSER_ACCESS_HEADER.to_string(), "true".to_string())])
            .collect(),
        BrowserAccess::Omitted => headers,
    }
}

pub fn optionally_handle_anthropic_oauth(headers: Headers, api_key: Option<&str>) -> OauthHandling {
    if let Some(token) =
        forwarded_oauth_bearer(&headers).map(|token| SecretValue::new(token.as_str()))
    {
        return OauthHandling::Bearer {
            headers: with_oauth_companions(
                headers,
                &[API_KEY_HEADER, AUTHORIZATION],
                BrowserAccess::Granted,
            ),
            token,
        };
    }
    if let Some(token) = api_key.and_then(OauthToken::parse) {
        return OauthHandling::Bearer {
            headers: with_oauth_companions(headers, &[API_KEY_HEADER], BrowserAccess::Granted),
            token: SecretValue::new(token.as_str()),
        };
    }
    OauthHandling::Untouched(headers)
}

pub fn is_tool_search_used(tools: Option<&[Recognized<MessagesTool>]>) -> bool {
    tools.into_iter().flatten().any(|tool| {
        matches!(
            tool,
            Recognized::Known(
                MessagesTool::ToolSearchRegex { .. } | MessagesTool::ToolSearchBm25 { .. }
            )
        )
    })
}

pub fn has_advisor_tool(tools: Option<&[Recognized<MessagesTool>]>) -> bool {
    tools
        .into_iter()
        .flatten()
        .any(|tool| matches!(tool, Recognized::Known(MessagesTool::Advisor { .. })))
}

pub fn requires_native_compaction_beta(
    compaction: Option<&Value>,
    messages: &[Recognized<Message>],
    unparsed: UnparsedMessages,
) -> bool {
    compaction.is_some()
        || messages
            .iter()
            .any(|message| carries_signed_compaction_block(message, unparsed))
}

/// Python's `carries_signed_compaction_block` reads any mapping, so where unparsed messages are
/// read a message that did not parse is scanned through its raw content too.
fn carries_signed_compaction_block(
    message: &Recognized<Message>,
    unparsed: UnparsedMessages,
) -> bool {
    match message {
        Recognized::Known(message) => message.blocks().iter().any(|block| {
            block.is_type(ContentBlockType::Compaction)
                && block.signature.as_deref().is_some_and(|s| !s.is_empty())
        }),
        Recognized::Unrecognized(Value::Object(fields)) if unparsed.read() => matches!(
            fields.get("content"),
            Some(Value::Array(blocks)) if blocks.iter().any(|block| {
                block.get("type").and_then(Value::as_str) == Some("compaction")
                    && block
                        .get("signature")
                        .and_then(Value::as_str)
                        .is_some_and(|signature| !signature.is_empty())
            })
        ),
        Recognized::Unrecognized(_) => false,
    }
}

fn is_blank(text: Option<&str>) -> bool {
    text.is_none_or(|text| text.trim().is_empty())
}

fn is_empty_text_block(block: &ContentBlock) -> bool {
    block.is_type(ContentBlockType::Text) && is_blank(block.text.as_deref())
}

pub fn is_empty_thinking_block(block: &ContentBlock) -> bool {
    block.is_type(ContentBlockType::Thinking) && is_blank(block.thinking.as_deref())
}

fn retain_blocks(messages: Vec<Message>, keep: impl Fn(&ContentBlock) -> bool) -> Vec<Message> {
    messages
        .into_iter()
        .filter_map(|message| match message.content {
            MessageContent::Text(_) => Some(message),
            MessageContent::Blocks(ref blocks) => {
                let kept: Vec<ContentBlock> =
                    blocks.iter().filter(|block| keep(block)).cloned().collect();
                if kept.len() == blocks.len() {
                    return Some(message);
                }
                (!kept.is_empty()).then(|| message.with_blocks(kept))
            }
        })
        .collect()
}

fn is_empty_unsigned_thinking_block(block: &ContentBlock) -> bool {
    // Anthropic sends redacted thinking blocks, and the signature is used to
    // match them to the appropriate thinking block on their servers. If there
    // is a signature, we need to keep it.
    is_empty_thinking_block(block) && block.signature.as_deref().is_none_or(str::is_empty)
}

pub fn strip_empty_content_blocks(messages: Vec<Message>) -> Vec<Message> {
    retain_blocks(messages, |block| {
        !is_empty_text_block(block) && !is_empty_unsigned_thinking_block(block)
    })
}

pub fn normalize_anthropic_tool_use_id(raw_id: &str) -> String {
    let base = raw_id
        .split_once(THOUGHT_SIGNATURE_SEPARATOR)
        .map_or(raw_id, |(base, _)| base);
    let sanitized: String = base
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "tool_use_id".to_string()
    } else {
        sanitized
    }
}

fn normalized_if_changed(raw_id: Option<&str>) -> Option<String> {
    let raw_id = raw_id?;
    let normalized = normalize_anthropic_tool_use_id(raw_id);
    (normalized != raw_id).then_some(normalized)
}

fn sanitize_tool_use_id_block(block: ContentBlock) -> ContentBlock {
    match block.block_type.as_ref() {
        Some(ContentBlockType::ToolUse | ContentBlockType::ServerToolUse) => {
            match normalized_if_changed(block.id.as_deref()) {
                Some(id) => ContentBlock {
                    id: Some(id),
                    ..block
                },
                None => block,
            }
        }
        Some(ContentBlockType::ToolResult) => {
            match normalized_if_changed(block.tool_use_id.as_deref()) {
                Some(tool_use_id) => ContentBlock {
                    tool_use_id: Some(tool_use_id),
                    ..block
                },
                None => block,
            }
        }
        _ => block,
    }
}

pub fn sanitize_tool_use_ids(messages: Vec<Message>) -> Vec<Message> {
    messages
        .into_iter()
        .map(|message| match message.content {
            MessageContent::Blocks(blocks) => Message {
                content: MessageContent::Blocks(
                    blocks.into_iter().map(sanitize_tool_use_id_block).collect(),
                ),
                ..message
            },
            MessageContent::Text(_) => message,
        })
        .collect()
}

pub fn strip_provider_specific_fields(messages: Vec<Message>) -> Vec<Message> {
    messages
        .into_iter()
        .map(|message| match message.content {
            MessageContent::Blocks(blocks) => Message {
                content: MessageContent::Blocks(
                    blocks
                        .into_iter()
                        .map(|block| ContentBlock {
                            provider_specific_fields: None,
                            ..block
                        })
                        .collect(),
                ),
                ..message
            },
            MessageContent::Text(_) => message,
        })
        .collect()
}

pub fn is_encrypted_reasoning_block(block: &ContentBlock) -> bool {
    let field = match block.block_type.as_ref() {
        Some(ContentBlockType::Thinking) => block.signature.as_deref(),
        Some(ContentBlockType::RedactedThinking) => block.data.as_deref(),
        _ => None,
    };
    field.is_some_and(|value| value.starts_with(ENCRYPTED_REASONING_SIGNATURE_PREFIX))
}

const BILLING_HEADER_PREFIX: &str = "x-anthropic-billing-header:";

fn is_billing_header_block(block: &ContentBlock) -> bool {
    block.block_type == Some(ContentBlockType::Text)
        && block
            .text
            .as_deref()
            .is_some_and(|text| text.starts_with(BILLING_HEADER_PREFIX))
}

fn is_raw_billing_header_block(block: &Value) -> bool {
    raw_block_type(block) == Some("text")
        && block
            .get("text")
            .and_then(Value::as_str)
            .is_some_and(|text| text.starts_with(BILLING_HEADER_PREFIX))
}

/// Python's `AnthropicMessagesConfig._filter_billing_headers_from_system`: the Claude Code
/// attribution blocks the first-party API reads, dropped for hosts that reject them. `None`
/// when nothing else was in the system prompt. A list holding blocks Rust could not parse is
/// filtered the same way, and any other value is left as sent.
pub fn filter_billing_headers_from_system(
    system: Recognized<SystemPrompt>,
) -> Option<Recognized<SystemPrompt>> {
    match system {
        Recognized::Known(SystemPrompt::Text(text)) if text.starts_with(BILLING_HEADER_PREFIX) => {
            None
        }
        Recognized::Known(SystemPrompt::Blocks(blocks)) => {
            let kept: Vec<ContentBlock> = blocks
                .into_iter()
                .filter(|block| !is_billing_header_block(block))
                .collect();
            (!kept.is_empty()).then_some(Recognized::Known(SystemPrompt::Blocks(kept)))
        }
        Recognized::Unrecognized(Value::Array(blocks)) => {
            let kept: Vec<Value> = blocks
                .into_iter()
                .filter(|block| !is_raw_billing_header_block(block))
                .collect();
            (!kept.is_empty()).then_some(Recognized::Unrecognized(Value::Array(kept)))
        }
        system => Some(system),
    }
}

/// The billing strip of Python's `AnthropicMessagesConfig.transform_anthropic_messages_request`,
/// which pops a system prompt the filter leaves empty, an empty string included.
pub fn strip_billing_metadata_from_system(
    system: Recognized<SystemPrompt>,
) -> Option<Recognized<SystemPrompt>> {
    filter_billing_headers_from_system(system).filter(|kept| match kept {
        Recognized::Known(SystemPrompt::Text(text)) => !text.is_empty(),
        Recognized::Unrecognized(Value::Object(fields)) => !fields.is_empty(),
        _ => true,
    })
}

/// Blocks carrying another provider's encrypted reasoning go, and a message left with no blocks
/// goes with them, as main's native route strips them.
pub fn strip_encrypted_reasoning_blocks(messages: Vec<Message>) -> Vec<Message> {
    retain_blocks(messages, |block| !is_encrypted_reasoning_block(block))
}

/// Python's `strip_encrypted_reasoning_blocks_from_anthropic_messages`: the same strip, and a
/// kept `redacted_thinking` block gets Claude's own `data` back from a Bedrock Converse wrapper.
pub fn strip_encrypted_reasoning_blocks_restoring_redacted_data(
    messages: Vec<Message>,
) -> Vec<Message> {
    messages
        .into_iter()
        .filter_map(|message| {
            let MessageContent::Blocks(blocks) = &message.content else {
                return Some(message);
            };
            let kept: Vec<ContentBlock> = blocks
                .iter()
                .filter(|block| !is_encrypted_reasoning_block(block))
                .cloned()
                .map(with_native_redacted_data)
                .collect();
            if &kept == blocks {
                return Some(message);
            }
            (!kept.is_empty()).then(|| message.with_blocks(kept))
        })
        .collect()
}

fn with_native_redacted_data(block: ContentBlock) -> ContentBlock {
    match block.data.as_deref().and_then(converse_wrapped_payload) {
        Some(data) if block.is_type(ContentBlockType::RedactedThinking) => ContentBlock {
            data: Some(data),
            ..block
        },
        _ => block,
    }
}

/// Python's `base64.b64decode(data, validate=True)`: the standard alphabet with its padding,
/// whatever the unused bits of the last character hold.
const PYTHON_BASE64: base64::engine::GeneralPurpose = base64::engine::GeneralPurpose::new(
    &base64::alphabet::STANDARD,
    base64::engine::GeneralPurposeConfig::new()
        .with_decode_allow_trailing_bits(true)
        .with_decode_padding_mode(base64::engine::DecodePaddingMode::RequireCanonical),
);

/// Python's `_converse_wrapped_payload`, kept only when non-empty as `native_redacted_thinking_data`
/// keeps it: Bedrock Converse wraps Claude's `data` in one more layer of base64, and Claude's own
/// value never decodes twice.
fn converse_wrapped_payload(data: &str) -> Option<String> {
    use base64::Engine;
    let payload = String::from_utf8(PYTHON_BASE64.decode(data).ok()?)
        .ok()
        .filter(|payload| payload.is_ascii() && !payload.is_empty())?;
    PYTHON_BASE64.decode(&payload).ok()?;
    Some(payload)
}

fn is_advisor_use(block: &ContentBlock) -> bool {
    block.is_type(ContentBlockType::ServerToolUse)
        && block.name.as_deref() == Some("advisor")
        && block.id.as_deref().is_some_and(|id| !id.is_empty())
}

pub fn strip_advisor_blocks(messages: Vec<Message>) -> Vec<Message> {
    messages
        .into_iter()
        .map(|message| {
            if message.role != "assistant" {
                return message;
            }
            let MessageContent::Blocks(blocks) = &message.content else {
                return message;
            };
            let advisor_ids: HashSet<&str> = blocks
                .iter()
                .filter(|block| is_advisor_use(block))
                .filter_map(|block| block.id.as_deref())
                .collect();
            if advisor_ids.is_empty() {
                return message;
            }
            let kept: Vec<ContentBlock> = blocks
                .iter()
                .filter(|block| {
                    let is_result = block.is_type(ContentBlockType::AdvisorToolResult)
                        && block
                            .tool_use_id
                            .as_deref()
                            .is_some_and(|id| advisor_ids.contains(id));
                    !is_advisor_use(block) && !is_result
                })
                .cloned()
                .collect();
            message.with_blocks(kept)
        })
        .collect()
}

#[derive(Deserialize)]
struct ReplayedWebSearchResult {
    #[serde(default)]
    url: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    snippet: String,
    #[serde(default)]
    encrypted_content: String,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ReplayedWebSearchContent {
    #[serde(rename = "web_search_result")]
    Result(ReplayedWebSearchResult),
    #[serde(rename = "web_search_tool_result_error")]
    Error {
        #[serde(default)]
        error_code: String,
    },
}

enum WebSearchResults {
    Results(Vec<ReplayedWebSearchResult>),
    Error(String),
}

fn flattenable_web_search_results(block: &ContentBlock) -> Option<(&str, WebSearchResults)> {
    if !block.is_type(ContentBlockType::WebSearchToolResult) {
        return None;
    }
    let tool_use_id = block.tool_use_id.as_deref()?;
    Some((
        tool_use_id,
        replayed_web_search_results(block.content.as_ref()?)?,
    ))
}

/// The results a replayed `web_search_tool_result` block carries, `None` when Anthropic minted
/// them, since only its search backend reads `encrypted_content`.
fn replayed_web_search_results(content: &Value) -> Option<WebSearchResults> {
    let results = match content {
        Value::Array(items) => {
            let results = items
                .iter()
                .map(|item| match serde_json::from_value(item.clone()).ok()? {
                    ReplayedWebSearchContent::Result(result) => Some(result),
                    ReplayedWebSearchContent::Error { .. } => None,
                })
                .collect::<Option<Vec<_>>>()?;
            if results
                .iter()
                .any(|result| !result.encrypted_content.is_empty())
            {
                return None;
            }
            WebSearchResults::Results(results)
        }
        error @ Value::Object(_) => match serde_json::from_value(error.clone()).ok()? {
            ReplayedWebSearchContent::Error { error_code } => WebSearchResults::Error(error_code),
            ReplayedWebSearchContent::Result(_) => return None,
        },
        _ => return None,
    };
    Some(results)
}

fn render_web_search_results(query: &str, results: &WebSearchResults) -> String {
    let header = if query.is_empty() {
        "Web search results:".to_string()
    } else {
        format!("Web search results for '{query}':")
    };
    match results {
        WebSearchResults::Error(code) => {
            let code = if code.is_empty() { "unavailable" } else { code };
            format!("{header}\n\nSearch failed: {code}")
        }
        WebSearchResults::Results(results) if results.is_empty() => {
            format!("{header}\n\nNo results were returned.")
        }
        WebSearchResults::Results(results) => {
            let body = results
                .iter()
                .map(|result| {
                    [
                        (!result.title.is_empty()).then(|| format!("Title: {}", result.title)),
                        (!result.url.is_empty()).then(|| format!("URL: {}", result.url)),
                        (!result.snippet.is_empty())
                            .then(|| format!("Snippet: {}", result.snippet)),
                    ]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .join("\n")
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            if body.is_empty() {
                header
            } else {
                format!("{header}\n\n{body}")
            }
        }
    }
}

fn server_tool_use_query(block: &ContentBlock) -> Option<(&str, &str)> {
    if !block.is_type(ContentBlockType::ServerToolUse) {
        return None;
    }
    let id = block.id.as_deref()?;
    let query = match block.input.as_ref() {
        None => "",
        Some(Value::Object(input)) => match input.get("query") {
            None => "",
            Some(query) => query.as_str()?,
        },
        Some(_) => return None,
    };
    Some((id, query))
}

fn flatten_web_search_results_in_blocks(blocks: Vec<ContentBlock>) -> Vec<ContentBlock> {
    let flattenable_ids: HashSet<&str> = blocks
        .iter()
        .filter_map(flattenable_web_search_results)
        .map(|(tool_use_id, _)| tool_use_id)
        .collect();
    if flattenable_ids.is_empty() {
        return blocks;
    }
    let queries: HashMap<&str, &str> = blocks.iter().filter_map(server_tool_use_query).collect();
    blocks
        .iter()
        .filter_map(|block| {
            if let Some((tool_use_id, results)) = flattenable_web_search_results(block) {
                let query = queries.get(tool_use_id).copied().unwrap_or("");
                return Some(ContentBlock::text(render_web_search_results(
                    query, &results,
                )));
            }
            if let Some((id, _)) = server_tool_use_query(block)
                && flattenable_ids.contains(id)
            {
                return None;
            }
            Some(block.clone())
        })
        .collect()
}

pub fn flatten_unencrypted_web_search_results(messages: Vec<Message>) -> Vec<Message> {
    messages
        .into_iter()
        .map(|message| match message.content {
            MessageContent::Blocks(blocks) => Message {
                content: MessageContent::Blocks(flatten_web_search_results_in_blocks(blocks)),
                ..message
            },
            MessageContent::Text(_) => message,
        })
        .collect()
}

const CONTENT: &str = "content";

/// Python's message rewrites read any mapping, so a message Rust could not parse gets the same
/// rewrite on its raw JSON: `known` rewrites each run of parsed messages and `raw` each unparsed
/// one, returning `None` where Python drops the message. A raw message the rewrite left
/// parseable is parsed again, so every later step reads it as Python does.
pub fn map_messages(
    messages: Vec<Recognized<Message>>,
    known: impl FnMut(Vec<Message>) -> Vec<Message>,
    raw: impl Fn(Value) -> Option<Value>,
) -> Vec<Recognized<Message>> {
    map_known_runs(messages, known)
        .into_iter()
        .filter_map(|message| match message {
            Recognized::Unrecognized(message) => raw(message).map(reparsed),
            known => Some(known),
        })
        .collect()
}

fn reparsed(message: Value) -> Recognized<Message> {
    match Message::deserialize(&message) {
        Ok(parsed) => Recognized::Known(parsed),
        Err(_) => Recognized::Unrecognized(message),
    }
}

/// Rewrites the content of a raw message whose content is a list, the only shape Python's
/// block rewrites touch. `None` from `rewrite` drops the message.
fn with_raw_blocks(
    message: Value,
    rewrite: impl FnOnce(Vec<Value>) -> Option<Vec<Value>>,
) -> Option<Value> {
    let Value::Object(mut fields) = message else {
        return Some(message);
    };
    let Some(Value::Array(blocks)) = fields.get_mut(CONTENT) else {
        return Some(Value::Object(fields));
    };
    let blocks = std::mem::take(blocks);
    fields.insert(CONTENT.into(), Value::Array(rewrite(blocks)?));
    Some(Value::Object(fields))
}

fn raw_block_type(block: &Value) -> Option<&str> {
    block.get("type").and_then(Value::as_str)
}

fn is_raw_blank(text: Option<&Value>) -> bool {
    is_blank(text.and_then(Value::as_str))
}

fn is_empty_raw_text_block(block: &Value) -> bool {
    raw_block_type(block) == Some("text") && is_raw_blank(block.get("text"))
}

fn is_empty_unsigned_raw_thinking_block(block: &Value) -> bool {
    raw_block_type(block) == Some("thinking")
        && is_raw_blank(block.get("thinking"))
        && !block.get("signature").is_some_and(python_truthy)
}

/// [`strip_empty_content_blocks`] on a raw message: a text block whose `text` is not a
/// non-blank string goes, so does an empty thinking block with no signature, and a message
/// left with no blocks is dropped.
pub fn strip_empty_raw_content_blocks(message: Value) -> Option<Value> {
    with_raw_blocks(message, |blocks| {
        let sent = blocks.len();
        let kept: Vec<Value> = blocks
            .into_iter()
            .filter(|block| {
                !is_empty_raw_text_block(block) && !is_empty_unsigned_raw_thinking_block(block)
            })
            .collect();
        (kept.len() == sent || !kept.is_empty()).then_some(kept)
    })
}

fn sanitize_raw_tool_use_id_block(block: Value) -> Value {
    let key = match raw_block_type(&block) {
        Some("tool_use" | "server_tool_use") => "id",
        Some("tool_result") => "tool_use_id",
        _ => return block,
    };
    let Value::Object(mut fields) = block else {
        return block;
    };
    if let Some(normalized) = normalized_if_changed(fields.get(key).and_then(Value::as_str)) {
        fields.insert(key.into(), Value::String(normalized));
    }
    Value::Object(fields)
}

/// [`sanitize_tool_use_ids`] on a raw message.
pub fn sanitize_raw_tool_use_ids(message: Value) -> Option<Value> {
    with_raw_blocks(message, |blocks| {
        Some(
            blocks
                .into_iter()
                .map(sanitize_raw_tool_use_id_block)
                .collect(),
        )
    })
}

fn raw_flattenable_web_search_results(block: &Value) -> Option<(&str, WebSearchResults)> {
    if raw_block_type(block) != Some("web_search_tool_result") {
        return None;
    }
    let tool_use_id = block.get("tool_use_id")?.as_str()?;
    Some((
        tool_use_id,
        replayed_web_search_results(block.get(CONTENT)?)?,
    ))
}

fn raw_server_tool_use_query(block: &Value) -> Option<(&str, &str)> {
    if raw_block_type(block) != Some("server_tool_use") {
        return None;
    }
    let id = block.get("id")?.as_str()?;
    let query = match block.get("input") {
        None => "",
        Some(Value::Object(input)) => match input.get("query") {
            None => "",
            Some(query) => query.as_str()?,
        },
        Some(_) => return None,
    };
    Some((id, query))
}

/// [`flatten_unencrypted_web_search_results`] on a raw message.
pub fn flatten_unencrypted_raw_web_search_results(message: Value) -> Option<Value> {
    with_raw_blocks(message, |blocks| {
        let flattenable_ids: HashSet<&str> = blocks
            .iter()
            .filter_map(raw_flattenable_web_search_results)
            .map(|(tool_use_id, _)| tool_use_id)
            .collect();
        if flattenable_ids.is_empty() {
            return Some(blocks);
        }
        let queries: HashMap<&str, &str> = blocks
            .iter()
            .filter_map(raw_server_tool_use_query)
            .collect();
        let flattened = blocks
            .iter()
            .filter_map(|block| {
                if let Some((tool_use_id, results)) = raw_flattenable_web_search_results(block) {
                    let query = queries.get(tool_use_id).copied().unwrap_or("");
                    return Some(
                        json!({"type": "text", "text": render_web_search_results(query, &results)}),
                    );
                }
                if let Some((id, _)) = raw_server_tool_use_query(block)
                    && flattenable_ids.contains(id)
                {
                    return None;
                }
                Some(block.clone())
            })
            .collect();
        Some(flattened)
    })
}

/// [`strip_provider_specific_fields`] on a raw message.
pub fn strip_raw_provider_specific_fields(message: Value) -> Option<Value> {
    with_raw_blocks(message, |blocks| {
        Some(
            blocks
                .into_iter()
                .map(|block| match block {
                    Value::Object(mut fields) => {
                        fields.shift_remove("provider_specific_fields");
                        Value::Object(fields)
                    }
                    block => block,
                })
                .collect(),
        )
    })
}

fn is_raw_advisor_use(block: &Value) -> bool {
    raw_block_type(block) == Some("server_tool_use")
        && block.get("name").and_then(Value::as_str) == Some("advisor")
}

/// A JSON value as a Python dict key: `True`, `1` and `1.0` are one key. Python cannot hash a
/// list or dict at all, so those keep JSON equality.
#[derive(PartialEq, Eq, Hash)]
enum PythonKey<'a> {
    Integer(i128),
    Float(u64),
    Json(&'a Value),
}

impl<'a> PythonKey<'a> {
    fn of(value: &'a Value) -> Self {
        const I128_RANGE: f64 = 170_141_183_460_469_231_731_687_303_715_884_105_728.0;
        match value {
            Value::Bool(flag) => Self::Integer(i128::from(*flag)),
            Value::Number(number) => match (number.as_i128(), number.as_f64()) {
                (Some(integer), _) => Self::Integer(integer),
                (None, Some(float)) if float.fract() == 0.0 && float.abs() < I128_RANGE => {
                    Self::Integer(float as i128)
                }
                (None, Some(float)) => Self::Float(float.to_bits()),
                (None, None) => Self::Json(value),
            },
            _ => Self::Json(value),
        }
    }
}

/// Python's `strip_advisor_blocks_from_messages` on a raw assistant message: every advisor
/// `server_tool_use` with a truthy id goes, with the `advisor_tool_result` answering it.
pub fn strip_raw_advisor_blocks(message: Value) -> Option<Value> {
    if message.get("role").and_then(Value::as_str) != Some("assistant") {
        return Some(message);
    }
    with_raw_blocks(message, |blocks| {
        let advisor_ids: HashSet<PythonKey> = blocks
            .iter()
            .filter(|block| is_raw_advisor_use(block))
            .filter_map(|block| block.get("id"))
            .filter(|id| python_truthy(id))
            .map(PythonKey::of)
            .collect();
        let answers_advisor =
            |id: Option<&Value>| id.is_some_and(|id| advisor_ids.contains(&PythonKey::of(id)));
        let kept = blocks
            .iter()
            .filter(|block| {
                let advisor_use = is_raw_advisor_use(block) && answers_advisor(block.get("id"));
                let advisor_result = raw_block_type(block) == Some("advisor_tool_result")
                    && answers_advisor(block.get("tool_use_id"));
                !advisor_use && !advisor_result
            })
            .cloned()
            .collect();
        Some(kept)
    })
}

fn is_raw_encrypted_reasoning_block(block: &Value) -> bool {
    let field = match raw_block_type(block) {
        Some("thinking") => block.get("signature"),
        Some("redacted_thinking") => block.get("data"),
        _ => None,
    };
    field
        .and_then(Value::as_str)
        .is_some_and(|value| value.starts_with(ENCRYPTED_REASONING_SIGNATURE_PREFIX))
}

fn with_native_raw_redacted_data(block: Value) -> Value {
    if raw_block_type(&block) != Some("redacted_thinking") {
        return block;
    }
    let Value::Object(mut fields) = block else {
        return block;
    };
    if let Some(data) = fields
        .get("data")
        .and_then(Value::as_str)
        .and_then(converse_wrapped_payload)
    {
        fields.insert("data".into(), Value::String(data));
    }
    Value::Object(fields)
}

/// [`strip_encrypted_reasoning_blocks`] on a raw message.
pub fn strip_raw_encrypted_reasoning_blocks(message: Value) -> Option<Value> {
    with_raw_blocks(message, |blocks| {
        let kept: Vec<Value> = blocks
            .iter()
            .filter(|block| !is_raw_encrypted_reasoning_block(block))
            .cloned()
            .map(with_native_raw_redacted_data)
            .collect();
        (kept == blocks || !kept.is_empty()).then_some(kept)
    })
}

pub fn is_anthropic_invalid_thinking_block_error(error_text: &str) -> bool {
    let lower = error_text.to_lowercase();
    if !lower.contains("thinking") {
        return false;
    }
    (lower.contains("signature") && (lower.contains("invalid") || lower.contains("valid string")))
        || lower.contains("must contain thinking")
}

pub fn is_anthropic_thinking_binding_mismatch_error(error_text: &str) -> bool {
    error_text
        .to_lowercase()
        .contains("bound to a different conversation")
}

pub fn anthropic_request_errors_on_thinking_binding_mismatch(request_body: &Value) -> bool {
    request_body
        .pointer("/thinking/block_binding/prefix_mismatch_behavior")
        .and_then(Value::as_str)
        == Some("error")
}

fn is_thinking_value_block(block: &Value) -> bool {
    matches!(
        block.get("type").and_then(Value::as_str),
        Some("thinking" | "redacted_thinking")
    )
}

fn without_thinking_value_blocks(message: Value) -> Option<Value> {
    let Value::Object(mut message) = message else {
        return Some(message);
    };
    if let Some(Value::Array(content)) = message.get_mut("content") {
        content.retain(|block| !is_thinking_value_block(block));
        if content.is_empty() {
            return None;
        }
    }
    Some(Value::Object(message))
}

pub fn strip_thinking_blocks_from_anthropic_messages_request(request_body: Value) -> Value {
    let Value::Object(mut body) = request_body else {
        return request_body;
    };
    if let Some(Value::Array(messages)) = body.get_mut("messages") {
        *messages = std::mem::take(messages)
            .into_iter()
            .filter_map(without_thinking_value_blocks)
            .collect();
    }
    body.shift_remove("thinking");
    Value::Object(body)
}

static CLAUDE_STRICT_TOOLS_REJECTION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"Too many strict tools \(\d+\)",
        r"|Schemas contains too many (?:optional parameters|parameters with union types)",
        r"|[Cc]ompiled grammar (?:is too large|size)",
        r"|tools\.\d+\.custom(?:\.strict)?: ",
        r"|Grammar compilation timed out",
    ))
    .expect("valid regex")
});

pub fn is_claude_strict_tools_rejection(status: u16, error_text: &str) -> bool {
    status == 400
        && CLAUDE_STRICT_TOOLS_REJECTION
            .is_match(error_text)
            .unwrap_or(false)
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolsWithoutStrict {
    pub tools: Vec<Value>,
    pub dropped_count: usize,
}

pub fn tools_without_strict(tools: Option<&Value>) -> Option<ToolsWithoutStrict> {
    let tools = tools?.as_array()?;
    if !tools.iter().all(Value::is_object) {
        return None;
    }
    let dropped_count = tools
        .iter()
        .filter(|tool| tool.get("strict") == Some(&Value::Bool(true)))
        .count();
    if dropped_count == 0 {
        return None;
    }
    Some(ToolsWithoutStrict {
        tools: tools
            .iter()
            .cloned()
            .map(|mut tool| {
                if let Value::Object(fields) = &mut tool {
                    fields.shift_remove("strict");
                }
                tool
            })
            .collect(),
        dropped_count,
    })
}

#[cfg(test)]
mod tests {
    use crate::base_llm::messages::context::SupportedEffortTiers;
    use rstest::{fixture, rstest};
    use serde_json::json;
    use strum::VariantArray;

    use super::*;

    fn apply(sanitizer: fn(Vec<Message>) -> Vec<Message>, messages: Value) -> Value {
        let parsed: Vec<Message> = serde_json::from_value(messages).unwrap();
        serde_json::to_value(sanitizer(parsed)).unwrap()
    }

    fn block(value: Value) -> ContentBlock {
        serde_json::from_value(value).unwrap()
    }

    fn history(messages: Value) -> Vec<Recognized<Message>> {
        serde_json::from_value(messages).unwrap()
    }

    fn tools(value: Option<Value>) -> Option<Vec<Recognized<MessagesTool>>> {
        value.map(|tools| serde_json::from_value(tools).unwrap())
    }

    fn headers(pairs: &[(&str, &str)]) -> Headers {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    fn betas(values: &[&str]) -> BetaSet {
        values.join(",").parse().unwrap()
    }

    fn env(vars: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            vars.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    const BOTH_BASE_ENVS: &[(&str, &str)] = &[
        (ANTHROPIC_API_BASE_ENV, "https://api-base.example.com"),
        (ANTHROPIC_BASE_URL_ENV, "https://base-url.example.com"),
    ];

    #[rstest]
    #[case::public_endpoint_by_default(None, &[], "https://api.anthropic.com")]
    #[case::explicit_api_base_beats_env(
        Some("https://explicit.example.com"),
        BOTH_BASE_ENVS,
        "https://explicit.example.com"
    )]
    #[case::explicit_api_base_is_trimmed(
        Some("  https://explicit.example.com  "),
        &[],
        "https://explicit.example.com"
    )]
    #[case::blank_api_base_falls_back_to_env(
        Some("  "),
        BOTH_BASE_ENVS,
        "https://api-base.example.com"
    )]
    #[case::api_base_env_beats_base_url_env(None, BOTH_BASE_ENVS, "https://api-base.example.com")]
    #[case::base_url_env_without_api_base_env(
        None,
        &[(ANTHROPIC_BASE_URL_ENV, "https://base-url.example.com")],
        "https://base-url.example.com"
    )]
    #[case::blank_api_base_env_falls_back_to_base_url_env(
        None,
        &[(ANTHROPIC_API_BASE_ENV, " \t "), (ANTHROPIC_BASE_URL_ENV, "https://base-url.example.com")],
        "https://base-url.example.com"
    )]
    #[case::blank_envs_fall_back_to_public_endpoint(
        None,
        &[(ANTHROPIC_API_BASE_ENV, ""), (ANTHROPIC_BASE_URL_ENV, "  ")],
        "https://api.anthropic.com"
    )]
    fn api_base_resolution(
        #[case] api_base: Option<&str>,
        #[case] vars: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        assert_eq!(resolve_anthropic_api_base(api_base, &env(vars)), expected);
    }

    #[rstest]
    #[case::forwarded_api_key(&[("X-Api-Key", "k")], true)]
    #[case::forwarded_bearer(&[("Authorization", "Bearer t")], true)]
    #[case::nothing_forwarded(&[("anthropic-version", "2023-06-01")], false)]
    fn forwarded_credential_is_detected_in_either_header(
        #[case] forwarded: &[(&str, &str)],
        #[case] expected: bool,
    ) {
        let headers: Headers = forwarded
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        assert_eq!(has_anthropic_credential(&headers), expected);
    }

    fn credential(auth: Option<AuthScheme>) -> Option<(&'static str, String)> {
        auth.map(|auth| match auth {
            AuthScheme::Credential { placement, secret } => {
                (placement.header_name(), secret.expose().to_string())
            }
            other => panic!("expected a credential, got {other:?}"),
        })
    }

    fn tagged(encrypted: &str) -> String {
        format!("{ENCRYPTED_REASONING_SIGNATURE_PREFIX}{encrypted}")
    }

    fn tiers(
        minimal: bool,
        low: bool,
        medium: bool,
        high: bool,
        xhigh: bool,
        max: bool,
    ) -> SupportedEffortTiers {
        SupportedEffortTiers {
            minimal,
            low,
            medium,
            high,
            xhigh,
            max,
        }
    }

    fn replayed_search_turn(results: Value) -> Value {
        json!([
            {"role": "user", "content": "when was Rome founded?"},
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "when"}},
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": results},
                {"type": "text", "text": "753 BC."}
            ]}
        ])
    }

    #[fixture]
    fn unmapped() -> MessagesModelCapabilities {
        MessagesModelCapabilities::default()
    }

    #[rstest]
    #[case::empty_text(json!({"type": "thinking", "thinking": ""}), true)]
    #[case::whitespace_only(json!({"type": "thinking", "thinking": " \n\t "}), true)]
    #[case::null_text(json!({"type": "thinking", "thinking": null}), true)]
    #[case::missing_text(json!({"type": "thinking"}), true)]
    #[case::empty_text_despite_signature(json!({"type": "thinking", "thinking": "", "signature": "sig_abc"}), true)]
    #[case::real_thinking(json!({"type": "thinking", "thinking": "plan", "signature": "sig"}), false)]
    #[case::padded_real_thinking(json!({"type": "thinking", "thinking": "  plan  "}), false)]
    #[case::redacted_thinking_is_a_different_type(json!({"type": "redacted_thinking", "data": "opaque"}), false)]
    #[case::empty_text_block(json!({"type": "text", "text": ""}), false)]
    #[case::untyped_block(json!({"thinking": ""}), false)]
    fn empty_thinking_block_detection(#[case] input: Value, #[case] expected: bool) {
        assert_eq!(is_empty_thinking_block(&block(input)), expected);
    }

    #[rstest]
    #[case::empty_text_beside_tool_use(
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": ""},
            {"type": "tool_use", "id": "x", "name": "Bash", "input": {}}
        ]}]),
        json!([{"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "Bash", "input": {}}]}])
    )]
    #[case::whitespace_text_beside_tool_use(
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "  \n "},
            {"type": "tool_use", "id": "x", "name": "Bash", "input": {}}
        ]}]),
        json!([{"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "Bash", "input": {}}]}])
    )]
    #[case::null_text(
        json!([{"role": "user", "content": [
            {"type": "text", "text": null},
            {"type": "tool_result", "tool_use_id": "x", "content": "y"}
        ]}]),
        json!([{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "y"}]}])
    )]
    #[case::missing_text(
        json!([{"role": "user", "content": [
            {"type": "text"},
            {"type": "tool_result", "tool_use_id": "x", "content": "y"}
        ]}]),
        json!([{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "y"}]}])
    )]
    #[case::empty_unsigned_thinking_beside_tool_use(
        json!([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": ""},
                {"type": "tool_use", "id": "toolu_01A", "name": "get_weather", "input": {"city": "Paris"}}
            ]}
        ]),
        json!([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_01A", "name": "get_weather", "input": {"city": "Paris"}}
            ]}
        ])
    )]
    #[case::empty_thinking_with_empty_signature(
        json!([{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "", "signature": ""},
            {"type": "redacted_thinking", "data": "opaque"}
        ]}]),
        json!([{"role": "assistant", "content": [{"type": "redacted_thinking", "data": "opaque"}]}])
    )]
    #[case::whitespace_thinking_beside_real_and_redacted_thinking(
        json!([{"role": "assistant", "content": [
            {"type": "thinking", "thinking": " \n "},
            {"type": "thinking", "thinking": "real plan", "signature": "sig"},
            {"type": "redacted_thinking", "data": "opaque"}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "real plan", "signature": "sig"},
            {"type": "redacted_thinking", "data": "opaque"}
        ]}])
    )]
    #[case::blank_text_beside_real_thinking(
        json!([{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "plan", "signature": "sig"},
            {"type": "text", "text": ""}
        ]}]),
        json!([{"role": "assistant", "content": [{"type": "thinking", "thinking": "plan", "signature": "sig"}]}])
    )]
    #[case::message_left_without_blocks_is_dropped(
        json!([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": ""}]}
        ]),
        json!([{"role": "user", "content": "hello"}])
    )]
    fn strip_empty_content_blocks_rewrites(#[case] input: Value, #[case] expected: Value) {
        assert_eq!(apply(strip_empty_content_blocks, input), expected);
    }

    #[rstest]
    #[case::non_empty_text(json!([{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]))]
    #[case::padded_text(json!([{"role": "assistant", "content": [{"type": "text", "text": "  hi  "}]}]))]
    #[case::empty_string_content(json!([{"role": "user", "content": ""}]))]
    #[case::textless_non_text_block(json!([{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}
    ]}]))]
    #[case::empty_signed_thinking_beside_tool_use(json!([{"role": "assistant", "content": [
        {"type": "thinking", "thinking": "", "signature": "sig_abc"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "tool_use", "id": "toolu_01A", "name": "get_weather", "input": {"city": "Paris"}}
    ]}]))]
    #[case::encrypted_reasoning_left_for_the_responses_bridge(json!([{"role": "assistant", "content": [
        {"type": "thinking", "thinking": "plan", "signature": tagged("gAAAA_1")},
        {"type": "redacted_thinking", "data": tagged("gAAAA_2")},
        {"type": "text", "text": "The answer."}
    ]}]))]
    fn strip_empty_content_blocks_leaves_untouched(#[case] input: Value) {
        assert_eq!(apply(strip_empty_content_blocks, input.clone()), input);
    }

    #[rstest]
    #[case::replayed_provider_id("functions.Bash:0", "functions_Bash_0")]
    #[case::thought_signature_suffix("call_abc123__thought__CiIBDDnWx+/a==", "call_abc123")]
    #[case::splits_at_first_thought_separator("call_1__thought__a__thought__b", "call_1")]
    #[case::valid_id("toolu_01-A_b", "toolu_01-A_b")]
    #[case::non_ascii_letter("café", "caf_")]
    #[case::only_invalid_characters("::", "__")]
    #[case::empty("", "tool_use_id")]
    #[case::thought_signature_only("__thought__CiIB", "tool_use_id")]
    fn normalize_anthropic_tool_use_id_cases(#[case] raw: &str, #[case] expected: &str) {
        assert_eq!(normalize_anthropic_tool_use_id(raw), expected);
    }

    #[rstest]
    #[case::tool_use_and_its_result(
        json!([
            {"role": "assistant", "content": [{"type": "tool_use", "id": "functions.Bash:0", "name": "Bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "functions.Bash:0", "content": "ok"}]}
        ]),
        json!([
            {"role": "assistant", "content": [{"type": "tool_use", "id": "functions_Bash_0", "name": "Bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "functions_Bash_0", "content": "ok"}]}
        ])
    )]
    #[case::server_tool_use(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "srv.1", "name": "web_search", "input": {}}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {}}
        ]}])
    )]
    #[case::tool_use_rewrites_only_its_id(
        json!([{"role": "assistant", "content": [
            {"type": "tool_use", "id": "a.b", "tool_use_id": "c.d", "name": "Bash", "input": {}}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "tool_use", "id": "a_b", "tool_use_id": "c.d", "name": "Bash", "input": {}}
        ]}])
    )]
    #[case::tool_result_rewrites_only_its_tool_use_id(
        json!([{"role": "user", "content": [
            {"type": "tool_result", "id": "a.b", "tool_use_id": "c.d", "content": "ok"}
        ]}]),
        json!([{"role": "user", "content": [
            {"type": "tool_result", "id": "a.b", "tool_use_id": "c_d", "content": "ok"}
        ]}])
    )]
    fn sanitize_tool_use_ids_rewrites(#[case] input: Value, #[case] expected: Value) {
        assert_eq!(apply(sanitize_tool_use_ids, input), expected);
    }

    #[rstest]
    #[case::valid_ids(json!([
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "ok"}]}
    ]))]
    #[case::id_mentioned_in_text(json!([{"role": "user", "content": [{"type": "text", "text": "id: functions.Bash:0"}]}]))]
    #[case::unknown_block_type(json!([{"role": "assistant", "content": [
        {"type": "future_tool_use", "id": "functions.Bash:0", "tool_use_id": "functions.Bash:0"}
    ]}]))]
    #[case::tool_use_without_id(json!([{"role": "assistant", "content": [{"type": "tool_use", "name": "Bash", "input": {}}]}]))]
    #[case::tool_result_without_tool_use_id(json!([{"role": "user", "content": [{"type": "tool_result", "content": "ok"}]}]))]
    #[case::string_content(json!([{"role": "user", "content": "functions.Bash:0"}]))]
    fn sanitize_tool_use_ids_leaves_untouched(#[case] input: Value) {
        assert_eq!(apply(sanitize_tool_use_ids, input.clone()), input);
    }

    #[rstest]
    #[case::thinking_block(
        json!([{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hm", "signature": "s", "provider_specific_fields": {"a": 1}}
        ]}]),
        json!([{"role": "assistant", "content": [{"type": "thinking", "thinking": "hm", "signature": "s"}]}])
    )]
    #[case::every_block_of_every_message(
        json!([
            {"role": "assistant", "content": [
                {"type": "text", "text": "a", "provider_specific_fields": {"x": 1}},
                {"type": "tool_use", "id": "t1", "name": "f", "input": {}, "provider_specific_fields": {"y": 2}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok", "provider_specific_fields": {}}
            ]}
        ]),
        json!([
            {"role": "assistant", "content": [
                {"type": "text", "text": "a"},
                {"type": "tool_use", "id": "t1", "name": "f", "input": {}}
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}
        ])
    )]
    fn strip_provider_specific_fields_rewrites(#[case] input: Value, #[case] expected: Value) {
        assert_eq!(apply(strip_provider_specific_fields, input), expected);
    }

    #[rstest]
    #[case::string_content(json!([{"role": "user", "content": "provider_specific_fields"}]))]
    #[case::blocks_without_the_field(json!([{"role": "assistant", "content": [{"type": "text", "text": "a"}]}]))]
    fn strip_provider_specific_fields_leaves_untouched(#[case] input: Value) {
        assert_eq!(apply(strip_provider_specific_fields, input.clone()), input);
    }

    #[rstest]
    #[case::tagged_thinking_signature(json!({"type": "thinking", "thinking": "x", "signature": tagged("g")}), true)]
    #[case::tagged_redacted_data(json!({"type": "redacted_thinking", "data": tagged("g")}), true)]
    #[case::bare_tag_signature(json!({"type": "thinking", "thinking": "x", "signature": tagged("")}), true)]
    #[case::bare_tag_data(json!({"type": "redacted_thinking", "data": tagged("")}), true)]
    #[case::anthropic_signature(json!({"type": "thinking", "thinking": "x", "signature": "ErcBCkgIValid"}), false)]
    #[case::anthropic_data(json!({"type": "redacted_thinking", "data": "EmwKAhgBEgy"}), false)]
    #[case::unsigned_thinking(json!({"type": "thinking", "thinking": "x"}), false)]
    #[case::tag_in_text_block(json!({"type": "text", "text": tagged("g")}), false)]
    #[case::tag_in_thinking_data(json!({"type": "thinking", "thinking": "x", "data": tagged("g")}), false)]
    #[case::tag_in_redacted_signature(
        json!({"type": "redacted_thinking", "data": "EmwKAhgBEgy", "signature": tagged("g")}),
        false
    )]
    #[case::tag_not_at_start(json!({"type": "thinking", "thinking": "x", "signature": format!("x{}", tagged("g"))}), false)]
    fn encrypted_reasoning_block_detection(#[case] input: Value, #[case] expected: bool) {
        assert_eq!(is_encrypted_reasoning_block(&block(input)), expected);
    }

    #[rstest]
    #[case::only_the_bridge_tagged_blocks(
        json!([
            {"role": "user", "content": "Solve it."},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "plan", "signature": tagged("gAAAA_1")},
                {"type": "redacted_thinking", "data": tagged("gAAAA_2")}
            ]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "plan", "signature": tagged("gAAAA_3")},
                {"type": "thinking", "thinking": "native", "signature": "EqQBCkYIAxgCIkA_anthropic_signed"},
                {"type": "redacted_thinking", "data": "EmwKAhgBEgy_anthropic_minted"},
                {"type": "text", "text": "The answer."}
            ]}
        ]),
        json!([
            {"role": "user", "content": "Solve it."},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "native", "signature": "EqQBCkYIAxgCIkA_anthropic_signed"},
                {"type": "redacted_thinking", "data": "EmwKAhgBEgy_anthropic_minted"},
                {"type": "text", "text": "The answer."}
            ]}
        ])
    )]
    #[case::bridge_turn_keeps_its_text(
        json!([
            {"role": "user", "content": "Solve it."},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "plan", "signature": tagged("gAAAA_1")},
                {"type": "redacted_thinking", "data": tagged("gAAAA_2")},
                {"type": "text", "text": "The answer."}
            ]},
            {"role": "user", "content": "And the next one?"}
        ]),
        json!([
            {"role": "user", "content": "Solve it."},
            {"role": "assistant", "content": [{"type": "text", "text": "The answer."}]},
            {"role": "user", "content": "And the next one?"}
        ])
    )]
    fn strip_encrypted_reasoning_blocks_rewrites(#[case] input: Value, #[case] expected: Value) {
        assert_eq!(apply(strip_encrypted_reasoning_blocks, input), expected);
    }

    #[rstest]
    #[case::anthropic_signed_blocks(json!([
        {"role": "user", "content": "Solve it."},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "plan", "signature": "EqQBCkYIAxgCIkA_anthropic_signed"},
            {"type": "redacted_thinking", "data": "EmwKAhgBEgy_anthropic_minted"},
            {"type": "text", "text": "The answer."}
        ]}
    ]))]
    #[case::string_content(json!([{"role": "user", "content": tagged("g")}]))]
    fn strip_encrypted_reasoning_blocks_leaves_untouched(#[case] input: Value) {
        assert_eq!(
            apply(strip_encrypted_reasoning_blocks, input.clone()),
            input
        );
    }

    #[rstest]
    #[case::bedrock_converse_wrapper("UVVKRA==", "QUJD")]
    #[case::claude_own_data("QUJD", "QUJD")]
    #[case::not_base64("EmwKAhgBEgy_anthropic_minted", "EmwKAhgBEgy_anthropic_minted")]
    #[case::wrapping_nothing("", "")]
    fn redacted_thinking_gets_claude_own_data_back(#[case] data: &str, #[case] expected: &str) {
        let turn = |data: &str| {
            json!([{"role": "assistant", "content": [
                {"type": "redacted_thinking", "data": data},
                {"type": "text", "text": "The answer."}
            ]}])
        };
        assert_eq!(
            apply(
                strip_encrypted_reasoning_blocks_restoring_redacted_data,
                turn(data)
            ),
            turn(expected)
        );
        assert_eq!(
            apply(strip_encrypted_reasoning_blocks, turn(data)),
            turn(data),
            "main's strip keeps a Bedrock Converse wrapper as sent"
        );
    }

    fn unparsed_block() -> Value {
        json!({"type": "text", "text": "Note.", "id": 5})
    }

    fn turn(blocks: Value) -> Value {
        json!({"role": "assistant", "content": blocks})
    }

    #[rstest]
    #[case::non_string_text(
        json!([{"type": "text", "text": 5}, unparsed_block()]),
        Some(json!([unparsed_block()]))
    )]
    #[case::blank_text(
        json!([{"type": "text", "text": " \n"}, unparsed_block()]),
        Some(json!([unparsed_block()]))
    )]
    #[case::empty_unsigned_thinking(
        json!([{"type": "thinking", "thinking": "", "signature": null}, unparsed_block()]),
        Some(json!([unparsed_block()]))
    )]
    #[case::empty_signed_thinking(
        json!([{"type": "thinking", "thinking": "", "signature": "sig"}, unparsed_block()]),
        Some(json!([{"type": "thinking", "thinking": "", "signature": "sig"}, unparsed_block()]))
    )]
    #[case::every_block_empty(json!([{"type": "text", "text": 5}, {"type": "text"}]), None)]
    #[case::no_blocks(json!([]), Some(json!([])))]
    fn strip_empty_raw_content_blocks_drops_what_python_drops(
        #[case] blocks: Value,
        #[case] expected: Option<Value>,
    ) {
        assert_eq!(
            strip_empty_raw_content_blocks(turn(blocks)),
            expected.map(turn)
        );
    }

    #[rstest]
    #[case::string_content(json!({"role": "user", "content": 5}))]
    #[case::not_an_object(json!("hi"))]
    fn raw_rewrites_leave_a_message_without_a_block_list(#[case] message: Value) {
        assert_eq!(
            strip_empty_raw_content_blocks(message.clone()),
            Some(message.clone())
        );
        assert_eq!(
            strip_raw_encrypted_reasoning_blocks(message.clone()),
            Some(message)
        );
    }

    #[rstest]
    fn sanitize_raw_tool_use_ids_rewrites_ids_python_rejects() {
        assert_eq!(
            sanitize_raw_tool_use_ids(turn(json!([
                unparsed_block(),
                {"type": "tool_use", "id": "functions.Bash:0", "name": "Bash", "input": {}},
                {"type": "server_tool_use", "id": "srvtoolu_ok", "name": "web_search"},
                {"type": "tool_result", "tool_use_id": "functions.Bash:0", "content": "ok"},
                {"type": "tool_result", "tool_use_id": 7}
            ]))),
            Some(turn(json!([
                unparsed_block(),
                {"type": "tool_use", "id": "functions_Bash_0", "name": "Bash", "input": {}},
                {"type": "server_tool_use", "id": "srvtoolu_ok", "name": "web_search"},
                {"type": "tool_result", "tool_use_id": "functions_Bash_0", "content": "ok"},
                {"type": "tool_result", "tool_use_id": 7}
            ])))
        );
    }

    #[rstest]
    #[case::replayed_results(
        json!([
            unparsed_block(),
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "paris"}},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
                {"type": "web_search_result", "url": "https://example.com", "title": "Paris", "snippet": "Capital."}
            ]}
        ]),
        json!([
            unparsed_block(),
            {"type": "text", "text": "Web search results for 'paris':\n\nTitle: Paris\nURL: https://example.com\nSnippet: Capital."}
        ])
    )]
    #[case::replayed_error(
        json!([
            unparsed_block(),
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": {"type": "web_search_tool_result_error"}}
        ]),
        json!([
            unparsed_block(),
            {"type": "text", "text": "Web search results:\n\nSearch failed: unavailable"}
        ])
    )]
    #[case::anthropic_issued(
        json!([
            unparsed_block(),
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
                {"type": "web_search_result", "url": "u", "encrypted_content": "blob"}
            ]}
        ]),
        json!([
            unparsed_block(),
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
                {"type": "web_search_result", "url": "u", "encrypted_content": "blob"}
            ]}
        ])
    )]
    fn flatten_unencrypted_raw_web_search_results_follows_python(
        #[case] blocks: Value,
        #[case] expected: Value,
    ) {
        assert_eq!(
            flatten_unencrypted_raw_web_search_results(turn(blocks)),
            Some(turn(expected))
        );
    }

    #[rstest]
    fn strip_raw_provider_specific_fields_drops_the_field_from_each_block() {
        assert_eq!(
            strip_raw_provider_specific_fields(turn(json!([
                unparsed_block(),
                {"type": "text", "text": "hi", "provider_specific_fields": {"x": 1}, "citations": null},
                "raw"
            ]))),
            Some(turn(json!([
                unparsed_block(),
                {"type": "text", "text": "hi", "citations": null},
                "raw"
            ])))
        );
    }

    fn advisor_exchange(id: Value) -> Value {
        json!([
            unparsed_block(),
            {"type": "server_tool_use", "id": id, "name": "advisor", "input": {}},
            {"type": "advisor_tool_result", "tool_use_id": id, "content": {"type": "advisor_result", "text": "Use channels."}},
            {"type": "text", "text": "Done."}
        ])
    }

    #[rstest]
    #[case::string_id(json!("srvtoolu_1"))]
    #[case::numeric_id(json!(7))]
    fn strip_raw_advisor_blocks_drops_the_exchange_from_an_assistant_turn(#[case] id: Value) {
        assert_eq!(
            strip_raw_advisor_blocks(turn(advisor_exchange(id))),
            Some(turn(
                json!([unparsed_block(), {"type": "text", "text": "Done."}])
            ))
        );
    }

    #[rstest]
    #[case::user_turn(json!({"role": "user", "content": advisor_exchange(json!("srvtoolu_1"))}))]
    #[case::empty_id(turn(advisor_exchange(json!(""))))]
    fn strip_raw_advisor_blocks_leaves_other_turns(#[case] message: Value) {
        assert_eq!(strip_raw_advisor_blocks(message.clone()), Some(message));
    }

    fn advisor_use(id: Value) -> Value {
        json!({"type": "server_tool_use", "id": id, "name": "advisor", "input": {}})
    }

    fn advisor_result(tool_use_id: Value) -> Value {
        json!({"type": "advisor_tool_result", "tool_use_id": tool_use_id, "content": {"type": "advisor_result", "text": "Advice."}})
    }

    #[rstest]
    #[case::int_answered_by_float(
        json!([advisor_use(json!(1)), advisor_result(json!(1.0)), unparsed_block()]),
        json!([unparsed_block()])
    )]
    #[case::float_answered_by_int(
        json!([advisor_use(json!(2.0)), advisor_result(json!(2)), unparsed_block()]),
        json!([unparsed_block()])
    )]
    #[case::true_answered_by_one(
        json!([advisor_use(json!(true)), advisor_result(json!(1)), unparsed_block()]),
        json!([unparsed_block()])
    )]
    #[case::one_answered_by_true(
        json!([advisor_use(json!(1)), advisor_result(json!(true)), unparsed_block()]),
        json!([unparsed_block()])
    )]
    #[case::string_not_answered_by_int(
        json!([advisor_use(json!("1")), advisor_result(json!(1)), unparsed_block()]),
        json!([advisor_result(json!(1)), unparsed_block()])
    )]
    #[case::int_beyond_float_precision(
        json!([advisor_use(json!(9_007_199_254_740_993_u64)), advisor_result(json!(9_007_199_254_740_992.0)), unparsed_block()]),
        json!([advisor_result(json!(9_007_199_254_740_992.0)), unparsed_block()])
    )]
    #[case::falsy_ids_are_kept(
        json!([
            advisor_use(json!(0)), advisor_result(json!(0)),
            advisor_use(json!(0.0)), advisor_result(json!(0.0)),
            advisor_use(json!(false)), advisor_result(json!(false)),
            advisor_use(Value::Null), advisor_result(Value::Null),
            {"type": "server_tool_use", "name": "advisor", "input": {}},
            {"type": "advisor_tool_result", "content": {}},
            advisor_use(json!("srvtoolu_1")), advisor_result(json!("srvtoolu_1"))
        ]),
        json!([
            advisor_use(json!(0)), advisor_result(json!(0)),
            advisor_use(json!(0.0)), advisor_result(json!(0.0)),
            advisor_use(json!(false)), advisor_result(json!(false)),
            advisor_use(Value::Null), advisor_result(Value::Null),
            {"type": "server_tool_use", "name": "advisor", "input": {}},
            {"type": "advisor_tool_result", "content": {}}
        ])
    )]
    #[case::only_advisor_blocks_answer_advisor_ids(
        json!([
            advisor_use(json!("a")),
            {"type": "server_tool_use", "id": "w", "name": "web_search", "input": {}},
            advisor_result(json!("w")),
            advisor_result(json!("unknown")),
            {"type": "tool_use", "id": "a", "name": "advisor", "input": {}},
            {"type": "tool_result", "tool_use_id": "a", "content": "ok"},
            advisor_result(json!("a"))
        ]),
        json!([
            {"type": "server_tool_use", "id": "w", "name": "web_search", "input": {}},
            advisor_result(json!("w")),
            advisor_result(json!("unknown")),
            {"type": "tool_use", "id": "a", "name": "advisor", "input": {}},
            {"type": "tool_result", "tool_use_id": "a", "content": "ok"}
        ])
    )]
    #[case::object_ids_compare_regardless_of_key_order(
        json!([advisor_use(json!({"a": 1, "b": 2})), advisor_result(json!({"b": 2, "a": 1})), unparsed_block()]),
        json!([unparsed_block()])
    )]
    #[case::nested_numbers_compare_as_json(
        json!([advisor_use(json!([1])), advisor_result(json!([1.0])), unparsed_block()]),
        json!([advisor_result(json!([1.0])), unparsed_block()])
    )]
    fn strip_raw_advisor_blocks_matches_ids_as_python_does(
        #[case] blocks: Value,
        #[case] expected: Value,
    ) {
        assert_eq!(strip_raw_advisor_blocks(turn(blocks)), Some(turn(expected)));
    }

    const MANY_BLOCKS: usize = 20_000;
    const MANY_BLOCKS_BUDGET: std::time::Duration = std::time::Duration::from_secs(1);

    fn many_ids() -> impl Iterator<Item = String> {
        (0..MANY_BLOCKS).map(|index| format!("srvtoolu_{index:08}"))
    }

    fn many_advisor_exchanges() -> Value {
        turn(Value::Array(
            many_ids()
                .map(|id| advisor_use(json!(id)))
                .chain(many_ids().map(|id| advisor_result(json!(id))))
                .chain([json!({"type": "text", "text": "Done."})])
                .collect(),
        ))
    }

    fn many_replayed_searches() -> Value {
        turn(Value::Array(
            many_ids()
                .map(|id| json!({"type": "server_tool_use", "id": id, "name": "web_search", "input": {"query": "q"}}))
                .chain(many_ids().map(
                    |id| json!({"type": "web_search_tool_result", "tool_use_id": id, "content": []}),
                ))
                .collect(),
        ))
    }

    fn timed<T>(rewrite: impl FnOnce() -> T) -> (T, std::time::Duration) {
        let started = std::time::Instant::now();
        let rewritten = rewrite();
        (rewritten, started.elapsed())
    }

    fn raw_blocks(message: Option<Value>) -> usize {
        message.unwrap()["content"].as_array().unwrap().len()
    }

    fn typed_blocks(messages: Vec<Message>) -> usize {
        match &messages[0].content {
            MessageContent::Blocks(blocks) => blocks.len(),
            MessageContent::Text(_) => 0,
        }
    }

    fn parsed(message: Value) -> Vec<Message> {
        serde_json::from_value(json!([message])).unwrap()
    }

    fn strip_raw_advisor(message: Value) -> (usize, std::time::Duration) {
        let (rewritten, elapsed) = timed(|| strip_raw_advisor_blocks(message));
        (raw_blocks(rewritten), elapsed)
    }

    fn strip_typed_advisor(message: Value) -> (usize, std::time::Duration) {
        let messages = parsed(message);
        let (rewritten, elapsed) = timed(|| strip_advisor_blocks(messages));
        (typed_blocks(rewritten), elapsed)
    }

    fn flatten_raw_searches(message: Value) -> (usize, std::time::Duration) {
        let (rewritten, elapsed) = timed(|| flatten_unencrypted_raw_web_search_results(message));
        (raw_blocks(rewritten), elapsed)
    }

    fn flatten_typed_searches(message: Value) -> (usize, std::time::Duration) {
        let messages = parsed(message);
        let (rewritten, elapsed) = timed(|| flatten_unencrypted_web_search_results(messages));
        (typed_blocks(rewritten), elapsed)
    }

    #[rstest]
    #[case::raw_advisor_exchanges(strip_raw_advisor, many_advisor_exchanges(), 1)]
    #[case::typed_advisor_exchanges(strip_typed_advisor, many_advisor_exchanges(), 1)]
    #[case::raw_replayed_searches(flatten_raw_searches, many_replayed_searches(), MANY_BLOCKS)]
    #[case::typed_replayed_searches(flatten_typed_searches, many_replayed_searches(), MANY_BLOCKS)]
    fn id_matching_scales_linearly_with_the_blocks(
        #[case] rewrite: fn(Value) -> (usize, std::time::Duration),
        #[case] message: Value,
        #[case] expected_blocks: usize,
    ) {
        let (blocks, elapsed) = rewrite(message);
        assert_eq!(blocks, expected_blocks);
        assert!(elapsed < MANY_BLOCKS_BUDGET, "{elapsed:?}");
    }

    #[rstest]
    #[case::bridge_tagged_blocks(
        json!([
            {"type": "thinking", "thinking": "plan", "signature": tagged("gAAAA_1")},
            {"type": "redacted_thinking", "data": tagged("gAAAA_2")},
            unparsed_block()
        ]),
        Some(json!([unparsed_block()]))
    )]
    #[case::converse_wrapped_redacted_data(
        json!([{"type": "redacted_thinking", "data": "UVVKRA=="}, unparsed_block()]),
        Some(json!([{"type": "redacted_thinking", "data": "QUJD"}, unparsed_block()]))
    )]
    #[case::only_bridge_tagged_blocks(
        json!([{"type": "thinking", "thinking": "plan", "signature": tagged("gAAAA_1")}]),
        None
    )]
    #[case::no_blocks(json!([]), Some(json!([])))]
    fn strip_raw_encrypted_reasoning_blocks_follows_python(
        #[case] blocks: Value,
        #[case] expected: Option<Value>,
    ) {
        assert_eq!(
            strip_raw_encrypted_reasoning_blocks(turn(blocks)),
            expected.map(turn)
        );
    }

    #[rstest]
    fn a_raw_message_the_rewrite_left_parseable_is_parsed_again() {
        let rewritten = map_messages(
            history(json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "text", "text": 5}, {"type": "text", "text": "ok"}]},
                {"role": "user", "content": [{"type": "text", "text": 5}]}
            ])),
            strip_empty_content_blocks,
            strip_empty_raw_content_blocks,
        );
        assert_eq!(
            rewritten,
            history(json!([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}
            ]))
        );
    }

    fn system(value: Value) -> Recognized<SystemPrompt> {
        serde_json::from_value(value).unwrap()
    }

    #[rstest]
    #[case::unparsed_list(
        json!([{"type": "text", "text": "x-anthropic-billing-header: cc_version=1"}, unparsed_block()]),
        Some(json!([unparsed_block()]))
    )]
    #[case::unparsed_list_of_billing_blocks(
        json!([{"type": "text", "text": "x-anthropic-billing-header: cc_version=1"}, 5, {"type": "text", "text": 5}]),
        Some(json!([5, {"type": "text", "text": 5}]))
    )]
    #[case::only_billing(json!([{"type": "text", "text": "x-anthropic-billing-header: cc_version=1", "id": 5}]), None)]
    #[case::object(json!({"type": "text", "text": "hi"}), Some(json!({"type": "text", "text": "hi"})))]
    #[case::empty_object(json!({}), None)]
    #[case::empty_text(json!(""), None)]
    fn strip_billing_metadata_from_a_system_prompt_as_sent(
        #[case] sent: Value,
        #[case] expected: Option<Value>,
    ) {
        assert_eq!(
            strip_billing_metadata_from_system(system(sent)),
            expected.map(system)
        );
    }

    #[rstest]
    #[case::advisor_exchange_between_texts(
        json!([
            {"role": "user", "content": "Build a worker pool."},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me consult the advisor."},
                {"type": "server_tool_use", "id": "srvtoolu_abc123", "name": "advisor", "input": {}},
                {"type": "advisor_tool_result", "tool_use_id": "srvtoolu_abc123",
                 "content": {"type": "advisor_result", "text": "Use channels."}},
                {"type": "text", "text": "Here is the implementation."}
            ]}
        ]),
        json!([
            {"role": "user", "content": "Build a worker pool."},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me consult the advisor."},
                {"type": "text", "text": "Here is the implementation."}
            ]}
        ])
    )]
    #[case::only_results_of_this_turns_advisor_calls(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "adv_1", "name": "advisor", "input": {}},
            {"type": "advisor_tool_result", "tool_use_id": "adv_1", "content": "advice"},
            {"type": "advisor_tool_result", "tool_use_id": "other", "content": "kept"},
            {"type": "tool_result", "tool_use_id": "adv_1", "content": "kept"},
            {"type": "text", "text": "answer"}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "advisor_tool_result", "tool_use_id": "other", "content": "kept"},
            {"type": "tool_result", "tool_use_id": "adv_1", "content": "kept"},
            {"type": "text", "text": "answer"}
        ]}])
    )]
    #[case::advisor_call_without_result(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "adv_1", "name": "advisor", "input": {}},
            {"type": "text", "text": "answer"}
        ]}]),
        json!([{"role": "assistant", "content": [{"type": "text", "text": "answer"}]}])
    )]
    #[case::advisor_only_turn_keeps_an_empty_block_list(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "adv_1", "name": "advisor", "input": {}},
            {"type": "advisor_tool_result", "tool_use_id": "adv_1", "content": "advice"}
        ]}]),
        json!([{"role": "assistant", "content": []}])
    )]
    fn strip_advisor_blocks_rewrites(#[case] input: Value, #[case] expected: Value) {
        assert_eq!(apply(strip_advisor_blocks, input), expected);
    }

    #[rstest]
    #[case::no_advisor_blocks(json!([
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Hi there"},
            {"type": "tool_use", "id": "toolu_abc", "name": "get_weather", "input": {"location": "SF"}}
        ]}
    ]))]
    #[case::user_turn(json!([{"role": "user", "content": [
        {"type": "server_tool_use", "id": "adv_2", "name": "advisor", "input": {}},
        {"type": "advisor_tool_result", "tool_use_id": "adv_2", "content": "advice"}
    ]}]))]
    #[case::other_server_tool(json!([{"role": "assistant", "content": [
        {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {}},
        {"type": "advisor_tool_result", "tool_use_id": "s1", "content": "advice"}
    ]}]))]
    #[case::client_tool_named_advisor(json!([{"role": "assistant", "content": [
        {"type": "tool_use", "id": "t1", "name": "advisor", "input": {}},
        {"type": "advisor_tool_result", "tool_use_id": "t1", "content": "advice"}
    ]}]))]
    #[case::advisor_call_with_empty_id(json!([{"role": "assistant", "content": [
        {"type": "server_tool_use", "id": "", "name": "advisor", "input": {}},
        {"type": "advisor_tool_result", "tool_use_id": "", "content": "advice"}
    ]}]))]
    #[case::advisor_call_without_id(json!([{"role": "assistant", "content": [
        {"type": "server_tool_use", "name": "advisor", "input": {}}
    ]}]))]
    #[case::string_content(json!([{"role": "assistant", "content": "advisor"}]))]
    fn strip_advisor_blocks_leaves_untouched(#[case] input: Value) {
        assert_eq!(apply(strip_advisor_blocks, input.clone()), input);
    }

    #[rstest]
    #[case::results_keep_their_evidence(
        json!([
            {"role": "user", "content": "latest version?"},
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "latest version"}},
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [
                    {"type": "web_search_result", "url": "https://example.com/releases", "title": "Releases",
                     "page_age": null, "encrypted_content": "", "snippet": "Latest release v1.95.0"}
                ]},
                {"type": "text", "text": "v1.95.0"}
            ]}
        ]),
        json!([
            {"role": "user", "content": "latest version?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Web search results for 'latest version':\n\nTitle: Releases\nURL: https://example.com/releases\nSnippet: Latest release v1.95.0"},
                {"type": "text", "text": "v1.95.0"}
            ]}
        ])
    )]
    #[case::each_result_lists_only_its_present_fields(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
                {"type": "web_search_result", "title": "A"},
                {"type": "web_search_result", "snippet": "b"},
                {"type": "web_search_result", "url": "https://c"}
            ]}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results for 'q':\n\nTitle: A\n\nSnippet: b\n\nURL: https://c"}
        ]}])
    )]
    #[case::result_without_fields_renders_the_header_only(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [{"type": "web_search_result"}]}
        ]}]),
        json!([{"role": "assistant", "content": [{"type": "text", "text": "Web search results for 'q':"}]}])
    )]
    #[case::resultless_search(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "who won"}},
            {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []},
            {"type": "text", "text": "I could not find that."}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results for 'who won':\n\nNo results were returned."},
            {"type": "text", "text": "I could not find that."}
        ]}])
    )]
    #[case::failed_search(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "q"}},
            {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
             "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results for 'q':\n\nSearch failed: max_uses_exceeded"}
        ]}])
    )]
    #[case::failed_search_without_error_code(
        json!([{"role": "assistant", "content": [
            {"type": "web_search_tool_result", "tool_use_id": "e1", "content": {"type": "web_search_tool_result_error"}}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results:\n\nSearch failed: unavailable"}
        ]}])
    )]
    #[case::server_tool_use_without_query(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search"},
            {"type": "server_tool_use", "id": "s2", "name": "web_search", "input": {}},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []},
            {"type": "web_search_tool_result", "tool_use_id": "s2", "content": []}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results:\n\nNo results were returned."},
            {"type": "text", "text": "Web search results:\n\nNo results were returned."}
        ]}])
    )]
    #[case::genuine_results_in_the_same_turn_stay(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "rust"}},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
                {"type": "web_search_result", "url": "https://r", "title": "Rust", "snippet": "fast"}
            ]},
            {"type": "server_tool_use", "id": "s2", "name": "web_search", "input": {"query": "real"}},
            {"type": "web_search_tool_result", "tool_use_id": "s2", "content": [
                {"type": "web_search_result", "url": "https://a", "title": "A", "snippet": "b", "encrypted_content": "enc"}
            ]},
            {"type": "text", "text": "done"}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results for 'rust':\n\nTitle: Rust\nURL: https://r\nSnippet: fast"},
            {"type": "server_tool_use", "id": "s2", "name": "web_search", "input": {"query": "real"}},
            {"type": "web_search_tool_result", "tool_use_id": "s2", "content": [
                {"type": "web_search_result", "url": "https://a", "title": "A", "snippet": "b", "encrypted_content": "enc"}
            ]},
            {"type": "text", "text": "done"}
        ]}])
    )]
    #[case::other_blocks_sharing_the_tool_use_id_stay(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []},
            {"type": "tool_result", "tool_use_id": "s1", "content": "x"}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results for 'q':\n\nNo results were returned."},
            {"type": "tool_result", "tool_use_id": "s1", "content": "x"}
        ]}])
    )]
    #[case::query_lookup_stays_within_the_message(
        json!([
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}}
            ]},
            {"role": "assistant", "content": [
                {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []}
            ]}
        ]),
        json!([
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Web search results:\n\nNo results were returned."}
            ]}
        ])
    )]
    #[case::result_without_any_field_keeps_its_slot(
        json!([{"role": "assistant", "content": [
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
                {"type": "web_search_result", "url": "https://a", "title": "A"},
                {"type": "web_search_result"},
                {"type": "web_search_result", "title": "B"}
            ]}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results:\n\nTitle: A\nURL: https://a\n\n\n\nTitle: B"}
        ]}])
    )]
    #[case::non_string_query_keeps_its_server_tool_use(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": 123}},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": 123}},
            {"type": "text", "text": "Web search results:\n\nNo results were returned."}
        ]}])
    )]
    #[case::non_object_input_keeps_its_server_tool_use(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": "q"},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": "q"},
            {"type": "text", "text": "Web search results:\n\nNo results were returned."}
        ]}])
    )]
    #[case::repeated_tool_use_id_renders_each_block_from_its_own_results(
        json!([{"role": "assistant", "content": [
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": {"type": "web_search_tool_result_error", "error_code": "max_uses"}}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results:\n\nNo results were returned."},
            {"type": "text", "text": "Web search results:\n\nSearch failed: max_uses"}
        ]}])
    )]
    #[case::encrypted_block_sharing_a_replayed_id_stays(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
                {"type": "web_search_result", "url": "https://a", "encrypted_content": "enc"}
            ]}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results for 'q':\n\nNo results were returned."},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
                {"type": "web_search_result", "url": "https://a", "encrypted_content": "enc"}
            ]}
        ]}])
    )]
    #[case::last_query_wins_for_a_repeated_server_tool_use_id(
        json!([{"role": "assistant", "content": [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "first"}},
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "second"}},
            {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "Web search results for 'second':\n\nNo results were returned."}
        ]}])
    )]
    fn flatten_unencrypted_web_search_results_rewrites(
        #[case] input: Value,
        #[case] expected: Value,
    ) {
        assert_eq!(
            apply(flatten_unencrypted_web_search_results, input),
            expected
        );
    }

    #[rstest]
    #[case::anthropic_issued_results(json!([{"role": "assistant", "content": [
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "q"}},
        {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [
            {"type": "web_search_result", "url": "https://example.com", "title": "Example",
             "page_age": null, "encrypted_content": "EqgfCioIARgBIiQ4"}
        ]}
    ]}]))]
    #[case::any_encrypted_result_marks_the_block_genuine(json!([{"role": "assistant", "content": [
        {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
            {"type": "web_search_result", "url": "https://a", "encrypted_content": ""},
            {"type": "web_search_result", "url": "https://b", "encrypted_content": "enc"}
        ]}
    ]}]))]
    #[case::result_without_tool_use_id(json!([{"role": "assistant", "content": [
        {"type": "web_search_tool_result", "content": []}
    ]}]))]
    #[case::foreign_item_in_results(json!([{"role": "assistant", "content": [
        {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
            {"type": "web_search_result", "url": "https://a"},
            {"type": "text", "text": "x"}
        ]}
    ]}]))]
    #[case::error_inside_result_array(json!([{"role": "assistant", "content": [
        {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
            {"type": "web_search_tool_result_error", "error_code": "unavailable"}
        ]}
    ]}]))]
    #[case::result_with_null_url(json!([{"role": "assistant", "content": [
        {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
            {"type": "web_search_result", "url": null, "title": "A"}
        ]}
    ]}]))]
    #[case::string_result_content(json!([{"role": "assistant", "content": [
        {"type": "web_search_tool_result", "tool_use_id": "s1", "content": "oops"}
    ]}]))]
    #[case::object_content_that_is_not_an_error(json!([{"role": "assistant", "content": [
        {"type": "web_search_tool_result", "tool_use_id": "s1", "content": {"type": "web_search_result", "url": "https://a"}}
    ]}]))]
    #[case::string_content(json!([{"role": "assistant", "content": "web_search_tool_result"}]))]
    fn flatten_unencrypted_web_search_results_leaves_untouched(#[case] input: Value) {
        assert_eq!(
            apply(flatten_unencrypted_web_search_results, input.clone()),
            input
        );
    }

    #[rstest]
    #[case::with_results(json!([{"type": "web_search_result", "url": "u", "title": "Rome", "snippet": "s", "page_age": null}]))]
    #[case::without_results(json!([]))]
    fn flatten_unencrypted_web_search_results_is_idempotent(#[case] results: Value) {
        let input = replayed_search_turn(results);
        let once = apply(flatten_unencrypted_web_search_results, input.clone());
        let twice = apply(flatten_unencrypted_web_search_results, once.clone());
        assert_ne!(once, input);
        assert_eq!(twice, once);
    }

    const OAUTH_TOKEN: &str = "sk-ant-oat01-token";
    const OAUTH_BEARER: &str = "Bearer sk-ant-oat01-token";
    const REGULAR_KEY: &str = "sk-ant-api03-regular";
    const OAUTH_BETA: &str = "oauth-2025-04-20";
    const BROWSER_ACCESS: (&str, &str) = ("anthropic-dangerous-direct-browser-access", "true");

    #[rstest]
    #[case::no_beta_header(&[("x-api-key", "k")], &[], &[("x-api-key", "k")])]
    #[case::blank_beta_header(&[("Anthropic-Beta", " , "), ("x-api-key", "k")], &[], &[("Anthropic-Beta", " , "), ("x-api-key", "k")])]
    #[case::added_to_no_header(&[("x-api-key", "k")], &["b"], &[("x-api-key", "k"), ("anthropic-beta", "b")])]
    #[case::added_to_blank_header(&[("anthropic-beta", "  ")], &["b"], &[("anthropic-beta", "b")])]
    #[case::sorted_after_merge(&[("anthropic-beta", "c,a")], &["b"], &[("anthropic-beta", "a,b,c")])]
    #[case::already_present(&[("anthropic-beta", "a,b")], &["a"], &[("anthropic-beta", "a,b")])]
    #[case::existing_normalized_without_additions(
        &[("Anthropic-Beta", "b, a ,b"), ("x-api-key", "k")],
        &[],
        &[("x-api-key", "k"), ("anthropic-beta", "a,b")]
    )]
    #[case::every_casing_unioned_into_one_lowercase_header(
        &[("anthropic-beta", "a"), ("ANTHROPIC-BETA", "c"), ("x-api-key", "k")],
        &["b"],
        &[("x-api-key", "k"), ("anthropic-beta", "a,b,c")]
    )]
    fn merge_beta_headers_replaces_the_header_with_the_sorted_union(
        #[case] input: &[(&str, &str)],
        #[case] added: &[&str],
        #[case] expected: &[(&str, &str)],
    ) {
        assert_eq!(
            merge_beta_headers(headers(input), betas(added)),
            headers(expected)
        );
    }

    #[rstest]
    #[case::raw_token(OAUTH_TOKEN, Some(OAUTH_TOKEN))]
    #[case::bare_prefix(ANTHROPIC_OAUTH_TOKEN_PREFIX, Some(ANTHROPIC_OAUTH_TOKEN_PREFIX))]
    #[case::bearer_token(OAUTH_BEARER, None)]
    #[case::api_key(REGULAR_KEY, None)]
    #[case::empty("", None)]
    #[case::uppercase_prefix("sk-ant-OAT01-abc123", None)]
    #[case::prefix_not_at_start(" sk-ant-oat01-abc123", None)]
    fn oauth_token_parses_only_the_raw_token(#[case] value: &str, #[case] expected: Option<&str>) {
        assert_eq!(OauthToken::parse(value).map(OauthToken::as_str), expected);
    }

    #[rstest]
    #[case::raw_token(OAUTH_TOKEN, Some(OAUTH_TOKEN))]
    #[case::bearer_token(OAUTH_BEARER, Some(OAUTH_TOKEN))]
    #[case::api_key(REGULAR_KEY, None)]
    #[case::bearer_api_key("Bearer sk-ant-api01-abc123", None)]
    #[case::empty("", None)]
    #[case::shouting_prefix("SK-ANT-OAT01-abc123", None)]
    #[case::lowercase_bearer("bearer sk-ant-oat01-abc123", None)]
    #[case::bearer_stripped_once("Bearer Bearer sk-ant-oat01-abc123", None)]
    fn oauth_key_parses_the_token_behind_an_optional_bearer(
        #[case] value: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(
            OauthToken::parse_key(value).map(OauthToken::as_str),
            expected
        );
    }

    #[rstest]
    #[case::bearer(&[("authorization", OAUTH_BEARER)], Some(OAUTH_TOKEN))]
    #[case::uppercase_header(&[("AUTHORIZATION", OAUTH_BEARER)], Some(OAUTH_TOKEN))]
    #[case::non_oauth_bearer(&[("authorization", "Bearer some-proxy-token")], None)]
    #[case::token_without_the_bearer_scheme(&[("authorization", OAUTH_TOKEN)], None)]
    #[case::lowercase_bearer_scheme(&[("authorization", "bearer sk-ant-oat01-token")], None)]
    #[case::token_in_x_api_key(&[("x-api-key", OAUTH_TOKEN)], None)]
    #[case::no_headers(&[], None)]
    fn forwarded_oauth_bearer_reads_the_authorization_header(
        #[case] forwarded: &[(&str, &str)],
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(
            forwarded_oauth_bearer(&headers(forwarded)).map(OauthToken::as_str),
            expected
        );
    }

    #[rstest]
    #[case::forwarded_bearer_drops_forwarded_and_deployment_keys(
        &[("X-Api-Key", REGULAR_KEY), ("Authorization", OAUTH_BEARER)],
        Some(REGULAR_KEY),
        &[],
    )]
    #[case::forwarded_bearer_keeps_unrelated_headers_in_place(
        &[("anthropic-version", "2023-06-01"), ("authorization", OAUTH_BEARER)],
        None,
        &[("anthropic-version", "2023-06-01")],
    )]
    #[case::forwarded_bearer_wins_over_an_oauth_api_key(
        &[("authorization", OAUTH_BEARER)],
        Some("sk-ant-oat01-deployment"),
        &[],
    )]
    #[case::api_key_alone(&[], Some(OAUTH_TOKEN), &[])]
    #[case::api_key_removes_a_forwarded_x_api_key(&[("x-api-key", OAUTH_TOKEN)], Some(OAUTH_TOKEN), &[])]
    #[case::api_key_keeps_a_forwarded_non_oauth_bearer(
        &[("Authorization", "Bearer some-proxy-token")],
        Some(OAUTH_TOKEN),
        &[("Authorization", "Bearer some-proxy-token")],
    )]
    fn oauth_token_is_the_whole_credential(
        #[case] forwarded: &[(&str, &str)],
        #[case] api_key: Option<&str>,
        #[case] kept: &[(&str, &str)],
    ) {
        let expected = kept
            .iter()
            .copied()
            .chain([("anthropic-beta", OAUTH_BETA), BROWSER_ACCESS])
            .collect::<Vec<_>>();
        assert_eq!(
            optionally_handle_anthropic_oauth(headers(forwarded), api_key),
            OauthHandling::Bearer {
                headers: headers(&expected),
                token: SecretValue::new(OAUTH_TOKEN),
            }
        );
    }

    #[rstest]
    #[case::forwarded_bearer_merges_a_differently_cased_beta_header(
        &[("Anthropic-Beta", "web-search-2025-03-05"), ("authorization", OAUTH_BEARER)],
        None,
    )]
    #[case::forwarded_bearer_dedupes_an_existing_oauth_beta(
        &[("anthropic-beta", "web-search-2025-03-05, oauth-2025-04-20"), ("authorization", OAUTH_BEARER)],
        None,
    )]
    #[case::api_key_merges_the_existing_beta_header(
        &[("anthropic-beta", " web-search-2025-03-05 ,")],
        Some(OAUTH_TOKEN),
    )]
    #[case::forwarded_bearer_unions_every_beta_header_casing(
        &[("anthropic-beta", "oauth-2025-04-20"), ("ANTHROPIC-BETA", "web-search-2025-03-05"), ("authorization", OAUTH_BEARER)],
        None,
    )]
    fn oauth_beta_merges_into_existing_betas(
        #[case] forwarded: &[(&str, &str)],
        #[case] api_key: Option<&str>,
    ) {
        assert_eq!(
            optionally_handle_anthropic_oauth(headers(forwarded), api_key),
            OauthHandling::Bearer {
                headers: headers(&[
                    ("anthropic-beta", "oauth-2025-04-20,web-search-2025-03-05"),
                    BROWSER_ACCESS,
                ]),
                token: SecretValue::new(OAUTH_TOKEN),
            }
        );
    }

    #[rstest]
    #[case::x_api_key(&[("x-api-key", "caller-key")], Some("sk-other"))]
    #[case::non_oauth_bearer(&[("Authorization", "Bearer some-proxy-token")], Some(REGULAR_KEY))]
    #[case::oauth_token_without_the_bearer_scheme(&[("authorization", OAUTH_TOKEN)], None)]
    #[case::bearer_prefixed_api_key(&[], Some(OAUTH_BEARER))]
    #[case::nothing(&[], None)]
    fn without_an_oauth_token_the_headers_are_untouched(
        #[case] forwarded: &[(&str, &str)],
        #[case] api_key: Option<&str>,
    ) {
        assert_eq!(
            optionally_handle_anthropic_oauth(headers(forwarded), api_key),
            OauthHandling::Untouched(headers(forwarded))
        );
    }

    #[rstest]
    #[case::api_key_param(Some("sk-param"), &[], Some(("x-api-key", "sk-param")))]
    #[case::api_key_param_over_env_key_and_auth_token(
        Some("sk-param"),
        &[("ANTHROPIC_API_KEY", "sk-env"), ("ANTHROPIC_AUTH_TOKEN", "env-token")],
        Some(("x-api-key", "sk-param")),
    )]
    #[case::env_key_without_a_param(None, &[("ANTHROPIC_API_KEY", "sk-env")], Some(("x-api-key", "sk-env")))]
    #[case::env_key_when_the_param_is_blank(Some("  "), &[("ANTHROPIC_API_KEY", "sk-env")], Some(("x-api-key", "sk-env")))]
    #[case::env_key_over_auth_token(
        None,
        &[("ANTHROPIC_API_KEY", "sk-env"), ("ANTHROPIC_AUTH_TOKEN", "env-token")],
        Some(("x-api-key", "sk-env")),
    )]
    #[case::auth_token_as_a_bearer(
        None,
        &[("ANTHROPIC_AUTH_TOKEN", "env-token")],
        Some(("Authorization", "env-token")),
    )]
    #[case::auth_token_stripped_of_whitespace(
        None,
        &[("ANTHROPIC_AUTH_TOKEN", "  env-token \t\n\u{1f}")],
        Some(("Authorization", "env-token")),
    )]
    #[case::auth_token_when_the_env_key_is_blank(
        None,
        &[("ANTHROPIC_API_KEY", " \t"), ("ANTHROPIC_AUTH_TOKEN", "env-token")],
        Some(("Authorization", "env-token")),
    )]
    #[case::oauth_param_as_a_bearer(Some(OAUTH_TOKEN), &[], Some(("Authorization", OAUTH_TOKEN)))]
    #[case::bearer_prefixed_oauth_env_key_as_a_bearer_once(
        None,
        &[("ANTHROPIC_API_KEY", OAUTH_BEARER)],
        Some(("Authorization", OAUTH_TOKEN)),
    )]
    #[case::env_key_stripped_of_whitespace(
        None,
        &[("ANTHROPIC_API_KEY", "  sk-env \t\n\u{1f}")],
        Some(("x-api-key", "sk-env")),
    )]
    #[case::api_key_param_stripped_of_separators(Some("\u{1c}sk-param "), &[], Some(("x-api-key", "sk-param")))]
    #[case::separator_only_env_key_falls_to_the_auth_token(
        None,
        &[("ANTHROPIC_API_KEY", "\u{1c}\u{1f}"), ("ANTHROPIC_AUTH_TOKEN", "env-token")],
        Some(("Authorization", "env-token")),
    )]
    #[case::no_credentials(None, &[], None)]
    #[case::separator_only_auth_token(None, &[("ANTHROPIC_AUTH_TOKEN", "\u{1c}\u{1f}")], None)]
    #[case::blank_everything(Some(""), &[("ANTHROPIC_API_KEY", "  "), ("ANTHROPIC_AUTH_TOKEN", " \t")], None)]
    fn auth_header_prefers_the_key_then_the_auth_token(
        #[case] api_key: Option<&str>,
        #[case] vars: &'static [(&'static str, &'static str)],
        #[case] expected: Option<(&str, &str)>,
    ) {
        assert_eq!(
            credential(get_auth_header(api_key, &env(vars))),
            expected.map(|(header, secret)| (header, secret.to_string()))
        );
    }

    #[rstest]
    #[case::param(Some("sk-param"), &[("ANTHROPIC_API_KEY", "sk-env")], Ok("sk-param"))]
    #[case::blank_param_falls_back_to_env(Some("  "), &[("ANTHROPIC_API_KEY", "sk-env")], Ok("sk-env"))]
    #[case::env_without_param(None, &[("ANTHROPIC_API_KEY", "sk-env")], Ok("sk-env"))]
    #[case::blank_env_is_missing(None, &[("ANTHROPIC_API_KEY", " ")], Err(()))]
    #[case::nothing_is_missing(None, &[], Err(()))]
    fn api_key_resolution(
        #[case] api_key: Option<&str>,
        #[case] vars: &'static [(&'static str, &'static str)],
        #[case] expected: Result<&str, ()>,
    ) {
        assert_eq!(
            resolve_anthropic_api_key(api_key, &env(vars)).map_err(|error| {
                assert!(matches!(
                    error,
                    litellm_auth::Error::MissingApiKey {
                        provider: "Anthropic",
                        environment_variable: "ANTHROPIC_API_KEY",
                    }
                ));
            }),
            expected.map(str::to_string)
        );
    }

    #[rstest]
    #[case::regex_tool(Some(json!([{"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}])), true)]
    #[case::bm25_tool(Some(json!([{"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}])), true)]
    #[case::after_other_tools(
        Some(json!([{"name": "get_weather", "input_schema": {}}, {"type": "tool_search_tool_bm25_20251119"}])),
        true
    )]
    #[case::function_tool(Some(json!([{"type": "function", "function": {"name": "get_weather"}}])), false)]
    #[case::name_without_type(Some(json!([{"name": "tool_search_tool_regex_20251119"}])), false)]
    #[case::empty_tools(Some(json!([])), false)]
    #[case::no_tools(None, false)]
    fn tool_search_detection(#[case] input: Option<Value>, #[case] expected: bool) {
        assert_eq!(is_tool_search_used(tools(input).as_deref()), expected);
    }

    #[rstest]
    #[case::advisor_tool(Some(json!([{"type": "advisor_20260301", "name": "advisor"}])), true)]
    #[case::after_other_tools(Some(json!([{"name": "f", "input_schema": {}}, {"type": "advisor_20260301"}])), true)]
    #[case::tool_named_advisor(Some(json!([{"name": "advisor", "input_schema": {}}])), false)]
    #[case::other_server_tool(Some(json!([{"type": "web_search_20250305", "name": "web_search"}])), false)]
    #[case::empty_tools(Some(json!([])), false)]
    #[case::no_tools(None, false)]
    fn advisor_tool_detection(#[case] input: Option<Value>, #[case] expected: bool) {
        assert_eq!(has_advisor_tool(tools(input).as_deref()), expected);
    }

    #[rstest]
    #[case::param_without_history(Some(json!({})), json!([]), true)]
    #[case::param_with_unsigned_history(
        Some(json!({"trigger": 1})),
        json!([{"role": "assistant", "content": [{"type": "compaction", "content": "c"}]}]),
        true
    )]
    #[case::signed_block(None, json!([{"role": "assistant", "content": [{"type": "compaction", "content": "c", "signature": "s"}]}]), true)]
    #[case::signed_block_later_in_history(
        None,
        json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "a"}, {"type": "compaction", "content": "c", "signature": "s"}]}
        ]),
        true
    )]
    #[case::unsigned_block(None, json!([{"role": "assistant", "content": [{"type": "compaction", "content": "c"}]}]), false)]
    #[case::empty_signature(None, json!([{"role": "assistant", "content": [{"type": "compaction", "content": "c", "signature": ""}]}]), false)]
    #[case::signed_non_compaction_block(
        None,
        json!([{"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": "s"}]}]),
        false
    )]
    #[case::string_content(None, json!([{"role": "user", "content": "compaction"}]), false)]
    #[case::neither(None, json!([]), false)]
    fn native_compaction_beta_requirement(
        #[case] compaction: Option<Value>,
        #[case] messages: Value,
        #[case] expected: bool,
    ) {
        for unparsed in [UnparsedMessages::Read, UnparsedMessages::Skipped] {
            assert_eq!(
                requires_native_compaction_beta(
                    compaction.as_ref(),
                    &history(messages.clone()),
                    unparsed
                ),
                expected
            );
        }
    }

    #[rstest]
    #[case::read(UnparsedMessages::Read, true)]
    #[case::skipped(UnparsedMessages::Skipped, false)]
    fn a_signed_compaction_block_in_an_unparsed_message_counts_only_where_unparsed_messages_are_read(
        #[case] unparsed: UnparsedMessages,
        #[case] expected: bool,
    ) {
        let messages = history(json!([{"role": "assistant", "content": [
            {"type": "text", "text": 5},
            {"type": "compaction", "content": "c", "signature": "s"}
        ]}]));
        assert_eq!(
            requires_native_compaction_beta(None, &messages, unparsed),
            expected
        );
    }

    #[rstest]
    #[case::minimal_only(tiers(true, false, false, false, false, false), [false, false, false, false, false])]
    #[case::low_only(tiers(false, true, false, false, false, false), [true, false, false, false, false])]
    #[case::medium_only(tiers(false, false, true, false, false, false), [false, true, false, false, false])]
    #[case::high_only(tiers(false, false, false, true, false, false), [false, false, true, false, false])]
    #[case::xhigh_only(tiers(false, false, false, false, true, false), [false, false, false, true, false])]
    #[case::max_only(tiers(false, false, false, false, false, true), [false, false, false, false, true])]
    fn supports_effort_tier_reads_the_matching_flag(
        #[case] effort_tiers: SupportedEffortTiers,
        #[case] expected: [bool; 5],
        unmapped: MessagesModelCapabilities,
    ) {
        let capabilities = MessagesModelCapabilities {
            effort_tiers,
            ..unmapped
        };
        assert_eq!(
            EffortLevel::VARIANTS
                .iter()
                .map(|level| supports_effort_tier(&capabilities, *level))
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[rstest]
    #[case::unmapped(false, false, false, SupportedEffortTiers::default(), false)]
    #[case::reasoning_and_adaptive_thinking_alone(
        true,
        true,
        false,
        SupportedEffortTiers::default(),
        false
    )]
    #[case::output_config_without_tiers(false, false, true, SupportedEffortTiers::default(), true)]
    #[case::minimal_tier(
        false,
        false,
        false,
        tiers(true, false, false, false, false, false),
        true
    )]
    #[case::low_tier(
        false,
        false,
        false,
        tiers(false, true, false, false, false, false),
        true
    )]
    #[case::medium_tier(
        false,
        false,
        false,
        tiers(false, false, true, false, false, false),
        true
    )]
    #[case::high_tier(
        false,
        false,
        false,
        tiers(false, false, false, true, false, false),
        true
    )]
    #[case::xhigh_tier(
        false,
        false,
        false,
        tiers(false, false, false, false, true, false),
        true
    )]
    #[case::max_tier(
        false,
        false,
        false,
        tiers(false, false, false, false, false, true),
        true
    )]
    fn supports_effort_param_cases(
        #[case] supports_reasoning: bool,
        #[case] supports_adaptive_thinking: bool,
        #[case] supports_output_config: bool,
        #[case] effort_tiers: SupportedEffortTiers,
        #[case] expected: bool,
        unmapped: MessagesModelCapabilities,
    ) {
        let capabilities = MessagesModelCapabilities {
            supports_reasoning,
            supports_adaptive_thinking,
            supports_output_config,
            effort_tiers,
            ..unmapped
        };
        assert_eq!(supports_effort_param(&capabilities), expected);
    }

    #[rstest]
    #[case::max_on_adaptive_thinking_model(
        true,
        SupportedEffortTiers::default(),
        EffortLevel::Max,
        true
    )]
    #[case::max_on_max_tier_model(
        false,
        tiers(false, false, false, false, false, true),
        EffortLevel::Max,
        true
    )]
    #[case::max_on_output_config_only_model(
        false,
        SupportedEffortTiers::default(),
        EffortLevel::Max,
        false
    )]
    #[case::max_on_xhigh_tier_model(
        false,
        tiers(false, false, false, false, true, false),
        EffortLevel::Max,
        false
    )]
    #[case::xhigh_on_xhigh_tier_model(
        false,
        tiers(false, false, false, false, true, false),
        EffortLevel::Xhigh,
        true
    )]
    #[case::xhigh_on_adaptive_thinking_model(
        true,
        SupportedEffortTiers::default(),
        EffortLevel::Xhigh,
        false
    )]
    #[case::xhigh_on_max_tier_model(
        false,
        tiers(false, false, false, false, false, true),
        EffortLevel::Xhigh,
        false
    )]
    #[case::high_on_unmapped_model(false, SupportedEffortTiers::default(), EffortLevel::High, true)]
    #[case::low_on_unmapped_model(false, SupportedEffortTiers::default(), EffortLevel::Low, true)]
    fn accepts_effort_cases(
        #[case] supports_adaptive_thinking: bool,
        #[case] effort_tiers: SupportedEffortTiers,
        #[case] level: EffortLevel,
        #[case] expected: bool,
        unmapped: MessagesModelCapabilities,
    ) {
        let capabilities = MessagesModelCapabilities {
            supports_output_config: true,
            supports_adaptive_thinking,
            effort_tiers,
            ..unmapped
        };
        assert_eq!(accepts_effort(&capabilities, level), expected);
    }

    #[rstest]
    fn unmapped_model_has_no_reasoning_features_but_accepts_sampling_params(
        unmapped: MessagesModelCapabilities,
    ) {
        assert_eq!(
            unmapped,
            MessagesModelCapabilities {
                supports_reasoning: false,
                supports_adaptive_thinking: false,
                thinking_always_on: false,
                thinking_on_by_default: false,
                supports_legacy_thinking: false,
                supports_output_config: false,
                supports_sampling_params: true,
                supports_speed: false,
                supports_mid_conversation_system: false,
                effort_tiers: tiers(false, false, false, false, false, false),
            }
        );
        assert_eq!(
            serde_json::from_value::<MessagesModelCapabilities>(json!({})).unwrap(),
            unmapped
        );
    }

    #[rstest]
    #[case::sampling_params_removed(
        json!({"supports_sampling_params": false}),
        MessagesModelCapabilities { supports_sampling_params: false, ..MessagesModelCapabilities::default() }
    )]
    #[case::fast_mode(
        json!({"supports_speed": true}),
        MessagesModelCapabilities { supports_speed: true, ..MessagesModelCapabilities::default() }
    )]
    #[case::partial_effort_tiers(
        json!({"supports_reasoning": true, "effort_tiers": {"xhigh": true}}),
        MessagesModelCapabilities {
            supports_reasoning: true,
            effort_tiers: tiers(false, false, false, false, true, false),
            ..MessagesModelCapabilities::default()
        }
    )]
    fn capabilities_fill_missing_flags_with_unmapped_defaults(
        #[case] input: Value,
        #[case] expected: MessagesModelCapabilities,
    ) {
        assert_eq!(
            serde_json::from_value::<MessagesModelCapabilities>(input).unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::bare_host("https://api.example.com", "https://api.example.com/v1/messages")]
    #[case::trailing_slash("https://api.example.com/", "https://api.example.com/v1/messages")]
    #[case::route_present(
        "https://api.example.com/v1/messages/",
        "https://api.example.com/v1/messages"
    )]
    #[case::route_and_query(
        "https://api.example.com/v1/messages?beta=true",
        "https://api.example.com/v1/messages?beta=true"
    )]
    #[case::query_without_route(
        "https://api.example.com/?x=1",
        "https://api.example.com?x=1/v1/messages"
    )]
    #[case::scheme_only("https://", "https:///v1/messages")]
    fn the_messages_url_follows_python_ensure_api_route(#[case] api_base: &str, #[case] url: &str) {
        assert_eq!(complete_anthropic_url(Some(api_base), &|_| None), url);
    }

    fn shared_cases(table: &str) -> Vec<(String, String, bool)> {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/anthropic_messages_http_error_retry.json"
        )))
        .unwrap();
        fixture[table]
            .as_array()
            .unwrap()
            .iter()
            .map(|case| {
                (
                    case["name"].as_str().unwrap().to_string(),
                    case["error_text"].as_str().unwrap().to_string(),
                    case["expected"].as_bool().unwrap(),
                )
            })
            .collect()
    }

    #[rstest]
    #[case::invalid_thinking_block(
        "invalid_thinking_block_error",
        is_anthropic_invalid_thinking_block_error
    )]
    #[case::thinking_binding_mismatch(
        "thinking_binding_mismatch_error",
        is_anthropic_thinking_binding_mismatch_error
    )]
    fn error_texts_match_the_cases_shared_with_python(
        #[case] table: &str,
        #[case] recognizes: fn(&str) -> bool,
    ) {
        let cases = shared_cases(table);
        assert!(!cases.is_empty());
        let mismatched: Vec<&String> = cases
            .iter()
            .filter(|(_, error_text, expected)| recognizes(error_text) != *expected)
            .map(|(name, _, _)| name)
            .collect();
        assert_eq!(mismatched, Vec::<&String>::new());
    }

    #[rstest]
    #[case::pinned(json!({"thinking": {"block_binding": {"prefix_mismatch_behavior": "error"}}}), true)]
    #[case::other_behavior(json!({"thinking": {"block_binding": {"prefix_mismatch_behavior": "drop"}}}), false)]
    #[case::string_binding(json!({"thinking": {"block_binding": "strict"}}), false)]
    #[case::no_thinking(json!({"messages": []}), false)]
    #[case::not_an_object(json!(["thinking"]), false)]
    fn a_pinned_binding_is_the_caller_asking_for_the_error(
        #[case] body: Value,
        #[case] expected: bool,
    ) {
        assert_eq!(
            anthropic_request_errors_on_thinking_binding_mismatch(&body),
            expected
        );
    }

    #[rstest]
    #[case::mixed_turn(
        json!({"model": "m", "thinking": {"type": "enabled"}, "messages": [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "t", "signature": "s"},
                {"type": "text", "text": "a"},
                {"type": "redacted_thinking", "data": "d"}
            ]}
        ]}),
        json!({"model": "m", "messages": [{"role": "assistant", "content": [{"type": "text", "text": "a"}]}]})
    )]
    #[case::thinking_only_and_empty_turns_are_dropped(
        json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "d"}]},
            {"role": "assistant", "content": []},
            {"role": "user", "content": "again"}
        ]}),
        json!({"messages": [{"role": "user", "content": "hi"}, {"role": "user", "content": "again"}]})
    )]
    #[case::non_object_entries_are_kept(
        json!({"messages": ["raw", {"role": "user", "content": ["raw", {"type": "thinking"}]}]}),
        json!({"messages": ["raw", {"role": "user", "content": ["raw"]}]})
    )]
    #[case::messages_not_a_list(
        json!({"messages": "hi", "thinking": {"type": "adaptive"}}),
        json!({"messages": "hi"})
    )]
    fn retrying_strips_thinking_from_the_replay_and_the_request(
        #[case] body: Value,
        #[case] expected: Value,
    ) {
        assert_eq!(
            strip_thinking_blocks_from_anthropic_messages_request(body),
            expected
        );
    }

    #[test]
    fn stripping_keeps_the_order_of_the_remaining_fields() {
        let stripped = strip_thinking_blocks_from_anthropic_messages_request(
            json!({"model": "m", "thinking": {}, "messages": [], "max_tokens": 1}),
        );
        let keys: Vec<&str> = stripped
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ["model", "messages", "max_tokens"]);
    }

    fn shared_table(table: &str) -> Vec<Value> {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/anthropic_messages_http_error_retry.json"
        )))
        .unwrap();
        fixture[table].as_array().unwrap().clone()
    }

    #[test]
    fn strict_tools_rejections_match_the_cases_shared_with_python() {
        let cases = shared_table("claude_strict_tools_rejection");
        assert!(!cases.is_empty());
        let mismatched: Vec<&str> = cases
            .iter()
            .filter(|case| {
                let status = u16::try_from(case["status"].as_u64().unwrap()).unwrap();
                is_claude_strict_tools_rejection(status, case["error_text"].as_str().unwrap())
                    != case["expected"].as_bool().unwrap()
            })
            .map(|case| case["name"].as_str().unwrap())
            .collect();
        assert_eq!(mismatched, Vec::<&str>::new());
    }

    #[test]
    fn tools_without_strict_matches_the_cases_shared_with_python() {
        let cases = shared_table("tools_without_strict");
        assert!(!cases.is_empty());
        let mismatched: Vec<&str> = cases
            .iter()
            .filter(|case| {
                let expected = case["expected"]
                    .as_object()
                    .map(|expected| ToolsWithoutStrict {
                        tools: expected["tools"].as_array().unwrap().clone(),
                        dropped_count: usize::try_from(expected["dropped_count"].as_u64().unwrap())
                            .unwrap(),
                    });
                tools_without_strict(Some(&case["tools"])) != expected
            })
            .map(|case| case["name"].as_str().unwrap())
            .collect();
        assert_eq!(mismatched, Vec::<&str>::new());
    }

    #[test]
    fn absent_tools_have_no_strict_to_drop() {
        assert_eq!(tools_without_strict(None), None);
    }
}
