use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_http::request::{has_header, header_value, without_headers};
use litellm_types::llms::{
    anthropic::{AnthropicBeta, BetaSet},
    anthropic_messages::anthropic_request::{
        AnthropicMessage, AnthropicTool, ContentBlock, EffortLevel, MessageContent,
    },
};
use litellm_types::recognized::Recognized;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    anthropic::ANTHROPIC_OAUTH_TOKEN_PREFIX,
    base_llm::auth::{AuthScheme, Headers},
};

pub const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
pub const ANTHROPIC_AUTH_TOKEN_ENV: &str = "ANTHROPIC_AUTH_TOKEN";
pub const ENCRYPTED_REASONING_SIGNATURE_PREFIX: &str = "litellm_encrypted_reasoning:";
const THOUGHT_SIGNATURE_SEPARATOR: &str = "__thought__";
const BETA_HEADER: &str = "anthropic-beta";
pub const ANTHROPIC_API_BASE_ENV: &str = "ANTHROPIC_API_BASE";
pub const ANTHROPIC_BASE_URL_ENV: &str = "ANTHROPIC_BASE_URL";
pub const DEFAULT_ANTHROPIC_API_BASE: &str = "https://api.anthropic.com";
pub const MESSAGES_PATH_SUFFIX: &str = "/v1/messages";
pub const API_KEY_PLACEMENT: CredentialPlacement = CredentialPlacement::Header("x-api-key");
const API_KEY_HEADER: &str = API_KEY_PLACEMENT.header_name();
const AUTHORIZATION: &str = CredentialPlacement::Bearer.header_name();
const DIRECT_BROWSER_ACCESS_HEADER: &str = "anthropic-dangerous-direct-browser-access";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportedEffortTiers {
    #[serde(default)]
    pub minimal: bool,
    #[serde(default)]
    pub low: bool,
    #[serde(default)]
    pub medium: bool,
    #[serde(default)]
    pub high: bool,
    #[serde(default)]
    pub xhigh: bool,
    #[serde(default)]
    pub max: bool,
}

impl SupportedEffortTiers {
    pub fn any(self) -> bool {
        self.minimal || self.low || self.medium || self.high || self.xhigh || self.max
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicModelCapabilities {
    #[serde(default)]
    pub supports_reasoning: bool,
    #[serde(default)]
    pub supports_adaptive_thinking: bool,
    #[serde(default)]
    pub thinking_always_on: bool,
    #[serde(default)]
    pub supports_legacy_thinking: bool,
    #[serde(default)]
    pub supports_output_config: bool,
    #[serde(default = "default_true")]
    pub supports_sampling_params: bool,
    #[serde(default)]
    pub supports_speed: bool,
    #[serde(default)]
    pub effort_tiers: SupportedEffortTiers,
}

fn default_true() -> bool {
    true
}

impl Default for AnthropicModelCapabilities {
    fn default() -> Self {
        Self {
            supports_reasoning: false,
            supports_adaptive_thinking: false,
            thinking_always_on: false,
            supports_legacy_thinking: false,
            supports_output_config: false,
            supports_sampling_params: true,
            supports_speed: false,
            effort_tiers: SupportedEffortTiers::default(),
        }
    }
}

impl AnthropicModelCapabilities {
    pub fn supports_effort_tier(&self, level: EffortLevel) -> bool {
        match level {
            EffortLevel::Low => self.effort_tiers.low,
            EffortLevel::Medium => self.effort_tiers.medium,
            EffortLevel::High => self.effort_tiers.high,
            EffortLevel::Xhigh => self.effort_tiers.xhigh,
            EffortLevel::Max => self.effort_tiers.max,
        }
    }

    pub fn supports_effort_param(&self) -> bool {
        self.supports_output_config || self.effort_tiers.any()
    }

    pub fn accepts_effort(&self, level: EffortLevel) -> bool {
        match level {
            EffortLevel::Max => self.supports_adaptive_thinking || self.effort_tiers.max,
            EffortLevel::Xhigh => self.effort_tiers.xhigh,
            EffortLevel::Low | EffortLevel::Medium | EffortLevel::High => true,
        }
    }
}

pub fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

pub fn non_empty_env(env_lookup: &dyn Fn(&str) -> Option<String>, name: &str) -> Option<String> {
    env_lookup(name).filter(|value| !value.trim().is_empty())
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
        AuthScheme::Credential {
            placement: CredentialPlacement::Bearer,
            secret: SecretValue::new(self.0),
        }
    }
}

/// Python's `AnthropicModelInfo.get_api_key`: the param, else `ANTHROPIC_API_KEY`.
pub fn get_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    non_empty(api_key)
        .map(str::to_string)
        .or_else(|| non_empty_env(env_lookup, ANTHROPIC_API_KEY_ENV))
}

pub fn get_auth_token(env_lookup: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    non_empty_env(env_lookup, ANTHROPIC_AUTH_TOKEN_ENV)
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
    non_empty(api_base)
        .map(str::to_string)
        .or_else(|| non_empty_env(env_lookup, ANTHROPIC_API_BASE_ENV))
        .or_else(|| non_empty_env(env_lookup, ANTHROPIC_BASE_URL_ENV))
        .unwrap_or_else(|| DEFAULT_ANTHROPIC_API_BASE.to_string())
}

pub fn complete_anthropic_url(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    let api_base = resolve_anthropic_api_base(api_base, env_lookup);

    let api_base = api_base.trim_end_matches('/');
    if api_base.ends_with(MESSAGES_PATH_SUFFIX) {
        return api_base.to_string();
    }
    format!("{api_base}{MESSAGES_PATH_SUFFIX}")
}

pub fn existing_betas(headers: &[(String, String)]) -> BetaSet {
    headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case(BETA_HEADER))
        .flat_map(|(_, value)| {
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
    without_headers(headers, &[BETA_HEADER])
        .into_iter()
        .chain([(BETA_HEADER.to_string(), merged.to_string())])
        .collect()
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

fn with_oauth_companions(headers: Headers, dropped: &[&str]) -> Headers {
    merge_beta_headers(
        without_headers(headers, dropped),
        BetaSet::from_iter([AnthropicBeta::Oauth20250420]),
    )
    .into_iter()
    .chain([(DIRECT_BROWSER_ACCESS_HEADER.to_string(), "true".to_string())])
    .collect()
}

pub fn optionally_handle_anthropic_oauth(headers: Headers, api_key: Option<&str>) -> OauthHandling {
    if let Some(token) =
        forwarded_oauth_bearer(&headers).map(|token| SecretValue::new(token.as_str()))
    {
        return OauthHandling::Bearer {
            headers: with_oauth_companions(headers, &[API_KEY_HEADER, AUTHORIZATION]),
            token,
        };
    }
    if let Some(token) = api_key.and_then(OauthToken::parse) {
        return OauthHandling::Bearer {
            headers: with_oauth_companions(headers, &[API_KEY_HEADER]),
            token: SecretValue::new(token.as_str()),
        };
    }
    OauthHandling::Untouched(headers)
}

pub fn is_tool_search_used(tools: Option<&[Recognized<AnthropicTool>]>) -> bool {
    tools.into_iter().flatten().any(|tool| {
        matches!(
            tool,
            Recognized::Known(
                AnthropicTool::ToolSearchRegex { .. } | AnthropicTool::ToolSearchBm25 { .. }
            )
        )
    })
}

pub fn has_advisor_tool(tools: Option<&[Recognized<AnthropicTool>]>) -> bool {
    tools
        .into_iter()
        .flatten()
        .any(|tool| matches!(tool, Recognized::Known(AnthropicTool::Advisor { .. })))
}

pub fn requires_native_compaction_beta(
    compaction: Option<&Value>,
    messages: &[AnthropicMessage],
) -> bool {
    compaction.is_some()
        || messages
            .iter()
            .flat_map(AnthropicMessage::blocks)
            .any(|block| {
                block.is_type("compaction")
                    && block.signature.as_deref().is_some_and(|s| !s.is_empty())
            })
}

fn is_blank(text: Option<&str>) -> bool {
    text.is_none_or(|text| text.trim().is_empty())
}

fn is_empty_text_block(block: &ContentBlock) -> bool {
    block.is_type("text") && is_blank(block.text.as_deref())
}

pub fn is_empty_thinking_block(block: &ContentBlock) -> bool {
    block.is_type("thinking") && is_blank(block.thinking.as_deref())
}

fn retain_blocks(
    messages: Vec<AnthropicMessage>,
    keep: impl Fn(&ContentBlock) -> bool,
) -> Vec<AnthropicMessage> {
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

pub fn strip_empty_content_blocks(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    retain_blocks(messages, |block| {
        !is_empty_text_block(block) && !is_empty_thinking_block(block)
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
    match block.block_type.as_deref() {
        Some("tool_use" | "server_tool_use") => match normalized_if_changed(block.id.as_deref()) {
            Some(id) => ContentBlock {
                id: Some(id),
                ..block
            },
            None => block,
        },
        Some("tool_result") => match normalized_if_changed(block.tool_use_id.as_deref()) {
            Some(tool_use_id) => ContentBlock {
                tool_use_id: Some(tool_use_id),
                ..block
            },
            None => block,
        },
        _ => block,
    }
}

fn map_blocks(
    messages: Vec<AnthropicMessage>,
    rewrite: impl Fn(Vec<ContentBlock>) -> Vec<ContentBlock>,
) -> Vec<AnthropicMessage> {
    messages
        .into_iter()
        .map(|message| match message.content {
            MessageContent::Blocks(blocks) => AnthropicMessage {
                content: MessageContent::Blocks(rewrite(blocks)),
                ..message
            },
            MessageContent::Text(_) => message,
        })
        .collect()
}

pub fn sanitize_tool_use_ids(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    map_blocks(messages, |blocks| {
        blocks.into_iter().map(sanitize_tool_use_id_block).collect()
    })
}

pub fn strip_provider_specific_fields(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    map_blocks(messages, |blocks| {
        blocks
            .into_iter()
            .map(|block| ContentBlock {
                provider_specific_fields: None,
                ..block
            })
            .collect()
    })
}

pub fn is_encrypted_reasoning_block(block: &ContentBlock) -> bool {
    let field = match block.block_type.as_deref() {
        Some("thinking") => block.signature.as_deref(),
        Some("redacted_thinking") => block.data.as_deref(),
        _ => None,
    };
    field.is_some_and(|value| value.starts_with(ENCRYPTED_REASONING_SIGNATURE_PREFIX))
}

pub fn is_anthropic_invalid_thinking_block_error(error_text: &str) -> bool {
    let lower = error_text.to_lowercase();
    lower.contains("thinking")
        && ((lower.contains("signature")
            && (lower.contains("invalid") || lower.contains("valid string")))
            || lower.contains("must contain thinking"))
}

pub fn strip_thinking_blocks_from_anthropic_messages(
    messages: Vec<AnthropicMessage>,
) -> Vec<AnthropicMessage> {
    retain_blocks(messages, |block| {
        !matches!(
            block.block_type.as_deref(),
            Some("thinking" | "redacted_thinking")
        )
    })
}

pub fn strip_encrypted_reasoning_blocks(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    retain_blocks(messages, |block| !is_encrypted_reasoning_block(block))
}

fn is_advisor_use(block: &ContentBlock) -> bool {
    block.is_type("server_tool_use")
        && block.name.as_deref() == Some("advisor")
        && block.id.as_deref().is_some_and(|id| !id.is_empty())
}

pub fn strip_advisor_blocks(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    messages
        .into_iter()
        .map(|message| {
            if message.role != "assistant" {
                return message;
            }
            let MessageContent::Blocks(blocks) = &message.content else {
                return message;
            };
            let advisor_ids: Vec<&str> = blocks
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
                    let is_result = block.is_type("advisor_tool_result")
                        && block
                            .tool_use_id
                            .as_deref()
                            .is_some_and(|id| advisor_ids.contains(&id));
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
    if !block.is_type("web_search_tool_result") {
        return None;
    }
    let tool_use_id = block.tool_use_id.as_deref()?;
    let results = match block.content.as_ref()? {
        Value::Array(items) => {
            let results = items
                .iter()
                .map(|item| {
                    (item.get("type").and_then(Value::as_str) == Some("web_search_result"))
                        .then(|| {
                            serde_json::from_value::<ReplayedWebSearchResult>(item.clone()).ok()
                        })
                        .flatten()
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
        },
        _ => return None,
    };
    Some((tool_use_id, results))
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
    if !block.is_type("server_tool_use") {
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
    let flattenable_ids: Vec<&str> = blocks
        .iter()
        .filter_map(flattenable_web_search_results)
        .map(|(tool_use_id, _)| tool_use_id)
        .collect();
    if flattenable_ids.is_empty() {
        return blocks;
    }
    let queries: Vec<(&str, &str)> = blocks.iter().filter_map(server_tool_use_query).collect();
    blocks
        .iter()
        .filter_map(|block| {
            if let Some((tool_use_id, results)) = flattenable_web_search_results(block) {
                let query = queries
                    .iter()
                    .rfind(|(id, _)| *id == tool_use_id)
                    .map_or("", |(_, query)| query);
                return Some(ContentBlock::text(render_web_search_results(
                    query, &results,
                )));
            }
            if let Some((id, _)) = server_tool_use_query(block)
                && flattenable_ids.contains(&id)
            {
                return None;
            }
            Some(block.clone())
        })
        .collect()
}

pub fn flatten_unencrypted_web_search_results(
    messages: Vec<AnthropicMessage>,
) -> Vec<AnthropicMessage> {
    map_blocks(messages, flatten_web_search_results_in_blocks)
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use serde_json::json;

    use super::*;

    const ALL_LEVELS: [EffortLevel; 5] = [
        EffortLevel::Low,
        EffortLevel::Medium,
        EffortLevel::High,
        EffortLevel::Xhigh,
        EffortLevel::Max,
    ];

    fn apply(
        sanitizer: fn(Vec<AnthropicMessage>) -> Vec<AnthropicMessage>,
        messages: Value,
    ) -> Value {
        let parsed: Vec<AnthropicMessage> = serde_json::from_value(messages).unwrap();
        serde_json::to_value(sanitizer(parsed)).unwrap()
    }

    fn block(value: Value) -> ContentBlock {
        serde_json::from_value(value).unwrap()
    }

    fn history(messages: Value) -> Vec<AnthropicMessage> {
        serde_json::from_value(messages).unwrap()
    }

    fn tools(value: Option<Value>) -> Option<Vec<Recognized<AnthropicTool>>> {
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
    fn unmapped() -> AnthropicModelCapabilities {
        AnthropicModelCapabilities::default()
    }

    #[rstest]
    #[case::invalid_signature("Invalid `signature` in `thinking` block", true)]
    #[case::missing_signature("thinking.signature.str: Input should be a valid string", true)]
    #[case::empty_thinking("each thinking block must contain thinking", true)]
    #[case::unrelated("invalid tool signature", false)]
    fn detects_recoverable_thinking_errors(#[case] message: &str, #[case] recoverable: bool) {
        assert_eq!(
            is_anthropic_invalid_thinking_block_error(message),
            recoverable
        );
    }

    #[test]
    fn stripping_replayed_thinking_preserves_tools_and_drops_empty_turns() {
        let input = json!([
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "opaque"}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "reason", "signature": "invalid"},
                {"type": "tool_use", "id": "call", "name": "lookup", "input": {}}
            ]}
        ]);
        assert_eq!(
            apply(strip_thinking_blocks_from_anthropic_messages, input.clone()),
            json!([
                input[0],
                {"role": "assistant", "content": [input[2]["content"][1]]}
            ])
        );
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
    #[case::empty_signed_thinking_beside_tool_use(
        json!([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "", "signature": "sig_abc"},
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
    #[case::no_credentials(None, &[], None)]
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
    #[case::absent(None, None)]
    #[case::blank(Some(" \t "), None)]
    #[case::padded(Some("  value "), Some("value"))]
    fn non_empty_trims_and_drops_blank_values(
        #[case] value: Option<&str>,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(non_empty(value), expected);
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
        assert_eq!(
            requires_native_compaction_beta(compaction.as_ref(), &history(messages)),
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
        unmapped: AnthropicModelCapabilities,
    ) {
        let capabilities = AnthropicModelCapabilities {
            effort_tiers,
            ..unmapped
        };
        assert_eq!(
            ALL_LEVELS.map(|level| capabilities.supports_effort_tier(level)),
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
        unmapped: AnthropicModelCapabilities,
    ) {
        let capabilities = AnthropicModelCapabilities {
            supports_reasoning,
            supports_adaptive_thinking,
            supports_output_config,
            effort_tiers,
            ..unmapped
        };
        assert_eq!(capabilities.supports_effort_param(), expected);
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
        unmapped: AnthropicModelCapabilities,
    ) {
        let capabilities = AnthropicModelCapabilities {
            supports_output_config: true,
            supports_adaptive_thinking,
            effort_tiers,
            ..unmapped
        };
        assert_eq!(capabilities.accepts_effort(level), expected);
    }

    #[rstest]
    fn unmapped_model_has_no_reasoning_features_but_accepts_sampling_params(
        unmapped: AnthropicModelCapabilities,
    ) {
        assert_eq!(
            unmapped,
            AnthropicModelCapabilities {
                supports_reasoning: false,
                supports_adaptive_thinking: false,
                thinking_always_on: false,
                supports_legacy_thinking: false,
                supports_output_config: false,
                supports_sampling_params: true,
                supports_speed: false,
                effort_tiers: tiers(false, false, false, false, false, false),
            }
        );
        assert_eq!(
            serde_json::from_value::<AnthropicModelCapabilities>(json!({})).unwrap(),
            unmapped
        );
    }

    #[rstest]
    #[case::sampling_params_removed(
        json!({"supports_sampling_params": false}),
        AnthropicModelCapabilities { supports_sampling_params: false, ..AnthropicModelCapabilities::default() }
    )]
    #[case::fast_mode(
        json!({"supports_speed": true}),
        AnthropicModelCapabilities { supports_speed: true, ..AnthropicModelCapabilities::default() }
    )]
    #[case::partial_effort_tiers(
        json!({"supports_reasoning": true, "effort_tiers": {"xhigh": true}}),
        AnthropicModelCapabilities {
            supports_reasoning: true,
            effort_tiers: tiers(false, false, false, false, true, false),
            ..AnthropicModelCapabilities::default()
        }
    )]
    fn capabilities_fill_missing_flags_with_unmapped_defaults(
        #[case] input: Value,
        #[case] expected: AnthropicModelCapabilities,
    ) {
        assert_eq!(
            serde_json::from_value::<AnthropicModelCapabilities>(input).unwrap(),
            expected
        );
    }
}
