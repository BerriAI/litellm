//! Admission for `/v1/messages`: the checks the Python proxy runs before a request reaches
//! the provider (identity, model access, size, token count, budget, rate limits).
//!
//! The body is parsed once from raw bytes. Identity resolution and tokenization run
//! concurrently, tokenization on the blocking pool behind a semaphore, and the budget and
//! per-minute checks reuse that single token count. Everything after the first request for a
//! key is an in-memory read.
//!
//! Proof of concept, not at parity with `user_api_key_auth` and the proxy hooks: identity
//! comes from the proxy's `/key/info` once per key, and budgets and TPM/RPM windows are
//! process-local.

pub mod extract;
pub mod identity;
pub mod limits;
pub mod tokenizer;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use serde_json::Value;

use crate::constants::{
    DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MAX_REQUEST_BYTES, DEFAULT_PROXY_BASE_URL,
    DEFAULT_TOKENIZER_CONCURRENCY,
};

pub use extract::Admit;
pub use identity::{Identity, IdentityCache, IdentityError, KeyLimits};
pub use limits::{LimitExceeded, Limits};
pub use tokenizer::{TokenCounter, TokenizerError};

#[derive(Debug, thiserror::Error)]
pub enum Rejection {
    #[error("missing or invalid bearer token")]
    Unauthorized,
    #[error("key lookup failed: {0}")]
    IdentityUnavailable(String),
    #[error("key is not allowed to call model '{0}'")]
    ModelNotAllowed(String),
    #[error("request body of {bytes} bytes exceeds the {max} byte limit")]
    RequestTooLarge { bytes: usize, max: usize },
    #[error("input of {tokens} tokens exceeds the {max} token limit")]
    ContextTooLarge { tokens: usize, max: usize },
    #[error("{0:?} limit exceeded")]
    LimitExceeded(LimitExceeded),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("tokenization failed: {0}")]
    Tokenizer(String),
}

impl From<IdentityError> for Rejection {
    fn from(error: IdentityError) -> Self {
        match error {
            IdentityError::Unauthorized => Rejection::Unauthorized,
            IdentityError::LookupFailed(reason) => Rejection::IdentityUnavailable(reason),
        }
    }
}

/// The request after the single parse: what admission needs plus the body to forward.
#[derive(Debug)]
pub struct ParsedRequest {
    pub body: Value,
    pub model: String,
    /// Everything the tokenizer sees: system prompt, message content, tool schemas.
    pub text: String,
}

impl ParsedRequest {
    pub fn parse(raw: &[u8]) -> Result<Self, Rejection> {
        let body: Value = serde_json::from_slice(raw)
            .map_err(|error| Rejection::InvalidRequest(format!("body is not JSON: {error}")))?;
        let Some(object) = body.as_object() else {
            return Err(Rejection::InvalidRequest(
                "body must be a JSON object".to_string(),
            ));
        };
        let model = object
            .get("model")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|model| !model.is_empty())
            .ok_or_else(|| Rejection::InvalidRequest("body requires a model".to_string()))?
            .to_string();
        let mut text = String::with_capacity(raw.len());
        if let Some(system) = object.get("system") {
            push_content(system, &mut text);
        }
        for message in object
            .get("messages")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(content) = message.get("content") {
                push_content(content, &mut text);
            }
        }
        if let Some(tools) = object.get("tools") {
            text.push_str(&tools.to_string());
        }
        Ok(Self { body, model, text })
    }
}

fn push_content(content: &Value, text: &mut String) {
    match content {
        Value::String(value) => {
            text.push_str(value);
            text.push('\n');
        }
        Value::Array(blocks) => {
            for block in blocks {
                match block.get("text").and_then(Value::as_str) {
                    Some(value) => {
                        text.push_str(value);
                        text.push('\n');
                    }
                    None => text.push_str(&block.to_string()),
                }
            }
        }
        other => text.push_str(&other.to_string()),
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

#[derive(Debug)]
pub struct Admitted {
    pub body: Value,
    pub identity: Identity,
    pub input_tokens: usize,
    pub elapsed_ms: f64,
}

pub struct Admission {
    identities: IdentityCache,
    limits: Limits,
    tokens: TokenCounter,
    max_request_bytes: usize,
    max_input_tokens: usize,
}

impl Admission {
    pub fn new(identities: IdentityCache, tokens: TokenCounter) -> Self {
        Self {
            identities,
            limits: Limits::default(),
            tokens,
            max_request_bytes: DEFAULT_MAX_REQUEST_BYTES,
            max_input_tokens: DEFAULT_MAX_INPUT_TOKENS,
        }
    }

    /// Build from `LITELLM_PROXY_BASE_URL`, `LITELLM_ANTHROPIC_TOKENIZER_PATH`,
    /// `LITELLM_TOKENIZER_CONCURRENCY`, `LITELLM_MAX_REQUEST_BYTES` and `LITELLM_MAX_INPUT_TOKENS`.
    /// Without a tokenizer path the token count is approximated from the input length.
    pub fn from_env(master_key: Option<Arc<str>>) -> Result<Self, TokenizerError> {
        let proxy_base_url = std::env::var("LITELLM_PROXY_BASE_URL")
            .ok()
            .map(|url| url.trim().to_string())
            .filter(|url| !url.is_empty())
            .unwrap_or_else(|| DEFAULT_PROXY_BASE_URL.to_string());
        let tokens = match std::env::var("LITELLM_ANTHROPIC_TOKENIZER_PATH") {
            Ok(path) if !path.trim().is_empty() => TokenCounter::from_file(
                Path::new(path.trim()),
                env_usize(
                    "LITELLM_TOKENIZER_CONCURRENCY",
                    DEFAULT_TOKENIZER_CONCURRENCY,
                ),
            )?,
            _ => TokenCounter::approximate(),
        };
        Ok(
            Self::new(IdentityCache::new(master_key, proxy_base_url), tokens).with_limits(
                env_usize("LITELLM_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES),
                env_usize("LITELLM_MAX_INPUT_TOKENS", DEFAULT_MAX_INPUT_TOKENS),
            ),
        )
    }

    pub fn with_limits(self, max_request_bytes: usize, max_input_tokens: usize) -> Self {
        Self {
            max_request_bytes,
            max_input_tokens,
            ..self
        }
    }

    pub fn identities(&self) -> &IdentityCache {
        &self.identities
    }

    pub fn tokens(&self) -> &TokenCounter {
        &self.tokens
    }

    pub fn max_request_bytes(&self) -> usize {
        self.max_request_bytes
    }

    pub async fn admit(&self, bearer: Option<&str>, raw: &[u8]) -> Result<Admitted, Rejection> {
        let started = Instant::now();
        let bearer = bearer
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .ok_or(Rejection::Unauthorized)?;
        if raw.len() > self.max_request_bytes {
            return Err(Rejection::RequestTooLarge {
                bytes: raw.len(),
                max: self.max_request_bytes,
            });
        }
        let ParsedRequest { body, model, text } = ParsedRequest::parse(raw)?;
        let (identity, input_tokens) = tokio::join!(
            self.identities.resolve(Some(bearer)),
            self.tokens.count(text)
        );
        let identity = identity?;
        let input_tokens = input_tokens
            .map_err(|error: TokenizerError| Rejection::Tokenizer(error.to_string()))?;
        if let Identity::VirtualKey { limits, .. } = &identity
            && !limits.allows_model(&model)
        {
            return Err(Rejection::ModelNotAllowed(model));
        }
        if input_tokens > self.max_input_tokens {
            return Err(Rejection::ContextTooLarge {
                tokens: input_tokens,
                max: self.max_input_tokens,
            });
        }
        self.limits
            .reserve(&identity, input_tokens)
            .map_err(Rejection::LimitExceeded)?;
        Ok(Admitted {
            body,
            identity,
            input_tokens,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::*;

    fn admission() -> Admission {
        let identities = IdentityCache::new(
            Some(Arc::from("sk-master")),
            "http://127.0.0.1:1".to_string(),
        );
        identities.insert(
            "sk-limited",
            KeyLimits {
                models: vec!["claude".to_string()],
                max_budget: None,
                spend: 0.0,
                tpm_limit: Some(100),
                rpm_limit: None,
            },
        );
        Admission::new(identities, TokenCounter::approximate())
    }

    fn body(model: &str, words: usize) -> Vec<u8> {
        json!({
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "word ".repeat(words)}]
        })
        .to_string()
        .into_bytes()
    }

    #[test]
    fn parse_collects_system_messages_and_tools_once() {
        let raw = json!({
            "model": "claude",
            "system": [{"type": "text", "text": "be brief"}],
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}
            ],
            "tools": [{"name": "t", "input_schema": {"type": "object"}}]
        })
        .to_string();
        let parsed = ParsedRequest::parse(raw.as_bytes()).unwrap();
        assert_eq!(parsed.model, "claude");
        assert!(parsed.text.contains("be brief\n"));
        assert!(parsed.text.contains("hi\n"));
        assert!(parsed.text.contains("hello\n"));
        assert!(parsed.text.contains("input_schema"));
        assert_eq!(parsed.body["messages"][0]["content"], "hi");
    }

    #[test]
    fn parse_rejects_non_object_and_missing_model() {
        assert!(matches!(
            ParsedRequest::parse(b"[]"),
            Err(Rejection::InvalidRequest(_))
        ));
        assert!(matches!(
            ParsedRequest::parse(br#"{"messages": []}"#),
            Err(Rejection::InvalidRequest(_))
        ));
        assert!(matches!(
            ParsedRequest::parse(b"{not json"),
            Err(Rejection::InvalidRequest(_))
        ));
    }

    #[tokio::test]
    async fn master_key_is_admitted_with_a_token_count() {
        let admitted = admission()
            .admit(Some("sk-master"), &body("claude", 40))
            .await
            .unwrap();
        assert_eq!(admitted.identity, Identity::Master);
        assert!(admitted.input_tokens > 0);
        assert_eq!(admitted.body["model"], "claude");
    }

    #[tokio::test]
    async fn missing_bearer_is_unauthorized() {
        assert!(matches!(
            admission().admit(None, &body("claude", 1)).await,
            Err(Rejection::Unauthorized)
        ));
    }

    #[tokio::test]
    async fn virtual_key_model_access_is_enforced() {
        assert!(matches!(
            admission().admit(Some("sk-limited"), &body("other", 1)).await,
            Err(Rejection::ModelNotAllowed(model)) if model == "other"
        ));
    }

    #[tokio::test]
    async fn one_token_count_feeds_the_tpm_window() {
        let admission = admission();
        let first = admission
            .admit(Some("sk-limited"), &body("claude", 40))
            .await
            .unwrap();
        assert!(first.input_tokens > 40, "{}", first.input_tokens);
        assert!(matches!(
            admission
                .admit(Some("sk-limited"), &body("claude", 40))
                .await,
            Err(Rejection::LimitExceeded(LimitExceeded::TokensPerMinute))
        ));
    }

    #[tokio::test]
    async fn oversized_bodies_are_rejected_before_parsing() {
        let admission = admission().with_limits(16, 1_000_000);
        assert!(matches!(
            admission
                .admit(Some("sk-master"), &body("claude", 10))
                .await,
            Err(Rejection::RequestTooLarge { .. })
        ));
    }

    #[tokio::test]
    async fn context_limit_uses_the_same_count() {
        let admission = admission().with_limits(1 << 20, 10);
        assert!(matches!(
            admission.admit(Some("sk-master"), &body("claude", 40)).await,
            Err(Rejection::ContextTooLarge { tokens, max: 10 }) if tokens > 10
        ));
    }
}
