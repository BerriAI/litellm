use std::{future::Future, pin::Pin, time::SystemTime};

use litellm_auth::SecretValue;

pub const DEFAULT_GITHUB_COPILOT_API_BASE: &str = "https://api.githubcopilot.com";

/// Copilot's static defaults. Every one only fills a header the caller did not set.
pub const COPILOT_DEFAULT_HEADERS: &[(&str, &str)] = &[
    ("content-type", "application/json"),
    ("copilot-integration-id", "vscode-chat"),
    ("editor-version", "vscode/1.95.0"),
    ("editor-plugin-version", "copilot-chat/0.26.7"),
    ("user-agent", "GitHubCopilotChat/0.26.7"),
    ("x-vscode-user-agent-library-version", "electron-fetch"),
];

/// The Copilot API key that GitHub mints from the user's OAuth access token, as cached in
/// `api-key.json`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopilotToken {
    pub token: SecretValue,
    pub expires_at: SystemTime,
    /// `endpoints.api`: the host this account's Copilot traffic must go to.
    pub api_endpoint: Option<String>,
}

pub type CopilotTokenFuture<'a> =
    Pin<Box<dyn Future<Output = Result<CopilotToken, litellm_auth::Error>> + Send + 'a>>;

/// The Copilot authenticator as the adapters see it.
pub trait CopilotTokenCache: std::fmt::Debug + Send + Sync {
    /// The cached token as is, even if expired, without refreshing or logging in.
    fn cached(&self) -> Option<CopilotToken>;
    /// A token that is valid now, refreshing it or running the device flow when needed.
    fn acquire(&self) -> CopilotTokenFuture<'_>;
}

pub fn resolve_copilot_api_base(tokens: &dyn CopilotTokenCache) -> String {
    tokens
        .cached()
        .and_then(|token| token.api_endpoint)
        .filter(|endpoint| !endpoint.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_GITHUB_COPILOT_API_BASE.to_string())
}
