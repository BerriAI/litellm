use std::{path::PathBuf, time::Duration};

use litellm_auth_types::{Error, SecretValue};
use serde::Deserialize;

use crate::{CopilotSession, CopilotSessionSource, SessionFuture};

const TOKEN_URL: &str = "https://api.github.com/copilot_internal/v2/token";

#[derive(Deserialize)]
struct SessionDocument {
    token: SecretValue,
    expires_at: u64,
    #[serde(default)]
    endpoints: Endpoints,
}

#[derive(Default, Deserialize)]
struct Endpoints {
    api: Option<String>,
}

impl SessionDocument {
    fn session(self) -> Result<CopilotSession, Error> {
        CopilotSession::new(self.token, self.endpoints.api.as_deref(), self.expires_at)
    }
}

pub struct PersistedSessionSource {
    http: Option<litellm_http::Client>,
    access_token_file: PathBuf,
    session_file: PathBuf,
    token_url: String,
}

impl PersistedSessionSource {
    pub fn new(
        http: litellm_http::Client,
        access_token_file: PathBuf,
        session_file: PathBuf,
        token_url: String,
    ) -> Self {
        Self {
            http: Some(http),
            access_token_file,
            session_file,
            token_url,
        }
    }

    pub fn from_environment(http: Option<litellm_http::Client>) -> Self {
        let directory = std::env::var_os("GITHUB_COPILOT_TOKEN_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                    .join(".config/litellm/github_copilot")
            });
        Self {
            http,
            access_token_file: directory.join(
                std::env::var_os("GITHUB_COPILOT_ACCESS_TOKEN_FILE")
                    .unwrap_or_else(|| "access-token".into()),
            ),
            session_file: directory.join(
                std::env::var_os("GITHUB_COPILOT_API_KEY_FILE")
                    .unwrap_or_else(|| "api-key.json".into()),
            ),
            token_url: std::env::var("GITHUB_COPILOT_API_KEY_URL")
                .unwrap_or_else(|_| TOKEN_URL.into()),
        }
    }

    async fn load_cached(&self, now: u64) -> Option<CopilotSession> {
        let bytes = tokio::fs::read(&self.session_file).await.ok()?;
        let document: SessionDocument = serde_json::from_slice(&bytes).ok()?;
        if document.expires_at <= now {
            return None;
        }
        document.session().ok()
    }

    async fn refresh(&self) -> Result<CopilotSession, Error> {
        let access_token = SecretValue::new(
            tokio::fs::read_to_string(&self.access_token_file)
                .await
                .map_err(|_| auth_error("Copilot login is required before starting the proxy"))?,
        );
        if access_token.expose().trim().is_empty() {
            return Err(auth_error("Copilot access token is empty"));
        }
        let client = self
            .http
            .as_ref()
            .ok_or_else(|| auth_error("Copilot token refresh requires a configured HTTP client"))?;
        let response = client
            .get(&self.token_url)
            .timeout(Duration::from_secs(30))
            .header(
                "authorization",
                format!("token {}", access_token.expose().trim()),
            )
            .header("accept", "application/json")
            .header("editor-version", "vscode/1.85.1")
            .header("editor-plugin-version", "copilot/1.155.0")
            .header("user-agent", "GithubCopilot/1.155.0")
            .send()
            .await
            .map_err(|_| auth_error("Copilot token request failed"))?;
        if !response.status().is_success() {
            return Err(auth_error("Copilot token request was rejected"));
        }
        let document = response
            .json::<SessionDocument>()
            .await
            .map_err(|_| auth_error("Copilot token response is invalid"))?;
        document.session()
    }
}

impl CopilotSessionSource for PersistedSessionSource {
    fn acquire(&self, now: u64) -> SessionFuture<'_> {
        Box::pin(async move {
            match self.load_cached(now).await {
                Some(session) => Ok(session),
                None => self.refresh().await,
            }
        })
    }
}

fn auth_error(message: &str) -> Error {
    Error::ProviderAuthentication(message.into())
}
