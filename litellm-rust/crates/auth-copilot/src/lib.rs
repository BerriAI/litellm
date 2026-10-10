#![forbid(unsafe_code)]

mod persisted;
pub use persisted::PersistedSessionSource;

use std::{future::Future, pin::Pin, sync::Arc, time::SystemTime};

use litellm_auth_types::{Error, SecretValue};
use tokio::sync::Mutex;

pub const DEFAULT_API_BASE: &str = "https://api.githubcopilot.com";

#[derive(Clone, Debug)]
pub struct CopilotSession {
    token: SecretValue,
    api_base: String,
    expires_at: u64,
}

impl CopilotSession {
    pub fn new(token: SecretValue, api_base: Option<&str>, expires_at: u64) -> Result<Self, Error> {
        if token.expose().trim().is_empty() {
            return Err(Error::ProviderAuthentication(
                "Copilot session token is empty".into(),
            ));
        }
        Ok(Self {
            token,
            api_base: trusted_api_base(api_base),
            expires_at,
        })
    }

    pub fn token(&self) -> &SecretValue {
        &self.token
    }

    pub fn api_base(&self) -> &str {
        &self.api_base
    }

    fn valid_at(&self, now: u64) -> bool {
        self.expires_at > now
    }
}

pub type SessionFuture<'a> =
    Pin<Box<dyn Future<Output = Result<CopilotSession, Error>> + Send + 'a>>;

pub trait CopilotSessionSource: Send + Sync {
    fn acquire(&self, now: u64) -> SessionFuture<'_>;
}

pub struct CopilotAuthService {
    source: Arc<dyn CopilotSessionSource>,
    cached: Mutex<Option<CopilotSession>>,
    now: Arc<dyn Fn() -> u64 + Send + Sync>,
}

impl CopilotAuthService {
    pub fn with_no_redirect_http(http: litellm_http::Client) -> Self {
        Self::from_persisted(Some(http))
    }

    fn from_persisted(http: Option<litellm_http::Client>) -> Self {
        Self::new(
            Arc::new(PersistedSessionSource::from_environment(http)),
            Arc::new(|| {
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map_or(0, |elapsed| elapsed.as_secs())
            }),
        )
    }

    pub fn new(
        source: Arc<dyn CopilotSessionSource>,
        now: Arc<dyn Fn() -> u64 + Send + Sync>,
    ) -> Self {
        Self {
            source,
            cached: Mutex::new(None),
            now,
        }
    }

    pub async fn session(&self) -> Result<CopilotSession, Error> {
        let mut cached = self.cached.lock().await;
        let now = (self.now)();
        if let Some(session) = cached.as_ref().filter(|session| session.valid_at(now)) {
            return Ok(session.clone());
        }
        let acquired = self.source.acquire(now).await?;
        if !acquired.valid_at((self.now)()) {
            return Err(Error::ProviderAuthentication(
                "Copilot session has expired".into(),
            ));
        }
        *cached = Some(acquired.clone());
        Ok(acquired)
    }
}

impl Default for CopilotAuthService {
    fn default() -> Self {
        Self::from_persisted(None)
    }
}

fn trusted_api_base(base: Option<&str>) -> String {
    let Some(base) = base else {
        return DEFAULT_API_BASE.into();
    };
    let Ok(url) = url::Url::parse(base) else {
        return DEFAULT_API_BASE.into();
    };
    let trusted_host = url
        .host_str()
        .is_some_and(|host| host == "githubcopilot.com" || host.ends_with(".githubcopilot.com"));
    if url.scheme() != "https"
        || !trusted_host
        || !url.username().is_empty()
        || url.password().is_some()
        || url.port().is_some_and(|port| port != 443)
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return DEFAULT_API_BASE.into();
    }
    url.as_str().trim_end_matches('/').into()
}
