use std::{
    path::Path,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use litellm_core_utils::settings::Lookup;
use litellm_http::Client;
use moka::future::Cache;
use serde::Deserialize;

use crate::{Error, SecretValue};

const GOOGLE_TOKEN_MAX_TTL: Duration = Duration::from_secs(3540);
const GITHUB_TOKEN_TTL: Duration = Duration::from_secs(295);
const TOKEN_EXPIRY_MARGIN_SECONDS: f64 = 60.0;
const CIRCLE_OIDC_TOKEN: &str = "CIRCLE_OIDC_TOKEN";
const CIRCLE_OIDC_TOKEN_V2: &str = "CIRCLE_OIDC_TOKEN_V2";
const AZURE_FEDERATED_TOKEN_FILE: &str = "AZURE_FEDERATED_TOKEN_FILE";
const ACTIONS_ID_TOKEN_REQUEST_URL: &str = "ACTIONS_ID_TOKEN_REQUEST_URL";
const ACTIONS_ID_TOKEN_REQUEST_TOKEN: &str = "ACTIONS_ID_TOKEN_REQUEST_TOKEN";
const OIDC_ALLOWED_CREDENTIAL_DIRS: &str = "LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS";
const DEFAULT_CREDENTIAL_DIRS: &str = "/var/run/secrets,/run/secrets";

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, strum::EnumString, strum::AsRefStr)]
#[strum(serialize_all = "snake_case")]
pub enum OidcProvider {
    Google,
    #[strum(serialize = "circleci")]
    CircleCi,
    #[strum(serialize = "circleci_v2")]
    CircleCiV2,
    Github,
    Azure,
    File,
    Env,
    EnvPath,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OidcReference<'a> {
    pub provider: OidcProvider,
    pub audience: &'a str,
}

impl<'a> TryFrom<&'a str> for OidcReference<'a> {
    type Error = Error;

    fn try_from(reference: &'a str) -> Result<Self, Error> {
        let (provider, audience) = reference
            .strip_prefix("oidc/")
            .and_then(|body| body.split_once('/'))
            .ok_or(Error::InvalidOidc)?;
        Ok(Self {
            provider: provider.parse().map_err(|_| Error::UnsupportedOidc)?,
            audience,
        })
    }
}

#[derive(Deserialize)]
struct OidcTokenClaims {
    exp: Option<NumericDate>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum NumericDate {
    Number(f64),
    String(String),
    Boolean(bool),
}

impl NumericDate {
    fn seconds(self) -> Option<f64> {
        match self {
            Self::Number(value) => Some(value),
            Self::String(value) => value.trim().parse().ok(),
            Self::Boolean(value) => Some(f64::from(u8::from(value))),
        }
        .filter(|value| value.is_finite())
    }
}

pub struct OidcResolver {
    client: Client,
    google_identity_endpoint: reqwest::Url,
    cache: Cache<String, (SecretValue, SystemTime)>,
    clock: fn() -> SystemTime,
    #[cfg(feature = "azure")]
    azure_token_provider: std::sync::Arc<dyn litellm_secrets_azure::AzureTokenProvider>,
}

const GOOGLE_IDENTITY_ENDPOINT: &str =
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity";

const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

impl OidcResolver {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            google_identity_endpoint: reqwest::Url::parse(GOOGLE_IDENTITY_ENDPOINT)
                .expect("static URL"),
            cache: Cache::builder()
                .max_capacity(200)
                .time_to_live(GOOGLE_TOKEN_MAX_TTL)
                .build(),
            clock: SystemTime::now,
            #[cfg(feature = "azure")]
            azure_token_provider: std::sync::Arc::new(
                litellm_secrets_azure::NativeAzureTokenProvider::default(),
            ),
        }
    }

    pub fn with_google_identity_endpoint(self, google_identity_endpoint: reqwest::Url) -> Self {
        Self {
            google_identity_endpoint,
            ..self
        }
    }

    #[cfg(feature = "azure")]
    pub fn with_azure_token_provider(
        self,
        provider: std::sync::Arc<dyn litellm_secrets_azure::AzureTokenProvider>,
    ) -> Self {
        Self {
            azure_token_provider: provider,
            ..self
        }
    }

    pub fn with_clock(self, clock: fn() -> SystemTime) -> Self {
        Self { clock, ..self }
    }

    pub async fn resolve(
        &self,
        reference: &str,
        environment: &(dyn Lookup + Send + Sync),
    ) -> Result<Option<SecretValue>, Error> {
        let OidcReference { provider, audience } = reference.try_into()?;
        match provider {
            OidcProvider::CircleCi => required_env(environment, CIRCLE_OIDC_TOKEN)
                .map(SecretValue::new)
                .map(Some),
            OidcProvider::CircleCiV2 => required_env(environment, CIRCLE_OIDC_TOKEN_V2)
                .map(SecretValue::new)
                .map(Some),
            OidcProvider::Env => required_env(environment, audience)
                .map(SecretValue::new)
                .map(Some),
            OidcProvider::EnvPath => read_file(&required_env(environment, audience)?)
                .await
                .map(Some),
            OidcProvider::File => read_allowed_file(audience, environment).await.map(Some),
            OidcProvider::Azure => {
                if let Some(path) = environment.get(AZURE_FEDERATED_TOKEN_FILE) {
                    return read_file(&path).await.map(Some);
                }
                #[cfg(feature = "azure")]
                {
                    self.azure_token_provider
                        .get_token(audience, environment)
                        .await
                        .map(Some)
                        .map_err(Error::Azure)
                }
                #[cfg(not(feature = "azure"))]
                Err(Error::UnsupportedOidc)
            }
            OidcProvider::Github => {
                let url = required_env(environment, ACTIONS_ID_TOKEN_REQUEST_URL)?;
                let authorization = required_env(environment, ACTIONS_ID_TOKEN_REQUEST_TOKEN)?;
                if let Some(value) = self.cached(reference).await {
                    return Ok(Some(value));
                }
                let response = self
                    .client
                    .get(url)
                    .timeout(REQUEST_TIMEOUT)
                    .query(&[("audience", audience)])
                    .bearer_auth(authorization)
                    .header("Accept", "application/json; api-version=2.0")
                    .send()
                    .await
                    .map_err(|_| Error::OidcHttp)?;
                if response.status() != reqwest::StatusCode::OK {
                    return Err(Error::OidcStatus(response.status().as_u16()));
                }
                #[derive(Deserialize)]
                struct Token {
                    value: Option<SecretValue>,
                }
                let token: Token = response.json().await.map_err(|_| Error::OidcResponse)?;
                if let Some(value) = &token.value {
                    self.cache
                        .insert(
                            reference.to_owned(),
                            (value.clone(), (self.clock)() + GITHUB_TOKEN_TTL),
                        )
                        .await;
                }
                Ok(token.value)
            }
            OidcProvider::Google => {
                if !cfg!(feature = "google") {
                    return Err(Error::UnsupportedOidc);
                }
                if let Some(value) = self.cached(reference).await {
                    return Ok(Some(value));
                }
                let response = self
                    .client
                    .get(self.google_identity_endpoint.clone())
                    .timeout(REQUEST_TIMEOUT)
                    .query(&[("audience", audience)])
                    .header("Metadata-Flavor", "Google")
                    .send()
                    .await
                    .map_err(|_| Error::OidcHttp)?;
                if response.status() != reqwest::StatusCode::OK {
                    return Err(Error::OidcStatus(response.status().as_u16()));
                }
                let token = response.text().await.map_err(|_| Error::OidcResponse)?;
                let now = (self.clock)();
                let ttl = oidc_token_cache_ttl(&token, now, GOOGLE_TOKEN_MAX_TTL);
                let value = SecretValue::new(token);
                if let Some(ttl) = ttl.filter(|ttl| !ttl.is_zero()) {
                    self.cache
                        .insert(reference.to_owned(), (value.clone(), now + ttl))
                        .await;
                }
                Ok(Some(value))
            }
        }
    }

    async fn cached(&self, reference: &str) -> Option<SecretValue> {
        self.cache
            .get(reference)
            .await
            .and_then(|(value, expires)| ((self.clock)() < expires).then_some(value))
    }
}

fn required_env(environment: &dyn Lookup, name: &str) -> Result<String, Error> {
    environment.get(name).ok_or(Error::MissingEnvironment)
}

async fn read_file(path: &str) -> Result<SecretValue, Error> {
    tokio::fs::read_to_string(path)
        .await
        .map(|value| SecretValue::new(value.replace("\r\n", "\n").replace('\r', "\n")))
        .map_err(|_| Error::OidcFile)
}

async fn read_allowed_file(
    path: &str,
    environment: &(dyn Lookup + Sync),
) -> Result<SecretValue, Error> {
    if !Path::new(path).is_absolute() {
        return Err(Error::UnsafeOidcPath);
    }
    let resolved = tokio::fs::canonicalize(path)
        .await
        .map_err(|_| Error::OidcFile)?;
    let allowed = environment
        .get(OIDC_ALLOWED_CREDENTIAL_DIRS)
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_CREDENTIAL_DIRS.into());
    for directory in allowed.split(',').map(str::trim).filter(|d| !d.is_empty()) {
        if let Ok(directory) = tokio::fs::canonicalize(directory).await
            && resolved.starts_with(directory)
        {
            return tokio::fs::read_to_string(&resolved)
                .await
                .map(|value| SecretValue::new(value.replace("\r\n", "\n").replace('\r', "\n")))
                .map_err(|_| Error::OidcFile);
        }
    }
    Err(Error::UnsafeOidcPath)
}

fn oidc_token_cache_ttl(token: &str, now: SystemTime, max_ttl: Duration) -> Option<Duration> {
    let fallback = Some(max_ttl);
    let segments: Vec<_> = token.split('.').collect();
    let [_, payload, _] = segments.as_slice() else {
        return fallback;
    };
    let Ok(decoded) = URL_SAFE_NO_PAD.decode(payload.trim_end_matches('=')) else {
        return fallback;
    };
    let Ok(claims) = serde_json::from_slice::<OidcTokenClaims>(&decoded) else {
        return fallback;
    };
    let Some(exp) = claims.exp.and_then(NumericDate::seconds) else {
        return fallback;
    };
    let seconds = exp.trunc()
        - now.duration_since(UNIX_EPOCH).ok()?.as_secs() as f64
        - TOKEN_EXPIRY_MARGIN_SECONDS;
    (seconds > 0.0).then(|| Duration::from_secs_f64(seconds.min(max_ttl.as_secs_f64())))
}
