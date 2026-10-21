mod client;
mod read;
mod write;

use std::{sync::Arc, time::Duration};

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_core_utils::settings::Lookup;
use litellm_http::{
    Client, ClientIdentity, ClientVariant, HttpClientConfig, HttpClientPool, TlsSource, Verify,
};
use litellm_secrets_types::{
    BaseSecretManager, CyberarkOperationContext, RotationError, SecretCache, SecretDeleter,
    SecretRotator, SecretValue, SecretWriteContext, SecretWriter, async_rotate_secret,
    validate_secret_name,
};
use moka::future::Cache;
use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, utf8_percent_encode};

use crate::Error;

const CYBERARK_API_BASE: &str = "CYBERARK_API_BASE";
const CYBERARK_ACCOUNT: &str = "CYBERARK_ACCOUNT";
const CYBERARK_USERNAME: &str = "CYBERARK_USERNAME";
const CYBERARK_API_KEY: &str = "CYBERARK_API_KEY";
const CYBERARK_CLIENT_CERT: &str = "CYBERARK_CLIENT_CERT";
const CYBERARK_CLIENT_KEY: &str = "CYBERARK_CLIENT_KEY";
const CYBERARK_SSL_VERIFY: &str = "CYBERARK_SSL_VERIFY";
const CYBERARK_REFRESH_INTERVAL: &str = "CYBERARK_REFRESH_INTERVAL";
const DEFAULT_API_BASE: &str = "http://127.0.0.1:8080";
const DEFAULT_ACCOUNT: &str = "default";
const DEFAULT_USERNAME: &str = "admin";
const DEFAULT_REFRESH_INTERVAL: Duration = Duration::from_secs(300);
const MAX_TOKEN_LIFETIME: Duration = Duration::from_secs(7 * 60);
const SECRET_NAME_SAFE: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'_')
    .remove(b'.')
    .remove(b'~');

#[derive(Clone)]
pub struct CyberArkSecretManager {
    client: Client,
    endpoint: reqwest::Url,
    account: String,
    username: String,
    api_key: SecretValue,
    token: Cache<(), SecretValue>,
    secrets: SecretCache<String, SecretValue>,
    authentication_lock: Arc<tokio::sync::Mutex<()>>,
    policy_load_lock: Arc<tokio::sync::Mutex<()>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeleteOutcome {
    NotSupported,
}

#[derive(Clone, Copy)]
pub enum AuthenticationRetry {
    Never,
    Unauthorized,
}

#[derive(veil::Redact)]
pub struct WriteFailure {
    pub source: Error,
    #[redact]
    pub request_url: Option<reqwest::Url>,
    pub authentication: bool,
}

impl WriteFailure {
    fn local(source: Error) -> Self {
        Self {
            source,
            request_url: None,
            authentication: false,
        }
    }

    fn request(source: Error, url: reqwest::Url) -> Self {
        Self {
            source,
            request_url: Some(url),
            authentication: false,
        }
    }
}

impl CyberArkSecretManager {
    fn secret_url(&self, name: &str) -> Result<reqwest::Url, Error> {
        let encoded = utf8_percent_encode(name, SECRET_NAME_SAFE);
        self.endpoint
            .join(&format!("secrets/{}/variable/{}", self.account, encoded))
            .map_err(|_| Error::Endpoint)
    }
}

fn with_timeout(
    request: reqwest::RequestBuilder,
    context: &CyberarkOperationContext,
) -> reqwest::RequestBuilder {
    match context.timeout {
        Some(timeout) => request.timeout(timeout),
        None => request,
    }
}
