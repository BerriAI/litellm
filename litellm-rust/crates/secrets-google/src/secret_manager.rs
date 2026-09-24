use std::{sync::Arc, time::Duration};

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_core_utils::settings::Lookup;
use litellm_secrets_types::{
    BaseSecretManager, GoogleOperationContext, Secret, SecretCache, SecretValue,
};
use serde::Deserialize;

use litellm_auth_gcp::GoogleCredentials;

use crate::{Error, auth};

const GOOGLE_SECRET_MANAGER_PROJECT_ID: &str = "GOOGLE_SECRET_MANAGER_PROJECT_ID";
const GOOGLE_SECRET_MANAGER_REFRESH_INTERVAL: &str = "GOOGLE_SECRET_MANAGER_REFRESH_INTERVAL";
const SECRET_MANAGER_REFRESH_INTERVAL: &str = "SECRET_MANAGER_REFRESH_INTERVAL";
const GOOGLE_SECRET_MANAGER_ALWAYS_READ_SECRET_MANAGER: &str =
    "GOOGLE_SECRET_MANAGER_ALWAYS_READ_SECRET_MANAGER";
const GCS_PATH_SERVICE_ACCOUNT: &str = "GCS_PATH_SERVICE_ACCOUNT";
const DEFAULT_REFRESH_INTERVAL: Duration = Duration::from_secs(86400);
const DEFAULT_CACHE_TTL: Duration = Duration::from_secs(600);
const CACHE_CAPACITY: u64 = 200;

#[derive(Clone)]
pub struct GoogleSecretManager {
    client: reqwest::Client,
    credentials: Arc<GoogleCredentials>,
    endpoint: reqwest::Url,
    project: String,
    cache: SecretCache<String, SecretValue>,
    python_misses: moka::future::Cache<String, ()>,
    always_read: bool,
}

#[derive(Deserialize)]
struct Response {
    payload: Option<Payload>,
}

#[derive(Deserialize)]
struct Payload {
    data: Option<String>,
    #[serde(rename = "dataCrc32c")]
    data_crc32c: Option<String>,
}

impl GoogleSecretManager {
    pub fn with_client(
        client: reqwest::Client,
        endpoint: reqwest::Url,
        project: String,
        environment: Arc<dyn Lookup + Send + Sync>,
        refresh_interval: Option<Duration>,
        always_read: bool,
    ) -> Result<Self, Error> {
        let credentials = auth::credentials(
            Some(project.clone()),
            environment
                .get(GCS_PATH_SERVICE_ACCOUNT)
                .map(SecretValue::new),
            environment,
        );
        let ttl = refresh_interval
            .filter(|ttl| !ttl.is_zero())
            .unwrap_or(DEFAULT_CACHE_TTL);
        let cache = SecretCache::new(CACHE_CAPACITY, ttl);
        Ok(Self {
            client,
            credentials: Arc::new(credentials),
            endpoint,
            project,
            cache,
            python_misses: moka::future::Cache::builder()
                .max_capacity(CACHE_CAPACITY)
                .time_to_live(ttl)
                .build(),
            always_read,
        })
    }

    pub fn new(
        environment: Arc<dyn Lookup + Send + Sync>,
        enterprise_enabled: bool,
    ) -> Result<Self, Error> {
        if !enterprise_enabled {
            return Err(Error::EnterpriseRequired);
        }
        let project = environment
            .get(GOOGLE_SECRET_MANAGER_PROJECT_ID)
            .ok_or(Error::MissingEnvironment(GOOGLE_SECRET_MANAGER_PROJECT_ID))?;
        let ttl = environment
            .get(GOOGLE_SECRET_MANAGER_REFRESH_INTERVAL)
            .filter(|v| !v.is_empty())
            .map(|v| v.parse::<i64>().map_err(|_| Error::RefreshInterval))
            .transpose()?
            .unwrap_or(
                environment
                    .get(SECRET_MANAGER_REFRESH_INTERVAL)
                    .map(|v| v.parse::<i64>().map_err(|_| Error::RefreshInterval))
                    .transpose()?
                    .unwrap_or(DEFAULT_REFRESH_INTERVAL.as_secs() as i64),
            );
        let always_read = environment
            .get(GOOGLE_SECRET_MANAGER_ALWAYS_READ_SECRET_MANAGER)
            .is_some_and(|v| v.eq_ignore_ascii_case("true"));
        Self::with_client(
            reqwest::Client::new(),
            reqwest::Url::parse("https://secretmanager.googleapis.com").expect("static URL"),
            project,
            environment,
            Some(if ttl < 0 {
                Duration::from_nanos(1)
            } else {
                Duration::from_secs(ttl as u64)
            }),
            always_read,
        )
    }

    pub async fn get_secret_from_google_secret_manager(
        &self,
        name: &str,
    ) -> Result<Option<Secret>, Error> {
        BaseSecretManager::async_read_secret(self, name, &GoogleOperationContext::default())
            .await
            .map(|value| value.map(Secret::String))
    }

    pub async fn get_secret_for_python(&self, name: &str) -> Result<Option<Secret>, Error> {
        if !self.always_read && self.python_misses.get(name).await.is_some() {
            return Ok(None);
        }
        let result = self.get_secret_from_google_secret_manager(name).await;
        if matches!(
            result,
            Ok(None) | Err(Error::Status(_) | Error::MissingPayload)
        ) {
            self.python_misses.insert(name.to_owned(), ()).await;
        }
        match result {
            Ok(None) => Err(Error::Status(404)),
            result => result,
        }
    }

    async fn read(&self, name: &str) -> Result<Option<SecretValue>, Error> {
        if self.always_read {
            return self.read_uncached(name).await;
        }
        self.cache
            .read(name.to_owned(), self.read_uncached(name))
            .await
    }

    async fn read_uncached(&self, name: &str) -> Result<Option<SecretValue>, Error> {
        let url = self
            .endpoint
            .join(&format!(
                "/v1/projects/{}/secrets/{}/versions/latest:access",
                percent_encoding::utf8_percent_encode(
                    &self.project,
                    percent_encoding::NON_ALPHANUMERIC
                ),
                percent_encoding::utf8_percent_encode(name, percent_encoding::NON_ALPHANUMERIC)
            ))
            .map_err(|_| Error::Endpoint)?;
        let response = self
            .client
            .get(url)
            .headers(self.credentials.request_headers().await?)
            .send()
            .await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if response.status() != reqwest::StatusCode::OK {
            return Err(Error::Status(response.status().as_u16()));
        }
        let response: Response = response.json().await?;
        let Some(payload) = response.payload else {
            return Err(Error::MissingPayload);
        };
        let Some(data) = payload.data else {
            return Err(Error::MissingPayload);
        };
        let bytes = STANDARD.decode(data)?;
        if let Some(expected) = payload.data_crc32c {
            let expected = expected.parse::<u32>().map_err(|_| Error::Checksum)?;
            if crc32c::crc32c(&bytes) != expected {
                return Err(Error::Checksum);
            }
        }
        let plaintext = String::from_utf8(bytes).map_err(|_| Error::Utf8)?;
        let value = SecretValue::new(plaintext);
        Ok(Some(value))
    }
}

impl BaseSecretManager for GoogleSecretManager {
    type Error = Error;
    type Context = GoogleOperationContext;

    async fn async_read_secret(
        &self,
        name: &str,
        context: &Self::Context,
    ) -> Result<Option<SecretValue>, Error> {
        match context.timeout {
            Some(timeout) => tokio::time::timeout(timeout, self.read(name))
                .await
                .map_err(|_| Error::Timeout)?,
            None => self.read(name).await,
        }
    }
}
