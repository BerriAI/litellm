use std::{sync::Arc, time::Duration};

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_core_utils::settings::Lookup;
use litellm_secrets_types::{Secret, SecretValue};
use moka::future::Cache;
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
    cache: Cache<String, SecretValue>,
    always_read: bool,
}

#[derive(Deserialize)]
struct Response {
    payload: Option<Payload>,
}

#[derive(Deserialize)]
struct Payload {
    data: Option<String>,
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
        let cache = Cache::builder()
            .max_capacity(CACHE_CAPACITY)
            .time_to_live(ttl)
            .build();
        Ok(Self {
            client,
            credentials: Arc::new(credentials),
            endpoint,
            project,
            cache,
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
        if !self.always_read
            && let Some(cached) = self.cache.get(name).await
        {
            return Ok(Some(Secret::String(cached)));
        }
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
        let Some(data) = response.payload.and_then(|payload| payload.data) else {
            return Err(Error::MissingPayload);
        };
        let bytes = STANDARD.decode(data)?;
        let plaintext = String::from_utf8(bytes).map_err(|_| Error::Utf8)?;
        let value = SecretValue::new(plaintext);
        self.cache.insert(name.to_owned(), value.clone()).await;
        Ok(Some(Secret::String(value)))
    }
}
