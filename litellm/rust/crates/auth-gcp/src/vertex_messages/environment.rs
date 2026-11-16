use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

use google_cloud_auth::credentials::mds;
use litellm_http::{
    ClientVariant, HttpClientPool, HttpSettings, Resolution, media::PublicDnsResolver,
};
use litellm_secret_value::SecretValue;
use serde_json::Value;

use super::{
    builders::{TokenSource, token_failure},
    failure::{GoogleLoginFailure, PythonException},
};
use crate::{
    constants::{CLOUD_PLATFORM_SCOPE, GOOGLE_OAUTH_TOKEN_ENDPOINT},
    identity::SdkTokenSource,
};

pub type EnvironmentFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

const METADATA_FLAVOR: &str = "Metadata-Flavor";
const GOOGLE: &str = "Google";
const DEFAULT_METADATA_IP: &str = "169.254.169.254";
const DEFAULT_METADATA_HOST: &str = "metadata.google.internal";
const DEFAULT_METADATA_TIMEOUT_SECONDS: u64 = 3;
const DEFAULT_METADATA_DETECT_RETRIES: u32 = 3;
const METADATA_PROJECT_RETRIES: u32 = 5;
const GCE_PRODUCT_NAME_FILE: &str = "/sys/class/dmi/id/product_name";

/// What google-auth's `default()` reads from the machine besides credential files: the process
/// environment, the gcloud CLI, the GCE metadata server and Cloud Resource Manager.
pub trait GoogleEnvironment: Send + Sync {
    /// `os.environ`, which google-auth reads directly rather than through LiteLLM's secrets.
    fn os_env(&self, name: &str) -> Option<String>;
    /// The endpoint google-auth refreshes an authorized user at, never the credentials' own.
    fn google_token_endpoint(&self) -> String;
    /// `gcloud config get project`, or `None` when the command fails or prints nothing.
    fn gcloud_project(&self) -> EnvironmentFuture<'_, Option<String>>;
    /// `_metadata.is_on_gce`.
    fn on_gce(&self) -> EnvironmentFuture<'_, bool>;
    /// `_metadata.get_project_id`, `None` on a transport failure.
    fn gce_project(&self) -> EnvironmentFuture<'_, Option<String>>;
    /// `compute_engine.Credentials`.
    fn gce_credentials(&self) -> Result<TokenSource, GoogleLoginFailure>;
    /// The `projectId` Cloud Resource Manager answers for a project number with HTTP 200.
    fn resource_manager_project(
        &self,
        url: String,
        token: SecretValue,
    ) -> EnvironmentFuture<'_, Result<Option<String>, GoogleLoginFailure>>;
}

pub struct SystemGoogleEnvironment {
    pool: HttpClientPool,
}

impl Default for SystemGoogleEnvironment {
    fn default() -> Self {
        Self {
            pool: HttpClientPool::new(Arc::new(PublicDnsResolver)),
        }
    }
}

impl SystemGoogleEnvironment {
    fn client(&self) -> Option<litellm_http::Client> {
        self.pool
            .client(
                &Resolution::from(&HttpSettings::default()).config,
                ClientVariant::Provider,
            )
            .ok()
    }

    fn numeric_env<T: std::str::FromStr>(&self, name: &str, default: T) -> T {
        self.os_env(name)
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or(default)
    }

    async fn ping(&self) -> bool {
        let root = format!(
            "http://{}",
            self.os_env("GCE_METADATA_IP")
                .unwrap_or_else(|| DEFAULT_METADATA_IP.into())
        );
        let timeout = Duration::from_secs(
            self.numeric_env("GCE_METADATA_TIMEOUT", DEFAULT_METADATA_TIMEOUT_SECONDS),
        );
        let attempts = self.numeric_env(
            "GCE_METADATA_DETECT_RETRIES",
            DEFAULT_METADATA_DETECT_RETRIES,
        );
        for _ in 0..attempts {
            let Some(client) = self.client() else {
                return false;
            };
            let response = client
                .get(&root)
                .header(METADATA_FLAVOR, GOOGLE)
                .timeout(timeout)
                .send()
                .await;
            if let Ok(response) = response {
                return response.status() == reqwest::StatusCode::OK
                    && response
                        .headers()
                        .get(METADATA_FLAVOR)
                        .is_some_and(|flavor| flavor == GOOGLE);
            }
        }
        false
    }

    fn metadata_host(&self) -> String {
        self.os_env("GCE_METADATA_HOST")
            .filter(|host| !host.is_empty())
            .or_else(|| self.os_env("GCE_METADATA_ROOT"))
            .unwrap_or_else(|| DEFAULT_METADATA_HOST.into())
    }
}

impl GoogleEnvironment for SystemGoogleEnvironment {
    fn os_env(&self, name: &str) -> Option<String> {
        std::env::var(name).ok()
    }

    fn google_token_endpoint(&self) -> String {
        GOOGLE_OAUTH_TOKEN_ENDPOINT.into()
    }

    fn gcloud_project(&self) -> EnvironmentFuture<'_, Option<String>> {
        Box::pin(async {
            let output = tokio::process::Command::new("gcloud")
                .args(["config", "get", "project"])
                .stderr(std::process::Stdio::null())
                .output()
                .await
                .ok()?;
            let project = String::from_utf8_lossy(&output.stdout).trim().to_string();
            (output.status.success() && !project.is_empty()).then_some(project)
        })
    }

    fn on_gce(&self) -> EnvironmentFuture<'_, bool> {
        Box::pin(async {
            if self.os_env("NO_GCE_CHECK").as_deref() == Some("true") {
                return false;
            }
            if self.ping().await {
                return true;
            }
            std::fs::read_to_string(GCE_PRODUCT_NAME_FILE)
                .is_ok_and(|product| product.trim().starts_with(GOOGLE))
        })
    }

    fn gce_project(&self) -> EnvironmentFuture<'_, Option<String>> {
        Box::pin(async {
            let url = format!(
                "http://{}/computeMetadata/v1/project/project-id",
                self.metadata_host()
            );
            for _ in 0..METADATA_PROJECT_RETRIES {
                let client = self.client()?;
                let Ok(response) = client
                    .get(&url)
                    .header(METADATA_FLAVOR, GOOGLE)
                    .timeout(Duration::from_secs(DEFAULT_METADATA_TIMEOUT_SECONDS))
                    .send()
                    .await
                else {
                    continue;
                };
                if response.status() != reqwest::StatusCode::OK {
                    return None;
                }
                return response.text().await.ok();
            }
            None
        })
    }

    fn gce_credentials(&self) -> Result<TokenSource, GoogleLoginFailure> {
        mds::Builder::default()
            .with_scopes([CLOUD_PLATFORM_SCOPE])
            .build_access_token_credentials()
            .map(|credentials| {
                std::sync::Arc::new(SdkTokenSource {
                    credentials,
                    project_id: None,
                }) as TokenSource
            })
            .map_err(|error| {
                GoogleLoginFailure::new(PythonException::DefaultCredentials, error.to_string())
            })
    }

    fn resource_manager_project(
        &self,
        url: String,
        token: SecretValue,
    ) -> EnvironmentFuture<'_, Result<Option<String>, GoogleLoginFailure>> {
        Box::pin(async move {
            let client = self.client().ok_or_else(|| {
                GoogleLoginFailure::new(PythonException::Transport, "no HTTP client")
            })?;
            let response = client
                .get(&url)
                .bearer_auth(token.expose())
                .send()
                .await
                .map_err(|error| token_failure(&error))?;
            if response.status() != reqwest::StatusCode::OK {
                return Ok(None);
            }
            let body: Value = response
                .json()
                .await
                .map_err(|error| token_failure(&error))?;
            Ok(body
                .get("projectId")
                .and_then(Value::as_str)
                .map(str::to_string))
        })
    }
}
