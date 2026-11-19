//! What the Vertex AI Messages login reads from the machine besides credential files: the
//! process environment, the gcloud CLI, the GCE metadata server and Cloud Resource Manager, and
//! the HTTP client litellm's own Google requests go through.

use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, LazyLock},
    time::Duration,
};

use litellm_http::{
    ClientIdentity, ClientVariant, HttpClientPool, HttpSettings, Resolution, Verify,
    media::PublicDnsResolver,
    requests_env::{ca_bundle, process_environment, proxies_for, requests_proxy},
};
use litellm_secret_value::SecretValue;
use serde::Deserialize;
use url::Url;

use super::failure::{GoogleLoginFailure, PythonException};
use crate::{GcpTokenSource, constants::GOOGLE_OAUTH_TOKEN_ENDPOINT, identity::SdkCredentials};

pub type EnvironmentFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

const METADATA_FLAVOR: &str = "Metadata-Flavor";
const GOOGLE: &str = "Google";
const DEFAULT_METADATA_IP: &str = "169.254.169.254";
const DEFAULT_METADATA_HOST: &str = "metadata.google.internal";
const METADATA_TIMEOUT: Duration = Duration::from_secs(3);
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
    fn gce_credentials(&self) -> Result<Arc<dyn GcpTokenSource>, GoogleLoginFailure>;
    /// The `projectId` Cloud Resource Manager answers for a project number with HTTP 200.
    fn resource_manager_project(
        &self,
        url: String,
        token: SecretValue,
    ) -> EnvironmentFuture<'_, Result<Option<String>, GoogleLoginFailure>>;
    /// The client `requests` would send a request for `url` through, with `identity` as its TLS
    /// client certificate: the proxy and CA bundle the process environment names now.
    fn http_client(
        &self,
        url: &str,
        identity: Option<ClientIdentity>,
    ) -> Result<litellm_http::Client, GoogleLoginFailure> {
        requests_client(&process_environment(), url, identity)
    }
}

/// The client `requests` would send a request for `url` through under `environment`.
pub(crate) fn requests_client(
    environment: &[(String, String)],
    url: &str,
    identity: Option<ClientIdentity>,
) -> Result<litellm_http::Client, GoogleLoginFailure> {
    static POOL: LazyLock<HttpClientPool> =
        LazyLock::new(|| HttpClientPool::new(Arc::new(PublicDnsResolver)));
    let parsed = Url::parse(url).map_err(|_| {
        GoogleLoginFailure::new(
            PythonException::Transport,
            format!("The token endpoint URL {url} has no scheme."),
        )
    })?;
    let mut config = Resolution::from(&HttpSettings::default()).config;
    config.verify = ca_bundle(environment)
        .map_err(|missing| GoogleLoginFailure::new(PythonException::OS, missing.to_string()))?
        .map_or(Verify::BuiltInRoots, Verify::CaBundle);
    config.proxies = proxies_for(&parsed, requests_proxy(environment, &parsed));
    config.client_certificate = identity;
    POOL.client(&config, ClientVariant::Provider)
        .map_err(|error| GoogleLoginFailure::new(PythonException::Transport, error.to_string()))
}

/// A failed request to a Google endpoint: a `TransportError`, as google-auth raises it.
pub(crate) fn unreachable(error: impl std::fmt::Display) -> GoogleLoginFailure {
    GoogleLoginFailure::new(PythonException::Transport, error.to_string())
}

/// Cloud Resource Manager's answer for a project.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ResourceManagerProject {
    project_id: Option<String>,
}

/// The machine this process runs on.
#[derive(Default)]
pub struct SystemGoogleEnvironment;

impl SystemGoogleEnvironment {
    async fn metadata_get(&self, url: &str) -> Option<reqwest::Response> {
        self.http_client(url, None)
            .ok()?
            .get(url)
            .header(METADATA_FLAVOR, GOOGLE)
            .timeout(METADATA_TIMEOUT)
            .send()
            .await
            .ok()
            .filter(|response| response.status() == reqwest::StatusCode::OK)
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
            let root = format!(
                "http://{}",
                self.os_env("GCE_METADATA_IP")
                    .unwrap_or_else(|| DEFAULT_METADATA_IP.into())
            );
            let pinged = self.metadata_get(&root).await.is_some_and(|response| {
                response
                    .headers()
                    .get(METADATA_FLAVOR)
                    .is_some_and(|flavor| flavor == GOOGLE)
            });
            pinged
                || std::fs::read_to_string(GCE_PRODUCT_NAME_FILE)
                    .is_ok_and(|product| product.trim().starts_with(GOOGLE))
        })
    }

    fn gce_project(&self) -> EnvironmentFuture<'_, Option<String>> {
        Box::pin(async {
            let url = format!(
                "http://{}/computeMetadata/v1/project/project-id",
                self.metadata_host()
            );
            self.metadata_get(&url).await?.text().await.ok()
        })
    }

    fn gce_credentials(&self) -> Result<Arc<dyn GcpTokenSource>, GoogleLoginFailure> {
        SdkCredentials::Metadata.build().map_err(|error| {
            GoogleLoginFailure::new(PythonException::DefaultCredentials, error.to_string())
        })
    }

    fn resource_manager_project(
        &self,
        url: String,
        token: SecretValue,
    ) -> EnvironmentFuture<'_, Result<Option<String>, GoogleLoginFailure>> {
        Box::pin(async move {
            let response = self
                .http_client(&url, None)?
                .get(&url)
                .bearer_auth(token.expose())
                .send()
                .await
                .map_err(unreachable)?;
            if response.status() != reqwest::StatusCode::OK {
                return Ok(None);
            }
            Ok(response
                .json::<ResourceManagerProject>()
                .await
                .ok()
                .and_then(|project| project.project_id))
        })
    }
}

/// A GCE machine whose metadata server grants `token` and names `project`, for tests of a login
/// that must not reach the network.
#[cfg(any(test, feature = "test-support"))]
pub struct OnGce {
    pub project: &'static str,
    pub token: &'static str,
}

#[cfg(any(test, feature = "test-support"))]
impl GoogleEnvironment for OnGce {
    fn os_env(&self, _name: &str) -> Option<String> {
        None
    }

    fn google_token_endpoint(&self) -> String {
        unreachable!("a GCE login refreshes at the metadata server")
    }

    fn gcloud_project(&self) -> EnvironmentFuture<'_, Option<String>> {
        Box::pin(async { None })
    }

    fn on_gce(&self) -> EnvironmentFuture<'_, bool> {
        Box::pin(async { true })
    }

    fn gce_project(&self) -> EnvironmentFuture<'_, Option<String>> {
        Box::pin(async { Some(self.project.to_string()) })
    }

    fn gce_credentials(&self) -> Result<Arc<dyn GcpTokenSource>, GoogleLoginFailure> {
        struct Granted(&'static str);

        impl GcpTokenSource for Granted {
            fn project_id(&self) -> crate::GcpAuthFuture<'_, String> {
                Box::pin(async {
                    unreachable!("a GCE login names its project from the metadata server")
                })
            }

            fn token(&self) -> crate::GcpAuthFuture<'_, SecretValue> {
                Box::pin(async { Ok(SecretValue::new(self.0)) })
            }
        }

        Ok(Arc::new(Granted(self.token)))
    }

    fn resource_manager_project(
        &self,
        _url: String,
        _token: SecretValue,
    ) -> EnvironmentFuture<'_, Result<Option<String>, GoogleLoginFailure>> {
        unreachable!("a GCE login names its project")
    }
}
