//! What a Google login reads from the machine besides credential files: the
//! process environment, the gcloud CLI, the GCE metadata server and Cloud Resource Manager, and
//! the HTTP client litellm's own Google requests go through.

use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, LazyLock},
    time::Duration,
};

use litellm_env::process_environment;
use litellm_http::{
    ClientIdentity, ClientVariant, HttpClientPool, HttpSettings, Resolution, Verify,
    media::PublicDnsResolver,
    requests_env::{ca_bundle, proxies_for, requests_proxy},
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
/// `_metadata.ping`'s `_METADATA_DETECT_RETRIES`, with `ExponentialBackoff`'s first wait.
const PING_RETRIES: Retries = Retries {
    attempts: 3,
    first_wait: Duration::from_secs(1),
};
/// `_metadata.get`'s `retry_count`, with `ExponentialBackoff`'s first wait.
const GET_RETRIES: Retries = Retries {
    attempts: 5,
    first_wait: Duration::from_secs(1),
};
/// `transport.DEFAULT_RETRYABLE_STATUS_CODES`.
const RETRYABLE_STATUSES: [u16; 5] = [500, 503, 504, 408, 429];

/// How many times google-auth's `_metadata` tries a request, and its wait before the second try,
/// which doubles before each try after it.
#[derive(Clone, Copy)]
struct Retries {
    attempts: u32,
    first_wait: Duration,
}

/// One try at a metadata server request: its outcome, or a failure worth trying again.
enum Attempt<T> {
    Done(T),
    Retry,
}

impl Retries {
    /// `attempt`'s outcome on the first try that does not ask to be retried, `None` when every
    /// try did.
    async fn run<T, F: Future<Output = Attempt<T>>>(self, attempt: impl Fn() -> F) -> Option<T> {
        for number in 0..self.attempts {
            if number > 0 {
                tokio::time::sleep(self.first_wait * 2u32.pow(number - 1)).await;
            }
            if let Attempt::Done(outcome) = attempt().await {
                return Some(outcome);
            }
        }
        None
    }
}

/// `_metadata.ping`: whether the metadata server at `root` answers as Google's, tried again only
/// when it cannot be reached.
async fn ping(client: &litellm_http::Client, root: &str, retries: Retries) -> bool {
    retries
        .run(|| async {
            match client
                .get(root)
                .header(METADATA_FLAVOR, GOOGLE)
                .timeout(METADATA_TIMEOUT)
                .send()
                .await
            {
                Err(_) => Attempt::Retry,
                Ok(response) => Attempt::Done(
                    response.status() == reqwest::StatusCode::OK
                        && response
                            .headers()
                            .get(METADATA_FLAVOR)
                            .is_some_and(|flavor| flavor == GOOGLE),
                ),
            }
        })
        .await
        .unwrap_or(false)
}

/// `_metadata.get` of the text at `url`, tried again when the server cannot be reached or answers
/// a retryable status; `None` once it gives up or answers anything but HTTP 200.
async fn metadata_text(
    client: &litellm_http::Client,
    url: &str,
    retries: Retries,
) -> Option<String> {
    retries
        .run(|| async {
            match client
                .get(url)
                .header(METADATA_FLAVOR, GOOGLE)
                .timeout(METADATA_TIMEOUT)
                .send()
                .await
            {
                Err(_) => Attempt::Retry,
                Ok(response) if RETRYABLE_STATUSES.contains(&response.status().as_u16()) => {
                    Attempt::Retry
                }
                Ok(response) if response.status() == reqwest::StatusCode::OK => {
                    Attempt::Done(response.text().await.ok())
                }
                Ok(_) => Attempt::Done(None),
            }
        })
        .await
        .flatten()
}

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
            let pinged = match self.http_client(&root, None) {
                Ok(client) => ping(&client, &root, PING_RETRIES).await,
                Err(_) => false,
            };
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
            metadata_text(&self.http_client(&url, None).ok()?, &url, GET_RETRIES).await
        })
    }

    fn gce_credentials(&self) -> Result<Arc<dyn GcpTokenSource>, GoogleLoginFailure> {
        SdkCredentials::Metadata.build().map_err(|_| {
            GoogleLoginFailure::new(
                PythonException::DefaultCredentials,
                super::failure::UNLOADABLE_CREDENTIALS,
            )
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

#[cfg(test)]
mod tests {
    use std::sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use rstest::rstest;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;

    /// What the fake metadata server does with one connection.
    #[derive(Clone, Copy, Debug)]
    enum Answer {
        /// Closes the connection unanswered, a transport failure.
        Drop,
        /// Answers this status as Google's metadata server, with the body `project`.
        Google(u16),
        /// Answers this status without the `Metadata-Flavor` header.
        Other(u16),
    }

    use Answer::{Drop, Google, Other};

    const QUICK: Retries = Retries {
        attempts: 0,
        first_wait: Duration::from_millis(1),
    };

    /// A metadata server answering each connection with the next of `answers`, and how many
    /// connections it took.
    async fn metadata_server(answers: Vec<Answer>) -> (String, Arc<AtomicUsize>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let taken = Arc::new(AtomicUsize::new(0));
        let counted = taken.clone();
        let answers = Mutex::new(answers.into_iter());
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                counted.fetch_add(1, Ordering::SeqCst);
                let answer = answers.lock().unwrap().next().unwrap_or(Drop);
                let mut request = [0u8; 1024];
                let _ = stream.read(&mut request).await;
                let reply = match answer {
                    Drop => continue,
                    Google(status) => format!(
                        "HTTP/1.1 {status} X\r\nMetadata-Flavor: Google\r\nContent-Length: 7\r\nConnection: close\r\n\r\nproject"
                    ),
                    Other(status) => format!(
                        "HTTP/1.1 {status} X\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    ),
                };
                let _ = stream.write_all(reply.as_bytes()).await;
            }
        });
        (url, taken)
    }

    fn client(url: &str) -> litellm_http::Client {
        requests_client(&[], url, None).unwrap()
    }

    #[rstest]
    #[case::first_try(vec![Google(200)], true, 1)]
    #[case::third_try(vec![Drop, Drop, Google(200)], true, 3)]
    #[case::unreachable_on_every_try(vec![Drop, Drop, Drop, Google(200)], false, 3)]
    #[case::not_google(vec![Other(200), Google(200)], false, 1)]
    #[case::error_status_is_not_retried(vec![Google(503), Google(200)], false, 1)]
    #[tokio::test]
    async fn the_ping_is_tried_three_times_when_unreachable(
        #[case] answers: Vec<Answer>,
        #[case] on_gce: bool,
        #[case] tries: usize,
    ) {
        let (url, taken) = metadata_server(answers).await;

        let pinged = ping(
            &client(&url),
            &url,
            Retries {
                attempts: PING_RETRIES.attempts,
                ..QUICK
            },
        )
        .await;

        assert_eq!((pinged, taken.load(Ordering::SeqCst)), (on_gce, tries));
    }

    #[rstest]
    #[case::first_try(vec![Google(200)], Some("project"), 1)]
    #[case::after_transient_failures(vec![Drop, Google(503), Google(429), Google(200)], Some("project"), 4)]
    #[case::gives_up_after_five_tries(vec![Google(500), Drop, Google(504), Google(408), Google(503), Google(200)], None, 5)]
    #[case::not_found_is_not_retried(vec![Google(404), Google(200)], None, 1)]
    #[tokio::test]
    async fn the_project_lookup_retries_as_google_auth_does(
        #[case] answers: Vec<Answer>,
        #[case] project: Option<&str>,
        #[case] tries: usize,
    ) {
        let (url, taken) = metadata_server(answers).await;

        let found = metadata_text(
            &client(&url),
            &url,
            Retries {
                attempts: GET_RETRIES.attempts,
                ..QUICK
            },
        )
        .await;

        assert_eq!(
            (found.as_deref(), taken.load(Ordering::SeqCst)),
            (project, tries)
        );
    }
}
