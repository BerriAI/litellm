//! google-auth's `transport.requests.Request`: every token request of a Vertex AI Messages login
//! goes through `requests`, which reads its proxy and CA bundle from the process environment on
//! each request and verifies against certifi's bundle when none is named.

use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
};

use litellm_http::{
    ClientVariant, HttpClientPool, HttpSettings, Resolution, Verify,
    media::PublicDnsResolver,
    requests_env::{ca_bundle, process_environment, proxies_for, requests_proxy},
};
use url::Url;

use super::failure::{GoogleLoginFailure, PythonException};

static CERTIFI_BUNDLE: OnceLock<PathBuf> = OnceLock::new();

/// Names the file `certifi.where()` returns in the host's Python, which `requests` verifies
/// against when the environment names no CA bundle. Only the first call counts.
pub fn set_certifi_bundle(path: PathBuf) {
    let _ = CERTIFI_BUNDLE.set(path);
}

type Environment = Arc<dyn Fn() -> Vec<(String, String)> + Send + Sync>;

#[derive(Clone)]
pub(crate) struct Requests {
    pool: Arc<HttpClientPool>,
    environment: Environment,
    certifi: Option<PathBuf>,
}

impl Requests {
    pub(crate) fn process() -> Self {
        Self {
            pool: Arc::new(HttpClientPool::new(Arc::new(PublicDnsResolver))),
            environment: Arc::new(process_environment),
            certifi: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_environment(variables: &[(&str, &str)]) -> Self {
        let variables: Vec<(String, String)> = variables
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        Self {
            environment: Arc::new(move || variables.clone()),
            ..Self::process()
        }
    }

    #[cfg(test)]
    fn with_certifi(self, path: &str) -> Self {
        Self {
            certifi: Some(path.into()),
            ..self
        }
    }

    /// `requests`' default bundle: the host's certifi file, else the same Mozilla roots built in.
    fn default_verify(&self) -> Verify {
        self.certifi
            .clone()
            .or_else(|| CERTIFI_BUNDLE.get().cloned())
            .map_or(Verify::BuiltInRoots, Verify::CaBundle)
    }

    fn verify(&self, environment: &[(String, String)]) -> Result<Verify, GoogleLoginFailure> {
        match ca_bundle(environment) {
            Ok(Some(bundle)) => Ok(Verify::CaBundle(bundle)),
            Ok(None) => Ok(self.default_verify()),
            Err(missing) => Err(GoogleLoginFailure::new(
                PythonException::OS,
                missing.to_string(),
            )),
        }
    }

    /// The client `requests` would send a request for `url` through.
    pub(crate) fn client(&self, url: &str) -> Result<litellm_http::Client, GoogleLoginFailure> {
        let parsed = Url::parse(url).map_err(|_| invalid_url(url))?;
        let environment = (self.environment)();
        let mut config = Resolution::from(&HttpSettings::default()).config;
        config.verify = self.verify(&environment)?;
        config.proxies = proxies_for(&parsed, requests_proxy(&environment, &parsed));
        self.pool
            .client(&config, ClientVariant::Provider)
            .map_err(|error| GoogleLoginFailure::new(PythonException::Transport, error.to_string()))
    }
}

/// `requests`' `MissingSchema`, which google-auth's transport raises as a `TransportError`.
fn invalid_url(url: &str) -> GoogleLoginFailure {
    GoogleLoginFailure::new(
        PythonException::Transport,
        format!("Invalid URL '{url}': No scheme supplied. Perhaps you meant https://{url}?"),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use axum::{Router, http::Uri, routing::any};
    use rstest::rstest;

    use super::*;

    const PRESENT: &str = env!("CARGO_MANIFEST_PATH");

    #[rstest]
    #[case::certifi_by_default(&[], Ok(Verify::CaBundle("/certifi/cacert.pem".into())))]
    #[case::requests_bundle(&[("REQUESTS_CA_BUNDLE", PRESENT)], Ok(Verify::CaBundle(PRESENT.into())))]
    #[case::curl_bundle(&[("CURL_CA_BUNDLE", PRESENT)], Ok(Verify::CaBundle(PRESENT.into())))]
    #[case::ssl_cert_file_is_not_read(&[("SSL_CERT_FILE", PRESENT)], Ok(Verify::CaBundle("/certifi/cacert.pem".into())))]
    #[case::missing_bundle(
        &[("REQUESTS_CA_BUNDLE", "/absent/ca.pem")],
        Err(GoogleLoginFailure::new(PythonException::OS, "Could not find a suitable TLS CA certificate bundle, invalid path: /absent/ca.pem"))
    )]
    fn the_ca_bundle_is_the_one_requests_verifies_against(
        #[case] variables: &[(&str, &str)],
        #[case] expected: Result<Verify, GoogleLoginFailure>,
    ) {
        let requests = Requests::with_environment(variables).with_certifi("/certifi/cacert.pem");
        let environment = (requests.environment)();
        assert_eq!(requests.verify(&environment), expected);
    }

    #[rstest]
    fn a_missing_scheme_is_requests_transport_error() {
        let Err(failure) = Requests::with_environment(&[]).client("None") else {
            panic!("a URL without a scheme has no client");
        };
        assert_eq!(
            failure,
            GoogleLoginFailure::new(
                PythonException::Transport,
                "Invalid URL 'None': No scheme supplied. Perhaps you meant https://None?"
            )
        );
    }

    /// An HTTP proxy that answers every request itself and records the target it was asked for.
    async fn proxy() -> (String, Arc<Mutex<Vec<String>>>, tokio::task::JoinHandle<()>) {
        let targets = Arc::new(Mutex::new(Vec::new()));
        let recorded = targets.clone();
        let app = Router::new().fallback(any(move |uri: Uri| {
            recorded.lock().unwrap().push(uri.to_string());
            async { "proxied" }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (url, targets, server)
    }

    #[rstest]
    #[case::proxied(&[], true)]
    #[case::no_proxy_host(&[("NO_PROXY", "127.0.0.1")], false)]
    #[case::no_proxy_network(&[("no_proxy", "127.0.0.0/8")], false)]
    #[tokio::test]
    async fn token_requests_go_through_the_environment_s_proxy(
        #[case] extra: &[(&str, &str)],
        #[case] proxied: bool,
    ) {
        let (proxy, targets, server) = proxy().await;
        let variables: Vec<(&str, &str)> = [("HTTP_PROXY", proxy.as_str())]
            .into_iter()
            .chain(extra.iter().copied())
            .collect();
        let client = Requests::with_environment(&variables)
            .client("http://127.0.0.1:9/token")
            .unwrap();

        let answer = client.post("http://127.0.0.1:9/token").send().await;

        assert_eq!(answer.is_ok(), proxied);
        assert_eq!(
            targets.lock().unwrap().clone(),
            if proxied {
                vec!["http://127.0.0.1:9/token".to_owned()]
            } else {
                vec![]
            }
        );
        server.abort();
    }
}
