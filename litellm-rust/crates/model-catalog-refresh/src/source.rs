#![allow(
    clippy::disallowed_types,
    reason = "the fork-local client uses HttpClientConfig without the shared pool"
)]

use std::{path::PathBuf, sync::Arc, time::Duration};

use async_trait::async_trait;
use litellm_http::HttpClientConfig;
use litellm_model_catalog::{FetchRequest, Reply, SourceId};
use reqwest::header::{ETAG, IF_NONE_MATCH, RETRY_AFTER};
use url::Url;

use crate::{SourceError, process_local::ProcessLocal};

pub const DEFAULT_SOURCE: &str =
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    Http(Url),
    File(PathBuf),
}

impl Source {
    pub fn parse(value: &str) -> Result<Self, SourceError> {
        let url = Url::parse(value).map_err(|error| SourceError::Invalid {
            reason: error.to_string(),
        })?;
        match url.scheme() {
            "http" | "https" => Ok(Self::Http(url)),
            "file" => url
                .to_file_path()
                .map(Self::File)
                .map_err(|()| SourceError::Invalid {
                    reason: "expected an absolute local file URL".to_owned(),
                }),
            scheme => Err(SourceError::UnsupportedScheme {
                scheme: scheme.to_owned(),
            }),
        }
    }

    pub fn id(&self) -> SourceId {
        SourceId(match self {
            Self::Http(url) => url.to_string(),
            Self::File(path) => Url::from_file_path(path).map_or_else(
                |()| format!("file://{}", path.display()),
                |url| url.to_string(),
            ),
        })
    }

    pub fn fetcher(&self, http: &HttpClientConfig) -> Result<Arc<dyn Fetch>, litellm_http::Error> {
        match self {
            Self::Http(url) => Ok(Arc::new(HttpSource::new(url.clone(), http)?)),
            Self::File(path) => Ok(Arc::new(FileSource::new(path.clone()))),
        }
    }
}

/// Implementations must discard process-bound transport state after a fork.
#[async_trait]
pub trait Fetch: Send + Sync + 'static {
    async fn fetch(&self, request: FetchRequest) -> Reply;
}

pub struct HttpSource {
    url: Url,
    config: HttpClientConfig,
    client: ProcessLocal<Result<reqwest::Client, litellm_http::Error>>,
}

impl HttpSource {
    pub fn new(url: Url, config: &HttpClientConfig) -> Result<Self, litellm_http::Error> {
        let source = Self {
            url,
            config: config.clone(),
            client: ProcessLocal::new(),
        };
        source.client()?;
        Ok(source)
    }

    fn client(&self) -> Result<&reqwest::Client, litellm_http::Error> {
        self.client
            .get_or_init(|| {
                reqwest::ClientBuilder::try_from(&self.config)?
                    .build()
                    .map_err(Into::into)
            })
            .as_ref()
            .map_err(Clone::clone)
    }
}

#[async_trait]
impl Fetch for HttpSource {
    async fn fetch(&self, request: FetchRequest) -> Reply {
        if !matches!(self.url.scheme(), "http" | "https") {
            return Reply::Failed {
                reason: "unsupported HTTP source protocol".to_owned(),
            };
        }
        let client = match self.client() {
            Ok(client) => client,
            Err(error) => {
                return Reply::Failed {
                    reason: error.to_string(),
                };
            }
        };
        let builder = client.get(self.url.clone()).timeout(request.timeout);
        let builder = match request.if_none_match {
            Some(validator) => builder.header(IF_NONE_MATCH, validator),
            None => builder,
        };
        let response = match builder.send().await {
            Ok(response) => response,
            Err(error) => return transport_error(error),
        };
        let status = response.status();
        if status.as_u16() == 304 {
            return Reply::NotModified;
        }
        if matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504) {
            let after = response
                .headers()
                .get(RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.trim().parse::<f64>().ok())
                .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok());
            return Reply::Retry {
                after,
                reason: format!("HTTP {} fetching model catalog", status.as_u16()),
            };
        }
        if !status.is_success() {
            return Reply::Failed {
                reason: format!("HTTP {} fetching model catalog", status.as_u16()),
            };
        }
        let validators: Vec<_> = response
            .headers()
            .get_all(ETAG)
            .iter()
            .filter_map(|value| value.to_str().ok())
            .collect();
        let validator = (!validators.is_empty()).then(|| validators.join(", "));
        match response.bytes().await {
            Ok(body) => Reply::Fetched { body, validator },
            Err(error) => transport_error(error),
        }
    }
}

fn transport_error(error: reqwest::Error) -> Reply {
    let failed = error.is_builder();
    let reason = error.without_url().to_string();
    if failed {
        Reply::Failed { reason }
    } else {
        Reply::Retry {
            after: None,
            reason,
        }
    }
}

pub struct FileSource {
    path: PathBuf,
}

impl FileSource {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

#[async_trait]
impl Fetch for FileSource {
    async fn fetch(&self, _request: FetchRequest) -> Reply {
        match tokio::fs::read(&self.path).await {
            Ok(body) => Reply::Fetched {
                body: body.into(),
                validator: None,
            },
            Err(error) => Reply::Failed {
                reason: error.to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::Body,
        http::{HeaderMap, Response, StatusCode, Uri},
        routing::get,
    };
    use litellm_http::{EnvironmentProxies, HttpSettings, Resolution, Verify};
    use rstest::{fixture, rstest};
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
        sync::mpsc,
        task::JoinHandle,
    };

    #[fixture]
    fn config() -> HttpClientConfig {
        Resolution::from(&HttpSettings::default()).config
    }

    #[fixture]
    fn request() -> FetchRequest {
        FetchRequest {
            if_none_match: Some("\"old\"".to_owned()),
            timeout: Duration::from_secs(2),
        }
    }

    struct Server {
        url: Url,
        task: JoinHandle<()>,
    }

    impl Drop for Server {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    async fn serve(
        status: u16,
        etag: Option<&'static str>,
        retry_after: Option<&'static str>,
    ) -> (Server, mpsc::UnboundedReceiver<(HeaderMap, Uri)>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = Url::parse(&format!(
            "http://{}/catalog",
            listener.local_addr().unwrap()
        ))
        .unwrap();
        let (sent, received) = mpsc::unbounded_channel();
        let router = Router::new().fallback(get(move |headers: HeaderMap, uri: Uri| {
            let sent = sent.clone();
            async move {
                sent.send((headers, uri)).unwrap();
                let mut response =
                    Response::builder().status(StatusCode::from_u16(status).unwrap());
                if let Some(etag) = etag {
                    response = response.header(ETAG, etag);
                }
                if let Some(retry_after) = retry_after {
                    response = response.header(RETRY_AFTER, retry_after);
                }
                response.body(Body::from("catalog bytes")).unwrap()
            }
        }));
        let task = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        (Server { url, task }, received)
    }

    #[rstest]
    #[case::https("https://example.test/map", Ok("http"))]
    #[case::http("http://example.test/map", Ok("http"))]
    #[case::file("file:///etc/litellm/my.json", Ok("file"))]
    #[case::escaped_file("file:///etc/litellm/my%20map.json", Ok("file"))]
    #[case::s3("s3://bucket/key", Err("s3"))]
    #[case::ftp("ftp://example.test/map", Err("ftp"))]
    #[case::invalid("not a URL", Err("invalid"))]
    #[case::missing_host("http://", Err("invalid"))]
    fn source_parsing(#[case] value: &str, #[case] expected: Result<&str, &str>) {
        match (Source::parse(value), expected) {
            (Ok(Source::Http(url)), Ok("http")) => assert_eq!(url.as_str(), value),
            (Ok(source @ Source::File(_)), Ok("file")) => {
                assert_eq!(Source::parse(&source.id().0).unwrap(), source);
                assert!(matches!(source, Source::File(ref path) if path.is_absolute()));
            }
            (Err(SourceError::UnsupportedScheme { scheme }), Err(expected)) => {
                assert_eq!(scheme, expected)
            }
            (Err(SourceError::Invalid { reason }), Err("invalid")) => assert!(!reason.is_empty()),
            result => panic!("unexpected parsing result: {result:?}"),
        }
    }

    #[rstest]
    #[case::ok(200, Some("\"new\""), None, "fetched", None)]
    #[case::other_success(201, None, None, "fetched", None)]
    #[case::not_modified(304, None, None, "not_modified", None)]
    #[case::rate_limit(429, None, Some("2"), "retry", Some(2.0))]
    #[case::internal_error(500, None, None, "retry", None)]
    #[case::bad_gateway(502, None, None, "retry", None)]
    #[case::unavailable(503, None, Some("2"), "retry", Some(2.0))]
    #[case::gateway_timeout(504, None, None, "retry", None)]
    #[case::fraction(503, None, Some("0.25"), "retry", Some(0.25))]
    #[case::zero(503, None, Some("0"), "retry", Some(0.0))]
    #[case::date(503, None, Some("Wed, 21 Oct 2015 07:28:00 GMT"), "retry", None)]
    #[case::negative(503, None, Some("-1"), "retry", None)]
    #[case::nan(503, None, Some("NaN"), "retry", None)]
    #[case::infinite(503, None, Some("inf"), "retry", None)]
    #[case::invalid(503, None, Some("later"), "retry", None)]
    #[case::not_implemented(501, None, None, "failed", None)]
    #[case::not_found(404, None, None, "failed", None)]
    #[tokio::test]
    async fn http_status_and_headers(
        config: HttpClientConfig,
        request: FetchRequest,
        #[case] status: u16,
        #[case] etag: Option<&'static str>,
        #[case] retry_after: Option<&'static str>,
        #[case] outcome: &str,
        #[case] delay: Option<f64>,
    ) {
        let (server, mut received) = serve(status, etag, retry_after).await;
        let fetch = HttpSource::new(server.url.clone(), &config).unwrap();
        let reply = fetch.fetch(request).await;
        assert_eq!(received.recv().await.unwrap().0[IF_NONE_MATCH], "\"old\"");
        match (reply, outcome) {
            (Reply::Fetched { body, validator }, "fetched") => {
                assert_eq!(body.as_ref(), b"catalog bytes");
                assert_eq!(validator.as_deref(), etag);
            }
            (Reply::NotModified, "not_modified") => {}
            (Reply::Retry { after, reason }, "retry") => {
                assert_eq!(after.map(|value| value.as_secs_f64()), delay);
                assert!(reason.contains(&status.to_string()));
            }
            (Reply::Failed { reason }, "failed") => assert!(reason.contains(&status.to_string())),
            result => panic!("unexpected response: {result:?}"),
        }
    }

    #[rstest]
    #[tokio::test]
    async fn configured_proxy_receives_the_request(
        mut config: HttpClientConfig,
        request: FetchRequest,
    ) {
        let (proxy, mut received) = serve(200, None, None).await;
        config.proxies = EnvironmentProxies::from_environment(&|key: &str| {
            (key == "http_proxy").then(|| proxy.url.to_string())
        });
        config.user_agent = Some("catalog-test".to_owned());
        let fetch = HttpSource::new(
            Url::parse("http://catalog.invalid/map?version=1").unwrap(),
            &config,
        )
        .unwrap();
        assert!(matches!(fetch.fetch(request).await, Reply::Fetched { .. }));
        let (headers, uri) = received.recv().await.unwrap();
        assert_eq!(uri.to_string(), "http://catalog.invalid/map?version=1");
        assert_eq!(headers[reqwest::header::USER_AGENT], "catalog-test");
    }

    #[rstest]
    #[case::cutoff(false)]
    #[case::timeout(true)]
    #[tokio::test]
    async fn transport_and_body_failures_retry(
        config: HttpClientConfig,
        mut request: FetchRequest,
        #[case] timeout: bool,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = Url::parse(&format!(
            "http://{}/catalog",
            listener.local_addr().unwrap()
        ))
        .unwrap();
        let task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0; 2048];
            assert!(stream.read(&mut request).await.unwrap() > 0);
            if timeout {
                std::future::pending::<()>().await;
            }
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort")
                .await
                .unwrap();
        });
        let server = Server { url, task };
        request.timeout = Duration::from_millis(100);
        let fetch = HttpSource::new(server.url.clone(), &config).unwrap();
        assert!(
            matches!(fetch.fetch(request).await, Reply::Retry { after: None, reason } if !reason.is_empty())
        );
    }

    #[rstest]
    #[case::disabled("disabled", true)]
    #[case::built_in_roots("roots", false)]
    #[case::ca_bundle("bundle", true)]
    #[tokio::test]
    async fn tls_verification_uses_shared_config(
        mut config: HttpClientConfig,
        request: FetchRequest,
        #[case] verify: &str,
        #[case] succeeds: bool,
    ) {
        let identity = rcgen::generate_simple_self_signed(vec!["localhost".to_owned()]).unwrap();
        let bundle = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(bundle.path(), identity.cert.pem()).unwrap();
        config.verify = match verify {
            "disabled" => Verify::Disabled,
            "bundle" => Verify::CaBundle(bundle.path().to_owned()),
            _ => Verify::BuiltInRoots,
        };
        let tls = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(
            vec![identity.cert.der().clone()],
            rustls::pki_types::PrivatePkcs8KeyDer::from(identity.signing_key.serialize_der())
                .into(),
        )
        .unwrap();
        let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(tls));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = Url::parse(&format!(
            "https://localhost:{}/catalog",
            listener.local_addr().unwrap().port()
        ))
        .unwrap();
        let task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            if let Ok(mut stream) = acceptor.accept(stream).await {
                let mut request = [0; 2048];
                assert!(stream.read(&mut request).await.unwrap() > 0);
                stream
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
                    )
                    .await
                    .unwrap();
                stream.shutdown().await.unwrap();
            }
        });
        let server = Server { url, task };
        let fetch = HttpSource::new(server.url.clone(), &config).unwrap();
        match (fetch.fetch(request).await, succeeds) {
            (Reply::Fetched { body, .. }, true) => assert_eq!(body.as_ref(), b"{}"),
            (Reply::Retry { reason, .. }, false) => assert!(!reason.is_empty()),
            result => panic!("unexpected TLS result: {result:?}"),
        }
    }

    #[rstest]
    #[case::unsupported(false)]
    #[case::invalid_header(true)]
    #[tokio::test]
    async fn invalid_requests_fail_without_retry(
        config: HttpClientConfig,
        mut request: FetchRequest,
        #[case] invalid_header: bool,
    ) {
        let url = if invalid_header {
            request.if_none_match = Some("invalid\nvalidator".to_owned());
            "https://catalog.invalid/map"
        } else {
            "ftp://catalog.invalid/map"
        };
        let fetch = HttpSource::new(Url::parse(url).unwrap(), &config).unwrap();
        assert!(
            matches!(fetch.fetch(request).await, Reply::Failed { reason } if !reason.is_empty())
        );
    }

    #[rstest]
    #[tokio::test]
    async fn files_always_reread_without_validators(
        config: HttpClientConfig,
        request: FetchRequest,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("my map.json");
        std::fs::write(&path, b"first").unwrap();
        let source = Source::parse(Url::from_file_path(&path).unwrap().as_str()).unwrap();
        let fetch = source.fetcher(&config).unwrap();
        assert!(
            matches!(fetch.fetch(request.clone()).await, Reply::Fetched { body, validator: None } if body.as_ref() == b"first")
        );
        assert!(
            matches!(fetch.fetch(request.clone()).await, Reply::Fetched { body, validator: None } if body.as_ref() == b"first")
        );
        std::fs::write(&path, b"changed").unwrap();
        assert!(
            matches!(fetch.fetch(request).await, Reply::Fetched { body, validator: None } if body.as_ref() == b"changed")
        );
    }

    #[rstest]
    #[case::missing(false)]
    #[case::unreadable_directory(true)]
    #[tokio::test]
    async fn unreadable_files_preserve_os_error(request: FetchRequest, #[case] directory: bool) {
        let root = tempfile::tempdir().unwrap();
        let path = if directory {
            root.path().to_owned()
        } else {
            root.path().join("missing.json")
        };
        let expected = std::fs::read(&path).unwrap_err().to_string();
        assert!(
            matches!(FileSource::new(path).fetch(request).await, Reply::Failed { reason } if reason == expected)
        );
    }
}
