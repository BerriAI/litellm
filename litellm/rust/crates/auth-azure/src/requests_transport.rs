use std::collections::BTreeMap;
use std::net::Ipv4Addr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use azure_core::error::ErrorKind;
use azure_core::http::headers::{HeaderName, HeaderValue, Headers};
use azure_core::http::{AsyncRawResponse, Body, HttpClient, Request, Url};
use futures_util::TryStreamExt;
use litellm_http::media::PublicDnsResolver;
use litellm_http::{ClientVariant, EnvironmentProxies, HttpClientConfig, HttpClientPool, Verify};

/// azure-core's `ConnectionConfiguration` defaults, which azure-identity's `requests` transport keeps.
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(300);
const READ_TIMEOUT: Duration = Duration::from_secs(300);
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(90);
const CA_BUNDLE_VARIABLES: [&str; 2] = ["REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"];

type Environment = [(String, String)];

/// What `requests` raises when the CA bundle its environment names does not exist.
#[derive(Debug, thiserror::Error)]
#[error("Could not find a suitable TLS CA certificate bundle, invalid path: {0}")]
pub struct MissingCaBundle(pub String);

/// A token request that failed after `requests` would have opened its connection, directly or
/// through the proxy it picked.
#[derive(Debug)]
pub struct RequestsSendFailure {
    proxy: Option<(String, u16)>,
    source: reqwest::Error,
}

impl RequestsSendFailure {
    pub fn proxy(&self) -> Option<(&str, u16)> {
        self.proxy
            .as_ref()
            .map(|(host, port)| (host.as_str(), *port))
    }

    pub fn error(&self) -> &reqwest::Error {
        &self.source
    }
}

/// How a token request failed, where `requests` reports it the same way every time.
#[derive(Debug, PartialEq, Eq)]
pub enum RequestsReason {
    ProxyUnreachable {
        host: String,
        port: u16,
        errno: i32,
    },
    ProxyUnresolved {
        host: String,
        port: u16,
        lookup: String,
    },
    ProxyAuthenticationRequired,
    ProxyDisconnected,
    ConnectionAborted {
        errno: Option<i32>,
    },
    TlsEof,
    CertificateRejected(CertificateRejection),
    ReadTimeout,
}

/// The certificate failures OpenSSL and rustls tell apart alike. An unknown issuer is not one:
/// OpenSSL names a self-signed leaf, a self-signed root in the chain or a missing local issuer.
#[derive(Debug, PartialEq, Eq)]
pub enum CertificateRejection {
    NameMismatch,
    Expired,
    NotYetValid,
}

const TUNNEL_CONNECT_FAILED: &str = "tunnel error: failed to create underlying connection";
const TUNNEL_PROXY_AUTH: &str = "tunnel error: proxy authorization required";
const TUNNEL_EOF: &str = "tunnel error: unexpected end of file";
const TUNNEL_IO: &str = "tunnel error: io error establishing tunnel";
const INCOMPLETE_MESSAGE: &str = "connection closed before message completed";
const HANDSHAKE_EOF: &str = "tls handshake eof";
const LOOKUP_PREFIX: &str = "failed to lookup address information: ";
const ECONNRESET: i32 = 104;

impl RequestsSendFailure {
    /// The reason for this failure as `requests` would report it, or `None` when what the native
    /// request observed does not decide it. A direct connect or name lookup failure is `None`.
    pub fn reason(&self) -> Option<RequestsReason> {
        let causes = causes(&self.source);
        let says = |text: &str| causes.iter().any(|cause| cause.to_string() == text);
        let io = || {
            causes
                .iter()
                .filter_map(|cause| cause.downcast_ref::<std::io::Error>())
        };
        let errno = io().find_map(std::io::Error::raw_os_error);
        let reset = errno == Some(ECONNRESET)
            || io().any(|cause| cause.kind() == std::io::ErrorKind::ConnectionReset);
        let eof = says(HANDSHAKE_EOF)
            || io().any(|cause| cause.kind() == std::io::ErrorKind::UnexpectedEof);
        if self.source.is_timeout() && !self.source.is_connect() {
            return Some(RequestsReason::ReadTimeout);
        }
        if !self.source.is_connect() {
            return (eof || says(INCOMPLETE_MESSAGE)).then_some(match self.proxy {
                Some(_) => RequestsReason::ProxyDisconnected,
                None => RequestsReason::ConnectionAborted { errno: None },
            });
        }
        if let Some((host, port)) = &self.proxy {
            if says(TUNNEL_CONNECT_FAILED) {
                return proxy_unreachable(host, *port, errno, &causes);
            }
            if says(TUNNEL_PROXY_AUTH) {
                return Some(RequestsReason::ProxyAuthenticationRequired);
            }
            if says(TUNNEL_EOF) {
                return Some(RequestsReason::ConnectionAborted { errno: None });
            }
            if says(TUNNEL_IO) {
                return (reset && errno.is_some())
                    .then_some(RequestsReason::ConnectionAborted { errno });
            }
        }
        if let Some(rejection) = causes
            .iter()
            .find_map(|cause| certificate_rejection(cause.downcast_ref::<rustls::Error>()?))
        {
            return Some(RequestsReason::CertificateRejected(rejection));
        }
        if eof {
            return Some(RequestsReason::TlsEof);
        }
        (reset && errno.is_some()).then_some(RequestsReason::ConnectionAborted { errno })
    }
}

fn proxy_unreachable(
    host: &str,
    port: u16,
    errno: Option<i32>,
    causes: &[&(dyn std::error::Error + 'static)],
) -> Option<RequestsReason> {
    if let Some(errno) = errno {
        return Some(RequestsReason::ProxyUnreachable {
            host: host.to_owned(),
            port,
            errno,
        });
    }
    let lookup = causes
        .iter()
        .find_map(|cause| Some(cause.to_string().strip_prefix(LOOKUP_PREFIX)?.to_owned()))?;
    Some(RequestsReason::ProxyUnresolved {
        host: host.to_owned(),
        port,
        lookup,
    })
}

fn certificate_rejection(error: &rustls::Error) -> Option<CertificateRejection> {
    let rustls::Error::InvalidCertificate(certificate) = error else {
        return None;
    };
    match certificate {
        rustls::CertificateError::NotValidForName
        | rustls::CertificateError::NotValidForNameContext { .. } => {
            Some(CertificateRejection::NameMismatch)
        }
        rustls::CertificateError::Expired | rustls::CertificateError::ExpiredContext { .. } => {
            Some(CertificateRejection::Expired)
        }
        rustls::CertificateError::NotValidYet
        | rustls::CertificateError::NotValidYetContext { .. } => {
            Some(CertificateRejection::NotYetValid)
        }
        _ => None,
    }
}

/// Every cause of `error`, including the error an `io::Error` wraps, which its own `source` skips.
fn causes<'a>(
    error: &'a (dyn std::error::Error + 'static),
) -> Vec<&'a (dyn std::error::Error + 'static)> {
    std::iter::successors(Some(error), |&cause| next_cause(cause)).collect()
}

fn next_cause<'a>(
    cause: &'a (dyn std::error::Error + 'static),
) -> Option<&'a (dyn std::error::Error + 'static)> {
    match cause
        .downcast_ref::<std::io::Error>()
        .and_then(std::io::Error::get_ref)
    {
        Some(inner) => Some(inner as &(dyn std::error::Error + 'static)),
        None => cause.source(),
    }
}

impl std::fmt::Display for RequestsSendFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.source.fmt(formatter)
    }
}

impl std::error::Error for RequestsSendFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.source()
    }
}

/// The HTTP stack Python's azure-identity sends a token request through: `requests` over urllib3,
/// reading its proxy and CA bundle from the process environment on every request.
pub(crate) struct RequestsTransport {
    pool: HttpClientPool,
    environment: Arc<dyn Fn() -> Vec<(String, String)> + Send + Sync>,
}

impl std::fmt::Debug for RequestsTransport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("RequestsTransport")
    }
}

impl RequestsTransport {
    pub(crate) fn new(environment: Arc<dyn Fn() -> Vec<(String, String)> + Send + Sync>) -> Self {
        Self {
            pool: HttpClientPool::new(Arc::new(PublicDnsResolver)),
            environment,
        }
    }

    pub(crate) fn process() -> Self {
        Self::new(Arc::new(|| {
            std::env::vars_os()
                .filter_map(|(name, value)| {
                    Some((name.into_string().ok()?, value.into_string().ok()?))
                })
                .collect()
        }))
    }
}

#[async_trait::async_trait]
impl HttpClient for RequestsTransport {
    async fn execute_request(&self, request: &Request) -> azure_core::Result<AsyncRawResponse> {
        let environment = (self.environment)();
        let proxy = requests_proxy(&environment, request.url());
        let config = client_config(&environment, request.url(), proxy.clone())
            .map_err(|missing| azure_core::Error::new(ErrorKind::Other, missing))?;
        let client = self
            .pool
            .client(&config, ClientVariant::NoRedirect)
            .map_err(|error| azure_core::Error::new(ErrorKind::Other, error))?;
        let proxy = proxy
            .and_then(|proxy| {
                Url::parse(&proxy)
                    .or_else(|_| Url::parse(&format!("http://{proxy}")))
                    .ok()
            })
            .and_then(|proxy| {
                let host = proxy
                    .host_str()?
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .to_owned();
                Some((host, proxy.port_or_known_default()?))
            });
        send(&client, request, proxy).await
    }
}

fn client_config(
    environment: &Environment,
    url: &Url,
    proxy: Option<String>,
) -> Result<HttpClientConfig, MissingCaBundle> {
    let variable = format!("{}_proxy", url.scheme());
    Ok(HttpClientConfig {
        verify: verify(environment)?,
        client_certificate: None,
        key_exchange_group: None,
        tls12_cipher_suites: None,
        force_ipv4: false,
        http2: false,
        user_agent: None,
        proxies: EnvironmentProxies::from_environment(&|name: &str| {
            (name == variable).then(|| proxy.clone()).flatten()
        }),
        connect_timeout: CONNECTION_TIMEOUT,
        tcp_keepalive: None,
        pool_idle_timeout: POOL_IDLE_TIMEOUT,
    })
}

/// `requests`' `merge_environment_settings` and `HTTPAdapter.cert_verify`: the first non-empty CA
/// bundle variable, else certifi's Mozilla roots. `SSL_CERT_FILE` is never read.
fn verify(environment: &Environment) -> Result<Verify, MissingCaBundle> {
    let bundle = CA_BUNDLE_VARIABLES
        .iter()
        .find_map(|name| variable(environment, name).filter(|value| !value.is_empty()));
    match bundle {
        None => Ok(Verify::BuiltInRoots),
        Some(path) if Path::new(path).exists() => Ok(Verify::CaBundle(path.into())),
        Some(path) => Err(MissingCaBundle(path.to_owned())),
    }
}

/// The proxy `requests` sends `url` through: urllib's environment proxies for its scheme, or `all`,
/// unless `requests`' `should_bypass_proxies` skips them for the host.
fn requests_proxy(environment: &Environment, url: &Url) -> Option<String> {
    if bypasses_proxies(environment, url) {
        return None;
    }
    let proxies = environment_proxies(environment);
    proxies
        .get(url.scheme())
        .or_else(|| proxies.get("all"))
        .cloned()
}

fn variable<'a>(environment: &'a Environment, name: &str) -> Option<&'a str> {
    environment
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.as_str())
}

/// urllib's `getproxies_environment`: every `*_proxy` variable in any case, then the lowercase
/// spellings again, where an empty one removes the scheme. A CGI request drops `HTTP_PROXY`.
fn environment_proxies(environment: &Environment) -> BTreeMap<String, String> {
    let any_case: BTreeMap<String, String> = environment
        .iter()
        .filter(|(_, value)| !value.is_empty())
        .filter_map(|(name, value)| {
            let scheme = name.to_lowercase().strip_suffix("_proxy")?.to_owned();
            Some((scheme, value.clone()))
        })
        .filter(|(scheme, _)| scheme != "http" || variable(environment, "REQUEST_METHOD").is_none())
        .collect();
    environment
        .iter()
        .filter_map(|(name, value)| Some((name.strip_suffix("_proxy")?.to_lowercase(), value)))
        .fold(any_case, |proxies, (scheme, value)| {
            proxies
                .into_iter()
                .filter(|(kept, _)| *kept != scheme)
                .chain((!value.is_empty()).then(|| (scheme.clone(), value.clone())))
                .collect()
        })
}

fn bypasses_proxies(environment: &Environment, url: &Url) -> bool {
    let Some(hostname) = url.host_str().map(|host| {
        host.trim_start_matches('[')
            .trim_end_matches(']')
            .to_lowercase()
    }) else {
        return true;
    };
    let no_proxy = variable(environment, "no_proxy")
        .filter(|value| !value.is_empty())
        .or_else(|| variable(environment, "NO_PROXY"))
        .unwrap_or_default()
        .replace(' ', "");
    let entries = || no_proxy.split(',').filter(|entry| !entry.is_empty());
    let listed = match hostname.parse::<Ipv4Addr>() {
        Ok(address) => entries().any(|entry| match cidr(entry) {
            Some((network, mask)) => address.to_bits() & mask == network.to_bits() & mask,
            None => entry == hostname,
        }),
        Err(_) => {
            let with_port = url
                .port()
                .map_or_else(|| hostname.clone(), |port| format!("{hostname}:{port}"));
            entries()
                .map(|entry| entry.trim_start_matches('.'))
                .any(|entry| {
                    let dotted = format!(".{entry}");
                    hostname == entry
                        || with_port == entry
                        || hostname.ends_with(&dotted)
                        || with_port.ends_with(&dotted)
                })
        }
    };
    listed || bypasses_environment(&environment_proxies(environment), &hostname)
}

/// `requests`' `is_valid_cidr` and `dotted_netmask`.
fn cidr(entry: &str) -> Option<(Ipv4Addr, u32)> {
    let (address, bits) = entry.split_once('/')?;
    let bits = bits
        .parse::<u32>()
        .ok()
        .filter(|bits| (1..=32).contains(bits))?;
    Some((address.parse().ok()?, u32::MAX << (32 - bits)))
}

/// urllib's `proxy_bypass_environment`, over the `no` entry of its environment proxies.
fn bypasses_environment(proxies: &BTreeMap<String, String>, host: &str) -> bool {
    let Some(no_proxy) = proxies.get("no") else {
        return false;
    };
    if no_proxy == "*" {
        return true;
    }
    let host_only = split_port(host);
    no_proxy
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(|name| name.trim_start_matches('.').to_lowercase())
        .any(|name| {
            let dotted = format!(".{name}");
            host_only == name
                || host == name
                || host_only.ends_with(&dotted)
                || host.ends_with(&dotted)
        })
}

/// urllib's `_splitport`, which reads the text after the last colon as a port when it is all digits.
fn split_port(host: &str) -> &str {
    host.rsplit_once(':')
        .filter(|(_, port)| port.bytes().all(|byte| byte.is_ascii_digit()))
        .map_or(host, |(host_only, _)| host_only)
}

async fn send(
    client: &litellm_http::Client,
    request: &Request,
    proxy: Option<(String, u16)>,
) -> azure_core::Result<AsyncRawResponse> {
    let builder = request.headers().iter().fold(
        client
            .request(
                reqwest::Method::from_bytes(request.method().as_str().as_bytes())
                    .map_err(|error| azure_core::Error::new(ErrorKind::Other, error))?,
                request.url().clone(),
            )
            .timeout(READ_TIMEOUT),
        |builder, (name, value)| builder.header(name.as_str(), value.as_str()),
    );
    let built = match request.body().clone() {
        Body::Bytes(bytes) => builder.body(bytes).build(),
        Body::SeekableStream(stream) => builder.body(reqwest::Body::wrap_stream(stream)).build(),
    }
    .map_err(|error| azure_core::Error::new(ErrorKind::Other, error))?;
    let response = client.execute(built).await.map_err(|error| {
        let kind = if error.is_connect() {
            ErrorKind::Connection
        } else {
            ErrorKind::Io
        };
        azure_core::Error::new(
            kind,
            RequestsSendFailure {
                proxy,
                source: error,
            },
        )
    })?;
    let status = response.status().as_u16();
    let headers = Headers::from(
        response
            .headers()
            .iter()
            .filter_map(|(name, value)| {
                Some((
                    HeaderName::from(name.as_str().to_owned()),
                    HeaderValue::from(value.to_str().ok()?.to_owned()),
                ))
            })
            .collect::<std::collections::HashMap<_, _>>(),
    );
    let body = response
        .bytes_stream()
        .map_err(|error| azure_core::Error::new(ErrorKind::Io, error));
    Ok(AsyncRawResponse::new(
        status.into(),
        headers,
        Box::pin(body),
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use azure_core::http::{HttpClient, Method, Request, Url};
    use litellm_http::Verify;
    use rstest::rstest;

    use super::{
        CertificateRejection, RequestsReason, RequestsSendFailure, RequestsTransport,
        requests_proxy, verify,
    };
    use crate::test_support::{Endpoint, Pki, Proxy, SERVED_TOKEN, TunnelProxy};

    const TOKEN_URL: &str = "https://login.microsoftonline.com/tenant/oauth2/v2.0/token";

    fn environment(variables: &[(&str, &str)]) -> Vec<(String, String)> {
        variables
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    #[rstest]
    #[case::uppercase(&[("HTTPS_PROXY", "http://upper:3128")], TOKEN_URL, Some("http://upper:3128"))]
    #[case::lowercase_wins(&[("HTTPS_PROXY", "http://upper:3128"), ("https_proxy", "http://lower:3128")], TOKEN_URL, Some("http://lower:3128"))]
    #[case::lowercase_wins_listed_first(&[("https_proxy", "http://lower:3128"), ("HTTPS_PROXY", "http://upper:3128")], TOKEN_URL, Some("http://lower:3128"))]
    #[case::empty_lowercase_clears_uppercase(&[("HTTPS_PROXY", "http://upper:3128"), ("https_proxy", "")], TOKEN_URL, None)]
    #[case::any_case_spelling(&[("Https_Proxy", "http://mixed:3128")], TOKEN_URL, Some("http://mixed:3128"))]
    #[case::all_proxy_fallback(&[("ALL_PROXY", "http://all:3128")], TOKEN_URL, Some("http://all:3128"))]
    #[case::https_proxy_before_all(&[("ALL_PROXY", "http://all:3128"), ("HTTPS_PROXY", "http://https:3128")], TOKEN_URL, Some("http://https:3128"))]
    #[case::http_proxy_is_not_for_https(&[("HTTP_PROXY", "http://http:3128")], TOKEN_URL, None)]
    #[case::no_proxy_host(&[("HTTPS_PROXY", "http://p:3128"), ("NO_PROXY", "login.microsoftonline.com")], TOKEN_URL, None)]
    #[case::no_proxy_dotted_parent(&[("HTTPS_PROXY", "http://p:3128"), ("NO_PROXY", ".microsoftonline.com")], TOKEN_URL, None)]
    #[case::no_proxy_parent(&[("HTTPS_PROXY", "http://p:3128"), ("NO_PROXY", "microsoftonline.com")], TOKEN_URL, None)]
    #[case::no_proxy_bare_suffix(&[("HTTPS_PROXY", "http://p:3128"), ("NO_PROXY", "soft.com")], TOKEN_URL, Some("http://p:3128"))]
    #[case::no_proxy_spaces(&[("HTTPS_PROXY", "http://p:3128"), ("NO_PROXY", "example.com, login.microsoftonline.com")], TOKEN_URL, None)]
    #[case::no_proxy_star(&[("HTTPS_PROXY", "http://p:3128"), ("NO_PROXY", "*")], TOKEN_URL, None)]
    #[case::no_proxy_star_in_a_list(&[("HTTPS_PROXY", "http://p:3128"), ("NO_PROXY", "example.com,*")], TOKEN_URL, Some("http://p:3128"))]
    #[case::no_proxy_upper_case_entry(&[("HTTPS_PROXY", "http://p:3128"), ("NO_PROXY", "LOGIN.MicrosoftOnline.com")], TOKEN_URL, None)]
    #[case::cgi_keeps_https(&[("REQUEST_METHOD", "GET"), ("HTTPS_PROXY", "http://p:3128")], TOKEN_URL, Some("http://p:3128"))]
    #[case::no_proxy_other_host(&[("HTTPS_PROXY", "http://p:3128"), ("NO_PROXY", "example.com")], TOKEN_URL, Some("http://p:3128"))]
    #[case::lowercase_no_proxy_wins(&[("HTTPS_PROXY", "http://p:3128"), ("no_proxy", "example.com"), ("NO_PROXY", "login.microsoftonline.com")], TOKEN_URL, Some("http://p:3128"))]
    #[case::empty_lowercase_no_proxy_reads_uppercase(&[("HTTPS_PROXY", "http://p:3128"), ("no_proxy", ""), ("NO_PROXY", "login.microsoftonline.com")], TOKEN_URL, None)]
    #[case::no_proxy_host_and_port(&[("HTTPS_PROXY", "http://p:3128"), ("NO_PROXY", "microsoftonline.com:8443")], "https://login.microsoftonline.com:8443/t", None)]
    #[case::no_proxy_cidr(&[("HTTPS_PROXY", "http://p:3128"), ("NO_PROXY", "10.0.0.0/8")], "https://10.1.2.3/t", None)]
    #[case::no_proxy_other_cidr(&[("HTTPS_PROXY", "http://p:3128"), ("NO_PROXY", "192.168.0.0/16")], "https://10.1.2.3/t", Some("http://p:3128"))]
    #[case::no_proxy_address(&[("HTTPS_PROXY", "http://p:3128"), ("NO_PROXY", "10.1.2.3")], "https://10.1.2.3/t", None)]
    fn the_proxy_is_the_one_requests_picks(
        #[case] variables: &[(&str, &str)],
        #[case] url: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(
            requests_proxy(&environment(variables), &Url::parse(url).unwrap()).as_deref(),
            expected
        );
    }

    const PRESENT: &str = env!("CARGO_MANIFEST_DIR");

    // The missing bundle text is requests 2.34.0 `HTTPAdapter.cert_verify`'s, checked 2026-11-14
    #[rstest]
    #[case::certifi_roots(&[], Ok(Verify::BuiltInRoots))]
    #[case::requests_bundle(&[("REQUESTS_CA_BUNDLE", PRESENT)], Ok(Verify::CaBundle(PRESENT.into())))]
    #[case::curl_bundle(&[("CURL_CA_BUNDLE", PRESENT)], Ok(Verify::CaBundle(PRESENT.into())))]
    #[case::requests_bundle_before_curl(&[("CURL_CA_BUNDLE", "/absent/curl.pem"), ("REQUESTS_CA_BUNDLE", PRESENT)], Ok(Verify::CaBundle(PRESENT.into())))]
    #[case::empty_requests_bundle_reads_curl(&[("REQUESTS_CA_BUNDLE", ""), ("CURL_CA_BUNDLE", PRESENT)], Ok(Verify::CaBundle(PRESENT.into())))]
    #[case::lowercase_is_not_read(&[("requests_ca_bundle", PRESENT)], Ok(Verify::BuiltInRoots))]
    #[case::ssl_cert_file_is_not_read(&[("SSL_CERT_FILE", PRESENT), ("SSL_CERT_DIR", PRESENT)], Ok(Verify::BuiltInRoots))]
    #[case::missing_bundle(&[("REQUESTS_CA_BUNDLE", "/absent/ca.pem")], Err("Could not find a suitable TLS CA certificate bundle, invalid path: /absent/ca.pem"))]
    fn the_trusted_roots_are_the_ones_requests_verifies_with(
        #[case] variables: &[(&str, &str)],
        #[case] expected: Result<Verify, &str>,
    ) {
        assert_eq!(
            verify(&environment(variables)).map_err(|error| error.to_string()),
            expected.map_err(str::to_owned)
        );
    }

    async fn post(transport: &RequestsTransport, url: &str) -> Result<String, azure_core::Error> {
        let mut request = Request::new(Url::parse(url).unwrap(), Method::Post);
        request.set_body("grant_type=client_credentials");
        let response = transport.execute_request(&request).await?;
        let (_, _, body) = response.deconstruct();
        Ok(String::from_utf8(body.collect().await?.to_vec()).unwrap())
    }

    fn transport(variables: Vec<(String, String)>) -> RequestsTransport {
        RequestsTransport::new(Arc::new(move || variables.clone()))
    }

    #[rstest]
    #[case::requests_bundle("REQUESTS_CA_BUNDLE", true)]
    #[case::curl_bundle("CURL_CA_BUNDLE", true)]
    #[case::ssl_cert_file("SSL_CERT_FILE", false)]
    #[tokio::test]
    async fn the_token_endpoint_is_trusted_through_the_bundle_requests_reads(
        #[case] variable: &str,
        #[case] trusted: bool,
    ) {
        let pki = Pki::new();
        let port = pki.token_endpoint().await;
        let transport = transport(vec![(variable.to_owned(), pki.ca.display().to_string())]);

        let answer = post(
            &transport,
            &format!("https://127.0.0.1:{port}/tenant/oauth2/v2.0/token"),
        )
        .await;

        assert_eq!(
            answer.is_ok_and(|body| body.contains(SERVED_TOKEN)),
            trusted
        );
    }

    #[rstest]
    #[case::https_proxy(&[("HTTPS_PROXY", "{proxy}")], true)]
    #[case::lowercase_proxy(&[("https_proxy", "{proxy}")], true)]
    #[case::bypassed_address(&[("HTTPS_PROXY", "{proxy}"), ("NO_PROXY", "127.0.0.1")], false)]
    #[case::empty_lowercase_clears_uppercase(&[("HTTPS_PROXY", "{proxy}"), ("https_proxy", "")], false)]
    #[case::http_proxy_only(&[("HTTP_PROXY", "{proxy}")], false)]
    #[tokio::test]
    async fn the_token_request_goes_through_the_proxy_requests_picks(
        #[case] variables: &[(&str, &str)],
        #[case] proxied: bool,
    ) {
        let pki = Pki::new();
        let port = pki.token_endpoint().await;
        let proxy = TunnelProxy::start(port).await;
        let proxy_url = format!("http://127.0.0.1:{}", proxy.port);
        let transport = transport(
            variables
                .iter()
                .map(|(name, value)| (name.to_string(), value.replace("{proxy}", &proxy_url)))
                .chain([(
                    "REQUESTS_CA_BUNDLE".to_owned(),
                    pki.ca.display().to_string(),
                )])
                .collect(),
        );

        let body = post(
            &transport,
            &format!("https://127.0.0.1:{port}/tenant/oauth2/v2.0/token"),
        )
        .await
        .unwrap();

        assert!(body.contains(SERVED_TOKEN), "{body}");
        assert_eq!(
            proxy.targets(),
            if proxied {
                vec![format!("127.0.0.1:{port}")]
            } else {
                Vec::new()
            }
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum Route {
        Direct,
        Untrusted,
        Named(&'static str),
        Expired,
        Tunnel,
        Refused,
        ProxyRefused,
        Proxy(Proxy),
    }

    async fn failed(endpoint: Endpoint, route: Route) -> Option<RequestsReason> {
        let pki = match route {
            Route::Expired => Pki::expired(),
            _ => Pki::new(),
        };
        let port = pki.endpoint(endpoint).await;
        let closed = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let proxy = match route {
            Route::Tunnel => Some(TunnelProxy::start(port).await.port),
            Route::Proxy(behaviour) => Some(TunnelProxy::answering(behaviour).await.port),
            Route::ProxyRefused => Some(closed),
            _ => None,
        };
        let (host, port) = match route {
            Route::Named(host) => (host, port),
            Route::Refused => ("127.0.0.1", closed),
            _ => ("127.0.0.1", port),
        };
        let bundle = match route {
            Route::Untrusted => Vec::new(),
            _ => vec![(
                "REQUESTS_CA_BUNDLE".to_owned(),
                pki.ca.display().to_string(),
            )],
        };
        let variables = bundle
            .into_iter()
            .chain(proxy.map(|proxy| {
                (
                    "HTTPS_PROXY".to_owned(),
                    format!("http://127.0.0.1:{proxy}"),
                )
            }))
            .collect();

        let error = post(
            &transport(variables),
            &format!("https://{host}:{port}/token"),
        )
        .await
        .unwrap_err();

        litellm_error::find::<RequestsSendFailure>(&error)
            .unwrap_or_else(|| panic!("{error:?}"))
            .reason()
    }

    #[rstest]
    #[case::direct_connect_is_worded_elsewhere(Endpoint::Token, Route::Refused, None)]
    #[case::reset_in_the_handshake(Endpoint::ResetBeforeTls, Route::Direct, Some(RequestsReason::ConnectionAborted { errno: Some(104) }))]
    #[case::closed_in_the_handshake(
        Endpoint::CloseBeforeTls,
        Route::Direct,
        Some(RequestsReason::TlsEof)
    )]
    #[case::closed_before_the_answer(Endpoint::CloseAfterRequest, Route::Direct, Some(RequestsReason::ConnectionAborted { errno: None }))]
    #[case::reset_before_the_answer(Endpoint::ResetAfterRequest, Route::Direct, None)]
    #[case::unknown_issuer(Endpoint::Token, Route::Untrusted, None)]
    #[case::name_mismatch(
        Endpoint::Token,
        Route::Named("localhost"),
        Some(RequestsReason::CertificateRejected(CertificateRejection::NameMismatch))
    )]
    #[case::expired(
        Endpoint::Token,
        Route::Expired,
        Some(RequestsReason::CertificateRejected(CertificateRejection::Expired))
    )]
    #[case::proxy_refused(Endpoint::Token, Route::ProxyRefused, Some(RequestsReason::ProxyUnreachable { host: "127.0.0.1".to_owned(), port: 0, errno: 111 }))]
    #[case::proxy_asks_for_credentials(
        Endpoint::Token,
        Route::Proxy(Proxy::Status("407 Proxy Authentication Required")),
        Some(RequestsReason::ProxyAuthenticationRequired)
    )]
    #[case::proxy_forbids(Endpoint::Token, Route::Proxy(Proxy::Status("403 Forbidden")), None)]
    #[case::proxy_closes(Endpoint::Token, Route::Proxy(Proxy::Close), Some(RequestsReason::ConnectionAborted { errno: None }))]
    #[case::proxy_resets(Endpoint::Token, Route::Proxy(Proxy::Reset), Some(RequestsReason::ConnectionAborted { errno: Some(104) }))]
    #[case::tunnel_closed_in_the_handshake(
        Endpoint::CloseBeforeTls,
        Route::Tunnel,
        Some(RequestsReason::TlsEof)
    )]
    #[case::tunnel_closed_before_the_answer(
        Endpoint::CloseAfterRequest,
        Route::Tunnel,
        Some(RequestsReason::ProxyDisconnected)
    )]
    #[case::tunnel_reset_before_the_answer(
        Endpoint::ResetAfterRequest,
        Route::Tunnel,
        Some(RequestsReason::ProxyDisconnected)
    )]
    #[tokio::test]
    async fn a_failed_token_request_is_read_as_requests_reports_it(
        #[case] endpoint: Endpoint,
        #[case] route: Route,
        #[case] expected: Option<RequestsReason>,
    ) {
        let reason = failed(endpoint, route).await;

        let ignoring_proxy_port = match reason {
            Some(RequestsReason::ProxyUnreachable { host, errno, .. }) => {
                Some(RequestsReason::ProxyUnreachable {
                    host,
                    port: 0,
                    errno,
                })
            }
            other => other,
        };
        assert_eq!(ignoring_proxy_port, expected);
    }

    #[rstest]
    #[case::direct(Route::Direct)]
    #[case::tunnel(Route::Tunnel)]
    #[tokio::test(start_paused = true)]
    async fn an_unanswered_token_request_is_a_read_timeout(#[case] route: Route) {
        assert_eq!(
            failed(Endpoint::Silent, route).await,
            Some(RequestsReason::ReadTimeout)
        );
    }
}
