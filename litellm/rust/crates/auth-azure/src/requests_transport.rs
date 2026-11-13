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
        let config = client_config(&environment, request.url())
            .map_err(|missing| azure_core::Error::new(ErrorKind::Other, missing))?;
        let client = self
            .pool
            .client(&config, ClientVariant::NoRedirect)
            .map_err(|error| azure_core::Error::new(ErrorKind::Other, error))?;
        send(&client, request).await
    }
}

fn client_config(
    environment: &Environment,
    url: &Url,
) -> Result<HttpClientConfig, MissingCaBundle> {
    let proxy = requests_proxy(environment, url);
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
        azure_core::Error::new(kind, error)
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

    use super::{RequestsTransport, requests_proxy, verify};
    use crate::test_support::{Pki, SERVED_TOKEN, TunnelProxy};

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
}
