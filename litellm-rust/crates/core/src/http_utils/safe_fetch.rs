//! Bounded downloads of user-controlled public URLs.
//!
//! This module owns transport policy only: SSRF protection, redirect handling,
//! timeouts, connection reuse, and response-size limits. Route code decides
//! whether the bytes become a data URI, multipart part, provider upload, or
//! something else.

use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::{StatusCode, Url, header::HeaderMap};

use crate::constants::{
    SAFE_FETCH_CONNECT_TIMEOUT_SECS, SAFE_FETCH_MAX_REDIRECTS, SAFE_FETCH_TIMEOUT_SECS,
};
use crate::error::{CoreError, CoreResult};

/// Required policy for one public-URL fetch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SafeFetchOptions {
    max_response_bytes: u64,
    timeout: Duration,
    max_redirects: usize,
}

impl SafeFetchOptions {
    /// Construct a fetch policy with an explicit route-appropriate byte limit.
    pub const fn new(max_response_bytes: u64) -> Self {
        Self {
            max_response_bytes,
            timeout: Duration::from_secs(SAFE_FETCH_TIMEOUT_SECS),
            max_redirects: SAFE_FETCH_MAX_REDIRECTS,
        }
    }

    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub const fn with_max_redirects(mut self, max_redirects: usize) -> Self {
        self.max_redirects = max_redirects;
        self
    }
}

/// Fully buffered, bounded response from a validated public URL.
#[derive(Debug)]
pub struct SafeFetchResponse {
    pub final_url: Url,
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

#[derive(Debug)]
struct PublicIpResolver;

impl Resolve for PublicIpResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_string();
        Box::pin(async move {
            let addresses = tokio::net::lookup_host((host.as_str(), 0))
                .await
                .map_err(boxed_io_error)?
                .collect::<Vec<_>>();
            if addresses.is_empty() {
                return Err(boxed_io_error(io::Error::new(
                    io::ErrorKind::NotFound,
                    "host resolved to no addresses",
                )));
            }
            if addresses.iter().any(|address| !is_public_ip(address.ip())) {
                return Err(boxed_io_error(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "host resolved to a non-public address",
                )));
            }
            Ok(Box::new(addresses.into_iter()) as Addrs)
        })
    }
}

fn boxed_io_error(error: io::Error) -> Box<dyn std::error::Error + Send + Sync> {
    Box::new(error)
}

fn safe_fetch_client() -> CoreResult<&'static reqwest::Client> {
    static CLIENT: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .dns_resolver(Arc::new(PublicIpResolver))
                // A proxy could resolve the target itself and bypass our resolver.
                .no_proxy()
                .connect_timeout(Duration::from_secs(SAFE_FETCH_CONNECT_TIMEOUT_SECS))
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|error| CoreError::Connect(error.clone()))
}

fn is_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => is_public_ipv4(ip),
        IpAddr::V6(ip) => ip
            .to_ipv4_mapped()
            .or_else(|| ip.to_ipv4())
            .map(is_public_ipv4)
            .unwrap_or_else(|| is_public_ipv6(ip)),
    }
}

fn is_public_ipv4(ip: Ipv4Addr) -> bool {
    let [a, b, c, d] = ip.octets();
    let is_shared = a == 100 && b & 0b1100_0000 == 0b0100_0000;
    let is_protocol_assignment = a == 192 && b == 0 && c == 0 && d != 9 && d != 10;
    let is_documentation = (a == 192 && b == 0 && c == 2)
        || (a == 198 && b == 51 && c == 100)
        || (a == 203 && b == 0 && c == 113);
    let is_benchmarking = a == 198 && matches!(b, 18 | 19);
    let is_deprecated_relay = a == 192 && b == 88 && c == 99;
    let is_cloud_metadata = [a, b, c, d] == [168, 63, 129, 16];

    !(a == 0
        || ip.is_private()
        || is_shared
        || ip.is_loopback()
        || ip.is_link_local()
        || is_protocol_assignment
        || is_documentation
        || is_benchmarking
        || is_deprecated_relay
        || a >= 224
        || is_cloud_metadata)
}

fn is_public_ipv6(ip: Ipv6Addr) -> bool {
    let segments = ip.segments();
    let value = u128::from_be_bytes(ip.octets());
    let is_ietf_assignment = segments[0] == 0x2001
        && segments[1] < 0x0200
        && !(value == 0x2001_0001_0000_0000_0000_0000_0000_0001
            || value == 0x2001_0001_0000_0000_0000_0000_0000_0002
            || segments[1] == 0x0003
            || (segments[1] == 0x0004 && segments[2] == 0x0112)
            || (0x0020..=0x003f).contains(&segments[1]));
    let is_documentation = (segments[0] == 0x2001 && segments[1] == 0x0db8)
        || (segments[0] == 0x3fff && segments[1] <= 0x0fff);

    !(ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_multicast()
        || matches!(segments, [0x0064, 0xff9b, 0x0001, _, _, _, _, _])
        || matches!(segments, [0x0100, 0, 0, 0, _, _, _, _])
        || is_ietf_assignment
        || segments[0] == 0x2002
        || is_documentation
        || segments[0] == 0x5f00
        || (segments[0] & 0xfe00) == 0xfc00
        || (segments[0] & 0xffc0) == 0xfe80)
}

fn invalid_url(message: impl Into<String>) -> CoreError {
    CoreError::InvalidRequest(format!("unsafe remote URL: {}", message.into()))
}

fn validate_url(url: &Url) -> CoreResult<()> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err(invalid_url("only http and https are allowed"));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(invalid_url("embedded credentials are not allowed"));
    }
    let host = url
        .host_str()
        .ok_or_else(|| invalid_url("hostname is required"))?;
    let ip_literal = host
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host);
    if ip_literal
        .parse::<IpAddr>()
        .is_ok_and(|ip| !is_public_ip(ip))
    {
        return Err(invalid_url("target address is not public"));
    }
    Ok(())
}

fn is_followable_redirect(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::MOVED_PERMANENTLY
            | StatusCode::FOUND
            | StatusCode::SEE_OTHER
            | StatusCode::TEMPORARY_REDIRECT
            | StatusCode::PERMANENT_REDIRECT
    )
}

fn redirect_url(current_url: &Url, location: Option<&str>) -> CoreResult<Url> {
    let location = location.ok_or_else(|| {
        CoreError::InvalidResponse("safe fetch redirect missing Location header".to_string())
    })?;
    current_url.join(location).map_err(|error| {
        CoreError::InvalidResponse(format!("safe fetch returned invalid redirect: {error}"))
    })
}

fn enforce_size(size: u64, max_bytes: u64) -> CoreResult<()> {
    if max_bytes == 0 {
        return Err(CoreError::InvalidRequest(
            "remote URL fetching is disabled by a zero-byte limit".to_string(),
        ));
    }
    if size > max_bytes {
        return Err(CoreError::InvalidRequest(format!(
            "remote response exceeds the {max_bytes}-byte limit"
        )));
    }
    Ok(())
}

fn append_bounded_chunk(
    body: &mut Vec<u8>,
    downloaded: &mut u64,
    chunk: &[u8],
    max_bytes: u64,
) -> CoreResult<()> {
    let next_size = downloaded
        .checked_add(chunk.len() as u64)
        .ok_or_else(|| CoreError::InvalidRequest("remote response size overflowed".to_string()))?;
    enforce_size(next_size, max_bytes)?;
    body.extend_from_slice(chunk);
    *downloaded = next_size;
    Ok(())
}

async fn read_bounded_response(
    mut response: reqwest::Response,
    max_bytes: u64,
) -> CoreResult<(StatusCode, HeaderMap, Vec<u8>)> {
    if let Some(content_length) = response.content_length() {
        enforce_size(content_length, max_bytes)?;
    } else {
        enforce_size(0, max_bytes)?;
    }

    let capacity = response
        .content_length()
        .and_then(|length| usize::try_from(length).ok())
        .unwrap_or_default();
    let status = response.status();
    let headers = response.headers().clone();
    let mut body = Vec::with_capacity(capacity);
    let mut downloaded = 0_u64;
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| CoreError::Network(error.to_string()))?
    {
        append_bounded_chunk(&mut body, &mut downloaded, &chunk, max_bytes)?;
    }
    Ok((status, headers, body))
}

fn map_send_error(error: reqwest::Error) -> CoreError {
    if error.is_connect() || error.is_builder() {
        CoreError::Connect(error.to_string())
    } else {
        CoreError::Network(error.to_string())
    }
}

async fn fetch_inner(url: &str, options: SafeFetchOptions) -> CoreResult<SafeFetchResponse> {
    enforce_size(0, options.max_response_bytes)?;
    let client = safe_fetch_client()?;
    let mut current_url =
        Url::parse(url).map_err(|error| invalid_url(format!("could not be parsed: {error}")))?;
    let mut redirects = 0_usize;

    loop {
        validate_url(&current_url)?;
        let response = client
            .get(current_url.clone())
            .send()
            .await
            .map_err(map_send_error)?;
        if !is_followable_redirect(response.status()) {
            let (status, headers, body) =
                read_bounded_response(response, options.max_response_bytes).await?;
            return Ok(SafeFetchResponse {
                final_url: current_url,
                status,
                headers,
                body,
            });
        }
        if redirects == options.max_redirects {
            return Err(CoreError::InvalidRequest(
                "remote URL exceeded the redirect limit".to_string(),
            ));
        }
        let location = response
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|value| value.to_str().ok());
        current_url = redirect_url(&current_url, location)?;
        redirects += 1;
    }
}

/// Fetch a user-controlled public URL under an explicit bounded policy.
///
/// DNS answers used by the connector are validated, proxies are disabled, and
/// every redirect target is checked before it is requested. The returned body
/// can never exceed `options.max_response_bytes`.
pub async fn safe_fetch(url: &str, options: SafeFetchOptions) -> CoreResult<SafeFetchResponse> {
    tokio::time::timeout(options.timeout, fetch_inner(url, options))
        .await
        .map_err(|_| CoreError::Network("safe fetch timed out".to_string()))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_public_addresses_are_allowed() {
        for blocked in [
            "0.0.0.0",
            "10.0.0.1",
            "100.64.0.1",
            "127.0.0.1",
            "168.63.129.16",
            "169.254.169.254",
            "192.0.2.1",
            "198.18.0.1",
            "224.0.0.1",
            "::1",
            "fc00::1",
            "fe80::1",
            "2001:db8::1",
            "::ffff:169.254.169.254",
        ] {
            assert!(!is_public_ip(blocked.parse().unwrap()), "allowed {blocked}");
        }
        for public in [
            "8.8.8.8",
            "1.1.1.1",
            "2001:4860:4860::8888",
            "::ffff:8.8.8.8",
        ] {
            assert!(is_public_ip(public.parse().unwrap()), "blocked {public}");
        }
    }

    #[test]
    fn validates_scheme_credentials_and_ip_literals() {
        assert!(validate_url(&Url::parse("ftp://example.com/file").unwrap()).is_err());
        assert!(validate_url(&Url::parse("https://user:pass@example.com/file").unwrap()).is_err());
        assert!(validate_url(&Url::parse("http://127.0.0.1/file").unwrap()).is_err());
        assert!(validate_url(&Url::parse("http://[::1]/file").unwrap()).is_err());
        assert!(validate_url(&Url::parse("https://example.com/file").unwrap()).is_ok());
    }

    #[test]
    fn resolves_relative_redirects_and_rejects_missing_locations() {
        let current = Url::parse("https://example.com/media/start").unwrap();
        assert_eq!(
            redirect_url(&current, Some("../next")).unwrap().as_str(),
            "https://example.com/next"
        );
        assert!(redirect_url(&current, None).is_err());

        let loopback = redirect_url(&current, Some("http://127.0.0.1/private")).unwrap();
        assert!(validate_url(&loopback).is_err());
    }

    #[test]
    fn enforces_declared_and_streamed_size_limits() {
        assert!(enforce_size(10, 10).is_ok());
        assert!(enforce_size(11, 10).is_err());
        assert!(enforce_size(0, 0).is_err());

        let mut body = Vec::new();
        let mut downloaded = 0;
        append_bounded_chunk(&mut body, &mut downloaded, b"123456", 10).unwrap();
        assert!(append_bounded_chunk(&mut body, &mut downloaded, b"78901", 10).is_err());
        assert_eq!(body, b"123456");
        assert_eq!(downloaded, 6);
    }

    #[test]
    fn reuses_one_hardened_client() {
        let first = safe_fetch_client().unwrap();
        let second = safe_fetch_client().unwrap();
        assert!(std::ptr::eq(first, second));
    }

    #[tokio::test]
    async fn resolver_rejects_non_public_dns_answers() {
        let name = "localhost".parse::<Name>().unwrap();
        let Err(error) = PublicIpResolver.resolve(name).await else {
            panic!("localhost must not resolve through the public-IP resolver");
        };
        assert!(error.to_string().contains("non-public address"));
    }
}
