use std::{
    future::Future,
    io,
    net::{IpAddr, SocketAddr},
    pin::Pin,
    sync::Arc,
    time::Duration,
};

use reqwest::{
    Url,
    dns::{Addrs, Name, Resolve, Resolving},
};

use crate::{Client, ClientVariant, HttpClientConfig, HttpClientPool};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("media URL rejected by network policy")]
    BlockedUrl,
    #[error("media download is disabled")]
    DownloadDisabled,
    #[error("media download exceeds the maximum size")]
    DownloadTooLarge,
    #[error("too many redirects while fetching media")]
    TooManyRedirects,
    #[error("media redirect is missing a Location header")]
    MissingRedirectLocation,
    #[error("invalid media redirect")]
    InvalidRedirect,
    #[error("media download failed with status {0}")]
    Http(u16),
    #[error("media download timed out")]
    Timeout,
    #[error("{0}")]
    Transport(#[from] crate::transport::Error),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UrlPolicy {
    pub validate: bool,
    pub allowed_hosts: Vec<String>,
}

impl Default for UrlPolicy {
    fn default() -> Self {
        Self {
            validate: true,
            allowed_hosts: Vec::new(),
        }
    }
}

impl UrlPolicy {
    fn allows(&self, host: &str, port: u16) -> bool {
        let host = normalize_host(host);
        self.allowed_hosts
            .iter()
            .filter_map(|entry| parse_allowed_host(entry))
            .any(|(entry_host, entry_port)| {
                entry_host == host && entry_port.is_none_or(|entry_port| entry_port == port)
            })
    }
}

pub fn normalize_host(host: &str) -> String {
    let host = host.trim().trim_end_matches('.');
    let host = host
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host);
    host.to_ascii_lowercase()
}

fn parse_allowed_host(entry: &str) -> Option<(String, Option<u16>)> {
    let entry = entry.trim();
    if let Some(entry) = entry.strip_prefix('[') {
        let (host, suffix) = entry.split_once(']')?;
        let port = match suffix {
            "" => None,
            suffix => Some(suffix.strip_prefix(':')?.parse().ok()?),
        };
        return Some((normalize_host(host), port));
    }
    let (host, port) = match entry.rsplit_once(':') {
        Some((host, port)) if !host.contains(':') => (host, Some(port.parse().ok()?)),
        _ => (entry, None),
    };
    Some((normalize_host(host), port))
}

type ProxyMatch = Arc<dyn Fn(&Url) -> bool + Send + Sync>;

#[derive(Clone)]
pub struct MediaFetcher {
    pinned: Client,
    unpinned: Client,
    uses_proxy: ProxyMatch,
    address_resolver: Arc<dyn AddressResolver>,
    url_policy: UrlPolicy,
    allow_private_network: bool,
}

type AddressResolution<'a> = Pin<Box<dyn Future<Output = io::Result<Vec<SocketAddr>>> + Send + 'a>>;

trait AddressResolver: Send + Sync {
    fn resolve<'a>(&'a self, host: &'a str, port: u16) -> AddressResolution<'a>;
}

#[derive(Clone, Copy)]
pub struct DownloadPolicy {
    pub timeout: Duration,
    pub max_bytes: u64,
    pub max_redirects: usize,
}

#[derive(Debug)]
pub struct DownloadedMedia {
    pub bytes: Vec<u8>,
    pub content_type: String,
}

impl MediaFetcher {
    pub fn new(
        pool: &HttpClientPool,
        config: &HttpClientConfig,
        url_policy: UrlPolicy,
    ) -> Result<Self, crate::Error> {
        let uses_proxy: ProxyMatch = Arc::new(config.proxies.matcher());
        Self::with_resolution(
            pool,
            config,
            url_policy,
            Arc::new(SystemAddressResolver),
            uses_proxy,
        )
    }

    fn with_resolution(
        pool: &HttpClientPool,
        config: &HttpClientConfig,
        url_policy: UrlPolicy,
        address_resolver: Arc<dyn AddressResolver>,
        uses_proxy: ProxyMatch,
    ) -> Result<Self, crate::Error> {
        Ok(Self {
            pinned: pool.client(config, ClientVariant::Media)?,
            unpinned: pool.client(config, ClientVariant::UnpinnedMedia)?,
            uses_proxy,
            address_resolver,
            url_policy,
            allow_private_network: false,
        })
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn for_test(client: Client) -> Self {
        Self {
            pinned: client.clone(),
            unpinned: client,
            uses_proxy: Arc::new(|_| false),
            address_resolver: Arc::new(AllowPrivateResolver),
            url_policy: UrlPolicy::default(),
            allow_private_network: true,
        }
    }

    pub async fn fetch(&self, url: Url, policy: DownloadPolicy) -> Result<DownloadedMedia, Error> {
        if policy.max_bytes == 0 {
            return Err(Error::DownloadDisabled);
        }
        tokio::time::timeout(policy.timeout, self.fetch_before_deadline(url, policy))
            .await
            .map_err(|_| Error::Timeout)?
    }

    async fn fetch_before_deadline(
        &self,
        mut url: Url,
        policy: DownloadPolicy,
    ) -> Result<DownloadedMedia, Error> {
        let mut redirects_followed = 0;
        loop {
            let mut response = self
                .client_for(&url)
                .await?
                .get(url.clone())
                .send()
                .await
                .map_err(crate::transport::Error::from)?;
            if response.status().is_redirection() {
                if redirects_followed == policy.max_redirects {
                    return Err(Error::TooManyRedirects);
                }
                let location = response
                    .headers()
                    .get(reqwest::header::LOCATION)
                    .and_then(|value| value.to_str().ok())
                    .ok_or(Error::MissingRedirectLocation)?;
                url = url.join(location).map_err(|_| Error::InvalidRedirect)?;
                redirects_followed += 1;
                continue;
            }
            if !response.status().is_success() {
                return Err(Error::Http(response.status().as_u16()));
            }
            enforce_download_size(response.content_length().unwrap_or(0), policy.max_bytes)?;
            let content_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.split(';').next())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or("application/octet-stream")
                .to_string();
            let mut bytes = Vec::new();
            while let Some(chunk) = response
                .chunk()
                .await
                .map_err(crate::transport::Error::from)?
            {
                enforce_download_size(bytes.len() as u64 + chunk.len() as u64, policy.max_bytes)?;
                bytes.extend_from_slice(&chunk);
            }
            return Ok(DownloadedMedia {
                bytes,
                content_type,
            });
        }
    }

    async fn client_for(&self, url: &Url) -> Result<&Client, Error> {
        if !self.url_policy.validate {
            return Ok(&self.unpinned);
        }
        if !matches!(url.scheme(), "http" | "https")
            || !url.username().is_empty()
            || url.password().is_some()
        {
            return Err(Error::BlockedUrl);
        }
        let host = url.host_str().ok_or(Error::BlockedUrl)?;
        if self.allow_private_network {
            return Ok(&self.pinned);
        }
        let port = url.port_or_known_default().ok_or(Error::BlockedUrl)?;
        if self.url_policy.allows(host, port) {
            return Ok(&self.unpinned);
        }
        self.validate_host(host, port).await?;
        Ok(if (self.uses_proxy)(url) {
            &self.unpinned
        } else {
            &self.pinned
        })
    }

    async fn validate_host(&self, host: &str, port: u16) -> Result<(), Error> {
        if let Ok(ip) = host
            .trim_start_matches('[')
            .trim_end_matches(']')
            .parse::<IpAddr>()
        {
            return (!is_blocked_ip(ip)).then_some(()).ok_or(Error::BlockedUrl);
        }
        let addresses = self
            .address_resolver
            .resolve(host, port)
            .await
            .map_err(|error| crate::transport::Error::Network(error.to_string()))?;
        validate_addresses(&addresses)
    }
}

fn enforce_download_size(length: u64, max_bytes: u64) -> Result<(), Error> {
    if length > max_bytes {
        return Err(Error::DownloadTooLarge);
    }
    Ok(())
}

fn validate_addresses(addresses: &[SocketAddr]) -> Result<(), Error> {
    if addresses.is_empty() || addresses.iter().any(|address| is_blocked_ip(address.ip())) {
        return Err(Error::BlockedUrl);
    }
    Ok(())
}

fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            let [first, second, third, _] = ip.octets();
            first == 0
                || first == 10
                || first == 127
                || (first == 100 && (64..=127).contains(&second))
                || (first == 169 && second == 254)
                || (first == 172 && (16..=31).contains(&second))
                || (first == 192 && second == 0 && (third == 0 || third == 2))
                || (first == 192 && second == 168)
                || (first == 192 && second == 88 && third == 99)
                || (first == 198 && (second == 18 || second == 19))
                || (first == 198 && second == 51 && third == 100)
                || (first == 203 && second == 0 && third == 113)
                || first >= 224
        }
        IpAddr::V6(ip) => {
            let segments = ip.segments();
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_multicast()
                || (segments[0] & 0xfe00) == 0xfc00
                || (segments[0] & 0xffc0) == 0xfe80
                || (segments[0] & 0xffc0) == 0xfec0
                || (segments[0] == 0x2001 && segments[1] == 0x0db8)
                || ip
                    .to_ipv4_mapped()
                    .or_else(|| ip.to_ipv4())
                    .map(|ipv4| is_blocked_ip(IpAddr::V4(ipv4)))
                    .unwrap_or(false)
        }
    }
}

#[derive(Default)]
pub struct PublicDnsResolver;

struct SystemAddressResolver;

impl AddressResolver for SystemAddressResolver {
    fn resolve<'a>(&'a self, host: &'a str, port: u16) -> AddressResolution<'a> {
        Box::pin(async move {
            Ok(tokio::net::lookup_host((host, port))
                .await?
                .collect::<Vec<_>>())
        })
    }
}

#[cfg(any(test, feature = "test-support"))]
struct AllowPrivateResolver;

#[cfg(any(test, feature = "test-support"))]
impl AddressResolver for AllowPrivateResolver {
    fn resolve<'a>(&'a self, _host: &'a str, port: u16) -> AddressResolution<'a> {
        Box::pin(async move { Ok(vec![SocketAddr::from(([8, 8, 8, 8], port))]) })
    }
}

impl Resolve for PublicDnsResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_string();
        Box::pin(async move {
            let addresses = tokio::net::lookup_host((host.as_str(), 0))
                .await
                .map_err(|error| Box::new(error) as Box<dyn std::error::Error + Send + Sync>)?
                .collect::<Vec<_>>();
            validate_addresses(&addresses).map_err(|_| {
                Box::new(io::Error::other("destination rejected by network policy"))
                    as Box<dyn std::error::Error + Send + Sync>
            })?;
            Ok(Box::new(addresses.into_iter()) as Addrs)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    use super::*;
    use crate::{HttpSettings, Resolution};

    async fn serve(response: &'static [u8]) -> (Url, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener binds");
        let address = listener.local_addr().expect("listener has address");
        let task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts request");
            let mut request = [0_u8; 1024];
            let bytes_read = socket.read(&mut request).await.expect("reads request");
            assert!(bytes_read > 0);
            socket.write_all(response).await.expect("writes response");
        });
        (
            Url::parse(&format!("http://{address}/document")).expect("valid test URL"),
            task,
        )
    }

    async fn serve_named(
        host: &str,
        responses: Vec<&'static [u8]>,
    ) -> (Url, tokio::task::JoinHandle<Vec<String>>, SocketAddr) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener binds");
        let address = listener.local_addr().expect("listener has address");
        let task = tokio::spawn(async move {
            let mut requests = Vec::with_capacity(responses.len());
            for response in responses {
                let (mut socket, _) = listener.accept().await.expect("accepts request");
                let mut request = [0_u8; 4096];
                let bytes_read = socket.read(&mut request).await.expect("reads request");
                requests.push(String::from_utf8_lossy(&request[..bytes_read]).into_owned());
                socket.write_all(response).await.expect("writes response");
            }
            requests
        });
        (
            Url::parse(&format!("http://{host}:{}/document", address.port()))
                .expect("valid test URL"),
            task,
            address,
        )
    }

    struct LoopbackDnsResolver(SocketAddr);

    impl Resolve for LoopbackDnsResolver {
        fn resolve(&self, _name: Name) -> Resolving {
            let address = self.0;
            Box::pin(async move { Ok(Box::new(vec![address].into_iter()) as Addrs) })
        }
    }

    struct TestAddressResolver {
        blocked_hosts: HashSet<&'static str>,
    }

    impl AddressResolver for TestAddressResolver {
        fn resolve<'a>(&'a self, host: &'a str, port: u16) -> AddressResolution<'a> {
            let blocked = self.blocked_hosts.contains(host);
            Box::pin(async move {
                let ip = if blocked {
                    IpAddr::from([127, 0, 0, 1])
                } else {
                    IpAddr::from([8, 8, 8, 8])
                };
                Ok(vec![SocketAddr::new(ip, port)])
            })
        }
    }

    fn policy_checked_fetcher(
        address: SocketAddr,
        blocked_hosts: HashSet<&'static str>,
    ) -> MediaFetcher {
        fetcher(address, blocked_hosts, UrlPolicy::default(), false)
    }

    fn fetcher(
        pinned_address: SocketAddr,
        blocked_hosts: HashSet<&'static str>,
        url_policy: UrlPolicy,
        uses_proxy: bool,
    ) -> MediaFetcher {
        let direct = Resolution::from(&HttpSettings::default()).config;
        MediaFetcher::with_resolution(
            &HttpClientPool::new(Arc::new(LoopbackDnsResolver(pinned_address))),
            &direct,
            url_policy,
            Arc::new(TestAddressResolver { blocked_hosts }),
            Arc::new(move |_| uses_proxy),
        )
        .expect("test fetcher builds")
    }

    const UNROUTABLE: SocketAddr =
        SocketAddr::new(IpAddr::V4(std::net::Ipv4Addr::new(192, 0, 2, 1)), 9);

    const OK_RESPONSE: &[u8] =
        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok";

    fn policy(max_bytes: u64, max_redirects: usize) -> DownloadPolicy {
        DownloadPolicy {
            timeout: Duration::from_secs(1),
            max_bytes,
            max_redirects,
        }
    }

    #[test]
    fn blocks_non_public_addresses() {
        for address in [
            "0.0.0.1",
            "10.0.0.1",
            "100.64.0.1",
            "127.0.0.1",
            "169.254.1.1",
            "172.16.0.1",
            "192.168.0.1",
            "198.18.0.1",
            "198.51.100.1",
            "203.0.113.1",
            "224.0.0.1",
            "::1",
            "fc00::1",
            "fe80::1",
            "2001:db8::1",
            "::ffff:127.0.0.1",
        ] {
            assert!(is_blocked_ip(address.parse().expect("valid test address")));
        }
        assert!(!is_blocked_ip(
            "8.8.8.8".parse().expect("valid public address")
        ));
    }

    #[tokio::test]
    async fn fetches_exact_limit_and_normalizes_content_type() {
        let (url, server) = serve(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/pdf; charset=binary\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc",
        )
        .await;
        let client = Client::no_redirect_for_test();
        let media = MediaFetcher::for_test(client)
            .fetch(url, policy(3, 0))
            .await
            .expect("download succeeds at exact limit");
        server.await.expect("server completes");
        assert_eq!(media.bytes, b"abc");
        assert_eq!(media.content_type, "application/pdf");
    }

    #[tokio::test]
    async fn rejects_declared_oversize_body() {
        let (url, server) = serve(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/pdf\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc",
        )
        .await;
        let client = Client::no_redirect_for_test();
        let error = MediaFetcher::for_test(client)
            .fetch(url, policy(2, 0))
            .await
            .expect_err("oversize body is rejected");
        server.await.expect("server completes");
        assert!(matches!(error, Error::DownloadTooLarge));
    }

    #[tokio::test]
    async fn rejects_streamed_oversize_body() {
        let (url, server) = serve(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n2\r\nab\r\n2\r\ncd\r\n0\r\n\r\n",
        )
        .await;
        let client = Client::no_redirect_for_test();
        let error = MediaFetcher::for_test(client)
            .fetch(url, policy(3, 0))
            .await
            .expect_err("stream crossing limit is rejected");
        server.await.expect("server completes");
        assert!(matches!(error, Error::DownloadTooLarge));
    }

    #[tokio::test]
    async fn follows_allowed_redirects_and_revalidates_each_destination() {
        let (url, server, address) = serve_named(
            "public.test",
            vec![
                b"HTTP/1.1 302 Found\r\nLocation: /final\r\nContent-Length: 0\r\n\r\n",
                b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok",
            ],
        )
        .await;
        let media = policy_checked_fetcher(address, HashSet::new())
            .fetch(url, policy(2, 1))
            .await
            .expect("redirected fetch succeeds");
        let requests = server.await.expect("server completes");
        assert_eq!(requests.len(), 2);
        assert!(requests[1].starts_with("GET /final "));
        assert_eq!(media.bytes, b"ok");
    }

    #[tokio::test]
    async fn blocks_redirected_private_destination_before_second_request() {
        let (url, server, address) = serve_named(
            "public.test",
            vec![b"HTTP/1.1 302 Found\r\nLocation: http://blocked.test/document\r\nContent-Length: 0\r\n\r\n"],
        )
        .await;
        let error = policy_checked_fetcher(address, HashSet::from(["blocked.test"]))
            .fetch(url, policy(10, 1))
            .await
            .expect_err("private redirect is rejected");
        let requests = server.await.expect("server completes");
        assert_eq!(requests.len(), 1);
        assert!(matches!(error, Error::BlockedUrl));
    }

    #[tokio::test]
    async fn enforces_total_timeout() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener binds");
        let address = listener.local_addr().expect("listener has address");
        let server = tokio::spawn(async move {
            let (_socket, _) = listener.accept().await.expect("accepts request");
            tokio::time::sleep(Duration::from_millis(100)).await;
        });
        let url = Url::parse(&format!("http://public.test:{}/document", address.port()))
            .expect("valid test URL");
        let error = policy_checked_fetcher(address, HashSet::new())
            .fetch(
                url,
                DownloadPolicy {
                    timeout: Duration::from_millis(20),
                    max_bytes: 10,
                    max_redirects: 0,
                },
            )
            .await
            .expect_err("fetch times out");
        server.await.expect("server completes");
        assert!(matches!(error, Error::Timeout));
    }

    #[tokio::test]
    async fn document_client_does_not_send_ambient_credentials() {
        let (url, server, address) = serve_named(
            "public.test",
            vec![b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"],
        )
        .await;
        policy_checked_fetcher(address, HashSet::new())
            .fetch(url, policy(2, 0))
            .await
            .expect("fetch succeeds");
        let requests = server.await.expect("server completes");
        assert!(!requests[0].to_ascii_lowercase().contains("authorization:"));
        assert!(!requests[0].to_ascii_lowercase().contains("api-key:"));
    }

    #[tokio::test]
    async fn rejects_url_credentials_before_network_access() {
        let fetcher = MediaFetcher::new(
            &HttpClientPool::new(Arc::new(PublicDnsResolver)),
            &Resolution::from(&HttpSettings::default()).config,
            UrlPolicy::default(),
        )
        .expect("media fetcher builds");
        let url =
            Url::parse("https://user:password@8.8.8.8/document").expect("credentialed URL parses");
        assert!(matches!(
            fetcher.fetch(url, policy(1, 0)).await,
            Err(Error::BlockedUrl)
        ));
    }

    #[tokio::test]
    async fn allowlisted_private_host_is_fetched_without_the_pinned_resolver() {
        let (url, server, _) = serve_named("localhost", vec![OK_RESPONSE]).await;
        let port = url.port().expect("test URL has a port");
        let allowed = UrlPolicy {
            validate: true,
            allowed_hosts: vec![format!("LOCALHOST:{port}")],
        };
        let media = fetcher(UNROUTABLE, HashSet::from(["localhost"]), allowed, false)
            .fetch(url, policy(2, 0))
            .await
            .expect("allowlisted host downloads");
        server.await.expect("server completes");
        assert_eq!(media.bytes, b"ok");
    }

    #[tokio::test]
    async fn allowlist_entry_for_another_port_does_not_open_the_host() {
        let (url, _server, _) = serve_named("localhost", vec![OK_RESPONSE]).await;
        let other_port = UrlPolicy {
            validate: true,
            allowed_hosts: vec!["localhost:1".into()],
        };
        let result = fetcher(UNROUTABLE, HashSet::from(["localhost"]), other_port, false)
            .fetch(url, policy(2, 0))
            .await;
        assert!(matches!(result, Err(Error::BlockedUrl)));
    }

    #[test]
    fn allowlist_matches_bracketed_ipv6_hosts_and_ports() {
        let policy = UrlPolicy {
            validate: true,
            allowed_hosts: vec!["[2001:db8::1]".into(), "[2001:db8::1]:8443".into()],
        };
        assert!(policy.allows("2001:db8::1", 443));
        assert!(policy.allows("2001:db8::1", 8443));
        let port_specific = UrlPolicy {
            validate: true,
            allowed_hosts: vec!["[2001:db8::1]:8443".into()],
        };
        assert!(!port_specific.allows("2001:db8::1", 9443));
    }

    #[tokio::test]
    async fn validation_off_fetches_private_hosts_and_follows_redirects() {
        let (url, server, _) = serve_named(
            "localhost",
            vec![
                b"HTTP/1.1 302 Found\r\nLocation: /moved\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                OK_RESPONSE,
            ],
        )
        .await;
        let off = UrlPolicy {
            validate: false,
            allowed_hosts: Vec::new(),
        };
        let media = fetcher(UNROUTABLE, HashSet::from(["localhost"]), off, false)
            .fetch(url, policy(2, 1))
            .await
            .expect("unvalidated download succeeds");
        let requests = server.await.expect("server completes");
        assert_eq!(media.bytes, b"ok");
        assert!(requests[1].starts_with("GET /moved "));
    }

    #[tokio::test]
    async fn proxied_urls_skip_the_pinned_resolver_but_keep_the_address_check() {
        let (url, server, _) = serve_named("localhost", vec![OK_RESPONSE]).await;
        let media = fetcher(UNROUTABLE, HashSet::new(), UrlPolicy::default(), true)
            .fetch(url.clone(), policy(2, 0))
            .await
            .expect("public host behind a proxy downloads");
        server.await.expect("server completes");
        assert_eq!(media.bytes, b"ok");

        let blocked = fetcher(
            UNROUTABLE,
            HashSet::from(["localhost"]),
            UrlPolicy::default(),
            true,
        )
        .fetch(url, policy(2, 0))
        .await;
        assert!(matches!(blocked, Err(Error::BlockedUrl)));
    }
}
