use std::{
    future::Future,
    io,
    net::{IpAddr, SocketAddr},
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};

use reqwest::{
    Url,
    dns::{Addrs, Name, Resolve, Resolving},
};

use crate::{Client, ClientVariant, Error, HttpClientConfig, HttpClientPool, error::Repr};

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

const RETRY_DELAYS: [Duration; 2] = [Duration::from_secs(1), Duration::from_secs(2)];

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
    ) -> Result<Self, Error> {
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
    ) -> Result<Self, Error> {
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
            return Err(Repr::DownloadDisabled.into());
        }
        tokio::time::timeout(policy.timeout, self.fetch_with_retries(url, policy))
            .await
            .map_err(Repr::DownloadTimeout)?
    }

    async fn fetch_with_retries(
        &self,
        url: Url,
        policy: DownloadPolicy,
    ) -> Result<DownloadedMedia, Error> {
        let started = Instant::now();
        for delay in RETRY_DELAYS {
            match self.fetch_before_deadline(url.clone(), policy).await {
                Err(error) if may_recover(&error) && started.elapsed() + delay < policy.timeout => {
                    tokio::time::sleep(delay).await;
                }
                result => return result,
            }
        }
        self.fetch_before_deadline(url, policy).await
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
                .map_err(Repr::download)?;
            if response.status().is_redirection() {
                if redirects_followed == policy.max_redirects {
                    return Err(Repr::TooManyRedirects.into());
                }
                let location = response
                    .headers()
                    .get(reqwest::header::LOCATION)
                    .ok_or(Repr::MissingRedirectLocation)?
                    .to_str()
                    .map_err(|error| Repr::InvalidRedirect(error.into()))?;
                url = url
                    .join(location)
                    .map_err(|error| Repr::InvalidRedirect(error.into()))?;
                redirects_followed += 1;
                continue;
            }
            if !response.status().is_success() {
                return Err(Repr::DownloadStatus(response.status().as_u16()).into());
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
            while let Some(chunk) = response.chunk().await.map_err(Repr::download)? {
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
            return Err(Repr::BlockedUrl.into());
        }
        let host = url.host_str().ok_or(Repr::BlockedUrl)?;
        if self.allow_private_network {
            return Ok(&self.pinned);
        }
        let port = url.port_or_known_default().ok_or(Repr::BlockedUrl)?;
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
            return (!is_blocked_ip(ip))
                .then_some(())
                .ok_or(Repr::BlockedUrl.into());
        }
        let addresses = self
            .address_resolver
            .resolve(host, port)
            .await
            .map_err(Repr::Resolve)?;
        validate_addresses(&addresses)
    }
}

fn may_recover(error: &Error) -> bool {
    match error.repr() {
        Repr::Resolve(_) => true,
        Repr::Download(cause) => !cause.is_timeout(),
        _ => false,
    }
}

fn enforce_download_size(length: u64, max_bytes: u64) -> Result<(), Error> {
    if length > max_bytes {
        return Err(Repr::DownloadTooLarge.into());
    }
    Ok(())
}

fn validate_addresses(addresses: &[SocketAddr]) -> Result<(), Error> {
    if addresses.is_empty() || addresses.iter().any(|address| is_blocked_ip(address.ip())) {
        return Err(Repr::BlockedUrl.into());
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
            validate_addresses(&addresses)
                .map_err(|error| Box::new(error) as Box<dyn std::error::Error + Send + Sync>)?;
            Ok(Box::new(addresses.into_iter()) as Addrs)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        sync::atomic::{AtomicUsize, Ordering},
    };

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

    const PATIENT: DownloadPolicy = DownloadPolicy {
        timeout: Duration::from_secs(30),
        max_bytes: 10,
        max_redirects: 0,
    };

    fn policy(max_bytes: u64, max_redirects: usize) -> DownloadPolicy {
        DownloadPolicy {
            timeout: Duration::from_secs(1),
            max_bytes,
            max_redirects,
        }
    }

    #[rstest::rstest]
    #[case::unspecified_v4("0.0.0.1")]
    #[case::private_v4("10.0.0.1")]
    #[case::carrier_grade_nat("100.64.0.1")]
    #[case::loopback_v4("127.0.0.1")]
    #[case::link_local_v4("169.254.1.1")]
    #[case::private_v4_second_range("172.16.0.1")]
    #[case::private_v4_third_range("192.168.0.1")]
    #[case::benchmarking_v4("198.18.0.1")]
    #[case::documentation_v4_first_range("198.51.100.1")]
    #[case::documentation_v4_second_range("203.0.113.1")]
    #[case::multicast_v4("224.0.0.1")]
    #[case::loopback_v6("::1")]
    #[case::unique_local_v6("fc00::1")]
    #[case::link_local_v6("fe80::1")]
    #[case::documentation_v6("2001:db8::1")]
    #[case::mapped_loopback_v6("::ffff:127.0.0.1")]
    fn blocks_non_public_addresses(#[case] address: &str) {
        assert!(is_blocked_ip(address.parse().expect("valid test address")));
    }

    #[rstest::rstest]
    #[case::public("8.8.8.8")]
    fn allows_a_public_address(#[case] address: &str) {
        assert!(!is_blocked_ip(
            address.parse().expect("valid public address")
        ));
    }

    #[tokio::test]
    async fn fetches_exact_limit_and_normalizes_content_type() {
        let (url, server) = serve(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/pdf; charset=binary\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc",
        )
        .await;
        let client = Client::for_test();
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
        let client = Client::for_test();
        let error = MediaFetcher::for_test(client)
            .fetch(url, policy(2, 0))
            .await
            .expect_err("oversize body is rejected");
        server.await.expect("server completes");
        assert!(matches!(error.repr(), Repr::DownloadTooLarge));
    }

    #[tokio::test]
    async fn rejects_streamed_oversize_body() {
        let (url, server) = serve(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n2\r\nab\r\n2\r\ncd\r\n0\r\n\r\n",
        )
        .await;
        let client = Client::for_test();
        let error = MediaFetcher::for_test(client)
            .fetch(url, policy(3, 0))
            .await
            .expect_err("stream crossing limit is rejected");
        server.await.expect("server completes");
        assert!(matches!(error.repr(), Repr::DownloadTooLarge));
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
        assert!(matches!(error.repr(), Repr::BlockedUrl));
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
        assert!(matches!(error.repr(), Repr::DownloadTimeout(_)));
        assert!(!error.dispatched());
    }

    const CUT_BODY: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\nConnection: close\r\n\r\nab";

    #[rstest::rstest]
    #[case::second_attempt(vec![CUT_BODY, OK_RESPONSE])]
    #[case::last_attempt(vec![CUT_BODY, CUT_BODY, OK_RESPONSE])]
    #[tokio::test]
    async fn retries_a_download_the_media_host_cut(#[case] answers: Vec<&'static [u8]>) {
        let attempts = answers.len();
        let (url, server, address) = serve_named("public.test", answers).await;
        let started = Instant::now();
        let media = policy_checked_fetcher(address, HashSet::new())
            .fetch(url, PATIENT)
            .await
            .expect("a later attempt downloads the media");
        assert_eq!(server.await.expect("server completes").len(), attempts);
        assert_eq!(media.bytes, b"ok");
        assert!(
            started.elapsed() >= RETRY_DELAYS[..attempts - 1].iter().sum::<Duration>(),
            "each retry waits out a brief outage first"
        );
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn stops_after_three_cut_downloads_as_the_callers_fault() {
        let (url, server, address) = serve_named(
            "public.test",
            vec![CUT_BODY, CUT_BODY, CUT_BODY, OK_RESPONSE],
        )
        .await;
        let error = policy_checked_fetcher(address, HashSet::new())
            .fetch(url, PATIENT)
            .await
            .expect_err("no fourth attempt is made");
        server.abort();
        assert!(matches!(error.repr(), Repr::Download(_)));
        assert!(error.is_request());
    }

    struct FlakyAddressResolver {
        failures: usize,
        lookups: AtomicUsize,
    }

    impl AddressResolver for FlakyAddressResolver {
        fn resolve<'a>(&'a self, _host: &'a str, port: u16) -> AddressResolution<'a> {
            let failed = self.lookups.fetch_add(1, Ordering::Relaxed) < self.failures;
            Box::pin(async move {
                if failed {
                    return Err(io::Error::new(io::ErrorKind::NotFound, "no such host"));
                }
                Ok(vec![SocketAddr::from(([8, 8, 8, 8], port))])
            })
        }
    }

    fn flaky_fetcher(
        address: SocketAddr,
        failures: usize,
    ) -> (MediaFetcher, Arc<FlakyAddressResolver>) {
        let resolver = Arc::new(FlakyAddressResolver {
            failures,
            lookups: AtomicUsize::new(0),
        });
        let fetcher = MediaFetcher::with_resolution(
            &HttpClientPool::new(Arc::new(LoopbackDnsResolver(address))),
            &Resolution::from(&HttpSettings::default()).config,
            UrlPolicy::default(),
            Arc::clone(&resolver) as Arc<dyn AddressResolver>,
            Arc::new(|_| false),
        )
        .expect("test fetcher builds");
        (fetcher, resolver)
    }

    #[rstest::rstest]
    #[case::second_lookup(1)]
    #[case::last_lookup(2)]
    #[tokio::test]
    async fn retries_a_host_that_resolves_later(#[case] failures: usize) {
        let (url, server, address) = serve_named("public.test", vec![OK_RESPONSE]).await;
        let (fetcher, resolver) = flaky_fetcher(address, failures);
        let media = fetcher
            .fetch(url, PATIENT)
            .await
            .expect("a later lookup resolves the host");
        server.await.expect("server completes");
        assert_eq!(media.bytes, b"ok");
        assert_eq!(resolver.lookups.load(Ordering::Relaxed), failures + 1);
    }

    #[rstest::rstest]
    #[case::time_for_every_retry(PATIENT, RETRY_DELAYS.len() + 1)]
    #[case::no_time_for_a_retry(
        DownloadPolicy { timeout: RETRY_DELAYS[0] / 2, ..PATIENT },
        1
    )]
    #[tokio::test]
    async fn an_unresolvable_host_is_the_callers_fault(
        #[case] policy: DownloadPolicy,
        #[case] lookups: usize,
    ) {
        let (fetcher, resolver) = flaky_fetcher(UNROUTABLE, usize::MAX);
        let error = fetcher
            .fetch(
                Url::parse("http://missing.test/document").expect("valid test URL"),
                policy,
            )
            .await
            .expect_err("the host does not resolve");
        assert!(matches!(error.repr(), Repr::Resolve(_)));
        assert!(error.is_request());
        assert_eq!(resolver.lookups.load(Ordering::Relaxed), lookups);
    }

    #[derive(Default)]
    struct HangingDnsResolver(AtomicUsize);

    impl Resolve for HangingDnsResolver {
        fn resolve(&self, _name: Name) -> Resolving {
            self.0.fetch_add(1, Ordering::Relaxed);
            Box::pin(std::future::pending())
        }
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn a_connect_timeout_is_the_callers_fault_and_is_not_retried() {
        let resolver = Arc::new(HangingDnsResolver::default());
        let config = Resolution::from(&HttpSettings {
            connect_timeout: Duration::from_millis(50),
            ..HttpSettings::default()
        })
        .config;
        let error = MediaFetcher::with_resolution(
            &HttpClientPool::new(Arc::clone(&resolver) as Arc<dyn Resolve>),
            &config,
            UrlPolicy::default(),
            Arc::new(TestAddressResolver {
                blocked_hosts: HashSet::new(),
            }),
            Arc::new(|_| false),
        )
        .expect("test fetcher builds")
        .fetch(
            Url::parse("http://public.test/document").expect("valid test URL"),
            PATIENT,
        )
        .await
        .expect_err("the connection never opens");
        assert!(matches!(error.repr(), Repr::Download(_)));
        assert!(error.is_request());
        assert_eq!(resolver.0.load(Ordering::Relaxed), 1);
    }

    #[rstest::rstest]
    #[case::missing_location(b"HTTP/1.1 302 Found\r\nContent-Length: 0\r\n\r\n")]
    #[case::unreadable_location(
        b"HTTP/1.1 302 Found\r\nLocation: \xff\r\nContent-Length: 0\r\n\r\n"
    )]
    #[tokio::test]
    async fn a_broken_redirect_is_the_callers_fault(#[case] answer: &'static [u8]) {
        let (url, server, address) = serve_named("public.test", vec![answer]).await;
        let error = policy_checked_fetcher(address, HashSet::new())
            .fetch(url, policy(10, 1))
            .await
            .expect_err("the redirect cannot be followed");
        server.await.expect("server completes");
        assert!(error.is_request());
    }

    #[rstest::rstest]
    #[case::closed_before_answering(b"")]
    #[case::reset_mid_body(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nab")]
    #[tokio::test]
    async fn a_failed_download_never_reached_the_provider(#[case] response: &'static [u8]) {
        let (url, server) = serve(response).await;
        let url = url.join("?sig=secret").expect("valid test URL");
        let error = MediaFetcher::for_test(Client::for_test())
            .fetch(url, policy(10, 0))
            .await
            .expect_err("download fails");
        server.await.expect("server completes");
        let report = litellm_error::Report::new(&error).to_string();
        assert!(!error.dispatched());
        assert!(error.is_request());
        assert!(!report.contains("secret"), "{report}");
        assert!(!report.contains("document"), "{report}");
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
            fetcher
                .fetch(url, policy(1, 0))
                .await
                .err()
                .as_ref()
                .map(Error::repr),
            Some(Repr::BlockedUrl)
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
        assert!(matches!(
            result.as_ref().err().map(Error::repr),
            Some(Repr::BlockedUrl)
        ));
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
        assert!(matches!(
            blocked.as_ref().err().map(Error::repr),
            Some(Repr::BlockedUrl)
        ));
    }
}
