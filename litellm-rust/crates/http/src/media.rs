use std::{
    future::Future,
    io,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
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

/// Ranges the IANA special-purpose registries mark as not globally reachable, plus multicast
/// and the Azure Wire Server. A superset of what `_is_blocked_ip` in
/// `litellm/litellm_core_utils/url_utils.py` rejects, pinned by `generated/blocked_ips.json`.
const BLOCKED_V4: [(Ipv4Addr, u32); 15] = [
    (Ipv4Addr::new(0, 0, 0, 0), 8),
    (Ipv4Addr::new(10, 0, 0, 0), 8),
    (Ipv4Addr::new(100, 64, 0, 0), 10),
    (Ipv4Addr::new(127, 0, 0, 0), 8),
    (Ipv4Addr::new(168, 63, 129, 16), 32),
    (Ipv4Addr::new(169, 254, 0, 0), 16),
    (Ipv4Addr::new(172, 16, 0, 0), 12),
    (Ipv4Addr::new(192, 0, 0, 0), 24),
    (Ipv4Addr::new(192, 0, 2, 0), 24),
    (Ipv4Addr::new(192, 88, 99, 0), 24),
    (Ipv4Addr::new(192, 168, 0, 0), 16),
    (Ipv4Addr::new(198, 18, 0, 0), 15),
    (Ipv4Addr::new(198, 51, 100, 0), 24),
    (Ipv4Addr::new(203, 0, 113, 0), 24),
    (Ipv4Addr::new(224, 0, 0, 0), 3),
];

const BLOCKED_V6: [(Ipv6Addr, u32); 13] = [
    (Ipv6Addr::UNSPECIFIED, 128),
    (Ipv6Addr::LOCALHOST, 128),
    (Ipv6Addr::new(0x64, 0xff9b, 1, 0, 0, 0, 0, 0), 48),
    (Ipv6Addr::new(0x100, 0, 0, 0, 0, 0, 0, 0), 64),
    (Ipv6Addr::new(0x2001, 0, 0, 0, 0, 0, 0, 0), 23),
    (Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 0), 32),
    (Ipv6Addr::new(0x2002, 0, 0, 0, 0, 0, 0, 0), 16),
    (Ipv6Addr::new(0x3fff, 0, 0, 0, 0, 0, 0, 0), 20),
    (Ipv6Addr::new(0x5f00, 0, 0, 0, 0, 0, 0, 0), 16),
    (Ipv6Addr::new(0xfc00, 0, 0, 0, 0, 0, 0, 0), 7),
    (Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 0), 10),
    (Ipv6Addr::new(0xfec0, 0, 0, 0, 0, 0, 0, 0), 10),
    (Ipv6Addr::new(0xff00, 0, 0, 0, 0, 0, 0, 0), 8),
];

fn in_network<const BITS: u32>(address: u128, network: u128, prefix: u32) -> bool {
    (address ^ network).checked_shr(BITS - prefix).unwrap_or(0) == 0
}

fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => BLOCKED_V4.iter().any(|(network, prefix)| {
            in_network::<32>(u32::from(ip).into(), u32::from(*network).into(), *prefix)
        }),
        IpAddr::V6(ip) => {
            BLOCKED_V6.iter().any(|(network, prefix)| {
                in_network::<128>(u128::from(ip), u128::from(*network), *prefix)
            }) || ip
                .to_ipv4_mapped()
                .or_else(|| ip.to_ipv4())
                .is_some_and(|ipv4| is_blocked_ip(IpAddr::V4(ipv4)))
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
    #[case::broadcast_v4("255.255.255.255")]
    #[case::ietf_protocol_assignments_v4("192.0.0.9")]
    #[case::azure_wire_server("168.63.129.16")]
    #[case::unspecified_v6("::")]
    #[case::loopback_v6("::1")]
    #[case::local_use_nat64("64:ff9b:1::1")]
    #[case::discard_only_v6("100::1")]
    #[case::teredo("2001::1")]
    #[case::documentation_v6("2001:db8::1")]
    #[case::six_to_four("2002:c000:204::1")]
    #[case::documentation_v6_second_range("3fff::1")]
    #[case::segment_routing_sids("5f00::1")]
    #[case::unique_local_v6("fc00::1")]
    #[case::link_local_v6("fe80::1")]
    #[case::site_local_v6("fec0::1")]
    #[case::multicast_v6("ff02::1")]
    #[case::mapped_loopback_v6("::ffff:127.0.0.1")]
    #[case::mapped_azure_wire_server("::ffff:168.63.129.16")]
    fn blocks_non_public_addresses(#[case] address: &str) {
        assert!(is_blocked_ip(address.parse().expect("valid test address")));
    }

    #[rstest::rstest]
    #[case::public("8.8.8.8")]
    #[case::next_to_azure_wire_server("168.63.129.17")]
    #[case::next_to_carrier_grade_nat("100.128.0.1")]
    #[case::last_unicast_before_multicast("223.255.255.255")]
    #[case::public_v6("2606:4700::1111")]
    #[case::above_ietf_protocol_assignments_v6("2001:200::1")]
    #[case::mapped_public_v6("::ffff:8.8.8.8")]
    fn allows_a_public_address(#[case] address: &str) {
        assert!(!is_blocked_ip(
            address.parse().expect("valid public address")
        ));
    }

    #[derive(serde::Deserialize)]
    struct BlockedIpFixture {
        rows: Vec<BlockedIpRow>,
    }

    #[derive(serde::Deserialize)]
    struct BlockedIpRow {
        address: IpAddr,
        blocked: bool,
    }

    /// Ranges Rust blocks where CPython's `ipaddress` reports global, with the reason. Each
    /// entry must still cover a diverging fixture row, so it is deleted once CPython catches up.
    const KNOWN_OVER_BLOCKED: &[(&str, &str)] = &[
        (
            "192.88.99.0/24",
            "deprecated 6to4 relay anycast, CPython never marked it non-global",
        ),
        (
            "2001:1::1/128",
            "Port Control Protocol anycast, carved out of 2001::/23 by CPython",
        ),
        (
            "2001:1::2/128",
            "TURN anycast, carved out of 2001::/23 by CPython",
        ),
        (
            "2001:3::/32",
            "AMT relays, carved out of 2001::/23 by CPython",
        ),
        (
            "2001:4:112::/48",
            "AS112-v6, carved out of 2001::/23 by CPython",
        ),
        (
            "2001:20::/28",
            "ORCHIDv2, carved out of 2001::/23 by CPython",
        ),
        (
            "2001:30::/28",
            "Drone Remote ID, carved out of 2001::/23 by CPython",
        ),
        ("5f00::/16", "SRv6 SIDs, newer than CPython's table"),
        (
            "fec0::/10",
            "deprecated site-local, which CPython only exposes as `is_site_local`",
        ),
        (
            "::/96",
            "deprecated IPv4-compatible addresses, Rust judges the embedded IPv4 address",
        ),
    ];

    fn in_cidr(ip: IpAddr, cidr: &str) -> bool {
        let (network, prefix) = cidr.split_once('/').expect("cidr carries a prefix length");
        let prefix: u32 = prefix.parse().expect("prefix length is numeric");
        match (ip, network.parse().expect("cidr network parses")) {
            (IpAddr::V4(ip), IpAddr::V4(network)) => {
                in_network::<32>(u32::from(ip).into(), u32::from(network).into(), prefix)
            }
            (IpAddr::V6(ip), IpAddr::V6(network)) => {
                in_network::<128>(ip.into(), network.into(), prefix)
            }
            _ => false,
        }
    }

    #[rstest::fixture]
    #[once]
    fn python_verdicts() -> BlockedIpFixture {
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/generated/blocked_ips.json"
        )))
        .expect("blocked_ips.json matches the fixture schema")
    }

    #[rstest::rstest]
    fn blocks_everything_the_python_policy_blocks(python_verdicts: &BlockedIpFixture) {
        let verdicts: Vec<(IpAddr, bool, bool)> = python_verdicts
            .rows
            .iter()
            .map(|row| (row.address, row.blocked, is_blocked_ip(row.address)))
            .collect();

        let under_blocked: Vec<IpAddr> = verdicts
            .iter()
            .filter(|(_, python, rust)| *python && !*rust)
            .map(|(address, ..)| *address)
            .collect();
        assert!(
            under_blocked.is_empty(),
            "Python blocks but Rust allows: {under_blocked:?}"
        );

        let over_blocked: Vec<IpAddr> = verdicts
            .iter()
            .filter(|(_, python, rust)| !*python && *rust)
            .map(|(address, ..)| *address)
            .collect();
        let unexplained: Vec<IpAddr> = over_blocked
            .iter()
            .copied()
            .filter(|address| {
                !KNOWN_OVER_BLOCKED
                    .iter()
                    .any(|(cidr, _)| in_cidr(*address, cidr))
            })
            .collect();
        assert!(
            unexplained.is_empty(),
            "Rust blocks but Python allows, missing from KNOWN_OVER_BLOCKED: {unexplained:?}"
        );

        let stale: Vec<&str> = KNOWN_OVER_BLOCKED
            .iter()
            .filter(|(cidr, _)| !over_blocked.iter().any(|address| in_cidr(*address, cidr)))
            .map(|(cidr, _)| *cidr)
            .collect();
        assert!(
            stale.is_empty(),
            "KNOWN_OVER_BLOCKED entries that no longer diverge from Python: {stale:?}"
        );
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
