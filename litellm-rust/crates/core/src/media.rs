use std::future::Future;
use std::io;
use std::net::{IpAddr, SocketAddr};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use reqwest::Url;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};

use crate::constants::MEDIA_CONNECT_TIMEOUT_SECS;
use crate::error::{MediaError, TransportError};

#[derive(Clone)]
pub(crate) struct MediaFetcher {
    client: reqwest::Client,
    address_resolver: Arc<dyn AddressResolver>,
    allow_private_network: bool,
}

type AddressResolution<'a> = Pin<Box<dyn Future<Output = io::Result<Vec<SocketAddr>>> + Send + 'a>>;

trait AddressResolver: Send + Sync {
    fn resolve<'a>(&'a self, host: &'a str, port: u16) -> AddressResolution<'a>;
}

#[derive(Clone, Copy)]
pub(crate) struct DownloadPolicy {
    pub(crate) timeout: Duration,
    pub(crate) max_bytes: u64,
    pub(crate) max_redirects: usize,
}

#[derive(Debug)]
pub(crate) struct DownloadedMedia {
    pub(crate) bytes: Vec<u8>,
    pub(crate) content_type: String,
}

impl MediaFetcher {
    pub(crate) fn new() -> Result<Self, reqwest::Error> {
        Self::with_resolvers(Arc::new(PublicDnsResolver), Arc::new(SystemAddressResolver))
    }

    fn with_resolvers<R>(
        transport_resolver: Arc<R>,
        address_resolver: Arc<dyn AddressResolver>,
    ) -> Result<Self, reqwest::Error>
    where
        R: Resolve + 'static,
    {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(MEDIA_CONNECT_TIMEOUT_SECS))
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .dns_resolver(transport_resolver)
            .build()?;
        Ok(Self {
            client,
            address_resolver,
            allow_private_network: false,
        })
    }

    #[cfg(test)]
    pub(crate) fn for_test(client: reqwest::Client) -> Self {
        Self {
            client,
            address_resolver: Arc::new(AllowPrivateResolver),
            allow_private_network: true,
        }
    }

    pub(crate) async fn fetch(
        &self,
        url: Url,
        policy: DownloadPolicy,
    ) -> Result<DownloadedMedia, MediaError> {
        if policy.max_bytes == 0 {
            return Err(MediaError::DownloadDisabled);
        }
        tokio::time::timeout(policy.timeout, self.fetch_before_deadline(url, policy))
            .await
            .map_err(|_| MediaError::Timeout)?
    }

    async fn fetch_before_deadline(
        &self,
        mut url: Url,
        policy: DownloadPolicy,
    ) -> Result<DownloadedMedia, MediaError> {
        let mut redirects_followed = 0;
        loop {
            self.validate_url(&url).await?;
            let mut response = self
                .client
                .get(url.clone())
                .send()
                .await
                .map_err(TransportError::from)?;
            if response.status().is_redirection() {
                if redirects_followed == policy.max_redirects {
                    return Err(MediaError::TooManyRedirects);
                }
                let location = response
                    .headers()
                    .get(reqwest::header::LOCATION)
                    .and_then(|value| value.to_str().ok())
                    .ok_or(MediaError::MissingRedirectLocation)?;
                url = url
                    .join(location)
                    .map_err(|_| MediaError::InvalidRedirect)?;
                redirects_followed += 1;
                continue;
            }
            if !response.status().is_success() {
                return Err(MediaError::Http(response.status().as_u16()));
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
            while let Some(chunk) = response.chunk().await.map_err(TransportError::from)? {
                enforce_download_size(bytes.len() as u64 + chunk.len() as u64, policy.max_bytes)?;
                bytes.extend_from_slice(&chunk);
            }
            return Ok(DownloadedMedia {
                bytes,
                content_type,
            });
        }
    }

    async fn validate_url(&self, url: &Url) -> Result<(), MediaError> {
        if !matches!(url.scheme(), "http" | "https")
            || !url.username().is_empty()
            || url.password().is_some()
        {
            return Err(MediaError::BlockedUrl);
        }
        let host = url.host_str().ok_or(MediaError::BlockedUrl)?;
        if self.allow_private_network {
            return Ok(());
        }
        if let Ok(ip) = host.parse::<IpAddr>() {
            return (!is_blocked_ip(ip))
                .then_some(())
                .ok_or(MediaError::BlockedUrl);
        }
        let port = url.port_or_known_default().ok_or(MediaError::BlockedUrl)?;
        let addresses = self
            .address_resolver
            .resolve(host, port)
            .await
            .map_err(|error| TransportError::Network(error.to_string()))?;
        validate_addresses(&addresses)
    }
}

fn enforce_download_size(length: u64, max_bytes: u64) -> Result<(), MediaError> {
    if length > max_bytes {
        return Err(MediaError::DownloadTooLarge);
    }
    Ok(())
}

fn validate_addresses(addresses: &[SocketAddr]) -> Result<(), MediaError> {
    if addresses.is_empty() || addresses.iter().any(|address| is_blocked_ip(address.ip())) {
        return Err(MediaError::BlockedUrl);
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
struct PublicDnsResolver;

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

#[cfg(test)]
struct AllowPrivateResolver;

#[cfg(test)]
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
    use super::*;
    use std::collections::HashSet;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

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
        MediaFetcher::with_resolvers(
            Arc::new(LoopbackDnsResolver(address)),
            Arc::new(TestAddressResolver { blocked_hosts }),
        )
        .expect("test fetcher builds")
    }

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
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("test client builds");
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
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("test client builds");
        let error = MediaFetcher::for_test(client)
            .fetch(url, policy(2, 0))
            .await
            .expect_err("oversize body is rejected");
        server.await.expect("server completes");
        assert!(matches!(error, MediaError::DownloadTooLarge));
    }

    #[tokio::test]
    async fn rejects_streamed_oversize_body() {
        let (url, server) = serve(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n2\r\nab\r\n2\r\ncd\r\n0\r\n\r\n",
        )
        .await;
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("test client builds");
        let error = MediaFetcher::for_test(client)
            .fetch(url, policy(3, 0))
            .await
            .expect_err("stream crossing limit is rejected");
        server.await.expect("server completes");
        assert!(matches!(error, MediaError::DownloadTooLarge));
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
        assert!(matches!(error, MediaError::BlockedUrl));
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
        assert!(matches!(error, MediaError::Timeout));
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
        let fetcher = MediaFetcher::new().expect("media fetcher builds");
        let url =
            Url::parse("https://user:password@8.8.8.8/document").expect("credentialed URL parses");
        assert!(matches!(
            fetcher.validate_url(&url).await,
            Err(MediaError::BlockedUrl)
        ));
    }
}
