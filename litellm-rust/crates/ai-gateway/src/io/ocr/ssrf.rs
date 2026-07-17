use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::Duration;

use litellm_core::error::CoreError;
use litellm_core::CoreResult;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::Url;
use url::Host;

use crate::constants::{DOCUMENT_FETCH_TIMEOUT_SECS, MAX_DOCUMENT_CLIENTS};

const BLOCKED_IPV6_CIDRS: &[(Ipv6Addr, u32)] = &[
    (Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 0), 96),
    (Ipv6Addr::new(0x64, 0xff9b, 0, 0, 0, 0, 0, 0), 96),
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

const BLOCKED_IPV4_CIDRS: &[(Ipv4Addr, u32)] = &[
    (Ipv4Addr::new(0, 0, 0, 0), 8),
    (Ipv4Addr::new(100, 64, 0, 0), 10),
    (Ipv4Addr::new(192, 0, 0, 0), 24),
    (Ipv4Addr::new(192, 0, 2, 0), 24),
    (Ipv4Addr::new(192, 88, 99, 0), 24),
    (Ipv4Addr::new(198, 18, 0, 0), 15),
    (Ipv4Addr::new(198, 51, 100, 0), 24),
    (Ipv4Addr::new(203, 0, 113, 0), 24),
    (Ipv4Addr::new(240, 0, 0, 0), 4),
];

pub(super) fn blocked_url_error() -> CoreError {
    CoreError::InvalidRequest("OCR document URL rejected by SSRF protection".to_string())
}

fn ipv4_in_cidr(ip: Ipv4Addr, network: Ipv4Addr, prefix_length: u32) -> bool {
    let mask = if prefix_length == 0 {
        0
    } else {
        u32::MAX << (32 - prefix_length)
    };
    u32::from(ip) & mask == u32::from(network) & mask
}

fn ipv6_in_cidr(ip: Ipv6Addr, network: Ipv6Addr, prefix_length: u32) -> bool {
    let mask = if prefix_length == 0 {
        0
    } else {
        u128::MAX << (128 - prefix_length)
    };
    u128::from(ip) & mask == u128::from(network) & mask
}

pub(super) fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            ip.is_private()
                || ip.is_loopback()
                || ip.is_link_local()
                || ip.is_broadcast()
                || ip.is_multicast()
                || ip.is_unspecified()
                || BLOCKED_IPV4_CIDRS
                    .iter()
                    .any(|(network, prefix)| ipv4_in_cidr(ip, *network, *prefix))
        }
        IpAddr::V6(ip) => {
            if let Some(v4) = ip.to_ipv4_mapped() {
                return is_blocked_ip(IpAddr::V4(v4));
            }
            BLOCKED_IPV6_CIDRS
                .iter()
                .any(|(network, prefix)| ipv6_in_cidr(ip, *network, *prefix))
        }
    }
}

pub(super) fn parse_fetchable_url(raw: &str) -> CoreResult<Url> {
    let url = Url::parse(raw).map_err(|_| blocked_url_error())?;
    ensure_allowed_url(&url)?;
    Ok(url)
}

fn ensure_allowed_url(url: &Url) -> CoreResult<()> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err(blocked_url_error());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(blocked_url_error());
    }
    Ok(())
}

pub(super) async fn resolve_validated(url: &Url) -> CoreResult<Vec<SocketAddr>> {
    ensure_allowed_url(url)?;
    let port = url.port_or_known_default().ok_or_else(blocked_url_error)?;
    let addresses: Vec<SocketAddr> = match url.host() {
        Some(Host::Ipv4(ip)) => vec![SocketAddr::from((ip, port))],
        Some(Host::Ipv6(ip)) => vec![SocketAddr::from((ip, port))],
        Some(Host::Domain(domain)) => tokio::net::lookup_host((domain, port))
            .await
            .map_err(|_| blocked_url_error())?
            .collect(),
        None => return Err(blocked_url_error()),
    };
    if addresses.is_empty() || addresses.iter().any(|address| is_blocked_ip(address.ip())) {
        return Err(blocked_url_error());
    }
    Ok(addresses)
}

type PinnedAddrs = Arc<Mutex<HashMap<String, Vec<SocketAddr>>>>;

#[derive(Debug, Clone)]
struct PinnedResolver {
    pins: PinnedAddrs,
}

impl Resolve for PinnedResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let pins = self.pins.clone();
        Box::pin(async move {
            let pinned = pins
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get(name.as_str())
                .cloned();
            match pinned {
                Some(addresses) if !addresses.is_empty() => {
                    Ok(Box::new(addresses.into_iter()) as Addrs)
                }
                _ => Err(Box::<dyn std::error::Error + Send + Sync>::from(
                    "OCR document host was not pinned to a validated address",
                )),
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DocumentClientKey {
    scheme: String,
    host: String,
    port: u16,
    addresses: Vec<SocketAddr>,
}

fn document_client_cache() -> &'static Mutex<HashMap<DocumentClientKey, reqwest::Client>> {
    static CACHE: OnceLock<Mutex<HashMap<DocumentClientKey, reqwest::Client>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(super) fn pinned_client(url: &Url, addresses: &[SocketAddr]) -> CoreResult<reqwest::Client> {
    let host = url.host_str().ok_or_else(blocked_url_error)?.to_owned();
    let port = url.port_or_known_default().ok_or_else(blocked_url_error)?;
    let mut sorted = addresses.to_vec();
    sorted.sort();
    let key = DocumentClientKey {
        scheme: url.scheme().to_owned(),
        host: host.clone(),
        port,
        addresses: sorted.clone(),
    };
    let cache = document_client_cache();
    if let Some(client) = cache
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&key)
        .cloned()
    {
        return Ok(client);
    }

    let mut pins = HashMap::new();
    pins.insert(host, sorted);
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(DOCUMENT_FETCH_TIMEOUT_SECS))
        .dns_resolver(Arc::new(PinnedResolver {
            pins: Arc::new(Mutex::new(pins)),
        }))
        .build()
        .map_err(|err| CoreError::Network(err.to_string()))?;

    let mut guard = cache.lock().unwrap_or_else(PoisonError::into_inner);
    if guard.len() >= MAX_DOCUMENT_CLIENTS {
        if let Some(evicted) = guard.keys().next().cloned() {
            guard.remove(&evicted);
        }
    }
    guard.insert(key, client.clone());
    Ok(client)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn validate_rejects_all_special_use_ranges() {
        let blocked = [
            "http://127.0.0.1/x",
            "http://10.0.0.1/x",
            "http://172.16.0.1/x",
            "http://192.168.1.1/x",
            "http://169.254.169.254/x",
            "http://100.64.0.1/x",
            "http://198.18.0.1/x",
            "http://192.0.2.1/x",
            "http://[::1]/x",
            "http://[fd00::1]/x",
            "http://[fe80::1]/x",
            "http://[fec0::1]/x",
            "http://[ff02::1]/x",
            "http://[::ffff:169.254.169.254]/x",
            "http://[::ffff:10.0.0.1]/x",
            "http://[::ffff:100.64.0.1]/x",
            "http://[::ffff:198.18.0.1]/x",
            "http://[::1.2.3.4]/x",
            "http://[2001:db8::1]/x",
            "http://[100::1]/x",
            "http://[2001::1]/x",
            "http://[2001:2::1]/x",
            "http://[2001:10::1]/x",
            "http://[2001:20::1]/x",
            "http://[2002:c0a8:101::1]/x",
            "http://[2002:a9fe:a9fe::1]/x",
            "http://[2002:808:808::1]/x",
            "http://[3fff::1]/x",
            "http://[5f00::1]/x",
            "http://[64:ff9b:1::1]/x",
            "http://[64:ff9b::192.168.1.1]/x",
            "http://[64:ff9b::169.254.169.254]/x",
            "http://[64:ff9b::8.8.8.8]/x",
            "ftp://8.8.8.8/x",
            "file:///etc/passwd",
            "gopher://8.8.8.8/x",
            "http://user:pass@8.8.8.8/x",
        ];
        for raw in blocked {
            let url = Url::parse(raw).unwrap();
            let error = resolve_validated(&url).await.unwrap_err();
            assert!(
                matches!(&error, CoreError::InvalidRequest(message) if message.contains("SSRF protection")),
                "{raw} should be rejected, got {error:?}"
            );
        }

        let allowed = Url::parse("http://8.8.8.8/x").unwrap();
        assert_eq!(
            resolve_validated(&allowed).await.unwrap(),
            vec![SocketAddr::from(([8, 8, 8, 8], 80))]
        );

        let allowed_v6 = [
            "http://[2606:4700:4700::1111]/x",
            "http://[::ffff:8.8.8.8]/x",
        ];
        for raw in allowed_v6 {
            let url = Url::parse(raw).unwrap();
            assert!(
                resolve_validated(&url).await.is_ok(),
                "{raw} should be allowed"
            );
        }
    }

    #[test]
    fn blocks_private_and_metadata_ips() {
        assert!(is_blocked_ip("127.0.0.1".parse().unwrap()));
        assert!(is_blocked_ip("10.0.0.1".parse().unwrap()));
        assert!(is_blocked_ip("169.254.169.254".parse().unwrap()));
        assert!(is_blocked_ip("100.64.0.1".parse().unwrap()));
        assert!(is_blocked_ip("198.18.0.1".parse().unwrap()));
        assert!(is_blocked_ip("192.0.2.1".parse().unwrap()));
        assert!(is_blocked_ip("::1".parse().unwrap()));
        assert!(is_blocked_ip("fd00::1".parse().unwrap()));
        assert!(is_blocked_ip("fe80::1".parse().unwrap()));
        assert!(is_blocked_ip("fec0::1".parse().unwrap()));
        assert!(is_blocked_ip("2001:db8::1".parse().unwrap()));
        assert!(is_blocked_ip("2002:c0a8:101::1".parse().unwrap()));
        assert!(is_blocked_ip("64:ff9b::8.8.8.8".parse().unwrap()));
        assert!(is_blocked_ip("3fff::1".parse().unwrap()));
        assert!(is_blocked_ip("5f00::1".parse().unwrap()));
        assert!(is_blocked_ip("::ffff:169.254.169.254".parse().unwrap()));
        assert!(is_blocked_ip("::ffff:10.0.0.1".parse().unwrap()));
        assert!(is_blocked_ip("::ffff:100.64.0.1".parse().unwrap()));
        assert!(is_blocked_ip("::ffff:198.18.0.1".parse().unwrap()));
        assert!(is_blocked_ip("::1.2.3.4".parse().unwrap()));
        assert!(!is_blocked_ip("8.8.8.8".parse().unwrap()));
        assert!(!is_blocked_ip("2606:4700:4700::1111".parse().unwrap()));
        assert!(!is_blocked_ip("::ffff:8.8.8.8".parse().unwrap()));
    }

    #[test]
    fn parse_fetchable_url_rejects_non_http_and_userinfo() {
        assert!(parse_fetchable_url("file:///etc/passwd").is_err());
        assert!(parse_fetchable_url("gopher://8.8.8.8/x").is_err());
        assert!(parse_fetchable_url("http://user:pass@example.com/x").is_err());
        assert_eq!(
            parse_fetchable_url("HTTP://example.com/x")
                .unwrap()
                .scheme(),
            "http"
        );
    }

    #[test]
    fn document_client_key_distinguishes_scheme_port_and_addresses() {
        let addr_a = SocketAddr::from(([203, 0, 113, 1], 443));
        let addr_b = SocketAddr::from(([203, 0, 113, 2], 443));
        let base = DocumentClientKey {
            scheme: "https".to_string(),
            host: "docs.example".to_string(),
            port: 443,
            addresses: vec![addr_a],
        };
        let other_scheme = DocumentClientKey {
            scheme: "http".to_string(),
            ..base.clone()
        };
        let other_port = DocumentClientKey {
            port: 8443,
            ..base.clone()
        };
        let other_addrs = DocumentClientKey {
            addresses: vec![addr_a, addr_b],
            ..base.clone()
        };

        assert_ne!(base, other_scheme);
        assert_ne!(base, other_port);
        assert_ne!(base, other_addrs);
        assert_eq!(base, base.clone());
    }
}
