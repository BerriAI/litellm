use std::net::{IpAddr, SocketAddr};

use axum::http::{HeaderMap, HeaderValue};
use ipnet::IpNet;

#[derive(Clone)]
pub(crate) struct TrustedProxies(Vec<IpNet>);

impl TrustedProxies {
    pub(crate) fn parse(value: &str) -> Result<Self, crate::Error> {
        if value.trim().is_empty() {
            return Ok(Self(Vec::new()));
        }
        value
            .split(',')
            .map(|value| {
                let value = value.trim();
                let network = value
                    .parse::<IpNet>()
                    .or_else(|_| value.parse::<IpAddr>().map(IpNet::from))
                    .map_err(|_| crate::Error::TrustedProxies)?;
                if network.prefix_len() == 0 {
                    return Err(crate::Error::TrustedProxies);
                }
                Ok(network)
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Self)
    }

    fn contains(&self, address: IpAddr) -> bool {
        self.0.iter().any(|network| network.contains(&address))
    }

    pub(crate) fn apply(&self, headers: &mut HeaderMap, peer: Option<SocketAddr>) {
        let trusted = peer.is_some_and(|peer| self.contains(peer.ip()));
        let chain = trusted
            .then(|| {
                headers
                    .get_all("x-forwarded-for")
                    .iter()
                    .map(|value| value.to_str().ok())
                    .collect::<Option<Vec<_>>>()
                    .and_then(|values| {
                        values
                            .iter()
                            .flat_map(|value| value.split(','))
                            .map(|value| value.trim().parse::<IpAddr>().ok())
                            .collect::<Option<Vec<_>>>()
                    })
            })
            .flatten();
        let client = chain
            .as_ref()
            .and_then(|chain| {
                chain
                    .iter()
                    .rev()
                    .find(|address| !self.contains(**address))
                    .or_else(|| chain.first())
                    .copied()
            })
            .or_else(|| peer.map(|peer| peer.ip()));
        let scheme = if trusted && headers.get_all("x-forwarded-proto").iter().count() == 1 {
            match headers
                .get("x-forwarded-proto")
                .and_then(|value| value.to_str().ok())
            {
                Some("https") => "https",
                _ => "http",
            }
        } else {
            "http"
        };
        for name in [
            "forwarded",
            "x-forwarded-for",
            "x-forwarded-proto",
            "x-forwarded-host",
            "x-forwarded-port",
            "x-forwarded-prefix",
            "x-real-ip",
        ] {
            headers.remove(name);
        }
        if let Some(client) = client
            && let Ok(value) = HeaderValue::from_str(&client.to_string())
        {
            headers.insert("x-forwarded-for", value);
        }
        headers.insert("x-forwarded-proto", HeaderValue::from_static(scheme));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::direct("", "127.0.0.2", "198.51.100.7", "https", "127.0.0.2", "http")]
    #[case::trusted(
        "127.0.0.2",
        "127.0.0.2",
        "198.51.100.7",
        "https",
        "198.51.100.7",
        "https"
    )]
    #[case::chain(
        "10.0.0.0/8",
        "10.0.0.1",
        "192.0.2.1, 198.51.100.7, 10.0.0.2",
        "https",
        "198.51.100.7",
        "https"
    )]
    #[case::invalid_chain(
        "10.0.0.0/8",
        "10.0.0.1",
        "198.51.100.7, unknown",
        "https",
        "10.0.0.1",
        "https"
    )]
    #[case::invalid_scheme(
        "127.0.0.2",
        "127.0.0.2",
        "198.51.100.7",
        "https,http",
        "198.51.100.7",
        "http"
    )]
    #[case::ipv6("fd00::/8", "fd00::1", "2001:db8::7", "https", "2001:db8::7", "https")]
    fn normalizes_proxy_identity(
        #[case] trusted: &str,
        #[case] peer: &str,
        #[case] chain: &str,
        #[case] proto: &str,
        #[case] client: &str,
        #[case] scheme: &str,
    ) {
        let mut headers = HeaderMap::new();
        headers.insert("host", HeaderValue::from_static("gateway.example"));
        headers.insert("origin", HeaderValue::from_static("https://app.example"));
        headers.insert("x-forwarded-for", HeaderValue::from_str(chain).unwrap());
        headers.insert("x-forwarded-proto", HeaderValue::from_str(proto).unwrap());
        headers.insert("forwarded", HeaderValue::from_static("proto=https"));
        TrustedProxies::parse(trusted).unwrap().apply(
            &mut headers,
            Some(SocketAddr::new(peer.parse().unwrap(), 1234)),
        );
        assert_eq!(headers["x-forwarded-for"], client);
        assert_eq!(headers["x-forwarded-proto"], scheme);
        assert_eq!(headers["host"], "gateway.example");
        assert_eq!(headers["origin"], "https://app.example");
        assert!(!headers.contains_key("forwarded"));
    }

    #[rstest]
    #[case("*")]
    #[case("0.0.0.0/0")]
    #[case("::/0")]
    #[case("127.0.0.1,")]
    #[case("proxy.example")]
    fn invalid_policy_fails_closed(#[case] value: &str) {
        assert!(matches!(
            TrustedProxies::parse(value),
            Err(crate::Error::TrustedProxies)
        ));
    }
}
