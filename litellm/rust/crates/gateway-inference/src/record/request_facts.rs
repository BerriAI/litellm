//! What the logging payload reads of the request itself, taken as it comes in on its route, since
//! the route consumes the request and only its record lasts until the payload is built.

use std::{collections::BTreeMap, net::SocketAddr};

use axum::{
    extract::{ConnectInfo, Request},
    http::header::USER_AGENT,
};

/// The request facts Python's proxy stamps on a request's metadata in
/// `add_litellm_data_to_request`, and the ones `get_standard_logging_object_payload` reads of
/// `proxy_server_request["headers"]`. `Debug` hides the custom headers, which a caller can put a
/// credential in.
#[derive(Clone, Default, PartialEq, Eq, veil::Redact)]
pub struct RequestFacts {
    /// `requester_ip_address`: the client's TCP peer, `""` without one. `use_x_forwarded_for` is
    /// a setting the gateway leaves to Python, so the peer is all it reads.
    pub requester_ip_address: String,
    /// `user_agent`: the request's first `User-Agent`, `""` without one.
    pub user_agent: String,
    /// The `User-Agent:` request tags, `StandardLoggingPayloadSetup._get_user_agent_tags`.
    pub user_agent_tags: Vec<String>,
    /// `requester_custom_headers`, as the enterprise metadata sets it.
    #[redact(with = "[REDACTED]")]
    pub requester_custom_headers: Option<BTreeMap<String, String>>,
    /// The `used_client_oauth_token` the proxy stamps from the headers, before
    /// `resolve_used_client_oauth_token` reads it against the served provider.
    pub client_sent_oauth_token: bool,
}

impl RequestFacts {
    /// The facts of `request`, whose serving loop puts the client's socket in its extensions as
    /// axum's `ConnectInfo`. A header value that is not visible ASCII is left out.
    pub(crate) fn of(request: &Request) -> Self {
        let headers: Vec<(String, String)> = request
            .headers()
            .iter()
            .filter_map(|(name, value)| {
                Some((name.as_str().to_owned(), value.to_str().ok()?.to_owned()))
            })
            .collect();
        Self {
            requester_ip_address: request
                .extensions()
                .get::<ConnectInfo<SocketAddr>>()
                .map(|ConnectInfo(peer)| peer.ip().to_string())
                .unwrap_or_default(),
            user_agent: request
                .headers()
                .get(USER_AGENT)
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
                .to_owned(),
            user_agent_tags: litellm_standard_logging::user_agent_tags(&headers),
            requester_custom_headers: litellm_standard_logging::requester_custom_headers(&headers),
            client_sent_oauth_token: litellm_standard_logging::client_sent_oauth_token(&headers),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

    use axum::body::Body;
    use rstest::rstest;

    use super::*;

    fn request(peer: Option<IpAddr>, sent: &[(&str, &str)]) -> Request {
        let request = sent
            .iter()
            .fold(Request::post("/v1/messages"), |request, (name, value)| {
                request.header(*name, *value)
            });
        let request = match peer {
            Some(ip) => request.extension(ConnectInfo(SocketAddr::new(ip, 51234))),
            None => request,
        };
        request.body(Body::empty()).unwrap()
    }

    #[rstest]
    #[case::ipv4(Some(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 7))), "10.0.0.7")]
    #[case::ipv6(Some(IpAddr::V6(Ipv6Addr::LOCALHOST)), "::1")]
    #[case::no_socket(None, "")]
    fn the_requester_is_the_peer_without_its_port(
        #[case] peer: Option<IpAddr>,
        #[case] address: &str,
    ) {
        assert_eq!(
            RequestFacts::of(&request(peer, &[])).requester_ip_address,
            address
        );
    }

    #[rstest]
    fn the_user_agent_is_the_first_and_its_tags_read_the_last() {
        let facts = RequestFacts::of(&request(
            None,
            &[
                ("user-agent", "claude-cli/2.1.0"),
                ("user-agent", "curl/8.0"),
            ],
        ));
        assert_eq!(
            (facts.user_agent.as_str(), facts.user_agent_tags),
            (
                "claude-cli/2.1.0",
                vec![
                    "User-Agent: curl".to_owned(),
                    "User-Agent: curl/8.0".to_owned()
                ]
            )
        );
    }

    #[rstest]
    fn a_request_without_headers_names_no_user_agent() {
        let facts = RequestFacts::of(&request(None, &[]));
        assert_eq!(
            (
                facts.user_agent.as_str(),
                facts.user_agent_tags.len(),
                facts.requester_custom_headers,
                facts.client_sent_oauth_token
            ),
            ("", 0, None, false)
        );
    }

    #[rstest]
    fn debug_hides_the_custom_headers() {
        let facts = RequestFacts::of(&request(None, &[("x-team-token", "credential-value")]));
        let debug = format!("{facts:?}");
        assert!(!debug.contains("credential-value") && debug.contains("[REDACTED]"));
    }
}
