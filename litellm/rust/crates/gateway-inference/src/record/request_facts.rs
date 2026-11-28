//! What the logging payload reads of the request itself, taken as it comes in on its route, since
//! the route consumes the request and only its record lasts until the payload is built.

use std::{collections::BTreeMap, net::SocketAddr};

use axum::{
    extract::{ConnectInfo, Request},
    http::header::USER_AGENT,
};
use litellm_gateway_auth_core::client_address::{Origin, peer_address};
use litellm_python_compat::unicode::decode_latin1;

const FORWARDED_FOR: &str = "x-forwarded-for";
const CALL_ID: &str = "x-litellm-call-id";

/// The `x-litellm-call-id` the client sent, before the gateway kept, replaced or added one, which
/// the call id layer puts in the request's extensions. Python logs the headers as the client sent
/// them, so the facts read this one in place of the header the request now carries.
#[derive(Clone, Debug, Default)]
pub struct SentCallId(pub Option<String>);

/// The request facts Python's proxy stamps on a request's metadata in
/// `add_litellm_data_to_request`, and the ones `get_standard_logging_object_payload` reads of
/// `proxy_server_request["headers"]`. `Debug` hides the custom headers, which a caller can put a
/// credential in.
#[derive(Clone, Default, PartialEq, Eq, veil::Redact)]
pub struct RequestFacts {
    /// `requester_ip_address`: `get_request_ip_address`, the `x-forwarded-for` header whole when
    /// `use_x_forwarded_for` is on and the request sent one, else the client's TCP peer, `""`
    /// without one.
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
    /// axum's `ConnectInfo`, read under `general_settings.use_x_forwarded_for`. A header value
    /// that is not visible ASCII is left out.
    pub(crate) fn of(request: &Request, use_x_forwarded_for: bool) -> Self {
        let headers = sent_headers(request);
        Self {
            requester_ip_address: requester_ip_address(request, use_x_forwarded_for),
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

/// The visible ASCII headers of `request` as its client sent them.
fn sent_headers(request: &Request) -> Vec<(String, String)> {
    let sent_call_id = request.extensions().get::<SentCallId>();
    let headers = request
        .headers()
        .iter()
        .filter(|(name, _)| sent_call_id.is_none() || name.as_str() != CALL_ID)
        .filter_map(|(name, value)| {
            Some((name.as_str().to_owned(), value.to_str().ok()?.to_owned()))
        });
    let sent = sent_call_id
        .and_then(|SentCallId(sent)| sent.clone())
        .map(|value| (CALL_ID.to_owned(), value));
    headers.chain(sent).collect()
}

/// `_get_request_ip_address`, which `add_litellm_data_to_request` stamps as the requester's
/// address.
fn requester_ip_address(request: &Request, use_x_forwarded_for: bool) -> String {
    let peer = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ConnectInfo(peer)| peer_address(*peer));
    let forwarded_for = request
        .headers()
        .get(FORWARDED_FOR)
        .map(|value| decode_latin1(value.as_bytes()));
    Origin {
        peer: peer.as_deref(),
        forwarded_for: forwarded_for.as_deref(),
    }
    .address(use_x_forwarded_for)
    .to_owned()
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
            RequestFacts::of(&request(peer, &[]), false).requester_ip_address,
            address
        );
    }

    #[rstest]
    #[case::read_when_on(true, "203.0.113.9, 10.0.0.1")]
    #[case::ignored_when_off(false, "10.0.0.7")]
    fn a_forwarded_address_is_the_requester_only_under_use_x_forwarded_for(
        #[case] use_x_forwarded_for: bool,
        #[case] address: &str,
    ) {
        let sent = request(
            Some(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 7))),
            &[("x-forwarded-for", "203.0.113.9, 10.0.0.1")],
        );
        assert_eq!(
            RequestFacts::of(&sent, use_x_forwarded_for).requester_ip_address,
            address
        );
    }

    #[rstest]
    fn the_user_agent_is_the_first_and_its_tags_read_the_last() {
        let facts = RequestFacts::of(
            &request(
                None,
                &[
                    ("user-agent", "claude-cli/2.1.0"),
                    ("user-agent", "curl/8.0"),
                ],
            ),
            false,
        );
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
        let facts = RequestFacts::of(&request(None, &[]), false);
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
    #[case::sent_and_kept(Some("call-1"), Some("call-1"), Some("call-1"))]
    #[case::added_by_the_gateway(None, Some("generated"), None)]
    #[case::replaced_by_the_gateway(Some(""), Some("generated"), Some(""))]
    #[case::read_without_the_layer(None, None, None)]
    fn the_call_id_is_logged_as_the_client_sent_it(
        #[case] sent: Option<&str>,
        #[case] carried: Option<&str>,
        #[case] logged: Option<&str>,
    ) {
        let mut request = request(None, &[("x-team", "ops")]);
        if let Some(carried) = carried {
            request
                .headers_mut()
                .insert(CALL_ID, carried.parse().unwrap());
            request
                .extensions_mut()
                .insert(SentCallId(sent.map(str::to_owned)));
        }
        let custom = RequestFacts::of(&request, false)
            .requester_custom_headers
            .unwrap();
        assert_eq!(
            (
                custom.get(CALL_ID).map(String::as_str),
                custom.get("x-team").map(String::as_str)
            ),
            (logged, Some("ops"))
        );
    }

    #[rstest]
    fn debug_hides_the_custom_headers() {
        let facts = RequestFacts::of(
            &request(None, &[("x-team-token", "credential-value")]),
            false,
        );
        let debug = format!("{facts:?}");
        assert!(!debug.contains("credential-value") && debug.contains("[REDACTED]"));
    }
}
