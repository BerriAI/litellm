#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid forwarding upstream URI")]
    UpstreamUri(#[source] axum::http::uri::InvalidUri),
    #[error(
        "LITELLM_FORWARD_UPSTREAM must be a loopback IPv4 http origin without a path, query or credentials"
    )]
    Upstream,
}
