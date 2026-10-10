use std::time::Duration;

use litellm_http::outbound::OutboundRequest;
use litellm_llms::base_llm::auth::Authenticated;
use serde_json::Value;

#[tracing::instrument(
    name = "litellm.provider.send",
    level = "debug",
    skip_all,
    fields(status)
)]
pub async fn send(
    request: OutboundRequest,
    client: &litellm_http::Client,
) -> Result<reqwest::Response, reqwest::Error> {
    request.send(client).await.inspect(|response| {
        tracing::Span::current().record("status", response.status().as_u16());
    })
}

/// Header credentials are already in `headers`; SigV4 is applied here, over the
/// bytes that are sent.
pub fn outbound_request(
    authenticated: Authenticated,
    url: String,
    body: &Value,
    timeout: Option<Duration>,
) -> Result<OutboundRequest, litellm_http::Error> {
    let Authenticated {
        headers,
        signer,
        url: authenticated_url,
    } = authenticated;
    let resolved_url = authenticated_url.unwrap_or(url);
    match signer {
        None => OutboundRequest::json(resolved_url, headers, body, timeout),
        Some(signer) => OutboundRequest::signed_json(resolved_url, headers, body, timeout, &signer),
    }
}
