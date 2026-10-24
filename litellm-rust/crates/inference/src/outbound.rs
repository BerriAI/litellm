use std::time::Duration;

use litellm_http::outbound::{json, signed};
use litellm_llms::base_llm::auth::Authenticated;
use serde_json::Value;

#[tracing::instrument(
    name = "litellm.provider.send",
    level = "debug",
    skip_all,
    fields(status)
)]
pub async fn send(
    request: reqwest::Request,
    client: &litellm_http::Client,
) -> Result<reqwest::Response, reqwest::Error> {
    client.execute(request).await.inspect(|response| {
        tracing::Span::current().record("status", response.status().as_u16());
    })
}

/// Header credentials are already in `headers`; SigV4 is applied here, over the
/// bytes that are sent.
pub fn outbound_request(
    authenticated: Authenticated,
    url: &str,
    body: &Value,
    timeout: Option<Duration>,
) -> Result<reqwest::Request, litellm_http::Error> {
    let Authenticated { headers, signer } = authenticated;
    let request = json(url, &headers, body, timeout)?;
    match signer {
        None => Ok(request),
        Some(signer) => signed(request, &signer),
    }
}
