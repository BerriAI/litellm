use std::time::Duration;

use bytes::Bytes;
use litellm_http::{outbound::OutboundRequest, response::Rejected, transport};
use litellm_llms::base_llm::auth::Authenticated;
use serde_json::Value;

use crate::RouteError;

/// Sends the request. A reply with a non-success status is read and returned as
/// [`RouteError::Rejected`], so every route reports a provider failure the same way.
/// Failing to connect means the request never went out and the host may still serve it;
/// anything after that, a timeout above all, may have reached the provider.
#[tracing::instrument(
    name = "litellm.provider.send",
    level = "debug",
    skip_all,
    fields(status)
)]
pub async fn send(
    request: OutboundRequest,
    client: &litellm_http::Client,
) -> Result<reqwest::Response, RouteError> {
    let response = request
        .send(client)
        .await
        .map_err(transport::Error::from_reqwest_before_dispatch)?;
    tracing::Span::current().record("status", response.status().as_u16());
    if response.status().is_success() {
        return Ok(response);
    }
    let reply = transport::read(response, None).await?;
    Err(RouteError::Rejected(Rejected::new(reply).truncated()))
}

/// Drains a successful reply.
pub async fn read(response: reqwest::Response) -> Result<http::Response<Bytes>, RouteError> {
    Ok(transport::read(response, None).await?)
}

/// The text of a reply body as the route decodes it.
pub fn text(body: &Bytes) -> String {
    String::from_utf8_lossy(body).into_owned()
}

/// Header credentials are already in `headers`; SigV4 is applied here, over the
/// bytes that are sent.
pub fn outbound_request(
    authenticated: Authenticated,
    url: String,
    body: &Value,
    timeout: Option<Duration>,
) -> Result<OutboundRequest, litellm_http::Error> {
    let Authenticated { headers, signer } = authenticated;
    match signer {
        None => OutboundRequest::json(url, headers, body, timeout),
        Some(signer) => OutboundRequest::signed_json(url, headers, body, timeout, &signer),
    }
}
