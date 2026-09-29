use reqwest::Method;
use serde_json::Value;
use std::time::Duration;

use litellm_core_utils::url_utils::{ApiUrl, Complete};
use litellm_host::interceptors::WireRequest;
use litellm_http::outbound::RequestSigner;

use litellm_http::outbound::OutboundRequest;
use litellm_llms::base_llm::auth::Authenticated;

#[tracing::instrument(
    name = "litellm.provider.send",
    level = "debug",
    skip_all,
    fields(status)
)]
pub(crate) async fn send(
    request: OutboundRequest,
    client: &litellm_http::Client,
) -> Result<reqwest::Response, reqwest::Error> {
    request.send(client).await.inspect(|response| {
        tracing::Span::current().record("status", response.status().as_u16());
    })
}

/// Header credentials are already in `headers`; SigV4 is applied here, over the
/// bytes that are sent.
pub(crate) fn outbound_request(
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

pub(crate) fn endpoint_request(
    authenticated: Authenticated,
    method: Method,
    url: ApiUrl<Complete>,
    body: &Value,
    timeout: Option<Duration>,
) -> Result<OutboundRequest, litellm_http::Error> {
    let Authenticated { headers, signer } = authenticated;
    OutboundRequest::endpoint_json(
        method,
        url,
        headers,
        body,
        timeout,
        signer.as_ref().map(|signer| signer as &dyn RequestSigner),
    )
}

pub(crate) fn validate_wire_url(
    original: ApiUrl<Complete>,
    wire: WireRequest,
) -> Result<(WireRequest, ApiUrl<Complete>), litellm_http::Error> {
    let url = if original.as_url().as_str() == wire.url {
        original
    } else {
        ApiUrl::parse_exact(&wire.url)?
    };
    Ok((
        WireRequest {
            url: url.as_url().to_string(),
            ..wire
        },
        url,
    ))
}
