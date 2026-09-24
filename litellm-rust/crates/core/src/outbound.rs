use std::time::Duration;

use litellm_auth::RequestAuth;
use litellm_auth_aws::SigV4Signer;
use litellm_http::outbound::OutboundRequest;
use serde_json::{Map, Value};

/// Header credentials are already in `headers`; SigV4 is applied here, over the
/// bytes that are sent.
pub(crate) async fn outbound_request<E>(
    auth: &RequestAuth,
    url: String,
    headers: Vec<(String, String)>,
    body: &Value,
    timeout: Option<Duration>,
    optional_params: &Map<String, Value>,
) -> Result<OutboundRequest, E>
where
    E: From<litellm_http::Error> + From<litellm_auth_aws::Error>,
{
    let RequestAuth::AwsSigV4 { region, service } = auth else {
        return Ok(OutboundRequest::json(url, headers, body, timeout)?);
    };
    let env_lookup = |key: &str| std::env::var(key).ok();
    let signer =
        SigV4Signer::resolve(region.clone(), service, optional_params, &env_lookup).await?;
    Ok(OutboundRequest::signed_json(
        url, headers, body, timeout, &signer,
    )?)
}
