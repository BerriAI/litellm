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

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serde_json::json;

    #[rstest]
    #[case::caller_url(None, "https://caller.example/messages")]
    #[case::authenticated_url(
        Some("https://session.example/messages"),
        "https://session.example/messages"
    )]
    fn authenticated_endpoints_outrank_the_requested_url(
        #[case] authenticated_url: Option<&str>,
        #[case] expected: &str,
    ) {
        let request = outbound_request(
            Authenticated {
                url: authenticated_url.map(str::to_string),
                headers: vec![("authorization".into(), "Bearer session".into())],
                signer: None,
            },
            "https://caller.example/messages".into(),
            &json!({"model": "native-test"}),
            None,
        )
        .unwrap();
        assert_eq!(request.url(), expected);
        assert_eq!(request.header("authorization"), Some("Bearer session"));
        assert_eq!(
            serde_json::from_slice::<Value>(request.body()).unwrap(),
            json!({"model": "native-test"})
        );
    }
}
