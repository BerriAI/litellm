use litellm_http::{outbound::OutboundRequest, request::truncate_error_body};
use litellm_llms::base_llm::chat::transformation::ProviderChatResponseData;
use litellm_types::utils::ChatCompletionsResponse;
use serde_json::Value;

use super::{Error, client::http_client, prepare::prepare_provider_request};
use crate::chat_completions::types::{
    ProviderChatCompletionsRequest, ResolvedChatCompletionsRequest,
};

pub(super) async fn execute_chat_completions_provider_call(
    request: ResolvedChatCompletionsRequest<'_>,
) -> Result<ChatCompletionsResponse, Error> {
    let request = prepare_provider_request(request)?;
    let outbound = outbound_request(&request).await?;

    let response = outbound.send(http_client()).await.map_err(|err| {
        // Failing to establish the connection means the request never went out,
        // so the host can still serve it. Everything else here, a timeout
        // above all, may have reached the provider and been answered.
        if err.is_connect() || err.is_builder() {
            Error::Transport(litellm_http::transport::Error::Connect(err.to_string()))
        } else {
            Error::Transport(litellm_http::transport::Error::Network(err.to_string()))
        }
    })?;

    let status = response.status();
    let text = response.text().await.map_err(|err| {
        Error::Transport(litellm_http::transport::Error::Network(err.to_string()))
    })?;

    if !status.is_success() {
        return Err(Error::Transport(litellm_http::transport::Error::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        }));
    }

    let body: Value = serde_json::from_str(&text).map_err(|err| {
        Error::InvalidResponse(format!("invalid chat completions response JSON: {err}"))
    })?;
    request
        .config
        .transform_response(&request.model, ProviderChatResponseData { body })
        .map_err(Error::from)
        .map_err(as_response_error)
}

/// Re-tag an error raised while normalizing a response the provider already
/// returned.
///
/// A config reports the same variants on either side of the call: a missing
/// field or an unsupported block can mean "this request cannot be translated"
/// during prepare and "this response cannot be normalized" here. Only the
/// second kind has already been billed, and a host that keeps a reference
/// implementation must not retry those, so collapse them to one variant that
/// can only mean the provider was already called.
pub(super) fn as_response_error(err: Error) -> Error {
    match err {
        already @ (Error::InvalidResponse(_)
        | Error::Transport(litellm_http::transport::Error::Http { .. })) => already,
        other => Error::InvalidResponse(other.to_string()),
    }
}

pub(super) async fn outbound_request(
    request: &ProviderChatCompletionsRequest,
) -> Result<OutboundRequest, Error> {
    crate::outbound::outbound_request(
        &request.auth,
        request.url.clone(),
        request.upstream_headers.clone(),
        &request.body,
        request.timeout,
        &request.optional_params,
    )
    .await
    .map_err(|error| match error {
        // Python drops the caller's copy and prefers a forwarded Authorization
        // over the signature, so leave the request to it.
        Error::Http(litellm_http::Error::ComputedHeader(_)) => {
            Error::Unsupported("request forwards a header AWS SigV4 computes")
        }
        other => other,
    })
}

#[cfg(test)]
mod tests {
    use super::{Error, as_response_error};

    #[test]
    fn response_errors_collapse_to_one_variant_that_can_only_mean_already_sent() {
        for original in [
            Error::MissingField("usage"),
            Error::Unsupported("non-text response content block"),
            Error::InvalidRequest("whatever".to_string()),
            Error::Auth(litellm_auth::Error::InvalidHeader),
        ] {
            let label = format!("{original:?}");
            assert!(
                matches!(as_response_error(original), Error::InvalidResponse(_)),
                "{label} must not stay retryable once the provider has answered"
            );
        }
        let upstream = Error::Transport(litellm_http::transport::Error::Http {
            status: 500,
            body: "boom".to_string(),
        });
        assert_eq!(as_response_error(upstream.clone()), upstream);
    }
}
