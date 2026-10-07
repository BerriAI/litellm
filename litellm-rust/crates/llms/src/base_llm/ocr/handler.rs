use std::sync::Arc;

use bytes::{Bytes, BytesMut};
use futures_util::future::BoxFuture;
use litellm_auth::AuthServices;
use litellm_host::interceptors::WireRequest;
use litellm_http::{
    Client, ClientVariant, HttpClientConfig, HttpClientPool,
    media::{MediaFetcher, UrlPolicy},
    outbound::{OutboundRequest, RequestSigner},
    transport,
};
use litellm_llms_types::formats::ocr::{LiteLLMOcrResponse, OcrDocument};
use litellm_secrets::source::SecretSource;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::base_llm::{
    call::{self, Failure},
    ocr::{
        error::Error,
        settings::OcrSettings,
        transformation::{
            BaseOcrConfig, DecodedOcrResponse, OcrResponseContext, PreparedOcrRequest,
            decode_request_value, decode_response,
        },
    },
};

/// The route's view of one call, handed to provider code that has to reach the
/// caller's hooks mid-flight (guardrails on the outgoing body, raw response events).
pub trait CallHooks<E>: Send + Sync {
    fn before_provider_request(&self, wire: WireRequest) -> BoxFuture<'_, Result<WireRequest, E>>;

    fn response_received<'a>(&'a self, body: &'a [u8]) -> BoxFuture<'a, Result<(), E>>;
}

#[derive(Clone)]
pub struct OcrClient {
    provider_http: Client,
    polling_http: Client,
    document_fetcher: MediaFetcher,
    auth: Arc<AuthServices>,
    settings: OcrSettings,
    secrets: Arc<dyn SecretSource>,
}

impl OcrClient {
    pub fn new(
        pool: &HttpClientPool,
        config: &HttpClientConfig,
        url_policy: UrlPolicy,
        auth: Arc<AuthServices>,
        settings: OcrSettings,
        secrets: Arc<dyn SecretSource>,
    ) -> Result<Self, litellm_http::Error> {
        Ok(Self {
            provider_http: pool.client(config, ClientVariant::Provider)?,
            polling_http: pool.client(config, ClientVariant::NoRedirect)?,
            document_fetcher: MediaFetcher::new(pool, config, url_policy)?,
            auth,
            settings,
            secrets,
        })
    }

    pub fn provider_http(&self) -> &Client {
        &self.provider_http
    }

    pub fn polling_http(&self) -> &Client {
        &self.polling_http
    }

    pub fn document_fetcher(&self) -> &MediaFetcher {
        &self.document_fetcher
    }

    pub fn auth(&self) -> &AuthServices {
        &self.auth
    }

    pub fn settings(&self) -> &OcrSettings {
        &self.settings
    }

    pub fn secret_source(&self) -> &Arc<dyn SecretSource> {
        &self.secrets
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn for_test(provider_http: Client, no_redirect_http: Client) -> Self {
        Self {
            secrets: Arc::new(
                litellm_secrets::source::EnvironmentSecrets::python_compatible(
                    provider_http.clone(),
                ),
            ),
            provider_http,
            polling_http: no_redirect_http.clone(),
            document_fetcher: MediaFetcher::for_test(no_redirect_http),
            auth: Arc::new(AuthServices::default()),
            settings: OcrSettings::default(),
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn with_settings(self, settings: OcrSettings) -> Self {
        Self { settings, ..self }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn with_secrets(self, secrets: Arc<dyn SecretSource>) -> Self {
        Self { secrets, ..self }
    }
}

/// Rust counterpart of `BaseLLMHTTPHandler.async_ocr`: prepare the provider request,
/// send it, and hand the response to the config for normalization.
pub async fn ocr<C: BaseOcrConfig>(
    config: &C,
    client: &OcrClient,
    request: &PreparedOcrRequest,
    hooks: &dyn CallHooks<Error>,
) -> Result<LiteLLMOcrResponse, Failure<Error>> {
    let (http, request_format) = call::prepare(async {
        let http = config.prepare_request(request, client, hooks).await?;
        Ok((http, request.response_format()?))
    })
    .await?;
    let url = http.url().to_string();
    let headers = http.headers().to_vec();
    let response = call::send(client.provider_http(), http)
        .await
        .map_err(|failure| {
            failure.map(|error| match error {
                Error::Upstream(upstream) => {
                    Error::Upstream(config.transform_upstream_error(upstream))
                }
                other => other,
            })
        })?;
    let context = OcrResponseContext {
        client,
        connection: &request.connection,
        hooks,
        request_format,
        url: &url,
        headers: &headers,
    };
    call::receive(config.async_transform_ocr_response(&request.model, response, context)).await
}

pub async fn read_json_response<T: DeserializeOwned>(
    response: reqwest::Response,
    native: bool,
    max_response_bytes: usize,
) -> Result<DecodedOcrResponse<T>, Error> {
    let bytes = read_response_bytes(response, max_response_bytes).await?;
    decode_response(&bytes, native)
}

/// Reads a success body of at most `limit` bytes. A non-success answer, which a follow-up
/// request such as a poll or an upload may receive, is reported as the provider's answer.
pub async fn read_response_bytes(
    mut response: reqwest::Response,
    limit: usize,
) -> Result<Bytes, Error> {
    if !response.status().is_success() {
        return Err(call::upstream_response(response, limit).await?.into());
    }
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err(Error::TooLarge { limit });
    }
    let mut bytes = BytesMut::new();
    while let Some(chunk) = response.chunk().await.map_err(transport::Error::from)? {
        if chunk.len() > limit - bytes.len() {
            return Err(Error::TooLarge { limit });
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes.freeze())
}

pub async fn transform_request_body<C: BaseOcrConfig, B: Serialize>(
    config: &C,
    request: &PreparedOcrRequest,
    url: &str,
    headers: &[(String, String)],
    body: B,
    signer: Option<&dyn RequestSigner>,
    hooks: &dyn CallHooks<Error>,
) -> Result<OutboundRequest, Error> {
    let composed = litellm_core_utils::call_arguments::compose_body(
        &request.optional_params,
        &body,
        config.get_supported_ocr_params(&request.model),
    )?;
    config.validate_request_body(&composed)?;
    let changed = hooks
        .before_provider_request(wire_request(url, headers, composed))
        .await?;
    if !changed.body.is_object() {
        return Err(Error::RequestField {
            path: "guardrail.body".into(),
        });
    }
    config.validate_request_body(&changed.body)?;
    let timeout = Some(request.connection.timeout);
    Ok(match signer {
        Some(signer) => OutboundRequest::signed_json(
            url.into(),
            changed.headers,
            &changed.body,
            timeout,
            signer,
        ),
        None => OutboundRequest::json(url.into(), changed.headers, &changed.body, timeout),
    }?)
}

fn wire_request(url: &str, headers: &[(String, String)], body: Value) -> WireRequest {
    WireRequest {
        url: url.into(),
        headers: headers.to_vec(),
        body,
    }
}

pub fn build_http_request(
    request: &PreparedOcrRequest,
    url: String,
    headers: Vec<(String, String)>,
    body: &impl Serialize,
) -> Result<OutboundRequest, Error> {
    Ok(OutboundRequest::json(
        url,
        headers,
        body,
        Some(request.connection.timeout),
    )?)
}

pub async fn guardrail_document(
    request: &PreparedOcrRequest,
    url: &str,
    headers: &[(String, String)],
    hooks: &dyn CallHooks<Error>,
) -> Result<(OcrDocument, Vec<(String, String)>), Error> {
    let body = serde_json::to_value(&request.document).map_err(|_| Error::RequestField {
        path: "document".into(),
    })?;
    let changed = hooks
        .before_provider_request(wire_request(url, headers, body))
        .await?;
    let document = decode_request_value(changed.body, "guardrail.document")?;
    Ok((document, changed.headers))
}

pub fn body_document(body: &Value) -> Result<OcrDocument, Error> {
    let document = body
        .get("document")
        .and_then(Value::as_object)
        .ok_or_else(|| Error::RequestField {
            path: "body.document".into(),
        })?;
    let source = document
        .iter()
        .filter(|(name, _)| matches!(name.as_str(), "type" | "image_url" | "document_url"))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    decode_request_value(Value::Object(source), "body.document")
}

#[cfg(test)]
mod tests {
    use litellm_host::failure::UpstreamResponse;

    use super::*;

    #[rstest::rstest]
    #[tokio::test]
    async fn a_follow_up_request_answered_with_an_error_is_the_providers_answer() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::any())
            .respond_with(wiremock::ResponseTemplate::new(503).set_body_string("busy"))
            .mount(&server)
            .await;
        let response = litellm_http::Client::plain_for_test()
            .get(server.uri())
            .send()
            .await
            .unwrap();
        let error = read_response_bytes(response, 1024).await.unwrap_err();
        let Error::Upstream(UpstreamResponse { status, body, .. }) = error else {
            panic!("{error:?}");
        };
        assert_eq!((status, body.as_str()), (503, "busy"));
    }
}
