use bytes::{Bytes, BytesMut};
use futures_util::future::BoxFuture;
use litellm_auth_gcp::VertexAuth;
use litellm_host::event::WireRequest;
use litellm_http::{ClientVariant, HttpClientConfig, HttpClientPool};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::{
    base_llm::ocr::{
        error::Error,
        transformation::{
            BaseOcrConfig, DecodedOcrResponse, LiteLLMOcrResponse, OcrDocument, OcrResponseContext,
            PreparedOcrRequest, decode_request_value, decode_response,
        },
    },
    custom_httpx::{
        http_handler::{HeaderPolicy, execute_http_request, with_headers},
        media::MediaFetcher,
        transport,
    },
};

/// The route's view of one call, handed to provider code that has to reach the
/// caller's hooks mid-flight (guardrails on the outgoing body, raw response events).
pub trait CallHooks<E>: Send + Sync {
    fn before_send(&self, wire: WireRequest) -> BoxFuture<'_, Result<WireRequest, E>>;

    fn response_received<'a>(&'a self, body: &'a [u8]) -> BoxFuture<'a, Result<(), E>>;
}

#[derive(Clone)]
pub struct OcrClient {
    provider_http: reqwest::Client,
    polling_http: reqwest::Client,
    document_fetcher: MediaFetcher,
    vertex_auth: VertexAuth,
}

impl OcrClient {
    pub fn new(
        pool: &HttpClientPool,
        config: &HttpClientConfig,
        vertex_auth: VertexAuth,
    ) -> Result<Self, transport::Error> {
        Ok(Self {
            provider_http: pool.client(config, ClientVariant::Provider)?,
            polling_http: pool.client(config, ClientVariant::NoRedirect)?,
            document_fetcher: MediaFetcher::new(pool, config)?,
            vertex_auth,
        })
    }

    pub fn provider_http(&self) -> &reqwest::Client {
        &self.provider_http
    }

    pub fn polling_http(&self) -> &reqwest::Client {
        &self.polling_http
    }

    pub fn document_fetcher(&self) -> &MediaFetcher {
        &self.document_fetcher
    }

    pub fn vertex_auth(&self) -> &VertexAuth {
        &self.vertex_auth
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn for_test(provider_http: reqwest::Client, document_http: reqwest::Client) -> Self {
        Self {
            provider_http,
            polling_http: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .expect("test polling client builds"),
            document_fetcher: MediaFetcher::for_test(document_http),
            vertex_auth: VertexAuth::default(),
        }
    }
}

/// Rust counterpart of `BaseLLMHTTPHandler.async_ocr`: prepare the provider request,
/// send it, and hand the response to the config for normalization.
pub async fn ocr<C: BaseOcrConfig>(
    config: &C,
    client: &OcrClient,
    request: &PreparedOcrRequest,
    hooks: &dyn CallHooks<Error>,
) -> Result<LiteLLMOcrResponse, Error> {
    let http = config.prepare_request(request, client, hooks).await?;
    let url = http.url().to_string();
    let headers = request_headers(&http)?;
    let response = execute_http_request(client.provider_http(), http)
        .await
        .map_err(transport_error)?;
    if !response.status().is_success() {
        let headers = response
            .headers()
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value| (name.to_string(), value.to_string()))
            })
            .collect();
        return match read_response_bytes(response, request.connection.max_response_bytes).await {
            Err(Error::Transport(transport::Error::Http { status, body })) => {
                Err(config.get_error_class(body, status, headers))
            }
            Err(error) => Err(error),
            Ok(_) => unreachable!("non-success response produces an HTTP error"),
        };
    }
    let context = OcrResponseContext {
        client,
        connection: &request.connection,
        hooks,
        request_format: request.response_format()?,
        url: &url,
        headers: &headers,
    };
    config
        .async_transform_ocr_response(&request.model, response, context)
        .await
}

fn request_headers(request: &reqwest::Request) -> Result<Vec<(String, String)>, Error> {
    request
        .headers()
        .iter()
        .map(|(name, value)| {
            value
                .to_str()
                .map(|value| (name.to_string(), value.to_string()))
                .map_err(|_| Error::RequestField {
                    path: "headers".into(),
                })
        })
        .collect()
}

pub async fn read_json_response<T: DeserializeOwned>(
    response: reqwest::Response,
    native: bool,
    max_response_bytes: usize,
) -> Result<DecodedOcrResponse<T>, Error> {
    let bytes = read_response_bytes(response, max_response_bytes).await?;
    decode_response(&bytes, native)
}

pub async fn read_response_bytes(
    mut response: reqwest::Response,
    limit: usize,
) -> Result<Bytes, Error> {
    let status = response.status();
    if status.is_success()
        && response
            .content_length()
            .is_some_and(|length| length > limit as u64)
    {
        return Err(Error::TooLarge { limit });
    }
    let mut bytes = BytesMut::new();
    while let Some(chunk) = response.chunk().await.map_err(transport_error)? {
        let remaining = limit.saturating_sub(bytes.len());
        if status.is_success() && chunk.len() > remaining {
            return Err(Error::TooLarge { limit });
        }
        bytes.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
        if !status.is_success() && bytes.len() == limit {
            break;
        }
    }
    if !status.is_success() {
        return Err(transport::Error::Http {
            status: status.as_u16(),
            body: String::from_utf8_lossy(&bytes).into_owned(),
        }
        .into());
    }
    Ok(bytes.freeze())
}

pub fn transport_error(error: reqwest::Error) -> Error {
    if error.is_timeout() {
        return Error::Transport(transport::Error::Http {
            status: 408,
            body: "OCR request timed out".into(),
        });
    }
    transport::Error::from(error).into()
}

pub async fn transform_request_body<C: BaseOcrConfig, B: Serialize>(
    config: &C,
    client: &OcrClient,
    request: &PreparedOcrRequest,
    url: &str,
    headers: &[(String, String)],
    body: B,
    hooks: &dyn CallHooks<Error>,
) -> Result<reqwest::Request, Error> {
    let composed = litellm_core_utils::call_arguments::compose_body(
        &request.optional_params,
        &body,
        config.get_supported_ocr_params(&request.model),
    )?;
    config.validate_request_body(&composed)?;
    let changed = hooks
        .before_send(wire_request(url, headers, composed))
        .await?;
    if !changed.body.is_object() {
        return Err(Error::RequestField {
            path: "guardrail.body".into(),
        });
    }
    config.validate_request_body(&changed.body)?;
    build_http_request(client, request, url, &changed.headers, &changed.body)
}

fn wire_request(url: &str, headers: &[(String, String)], body: Value) -> WireRequest {
    WireRequest {
        url: url.into(),
        headers: headers.to_vec(),
        body,
    }
}

pub fn build_http_request<B: Serialize>(
    client: &OcrClient,
    request: &PreparedOcrRequest,
    url: &str,
    headers: &[(String, String)],
    body: &B,
) -> Result<reqwest::Request, Error> {
    let builder = client
        .provider_http()
        .post(url)
        .json(body)
        .timeout(request.connection.timeout);
    with_headers(builder, headers, HeaderPolicy::All)
        .build()
        .map_err(transport::Error::from)
        .map_err(Error::from)
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
    let changed = hooks.before_send(wire_request(url, headers, body)).await?;
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
    use std::time::Duration;

    use super::*;

    #[tokio::test]
    async fn request_timeout_has_an_http_408_status() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let _connection = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_secs(1)).await;
        });
        let error = reqwest::Client::new()
            .get(format!("http://{address}"))
            .timeout(Duration::from_millis(10))
            .send()
            .await
            .unwrap_err();
        assert!(matches!(
            transport_error(error),
            Error::Transport(transport::Error::Http { status: 408, .. })
        ));
        server.abort();
    }
}
