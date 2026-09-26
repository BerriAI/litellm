use azure_core::{
    error::ErrorKind,
    http::{
        AsyncRawResponse, Body, HttpClient, Request,
        headers::{HeaderName, HeaderValue, Headers},
    },
};
use futures_util::TryStreamExt;

#[derive(Debug)]
pub struct ReqwestTransport(pub litellm_http::Client);

#[async_trait::async_trait]
impl HttpClient for ReqwestTransport {
    async fn execute_request(&self, request: &Request) -> azure_core::Result<AsyncRawResponse> {
        let method = reqwest::Method::from_bytes(request.method().as_ref().as_bytes())
            .map_err(|error| azure_core::Error::new(ErrorKind::Other, error))?;
        let mut outgoing = self.0.request(method, request.url().as_str());
        for (name, value) in request.headers().iter() {
            outgoing = outgoing.header(name.as_str(), value.as_str());
        }
        let outgoing = match request.body().clone() {
            Body::Bytes(bytes) => outgoing.body(bytes),
            Body::SeekableStream(stream) => outgoing.body(reqwest::Body::wrap_stream(stream)),
        };
        let response = outgoing.send().await.map_err(|error| {
            let kind = if error.is_connect() {
                ErrorKind::Connection
            } else {
                ErrorKind::Io
            };
            azure_core::Error::new(kind, error)
        })?;
        let status = response.status().as_u16().into();
        let mut headers = Headers::new();
        for (name, value) in response.headers() {
            if let Ok(value) = value.to_str() {
                headers.insert(
                    HeaderName::from(name.as_str().to_owned()),
                    HeaderValue::from(value.to_owned()),
                );
            }
        }
        let body = response
            .bytes_stream()
            .map_err(|error| azure_core::Error::new(ErrorKind::Io, error));
        Ok(AsyncRawResponse::new(status, headers, Box::pin(body)))
    }
}
