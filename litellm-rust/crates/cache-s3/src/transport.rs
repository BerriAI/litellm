use aws_smithy_runtime_api::client::{
    http::{
        HttpClient, HttpConnector, HttpConnectorFuture, HttpConnectorSettings, SharedHttpConnector,
    },
    orchestrator::HttpRequest,
    result::ConnectorError,
    runtime_components::RuntimeComponents,
};
use aws_smithy_types::body::SdkBody;

/// Sends the SDK's requests through a host-supplied `reqwest::Client`, so proxy, TLS, timeout
/// and pool settings come from the host instead of the SDK's own hyper client.
#[derive(Clone, Debug)]
pub(crate) struct ReqwestHttpClient(pub(crate) reqwest::Client);

impl HttpClient for ReqwestHttpClient {
    fn http_connector(
        &self,
        _: &HttpConnectorSettings,
        _: &RuntimeComponents,
    ) -> SharedHttpConnector {
        SharedHttpConnector::new(self.clone())
    }
}

impl HttpConnector for ReqwestHttpClient {
    fn call(&self, request: HttpRequest) -> HttpConnectorFuture {
        let client = self.0.clone();
        HttpConnectorFuture::new(async move {
            let request = request
                .try_into_http1x()
                .map_err(|error| ConnectorError::other(error.into(), None))?
                .map(reqwest::Body::wrap);
            let request = reqwest::Request::try_from(request)
                .map_err(|error| ConnectorError::other(error.into(), None))?;
            let response = client.execute(request).await.map_err(|error| {
                if error.is_timeout() {
                    ConnectorError::timeout(error.into())
                } else {
                    ConnectorError::io(error.into())
                }
            })?;
            let response = http::Response::from(response).map(SdkBody::from_body_1_x);
            response
                .try_into()
                .map_err(|error: aws_smithy_runtime_api::http::HttpError| {
                    ConnectorError::other(error.into(), None)
                })
        })
    }
}
