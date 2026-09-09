pub mod request;
pub mod transformation;
pub mod types;

use serde_json::Value;
use std::time::Duration;

use reqwest::header::{CONTENT_ENCODING, HeaderMap, HeaderName, HeaderValue};

use crate::Error;
use crate::error::json_type_name;
use crate::http_utils::{HttpClientProfile, has_header, http_client, http_request};

pub use types::{
    OcrAdmissionRequest, OcrEndpoint, OcrPreCallRequest, OcrResponseData, SettledOcrRequest,
};
use types::{OcrDocument, OcrDocumentProjection};

use crate::lifecycle::{
    CallLifecycleContext, Clock, DeploymentFailureHooks, ExecutedCall, TerminalDispatcher,
};

pub trait OcrServices: TerminalDispatcher + Clock + DeploymentFailureHooks {}

impl<T> OcrServices for T where T: TerminalDispatcher + Clock + DeploymentFailureHooks {}

pub struct DefaultOcrServices;

impl DeploymentFailureHooks for DefaultOcrServices {}

impl Default for DefaultOcrServices {
    fn default() -> Self {
        Self
    }
}

impl Clock for DefaultOcrServices {
    fn now(&self) -> f64 {
        crate::lifecycle::SystemClock.now()
    }
}

impl TerminalDispatcher for DefaultOcrServices {
    fn dispatch<'a>(
        &'a self,
        _: &'a crate::lifecycle::TerminalRecord,
    ) -> crate::integrations::custom_logger::LogFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

pub async fn ocr<S: OcrServices>(
    services: &S,
    permit: crate::lifecycle::program::ProviderPermit<crate::lifecycle::ocr::OcrRoute>,
    request: SettledOcrRequest,
    context: CallLifecycleContext,
) -> ExecutedCall<Value, Error> {
    let start_time = services.now();
    let mut completion = crate::lifecycle::completion::CompletionOwner::new(
        context.clone(),
        start_time,
        services,
        services,
    );
    let result = permit.ocr(request, context.clone()).await;
    if let ExecutedCall::Failure { error, .. } = &result {
        let _ = services
            .async_post_call_failure_deployment_hook(&context, error)
            .await;
    }
    completion.finish(result.terminal());
    crate::lifecycle::completion::dispatch(services, result.terminal()).await;
    result
}

pub(crate) async fn send(request: SettledOcrRequest) -> Result<OcrResponseData, Error> {
    let SettledOcrRequest { endpoint, http } = request;
    let body: Value = serde_json::from_slice(http.body())
        .map_err(|_| Error::InvalidRequest("could not decode settled OCR request".into()))?;
    let config = request::provider_config(&endpoint.custom_llm_provider, &endpoint.model)?;
    request::validate_capabilities(config)?;
    let object = body.as_object().ok_or_else(|| Error::InvalidType {
        expected: "object",
        actual: json_type_name(&body),
    })?;
    if config.document_projection() != OcrDocumentProjection::Transformed {
        let document = object
            .get("document")
            .ok_or(Error::MissingField("document"))?;
        let document: OcrDocument = serde_json::from_value(document.clone())
            .map_err(|_| Error::InvalidRequest("invalid OCR document".into()))?;
        document.validate(config.requires_data_uri_document())?;
    }
    if object.get("stream").and_then(Value::as_bool) == Some(true) {
        return Err(Error::Unsupported("OCR streaming response handling"));
    }
    if http.headers().iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("content-encoding") && !value.eq_ignore_ascii_case("identity")
    }) {
        return Err(Error::Unsupported("compressed OCR request"));
    }
    let headers = http
        .headers()
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .chain(
            [
                ("Content-Type", "application/json"),
                ("Accept-Encoding", "identity"),
            ]
            .into_iter()
            .filter(|(name, _)| !has_header(http.headers(), name)),
        )
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect();
    let timeout = Duration::try_from_secs_f64(endpoint.timeout_seconds)
        .ok()
        .filter(|timeout| !timeout.is_zero())
        .ok_or_else(|| Error::InvalidRequest("timeout must be positive and finite".into()))?;
    let (body, headers) = http.replace_headers(headers).into_parts();
    let headers = headers
        .into_iter()
        .try_fold(HeaderMap::new(), |mut map, (name, value)| {
            let name = HeaderName::from_bytes(name.as_bytes())
                .map_err(|_| Error::InvalidRequest("invalid header name".into()))?;
            let value = HeaderValue::from_bytes(value.as_bytes())
                .map_err(|_| Error::InvalidRequest("invalid header value".into()))?;
            map.append(name, value);
            Ok::<_, Error>(map)
        })?;
    let client = http_client(HttpClientProfile::NoRedirectsOrDecompression)
        .map_err(|_| Error::Network("could not initialize HTTP client".into()))?;
    let response = http_request(
        client
            .post(&endpoint.url)
            .headers(headers)
            .body(body)
            .timeout(timeout),
    )
    .await
    .map_err(|_| Error::Network("transport failed".into()))?;
    let status = response.status();
    let response_headers = response.headers().clone();
    let content = response
        .bytes()
        .await
        .map_err(|_| Error::Network("could not read response".into()))?;
    if !status.is_success() {
        return Err(Error::Http {
            status: status.as_u16(),
            body: "OCR provider request failed".into(),
        });
    }
    if response_headers
        .get_all(CONTENT_ENCODING)
        .iter()
        .any(|value| {
            value
                .as_bytes()
                .split(|byte| *byte == b',')
                .any(|encoding| !encoding.trim_ascii().eq_ignore_ascii_case(b"identity"))
        })
    {
        return Err(Error::Unsupported("compressed OCR response"));
    }
    let response_json = serde_json::from_slice(&content)
        .map_err(|_| Error::InvalidResponse("invalid OCR JSON response".into()))?;
    config.transform_ocr_response(&endpoint.model, response_json)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::net::TcpListener;

    use super::*;
    use crate::lifecycle::{AuthorizedBody, WireBody};

    #[derive(Default)]
    struct FailureObserver(Mutex<Vec<&'static str>>);

    impl Clock for FailureObserver {
        fn now(&self) -> f64 {
            1.0
        }
    }

    impl crate::lifecycle::DeploymentFailureHooks for FailureObserver {
        fn async_post_call_failure_deployment_hook<'a>(
            &'a self,
            _: &'a CallLifecycleContext,
            _: &'a Error,
        ) -> crate::lifecycle::CallbackFuture<'a, Result<(), Error>> {
            Box::pin(async move {
                self.0.lock().unwrap().push("deployment_failure");
                Err(Error::InvalidRequest("observer failed".into()))
            })
        }
    }

    impl TerminalDispatcher for FailureObserver {
        fn dispatch<'a>(
            &'a self,
            _: &'a crate::lifecycle::TerminalRecord,
        ) -> crate::integrations::custom_logger::LogFuture<'a> {
            Box::pin(async move {
                self.0.lock().unwrap().push("terminal");
                Ok(())
            })
        }
    }

    #[tokio::test]
    async fn validation_failure_notifies_supplied_deployment_hook() {
        let services = FailureObserver::default();
        let request = SettledOcrRequest {
            endpoint: OcrEndpoint {
                model: "mistral-ocr-latest".into(),
                custom_llm_provider: "mistral".into(),
                url: "http://127.0.0.1:1".into(),
                timeout_seconds: 1.0,
            },
            http: AuthorizedBody::new(
                WireBody::from_serialized("{\"stream\":true,\"document\":{\"type\":\"document_url\",\"document_url\":\"data:application/pdf;base64,cGRm\"}}".into()),
                vec![],
            ).settle(),
        };
        let mut owner = crate::lifecycle::CallLifecycle::asynchronous();
        while owner.operation() != crate::lifecycle::program::Operation::Send {
            let ticket = owner.issue().unwrap();
            owner
                .complete_operation(
                    ticket,
                    crate::lifecycle::Outcome::Success,
                    Default::default(),
                )
                .unwrap();
        }
        let ticket = owner.issue().unwrap();
        let permit = owner.provider_permit(&ticket).unwrap();
        let result = ocr(
            &services,
            permit,
            request,
            CallLifecycleContext::new("ocr", "mistral-ocr-latest", "mistral", "contract-call"),
        )
        .await
        .into_result();

        assert!(matches!(
            result,
            Err(Error::Unsupported("OCR streaming response handling"))
        ));
        assert_eq!(
            *services.0.lock().unwrap(),
            ["deployment_failure", "terminal"]
        );
    }

    #[tokio::test]
    async fn validation_does_not_reserialize_settled_bytes() {
        let bytes = "{ \"document\": {\"type\":\"document_url\",\"document_url\":\"data:application/pdf;base64,cGRm\"}, \"model\": \"mistral-ocr-latest\" }\n";
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1/ocr", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut socket = BufReader::new(socket);
            let mut line = String::new();
            loop {
                line.clear();
                assert_ne!(socket.read_line(&mut line).await.unwrap(), 0);
                if line == "\r\n" {
                    break;
                }
            }
            let mut received = vec![0; bytes.len()];
            socket.read_exact(&mut received).await.unwrap();
            assert_eq!(received, bytes.as_bytes());
            socket
                .get_mut()
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
                .await
                .unwrap();
        });
        let request = SettledOcrRequest {
            endpoint: OcrEndpoint {
                model: "mistral-ocr-latest".into(),
                custom_llm_provider: "mistral".into(),
                url,
                timeout_seconds: 2.0,
            },
            http: AuthorizedBody::new(WireBody::from_serialized(bytes.into()), vec![]).settle(),
        };
        send(request).await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap();
    }
}
