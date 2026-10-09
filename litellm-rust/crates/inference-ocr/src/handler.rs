use futures_util::future::BoxFuture;
use litellm_host::interceptors::{Interceptors, RawResponse, RequestContext, WireRequest};
use litellm_host::{lifecycle::ExecutionEvent, observation::ObservationSender};
use litellm_inference::provider::ResolvedProvider;
use litellm_llms::base_llm::ocr::{
    error::Error,
    handler::{CallHooks, OcrClient},
    transformation::{OcrConnection, PreparedOcrRequest},
};
use litellm_llms_types::formats::ocr::LiteLLMOcrResponse;
use litellm_secrets::source::Secrets;
use serde_json::Value;

use super::{arguments::is_secret_param, provider_config::OcrConfigKind};
use crate::types::{LiteLLMOcrRequest, ResolvedOcrRequest};

#[allow(clippy::too_many_arguments)] // Required by the shared route execution signature.
pub(crate) async fn execute(
    client: &OcrClient,
    config: OcrConfigKind,
    provider: ResolvedProvider<'_>,
    call: ResolvedOcrRequest,
    secrets: Secrets,
    host: &impl Interceptors<Error>,
    caller_document: bool,
    observers: Option<&ObservationSender>,
) -> Result<LiteLLMOcrResponse, Error> {
    call.response_format()?;
    let LiteLLMOcrRequest {
        document,
        credentials,
        transport,
        optional_params,
        input_sources,
        azure_ad_token_provider,
        ..
    } = call;
    let resolved = config.resolve_credentials(credentials, &secrets);
    let request = PreparedOcrRequest {
        model: provider.model.to_owned(),
        document,
        connection: OcrConnection::new(resolved, transport, client.settings().clone(), secrets),
        caller_document,
        optional_params,
        input_sources,
        azure_ad_token_provider,
    };
    let interceptors = OcrCallHooks::new(host, &request, config, observers);
    config.ocr(client, &request, &interceptors).await
}

struct OcrCallHooks<'a, H> {
    interceptors: &'a H,
    context: RequestContext,
    observers: Option<&'a ObservationSender>,
}

impl<'a, H> OcrCallHooks<'a, H> {
    fn new(
        interceptors: &'a H,
        request: &PreparedOcrRequest,
        config: OcrConfigKind,
        observers: Option<&'a ObservationSender>,
    ) -> Self {
        Self {
            interceptors,
            observers,
            context: RequestContext {
                model: request.model.clone(),
                custom_llm_provider: <&str>::from(config.provider()).to_owned(),
                optional_params: Value::Object(request.optional_params.clone().into()),
                secret_fields: request
                    .optional_params
                    .keys()
                    .filter(|name| is_secret_param(name))
                    .cloned()
                    .collect(),
                api_key: request.connection.api_key.clone(),
            },
        }
    }
}

impl<H: Interceptors<Error>> CallHooks<Error> for OcrCallHooks<'_, H> {
    fn before_provider_request(
        &self,
        wire: WireRequest,
    ) -> BoxFuture<'_, Result<WireRequest, Error>> {
        Box::pin(
            self.interceptors
                .before_provider_request(wire, self.context.clone()),
        )
    }

    fn response_received<'a>(&'a self, body: &'a [u8]) -> BoxFuture<'a, Result<(), Error>> {
        let raw = RawResponse {
            body: String::from_utf8_lossy(body).into_owned(),
        };
        if let Some(observers) = self.observers {
            observers.emit(litellm_host::lifecycle::CallEvent::Execution(
                ExecutionEvent::ProviderResponseReceived { raw: raw.clone() },
            ));
        }
        Box::pin(self.interceptors.after_provider_response(raw))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use litellm_core_utils::call_arguments::{CallArguments, compose_body, parse_options};
    use litellm_llms_types::formats::ocr::OcrResponseFormat;
    use rstest::rstest;
    use serde_json::{Value, json};
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::any};

    use super::*;
    use crate::{
        client::OcrRoute,
        types::LiteLLMOcrRequest,
        wire::{OcrWireRequest, decode_request},
    };

    fn client() -> OcrClient {
        OcrClient::for_test(
            litellm_http::Client::plain_for_test(),
            litellm_http::Client::no_redirect_for_test(),
        )
        .with_secrets(litellm_inference_testing::no_secrets())
    }

    fn request(model: &str, base: &str, document: Value, options: Value) -> LiteLLMOcrRequest {
        decode_request(OcrWireRequest {
            model: model.into(),
            document,
            api_key: Some(litellm_auth::SecretValue::new("test-key")),
            api_base: Some(base.into()),
            custom_llm_provider: None,
            extra_headers: None,
            optional_params: options.as_object().unwrap().clone(),
            input_sources: Default::default(),
            timeout_seconds: Some(2.0),
        })
        .unwrap()
    }

    async fn run_request(request: LiteLLMOcrRequest) -> Result<LiteLLMOcrResponse, Error> {
        OcrRoute::new(client()).execute(request, &(), None).await
    }

    async fn upstream(request_count: u64, response: Value) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(any())
            .respond_with(ResponseTemplate::new(200).set_body_json(response))
            .up_to_n_times(request_count)
            .mount(&server)
            .await;
        server
    }

    fn image(url: &str) -> Value {
        json!({"type": "image_url", "image_url": url})
    }

    fn body(request: &wiremock::Request) -> Value {
        serde_json::from_slice(&request.body).unwrap()
    }

    fn assert_mistral_request(request: &wiremock::Request) {
        assert_eq!(
            request
                .headers
                .get("authorization")
                .unwrap()
                .to_str()
                .unwrap(),
            "Bearer test-key"
        );
        assert_eq!(
            request
                .headers
                .get("content-type")
                .unwrap()
                .to_str()
                .unwrap(),
            "application/json"
        );
        assert_eq!(
            body(request),
            json!({
                "model": "mistral-ocr-maas",
                "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
                "pages": [0, 2],
                "include_image_base64": true,
                "unknown": "preserved"
            })
        );
    }

    #[rstest]
    #[tokio::test]
    async fn cohere_body_keeps_native_document_fields_and_untyped_overrides() {
        let upstream = upstream(
            1,
            json!({"pages": [{"index": 0, "markdown": {"content": "hello"}}]}),
        )
        .await;
        let request = request(
            "cohere/parse",
            &upstream.uri(),
            image("https://example.com/original.png"),
            json!({
                "output_format": "markdown", "timeout": 30,
                "extra_body": {
                    "output_format": {"future": true},
                    "document": {"type": "image_url", "image_url": "https://example.com/a.png",
                        "provider_options": {"nested": [false, 0, null]}}
                }
            }),
        );

        run_request(request).await.unwrap();

        let [request] = upstream
            .received_requests()
            .await
            .unwrap()
            .try_into()
            .unwrap();
        assert_eq!(
            body(&request),
            json!({
                "model": "parse", "output_format": {"future": true},
                "document": {"type": "image_url", "image_url": "https://example.com/a.png",
                    "provider_options": {"nested": [false, 0, null]}}
            })
        );
    }

    #[rstest]
    #[tokio::test]
    async fn explicit_null_options_use_defaults_before_http() {
        let upstream = upstream(
            1,
            json!({"pages": [{"index": 0, "markdown": {"content": "hello"}}]}),
        )
        .await;
        let request = request(
            "cohere/parse",
            &upstream.uri(),
            image("https://example.com/a.png"),
            json!({"output_format": null, "req_format": null}),
        );
        assert_eq!(
            request.response_format().unwrap(),
            OcrResponseFormat::Litellm
        );

        run_request(request).await.unwrap();

        let [request] = upstream
            .received_requests()
            .await
            .unwrap()
            .try_into()
            .unwrap();
        let body = body(&request);
        assert_eq!(body["output_format"], "markdown");
        assert!(body.get("req_format").is_none());
    }

    #[rstest]
    #[tokio::test]
    async fn direct_and_vertex_mistral_build_the_same_request_and_share_normalization() {
        let upstream = upstream(
            2,
            json!({
                "pages": [{"index": 0, "markdown": "hello"}],
                "extra": "preserved"
            }),
        )
        .await;
        let options = json!({
            "pages": [0, 2],
            "include_image_base64": true,
            "vertex_project": "project-1",
            "vertex_location": "us-central1",
            "unknown": "preserved"
        });
        let document =
            json!({"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"});
        let direct_call = request(
            "mistral/mistral-ocr-maas",
            &upstream.uri(),
            document.clone(),
            options.clone(),
        );
        let vertex_call = request(
            "vertex_ai/mistral-ocr-maas",
            &upstream.uri(),
            document,
            options,
        );
        assert_eq!(direct_call.transport.timeout, Some(Duration::from_secs(2)));
        assert_eq!(vertex_call.transport.timeout, Some(Duration::from_secs(2)));
        let direct_response = run_request(direct_call).await.unwrap().into_json();
        let vertex_response = run_request(vertex_call).await.unwrap().into_json();

        let [direct_http, vertex_http] = upstream
            .received_requests()
            .await
            .unwrap()
            .try_into()
            .unwrap();
        assert_eq!(direct_http.url.path(), "/v1/ocr");
        assert_eq!(
            vertex_http.url.path(),
            "/v1/projects/project-1/locations/us-central1/publishers/mistralai/models/mistral-ocr-maas:rawPredict"
        );
        assert_mistral_request(&direct_http);
        assert_mistral_request(&vertex_http);
        assert_eq!(direct_response, vertex_response);
        assert_eq!(direct_response["model"], "mistral-ocr-maas");
        assert_eq!(direct_response["object"], "ocr");
        assert_eq!(direct_response["extra"], "preserved");
    }

    #[derive(serde::Deserialize)]
    struct KnownParams {
        pages: Option<Vec<i64>>,
    }

    #[rstest]
    fn parsed_provider_params_separates_known_and_extra_params() {
        let arguments: CallArguments = serde_json::from_value(json!({
            "pages": [0, 2],
            "future_ocr_option": true,
            "extra_body": {"provider_option": "value"}
        }))
        .unwrap();
        let known: KnownParams = parse_options(&arguments).unwrap();
        assert_eq!(known.pages, Some(vec![0, 2]));
        assert_eq!(arguments["future_ocr_option"], true);
        assert_eq!(arguments["extra_body"], json!({"provider_option": "value"}));
        assert_eq!(
            arguments
                .iter()
                .filter(|(name, _)| name.as_str() != "pages")
                .count(),
            2
        );
        assert_eq!(
            compose_body(&arguments, &json!({"pages": known.pages}), &["pages"]).unwrap(),
            json!({
                "pages": [0, 2], "future_ocr_option": true, "provider_option": "value"
            })
        );
    }
}
