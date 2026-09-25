use litellm_auth::{InputSource, SecretValue, Sourced};
use litellm_llms::base_llm::ocr::{
    handler::OcrClient,
    transformation::{OcrConnection, OcrCredentialInputs, PreparedOcrRequest},
};
use litellm_secrets::source::Secrets;

use super::provider_config::OcrProvider;
use crate::ocr::types::{LiteLLMOcrRequest, ResolvedOcrRequest};

pub(crate) fn prepare_request(
    request: ResolvedOcrRequest,
    caller_document: bool,
    client: &OcrClient,
    secrets: Secrets,
) -> PreparedOcrRequest {
    let credentials = request.credentials.clone();
    let (preferred_api_key_env, api_base_env) = match request.config.provider() {
        OcrProvider::Mistral => (
            Some("MISTRAL_AZURE_API_KEY"),
            Some("MISTRAL_AZURE_API_BASE"),
        ),
        OcrProvider::AzureAi => (None, Some("AZURE_AI_API_BASE")),
        OcrProvider::AwsTextract
        | OcrProvider::Cohere
        | OcrProvider::Reducto
        | OcrProvider::VertexAi => (None, None),
    };
    let secret = |name: &str| secrets.truthy(name);
    let dynamic_api_key = credentials.dynamic_api_key.or_else(|| {
        credentials.api_key.clone().or_else(|| {
            preferred_api_key_env
                .into_iter()
                .chain(request.config.get_api_key_env_var())
                .find_map(secret)
                .map(|value| Sourced::new(SecretValue::new(value), InputSource::Environment))
        })
    });
    let dynamic_api_base = credentials.dynamic_api_base.or_else(|| {
        credentials.api_base.clone().or_else(|| {
            api_base_env
                .and_then(secret)
                .map(|value| Sourced::new(value, InputSource::Environment))
        })
    });
    let resolved = request
        .config
        .resolve_connection_params(OcrCredentialInputs {
            dynamic_api_key,
            dynamic_api_base,
            ..credentials
        });
    let LiteLLMOcrRequest {
        model,
        document,
        transport,
        optional_params,
        input_sources,
        azure_ad_token_provider,
        ..
    } = request;
    PreparedOcrRequest {
        model,
        document,
        connection: OcrConnection::new(resolved, transport, client.settings().clone(), secrets),
        caller_document,
        optional_params,
        input_sources,
        azure_ad_token_provider,
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures_util::future::BoxFuture;
    use litellm_core_utils::call_arguments::{CallArguments, compose_body, parse_options};
    use litellm_host::event::WireRequest;
    use litellm_llms::{
        base_llm::ocr::{
            error::Error,
            handler::{CallHooks, OcrClient},
            transformation::{BaseOcrConfig, OcrResponseFormat},
        },
        cohere::ocr::transformation::CohereParseConfig,
        mistral::ocr::transformation::MistralOcrConfig,
        vertex_ai::ocr::transformation::VertexAiOcrConfig,
    };
    use serde_json::{Value, json};

    use super::*;
    use crate::ocr::{
        document::prepare_document,
        types::LiteLLMOcrRequest,
        wire::{OcrWireRequest, decode_request},
    };

    /// Stands in for a host with no hooks registered.
    struct NoHooks;

    impl CallHooks<Error> for NoHooks {
        fn before_send(&self, wire: WireRequest) -> BoxFuture<'_, Result<WireRequest, Error>> {
            Box::pin(async move { Ok(wire) })
        }

        fn response_received<'a>(&'a self, _body: &'a [u8]) -> BoxFuture<'a, Result<(), Error>> {
            Box::pin(async { Ok(()) })
        }
    }

    fn client() -> OcrClient {
        OcrClient::for_test(reqwest::Client::new(), reqwest::Client::new())
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

    fn prepared(request: LiteLLMOcrRequest) -> PreparedOcrRequest {
        prepare_request(
            request.map_document(prepare_document).unwrap(),
            true,
            &client(),
            std::sync::Arc::new(litellm_core_utils::settings::ProcessEnvironment),
        )
    }

    fn image(url: &str) -> Value {
        json!({"type": "image_url", "image_url": url})
    }

    #[tokio::test]
    async fn cohere_body_keeps_native_document_fields_and_untyped_overrides() {
        let request = request(
            "cohere/parse",
            "https://example.com",
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

        let http = CohereParseConfig
            .prepare_request(&prepared(request), &client(), &NoHooks)
            .await
            .unwrap();

        let body: Value = serde_json::from_slice(http.body()).unwrap();
        assert_eq!(
            body,
            json!({
                "model": "parse", "output_format": {"future": true},
                "document": {"type": "image_url", "image_url": "https://example.com/a.png",
                    "provider_options": {"nested": [false, 0, null]}}
            })
        );
    }

    #[tokio::test]
    async fn explicit_null_options_use_defaults_before_http() {
        let request = request(
            "cohere/parse",
            "https://example.com",
            image("https://example.com/a.png"),
            json!({"output_format": null, "req_format": null}),
        );
        assert_eq!(
            request.response_format().unwrap(),
            OcrResponseFormat::Litellm
        );

        let http = CohereParseConfig
            .prepare_request(&prepared(request), &client(), &NoHooks)
            .await
            .unwrap();

        let body: Value = serde_json::from_slice(http.body()).unwrap();
        assert_eq!(body["output_format"], "markdown");
        assert!(body.get("req_format").is_none());
    }

    #[tokio::test]
    async fn direct_and_vertex_mistral_build_the_same_request_and_share_normalization() {
        let options = json!({
            "pages": [0, 2],
            "include_image_base64": true,
            "vertex_project": "project-1",
            "vertex_location": "us-central1",
            "unknown": "preserved"
        });
        let document =
            json!({"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"});
        let direct = prepared(request(
            "mistral/mistral-ocr-maas",
            "https://mistral.test",
            document.clone(),
            options.clone(),
        ));
        let vertex = prepared(request(
            "vertex_ai/mistral-ocr-maas",
            "https://vertex.test",
            document,
            options,
        ));

        let direct_http = MistralOcrConfig
            .prepare_request(&direct, &client(), &NoHooks)
            .await
            .unwrap();
        let vertex_http = VertexAiOcrConfig
            .prepare_request(&vertex, &client(), &NoHooks)
            .await
            .unwrap();

        assert_eq!(direct_http.url(), "https://mistral.test/v1/ocr");
        assert_eq!(
            vertex_http.url(),
            "https://vertex.test/v1/projects/project-1/locations/us-central1/publishers/mistralai/models/mistral-ocr-maas:rawPredict"
        );
        for http in [&direct_http, &vertex_http] {
            assert_eq!(http.header("authorization").unwrap(), "Bearer test-key");
            assert_eq!(http.header("content-type").unwrap(), "application/json");
            assert_eq!(http.timeout(), Some(Duration::from_secs(2)));
            let body: Value = serde_json::from_slice(http.body()).unwrap();
            assert_eq!(
                body,
                json!({
                    "model": "mistral-ocr-maas",
                    "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
                    "pages": [0, 2],
                    "include_image_base64": true,
                    "unknown": "preserved"
                })
            );
        }
        let payload = serde_json::to_vec(
            &json!({"pages": [{"index": 0, "markdown": "hello"}], "extra": "preserved"}),
        )
        .unwrap();
        let direct_response = MistralOcrConfig
            .transform_ocr_response(&direct.model, &payload, OcrResponseFormat::Litellm)
            .unwrap()
            .into_json();
        let vertex_response = VertexAiOcrConfig
            .transform_ocr_response(&vertex.model, &payload, OcrResponseFormat::Litellm)
            .unwrap()
            .into_json();
        assert_eq!(direct_response, vertex_response);
        assert_eq!(direct_response["model"], "mistral-ocr-maas");
        assert_eq!(direct_response["object"], "ocr");
        assert_eq!(direct_response["extra"], "preserved");
    }

    #[derive(serde::Deserialize)]
    struct KnownParams {
        pages: Option<Vec<i64>>,
    }

    #[test]
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
