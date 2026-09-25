use base64::Engine;
use litellm_core::ocr::types::OcrDocumentInput;
use litellm_host::event::WireRequest;
use rstest::rstest;
use wiremock::{Mock, matchers::any};

use super::*;

const SERVED_DOCUMENT: &[u8] = b"\x89PNG served document";
const REPLACED_DOCUMENT: &str = "data:image/png;base64,cmVwbGFjZWQ=";

#[derive(Clone, Copy, Debug)]
enum Route {
    Mistral,
    AzureAi,
    VertexMistral,
    AzureCohereParse,
    Cohere,
}

impl Route {
    fn model(self) -> &'static str {
        match self {
            Self::Mistral => "mistral/model",
            Self::AzureAi => "azure_ai/model",
            Self::VertexMistral => "vertex_ai/mistral-ocr-maas",
            Self::AzureCohereParse => "azure_ai/cohere-parse",
            Self::Cohere => "cohere/model",
        }
    }

    fn document_type(self) -> &'static str {
        match self {
            Self::Mistral | Self::AzureAi | Self::VertexMistral => "document_url",
            Self::AzureCohereParse | Self::Cohere => "image_url",
        }
    }

    fn options(self) -> Value {
        match self {
            Self::Mistral | Self::AzureAi => json!({"pages": [0]}),
            Self::VertexMistral => json!({"pages": [0], "vertex_project": "project-1"}),
            Self::AzureCohereParse | Self::Cohere => json!({"output_format": "markdown"}),
        }
    }
}

/// What the host does to the wire request in `before_send`.
#[derive(Clone, Copy, Debug)]
enum Guardrail {
    Detached,
    ReplacesDocument,
}

impl Guardrail {
    fn before_send(self, wire: WireRequest) -> WireRequest {
        let Value::Object(fields) = wire.body else {
            return wire;
        };
        let body = fields
            .into_iter()
            .map(|(name, value)| match self {
                Self::ReplacesDocument if name == "document" => {
                    let document_type = value["type"].clone();
                    let key = document_type.as_str().unwrap_or_default().to_string();
                    (name, json!({"type": document_type, key: REPLACED_DOCUMENT}))
                }
                Self::Detached | Self::ReplacesDocument => (name, value),
            })
            .collect();
        WireRequest {
            body: Value::Object(body),
            ..wire
        }
    }
}

/// Serves [`SERVED_DOCUMENT`] as `image/png` to every request.
async fn document_server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(any())
        .respond_with(ResponseTemplate::new(200).set_body_raw(SERVED_DOCUMENT, "image/png"))
        .mount(&server)
        .await;
    server
}

/// Sends a remote document through `route` and returns the document the provider saw.
async fn provider_document(route: Route, guardrail: Guardrail) -> Value {
    let documents = document_server().await;
    let upstream = upstream([pages_response()]).await;
    let document_type = route.document_type();
    let request = ocr_request_with_document(
        route.model(),
        &upstream.uri(),
        json!({"type": document_type, document_type: format!("{}/scan.png", documents.uri())}),
        route.options(),
    );
    let host =
        LocalOcrHost::new(request).with_before_send(move |wire, _| Ok(guardrail.before_send(wire)));

    perform_with(host).await.unwrap();

    only_request(&upstream).await.json()["document"][document_type].clone()
}

#[rstest]
#[case::azure_ai(Route::AzureAi)]
#[case::vertex_mistral(Route::VertexMistral)]
#[case::azure_cohere_parse(Route::AzureCohereParse)]
#[tokio::test]
async fn inlining_routes_send_the_downloaded_document(#[case] route: Route) {
    let expected = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(SERVED_DOCUMENT)
    );

    assert_eq!(
        provider_document(route, Guardrail::Detached).await,
        expected
    );
}

#[rstest]
#[tokio::test]
async fn a_document_replaced_by_the_host_reaches_the_provider(
    #[values(
        Route::Mistral,
        Route::AzureAi,
        Route::VertexMistral,
        Route::AzureCohereParse,
        Route::Cohere
    )]
    route: Route,
) {
    assert_eq!(
        provider_document(route, Guardrail::ReplacesDocument).await,
        REPLACED_DOCUMENT
    );
}

#[tokio::test]
async fn an_empty_byte_document_fails_before_sending() {
    let upstream = upstream([pages_response()]).await;
    let request = ocr_request("mistral/model", &upstream.uri(), json!({})).with_document(
        OcrDocumentInput::Bytes {
            bytes: Default::default(),
            file_name: None,
            mime_type: None,
        },
    );

    let error = perform(request).await.unwrap_err();

    assert!(matches!(error, Error::EmptyFile), "{error:?}");
    assert!(received(&upstream).await.is_empty());
}

#[tokio::test]
async fn a_missing_path_document_fails_before_sending() {
    let upstream = upstream([pages_response()]).await;
    let path =
        std::env::temp_dir().join(format!("litellm-ocr-missing-{}.png", rand::random::<u64>()));
    let request = ocr_request("mistral/model", &upstream.uri(), json!({})).with_document(
        OcrDocumentInput::Path {
            path: path.clone(),
            mime_type: None,
        },
    );

    let error = perform(request).await.unwrap_err();

    assert!(
        matches!(
            &error,
            Error::FileRead { path: failed, source }
                if *failed == path && source.kind() == std::io::ErrorKind::NotFound
        ),
        "{error:?}"
    );
    assert!(received(&upstream).await.is_empty());
}
