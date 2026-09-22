use litellm_host::event::WireRequest;
use litellm_llms::base_llm::ocr::error::Error;
use rstest::rstest;
use serde_json::{Value, json};

use super::test_support::{
    MockResponse, SERVED_DOCUMENT, document_server, mock_server, perform_ocr_with, request_body,
    wire_request_with_document,
};
use crate::ocr::route::LocalOcrHost;

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
enum Host {
    Detached,
    ReplacesDocument,
}

const REPLACED_DOCUMENT: &str = "data:image/png;base64,cmVwbGFjZWQ=";

impl Host {
    fn before_send(self, wire: WireRequest) -> WireRequest {
        let Value::Object(fields) = wire.body else {
            return wire;
        };
        let body = fields
            .into_iter()
            .map(|(name, value)| match self {
                Self::Detached => (name, value),
                Self::ReplacesDocument if name == "document" => {
                    let document_type = value["type"].clone();
                    let key = document_type.as_str().unwrap_or_default().to_string();
                    (name, json!({"type": document_type, key: REPLACED_DOCUMENT}))
                }
                Self::ReplacesDocument => (name, value),
            })
            .collect();
        WireRequest {
            body: Value::Object(body),
            ..wire
        }
    }
}

struct Sent {
    result: Result<(), Error>,
    provider_body: Option<Value>,
}

async fn send(route: Route, host: Host, document_base: &str) -> Sent {
    let (base, seen, provider) = mock_server(vec![MockResponse::json(json!({"pages": []}))]).await;
    let document_type = route.document_type();
    let document =
        json!({"type": document_type, document_type: format!("{document_base}/scan.png")});
    let request = wire_request_with_document(route.model(), &base, document, route.options());
    let local =
        LocalOcrHost::new(request).with_before_send(move |wire, _| Ok(host.before_send(wire)));
    let result = perform_ocr_with(local).await.map(|_| ());
    match result {
        Ok(()) => provider.await.unwrap(),
        Err(_) => provider.abort(),
    }
    let provider_body = seen
        .lock()
        .unwrap()
        .first()
        .map(|request| request_body(request));
    Sent {
        result,
        provider_body,
    }
}

fn served_document_uri() -> String {
    use base64::Engine;
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(SERVED_DOCUMENT)
    )
}

#[rstest]
#[case::azure_ai(Route::AzureAi)]
#[case::vertex_mistral(Route::VertexMistral)]
#[case::azure_cohere_parse(Route::AzureCohereParse)]
#[tokio::test]
async fn inlining_routes_send_the_downloaded_document(#[case] route: Route) {
    let (document_base, _documents) = document_server().await;
    let sent = send(route, Host::Detached, &document_base).await;
    sent.result.unwrap();
    assert_eq!(
        sent.provider_body.unwrap()["document"][route.document_type()],
        json!(served_document_uri())
    );
}

#[rstest]
#[tokio::test]
async fn document_replaced_by_the_host_reaches_the_provider(
    #[values(
        Route::Mistral,
        Route::AzureAi,
        Route::VertexMistral,
        Route::AzureCohereParse,
        Route::Cohere
    )]
    route: Route,
) {
    let (document_base, _documents) = document_server().await;
    let sent = send(route, Host::ReplacesDocument, &document_base).await;
    sent.result.unwrap();
    assert_eq!(
        sent.provider_body.unwrap()["document"][route.document_type()],
        json!(REPLACED_DOCUMENT)
    );
}
