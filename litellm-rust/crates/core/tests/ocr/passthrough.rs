use std::sync::{Arc, Mutex};

use litellm_callbacks::event::{RequestContext, WireRequest};
use rstest::rstest;
use rstest_reuse::{self, apply, template};
use serde_json::{Map, Value, json};

use super::LocalOcrHost;
use super::test_support::{
    MockResponse, SERVED_DOCUMENT, document_server, mock_server, perform_ocr_with, request_body,
    wire_request_with_document,
};

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

#[derive(Clone, Copy, Debug)]
enum Source {
    Inline,
    Remote,
    RemoteWithExtraField,
}

/// What the host does to the wire request in `before_send`.
#[derive(Clone, Copy, Debug)]
enum Host {
    Detached,
    /// What `litellm-callbacks-legacy` does before `pre_call`: every passthrough body key
    /// is replaced by the caller's own value.
    Realiasing,
    ReplacesDocument,
}

const REPLACED_DOCUMENT: &str = "data:image/png;base64,cmVwbGFjZWQ=";

impl Host {
    fn before_send(
        self,
        caller: &Map<String, Value>,
        wire: WireRequest,
        context: &RequestContext,
    ) -> WireRequest {
        let Value::Object(fields) = wire.body else {
            return wire;
        };
        let body = fields
            .into_iter()
            .map(|(name, value)| match self {
                Self::Detached => (name, value),
                Self::Realiasing => {
                    let aliased = context
                        .passthrough_fields
                        .contains(&name)
                        .then(|| caller.get(&name).cloned())
                        .flatten()
                        .unwrap_or(value);
                    (name, aliased)
                }
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
    caller: Map<String, Value>,
    result: Result<(), crate::ocr::Error>,
    before_send: Option<(WireRequest, RequestContext)>,
    provider_body: Option<Value>,
}

fn caller_document(route: Route, source: Source, document_base: &str) -> Value {
    let document_type = route.document_type();
    let remote = format!("{document_base}/scan.png");
    match source {
        Source::Inline => {
            json!({"type": document_type, document_type: "data:image/png;base64,YWJj"})
        }
        Source::Remote => json!({"type": document_type, document_type: remote}),
        Source::RemoteWithExtraField => {
            json!({"type": document_type, document_type: remote, "document_name": "scan.png"})
        }
    }
}

async fn send(route: Route, source: Source, host: Host, document_base: &str) -> Sent {
    let (base, seen, provider) = mock_server(vec![MockResponse::json(json!({"pages": []}))]).await;
    let document = caller_document(route, source, document_base);
    let caller: Map<String, Value> = route
        .options()
        .as_object()
        .unwrap()
        .clone()
        .into_iter()
        .chain([("document".to_string(), document.clone())])
        .collect();
    let observed = Arc::new(Mutex::new(None));
    let captured = observed.clone();
    let host_caller = caller.clone();
    let request = wire_request_with_document(route.model(), &base, document, route.options());
    let local = LocalOcrHost::new(request).with_before_send(move |wire, context| {
        *captured.lock().unwrap() = Some((wire.clone(), context.clone()));
        Ok(host.before_send(&host_caller, wire, context))
    });
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
    let before_send = observed.lock().unwrap().take();
    Sent {
        caller,
        result,
        before_send,
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

#[template]
#[rstest]
fn every_route_and_source(
    #[values(
        Route::Mistral,
        Route::AzureAi,
        Route::VertexMistral,
        Route::AzureCohereParse,
        Route::Cohere
    )]
    route: Route,
    #[values(Source::Inline, Source::Remote, Source::RemoteWithExtraField)] source: Source,
) {
}

#[template]
#[rstest]
fn every_route(
    #[values(
        Route::Mistral,
        Route::AzureAi,
        Route::VertexMistral,
        Route::AzureCohereParse,
        Route::Cohere
    )]
    route: Route,
) {
}

#[template]
#[rstest]
#[case::azure_ai(Route::AzureAi)]
#[case::vertex_mistral(Route::VertexMistral)]
#[case::azure_cohere_parse(Route::AzureCohereParse)]
fn inlining_routes(#[case] route: Route) {}

#[apply(every_route_and_source)]
#[tokio::test]
async fn passthrough_fields_hold_the_callers_values(route: Route, source: Source) {
    let (document_base, _documents) = document_server().await;
    let sent = send(route, source, Host::Detached, &document_base).await;
    sent.result.unwrap();
    let (wire, context) = sent.before_send.unwrap();
    let rewritten: Vec<_> = context
        .passthrough_fields
        .iter()
        .filter(|name| wire.body.get(*name) != sent.caller.get(*name))
        .collect();
    assert!(
        rewritten.is_empty(),
        "passthrough fields the route rewrote: {rewritten:?}\nbody: {:#}\ncaller: {:#}",
        wire.body,
        Value::Object(sent.caller)
    );
}

#[apply(every_route_and_source)]
#[tokio::test]
async fn realiasing_leaves_the_provider_request_unchanged(route: Route, source: Source) {
    let (document_base, _documents) = document_server().await;
    let detached = send(route, source, Host::Detached, &document_base).await;
    let realiased = send(route, source, Host::Realiasing, &document_base).await;
    detached.result.unwrap();
    realiased.result.unwrap();
    assert_eq!(realiased.provider_body, detached.provider_body);
}

#[apply(inlining_routes)]
#[tokio::test]
async fn inlining_routes_send_the_downloaded_document(
    route: Route,
    #[values(Host::Detached, Host::Realiasing)] host: Host,
) {
    let (document_base, _documents) = document_server().await;
    let sent = send(route, Source::Remote, host, &document_base).await;
    sent.result.unwrap();
    assert_eq!(
        sent.provider_body.unwrap()["document"][route.document_type()],
        json!(served_document_uri())
    );
}

#[apply(every_route)]
#[tokio::test]
async fn document_replaced_by_the_host_reaches_the_provider(route: Route) {
    let (document_base, _documents) = document_server().await;
    let sent = send(
        route,
        Source::Remote,
        Host::ReplacesDocument,
        &document_base,
    )
    .await;
    sent.result.unwrap();
    assert_eq!(
        sent.provider_body.unwrap()["document"][route.document_type()],
        json!(REPLACED_DOCUMENT)
    );
}
