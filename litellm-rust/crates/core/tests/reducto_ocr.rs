use std::sync::Arc;

use rstest::rstest;
use serde_json::{Value, json};

use super::hooks::{OcrDuringCallRequest, OcrHookFuture, OcrHooks};
use super::test_support::{MockResponse, mock_server, perform_ocr, wire_request};

fn request_body(request: &str) -> Value {
    serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap()
}

#[rstest]
#[case(
    "reducto/parse-v3",
    json!({
        "formatting":{"table_output_format":"html"},
        "retrieval":{"chunk_mode":"section"},
        "settings":{"ocr_system":"standard"}
    }),
    "reducto://already.pdf",
    json!({
        "input":"reducto://already.pdf",
        "formatting":{"table_output_format":"html"},
        "retrieval":{"chunk_mode":"section"},
        "settings":{"ocr_system":"standard"}
    })
)]
#[case(
    "reducto/parse-legacy",
    json!({"enhance":{"agentic":[{"type":"table"}]}}),
    "reducto://legacy.pdf",
    json!({
        "document_url":"reducto://legacy.pdf",
        "options":{"enhance":{"agentic":[{"type":"table"}]}}
    })
)]
#[tokio::test]
async fn request_mapping_matches_python(
    #[case] model: &str,
    #[case] options: Value,
    #[case] source: &str,
    #[case] expected: Value,
) {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
        "result":{"chunks":[]}
    }))])
    .await;
    let mut request = wire_request(model, &base, options);
    request.document = request.document.with_source(source.into());

    perform_ocr(request).await.unwrap();
    server.await.unwrap();
    let requests = seen.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].starts_with("POST /parse "));
    assert_eq!(request_body(&requests[0]), expected);
}

#[rstest]
#[case("parse-v3")]
#[case("parse-legacy")]
#[tokio::test]
async fn data_uri_upload_preserves_multipart_headers(#[case] model: &str) {
    let (base, seen, server) = mock_server(vec![
        MockResponse::json(json!({"file_id":"reducto://uploaded.pdf"})),
        MockResponse::json(json!({"result":{"chunks":[{"content":"hello"}]}})),
    ])
    .await;
    let mut request = wire_request(&format!("reducto/{model}"), &base, json!({}));
    request.connection.extra_headers = vec![
        ("Content-Type".into(), "application/json".into()),
        ("X-Trace".into(), "upload-test".into()),
    ];

    let response = perform_ocr(request).await.unwrap();
    server.await.unwrap();
    assert_eq!(response.pages[0]["markdown"], "hello");
    let requests = seen.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests[0].starts_with("POST /upload "));
    assert!(
        requests[0]
            .to_ascii_lowercase()
            .contains("content-type: multipart/form-data; boundary=")
    );
    assert!(requests[0].contains("x-trace: upload-test"));
    assert!(requests[0].contains("application/pdf"));
    assert!(requests[0].contains("abc"));
    assert!(requests[1].starts_with("POST /parse "));
}

#[rstest]
#[case(json!({"file_id":""}))]
#[case(json!({}))]
#[case(json!({"file_id":null}))]
#[tokio::test]
async fn invalid_upload_ids_stop_before_parse(#[case] response: Value) {
    let (base, seen, server) = mock_server(vec![MockResponse::json(response)]).await;
    let error = perform_ocr(wire_request("reducto/parse-v3", &base, json!({})))
        .await
        .unwrap_err();
    server.await.unwrap();
    assert!(error.to_string().contains("file_id"));
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn upload_failure_stops_before_parse() {
    let (base, seen, server) = mock_server(vec![MockResponse {
        status: 503,
        headers: vec![],
        body: json!({"error":"unavailable"}),
    }])
    .await;
    assert!(
        perform_ocr(wire_request("reducto/parse-v3", &base, json!({})))
            .await
            .is_err()
    );
    server.await.unwrap();
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[rstest]
#[case("https://example.com/a.pdf")]
#[case("reducto://")]
#[case("data:application/pdf;base64")]
#[case("data:application/pdf;base64,INVALID!")]
#[tokio::test]
async fn rejects_invalid_document_sources_before_network(#[case] source: &str) {
    let mut request = wire_request("reducto/parse-v3", "http://127.0.0.1:1", json!({}));
    request.document = request.document.with_source(source.into());
    assert!(perform_ocr(request).await.is_err());
}

#[test]
fn response_normalization_groups_blocks_and_distinguishes_null_result() {
    use crate::ocr::codecs::reducto::{ReductoResponse, transform_ocr_response};

    let raw = json!({"usage":{"num_pages":"2","credits":"3"},"result":{"chunks":[
        {"blocks":[{"content":"B","bbox":{"page":2},"kind":"table"}]},
        {"blocks":[{"content":"A","bbox":{"page":1},"kind":"text"},{"content":"C","bbox":{"page":1}}]}
    ]}});
    let response: ReductoResponse = serde_json::from_value(raw).unwrap();
    let normalized = transform_ocr_response("parse-v3", response)
        .unwrap()
        .into_json();
    assert_eq!(normalized["pages"][0]["markdown"], "A\n\nC");
    assert_eq!(normalized["pages"][1]["markdown"], "B");
    assert_eq!(normalized["pages"][1]["blocks"][0]["kind"], "table");
    assert_eq!(normalized["usage_info"]["pages_processed"], 2);
    assert_eq!(normalized["usage_info"]["credits"], 3.0);

    let missing: ReductoResponse =
        serde_json::from_value(json!({"chunks":[{"content":"text"}]})).unwrap();
    let missing = transform_ocr_response("parse-v3", missing).unwrap();
    assert_eq!(missing.pages[0]["markdown"], "text");
    let null: ReductoResponse = serde_json::from_value(
        json!({"result":null,"chunks":[{"content":"ignored"}],"usage":null}),
    )
    .unwrap();
    let null = transform_ocr_response("parse-v3", null).unwrap();
    assert!(null.pages.is_empty());
}

#[tokio::test]
async fn facade_omits_native_response_by_default_and_preserves_auth_priority() {
    let raw = json!({"job_id":"job-1","result":{"chunks":[]}});
    let (base, seen, server) = mock_server(vec![MockResponse::json(raw)]).await;
    let mut request = wire_request("reducto/parse-v3", &base, json!({}));
    request.document = request.document.with_source("reducto://ready.pdf".into());
    request.connection.extra_headers = vec![("authorization".into(), "Bearer existing".into())];

    let response = perform_ocr(request).await.unwrap();
    server.await.unwrap();
    assert_eq!(response.provider_native_response, None);
    assert!(
        seen.lock().unwrap()[0]
            .to_ascii_lowercase()
            .contains("authorization: bearer existing")
    );
}

struct RewriteDocument;

impl OcrHooks for RewriteDocument {
    fn has_guardrails(&self) -> bool {
        true
    }

    fn during_call(
        &self,
        request: OcrDuringCallRequest,
    ) -> OcrHookFuture<'_, OcrDuringCallRequest> {
        Box::pin(async move {
            assert_eq!(
                request.body["document_url"],
                "data:application/pdf;base64,YWJj"
            );
            Ok(OcrDuringCallRequest {
                body: json!({"type":"document_url","document_url":"reducto://guarded.pdf"}),
                ..request
            })
        })
    }
}

#[tokio::test]
async fn guardrail_rewrites_document_before_upload() {
    let (base, seen, server) =
        mock_server(vec![MockResponse::json(json!({"result":{"chunks":[]}}))]).await;
    let mut request = wire_request("reducto/parse-v3", &base, json!({}));
    request.hooks = Arc::new(RewriteDocument);

    perform_ocr(request).await.unwrap();
    server.await.unwrap();
    let requests = seen.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].starts_with("POST /parse "));
    assert!(requests[0].contains("reducto://guarded.pdf"));
}
