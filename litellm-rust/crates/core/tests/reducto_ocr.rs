use std::sync::Arc;

use rstest::rstest;
use serde_json::{Value, json};

use super::hooks::{OcrDuringCallRequest, OcrHookFuture, OcrHooks, OcrPostCallRequest};
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
        "settings":{"ocr_system":"standard"},
        "future_ocr_option":true,
        "extra_body":{"provider_option":"value"}
    }),
    "reducto://already.pdf",
    json!({
        "input":"reducto://already.pdf",
        "formatting":{"table_output_format":"html"},
        "retrieval":{"chunk_mode":"section"},
        "settings":{"ocr_system":"standard"},
        "future_ocr_option":true,
        "provider_option":"value"
    })
)]
#[case(
    "reducto/parse-legacy",
    json!({
        "enhance":{"agentic":[{"type":"table"}]},
        "future_ocr_option":true,
        "extra_body":{"provider_option":"value"}
    }),
    "reducto://legacy.pdf",
    json!({
        "document_url":"reducto://legacy.pdf",
        "options":{"enhance":{"agentic":[{"type":"table"}]}},
        "future_ocr_option":true,
        "provider_option":"value"
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
    let request = super::test_support::with_source(wire_request(model, &base, options), source);

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
async fn data_uri_upload_preserves_multipart_headers(
    #[case] model: &str,
    #[values("application/pdf", "image/png")] mime_type: &str,
) {
    let (base, seen, server) = mock_server(vec![
        MockResponse::json(json!({"file_id":"reducto://uploaded.pdf"})),
        MockResponse::json(json!({"result":{"chunks":[{"content":"hello"}]}})),
    ])
    .await;
    let document = if mime_type.starts_with("image/") {
        json!({"type":"image_url","image_url":format!("data:{mime_type};base64,YWJj")})
    } else {
        json!({"type":"document_url","document_url":format!("data:{mime_type};base64,YWJj")})
    };
    let mut request = super::LiteLLMOcrRequest {
        document: serde_json::from_value::<super::OcrDocument>(document)
            .unwrap()
            .into(),
        ..wire_request(&format!("reducto/{model}"), &base, json!({}))
    };
    request.transport.extra_headers = vec![
        ("Content-Type".into(), "application/json".into()),
        ("X-Trace".into(), "upload-test".into()),
    ];

    let response = perform_ocr(request).await.unwrap();
    server.await.unwrap();
    assert_eq!(response.pages[0].markdown, "hello");
    let requests = seen.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests[0].starts_with("POST /upload "));
    assert!(
        requests[0]
            .to_ascii_lowercase()
            .contains("content-type: multipart/form-data; boundary=")
    );
    assert!(requests[0].contains("x-trace: upload-test"));
    let multipart = requests[0].split_once("\r\n\r\n").unwrap().1;
    assert!(multipart.contains(&format!("Content-Type: {mime_type}\r\n")));
    assert!(multipart.contains("\r\n\r\nabc\r\n--"));
    assert!(requests[1].starts_with("POST /parse "));
    let source_field = if model == "parse-legacy" {
        "document_url"
    } else {
        "input"
    };
    assert_eq!(
        request_body(&requests[1]),
        json!({source_field:"reducto://uploaded.pdf"})
    );
    for request in requests.iter() {
        assert!(
            request
                .to_ascii_lowercase()
                .contains("authorization: bearer test-key\r\n")
        );
    }
}

struct ParseBoundary {
    request_count: Arc<std::sync::Mutex<Vec<String>>>,
}

impl OcrHooks for ParseBoundary {
    fn post_call(&self, request: OcrPostCallRequest) -> OcrHookFuture<'_, OcrPostCallRequest> {
        Box::pin(async move {
            assert_eq!(self.request_count.lock().unwrap().len(), 2);
            assert_eq!(
                request.original_response,
                json!(r#"{"result":{"chunks":[]}}"#)
            );
            Ok(request)
        })
    }
}

#[tokio::test]
async fn post_call_stays_after_reducto_upload_and_parse() {
    let (base, seen, server) = mock_server(vec![
        MockResponse::json(json!({"file_id":"reducto://uploaded.pdf"})),
        MockResponse::json(json!({"result":{"chunks":[]}})),
    ])
    .await;
    let request = super::LiteLLMOcrRequest {
        hooks: Arc::new(ParseBoundary {
            request_count: seen.clone(),
        }),
        ..wire_request("reducto/parse-v3", &base, json!({}))
    };

    perform_ocr(request).await.unwrap();
    server.await.unwrap();
    assert_eq!(seen.lock().unwrap().len(), 2);
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
#[case("https://example.com/a.pdf", crate::ocr::Error::ReductoSource)]
#[case("reducto://", crate::ocr::Error::RequestField { path: "document file id".into() })]
#[case("data:application/pdf;base64", crate::ocr::Error::InvalidDataUri)]
#[case(
    "data:application/pdf;base64,INVALID!",
    crate::ocr::Error::InvalidDataUri
)]
#[tokio::test]
async fn rejects_invalid_document_sources_before_network(
    #[case] source: &str,
    #[case] expected: super::Error,
) {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({}))]).await;
    let request = super::test_support::with_source(
        wire_request("reducto/parse-v3", &base, json!({})),
        source,
    );
    let result = perform_ocr(request).await;
    server.abort();
    let _ = server.await;
    assert!(
        seen.lock().unwrap().is_empty(),
        "sent invalid source: {source}"
    );
    let error = result.unwrap_err();
    assert_eq!(
        std::mem::discriminant(&error),
        std::mem::discriminant(&expected)
    );
    assert_eq!(error.http_status_code(), Some(400));
    assert_eq!(error.to_string(), expected.to_string());
}

#[test]
fn response_normalization_groups_blocks_and_distinguishes_null_result() {
    use crate::llms::reducto::ocr::transformation::{
        ReductoResponse, normalize_response as transform_ocr_response,
    };

    let raw = json!({"usage":{"num_pages":"2","credits":"3"},"result":{"type":"full","chunks":[
        {"blocks":[{
            "type":"Table",
            "content":"B",
            "bbox":{"left":0.1,"top":0.2,"width":0.8,"height":0.3,"page":2,"original_page":4},
            "confidence":"high",
            "granular_confidence":{"parse_confidence":0.95,"extract_confidence":null},
            "image_url":null
        }]},
        {"blocks":[{"content":"A","bbox":{"page":1},"type":"Text"},{"content":"C","bbox":{"page":1}}]}
    ]}});
    let response: ReductoResponse = serde_json::from_value(raw).unwrap();
    let normalized = transform_ocr_response("parse-v3", response)
        .unwrap()
        .into_json();
    assert_eq!(normalized["pages"][0]["markdown"], "A\n\nC");
    assert_eq!(normalized["pages"][1]["markdown"], "B");
    assert_eq!(normalized["pages"][1]["blocks"][0]["type"], "Table");
    assert_eq!(
        normalized["pages"][1]["blocks"][0]["bbox"],
        json!({"left":0.1,"top":0.2,"width":0.8,"height":0.3,"page":2,"original_page":4})
    );
    assert_eq!(normalized["pages"][1]["blocks"][0]["confidence"], "high");
    assert_eq!(
        normalized["pages"][1]["blocks"][0]["granular_confidence"]["parse_confidence"],
        0.95
    );
    assert!(normalized["pages"][1]["blocks"][0]["image_url"].is_null());
    assert_eq!(normalized["usage_info"]["pages_processed"], 2);
    assert_eq!(normalized["usage_info"]["credits"], 3.0);

    let missing: ReductoResponse =
        serde_json::from_value(json!({"chunks":[{"content":"text"}]})).unwrap();
    let missing = transform_ocr_response("parse-v3", missing).unwrap();
    assert_eq!(missing.pages[0].markdown, "text");
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
    let mut request = super::test_support::with_source(
        wire_request("reducto/parse-v3", &base, json!({})),
        "reducto://ready.pdf",
    );
    request.transport.extra_headers = vec![("authorization".into(), "Bearer existing".into())];

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
    fn intercepts_requests(&self) -> bool {
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
