use std::sync::{Arc, Mutex};

use litellm_host::event::{CallEvent, MachineEvent, WireRequest};
use rstest::rstest;

use super::*;

fn upload_response() -> ResponseTemplate {
    json_response(json!({"file_id": "reducto://uploaded.pdf"}))
}

fn chunks_response(chunks: Value) -> ResponseTemplate {
    json_response(json!({"result": {"chunks": chunks}}))
}

fn source_field(model: &str) -> &'static str {
    match model.ends_with("parse-legacy") {
        true => "document_url",
        false => "input",
    }
}

#[rstest]
#[case::v3(
    "reducto/parse-v3",
    json!({
        "formatting": {"table_output_format": "html"},
        "retrieval": {"chunk_mode": "section"},
        "settings": {"ocr_system": "standard"},
        "future_ocr_option": true,
        "extra_body": {"provider_option": "value"}
    }),
    "reducto://already.pdf",
    json!({
        "input": "reducto://already.pdf",
        "formatting": {"table_output_format": "html"},
        "retrieval": {"chunk_mode": "section"},
        "settings": {"ocr_system": "standard"},
        "future_ocr_option": true,
        "provider_option": "value"
    })
)]
#[case::legacy(
    "reducto/parse-legacy",
    json!({
        "enhance": {"agentic": [{"type": "table"}]},
        "future_ocr_option": true,
        "extra_body": {"provider_option": "value"}
    }),
    "reducto://legacy.pdf",
    json!({
        "document_url": "reducto://legacy.pdf",
        "options": {"enhance": {"agentic": [{"type": "table"}]}},
        "future_ocr_option": true,
        "provider_option": "value"
    })
)]
#[tokio::test]
async fn an_uploaded_document_is_parsed_with_mapped_options(
    #[case] model: &str,
    #[case] options: Value,
    #[case] source: &str,
    #[case] expected: Value,
) {
    let upstream = upstream([chunks_response(json!([]))]).await;

    perform(with_source(
        ocr_request(model, &upstream.uri(), options),
        source,
    ))
    .await
    .unwrap();

    let sent = only_request(&upstream).await;
    assert_eq!(sent.url.path(), "/parse");
    assert_eq!(sent.json(), expected);
}

#[rstest]
#[tokio::test]
async fn an_inline_document_is_uploaded_as_multipart_then_parsed(
    #[values("parse-v3", "parse-legacy")] model: &str,
    #[values("application/pdf", "image/png")] mime_type: &str,
) {
    let upstream = upstream([
        upload_response(),
        chunks_response(json!([{"content": "hello"}])),
    ])
    .await;
    let data_uri = format!("data:{mime_type};base64,YWJj");
    let document = match mime_type.starts_with("image/") {
        true => json!({"type": "image_url", "image_url": data_uri}),
        false => json!({"type": "document_url", "document_url": data_uri}),
    };
    let request = with_headers(
        ocr_request_with_document(
            &format!("reducto/{model}"),
            &upstream.uri(),
            document,
            json!({}),
        ),
        &[
            ("Content-Type", "application/json"),
            ("X-Trace", "upload-test"),
        ],
    );

    let response = perform(request).await.unwrap();

    assert_eq!(response.pages[0].markdown, "hello");
    let requests = received(&upstream).await;
    let [upload, parse] = requests.as_slice() else {
        panic!(
            "expected an upload and a parse, got {} requests",
            requests.len()
        );
    };
    assert_eq!(upload.url.path(), "/upload");
    assert!(
        upload
            .header("content-type")
            .is_some_and(|value| value.starts_with("multipart/form-data; boundary=")),
        "{:?}",
        upload.header("content-type")
    );
    assert_eq!(upload.header("x-trace"), Some("upload-test"));
    let multipart = upload.body_text();
    assert!(
        multipart.contains(&format!("Content-Type: {mime_type}\r\n")),
        "{multipart}"
    );
    assert!(multipart.contains("\r\n\r\nabc\r\n--"), "{multipart}");
    assert_eq!(parse.url.path(), "/parse");
    assert_eq!(
        parse.json(),
        json!({source_field(model): "reducto://uploaded.pdf"})
    );
    for request in &requests {
        assert_eq!(request.header("authorization"), Some("Bearer test-key"));
    }
}

#[tokio::test]
async fn response_received_fires_once_for_the_parse_response() {
    let upstream = upstream([upload_response(), chunks_response(json!([]))]).await;
    let observed = Arc::new(Mutex::new(Vec::new()));
    let recorder = observed.clone();
    let host = LocalOcrHost::new(ocr_request("reducto/parse-v3", &upstream.uri(), json!({})))
        .with_observer(move |event| {
            if let CallEvent::Machine(MachineEvent::ResponseReceived { raw }) = event {
                recorder.lock().unwrap().push(raw.body.clone());
            }
        });

    perform_with(host).await.unwrap();

    assert_eq!(received(&upstream).await.len(), 2);
    assert_eq!(*observed.lock().unwrap(), [r#"{"result":{"chunks":[]}}"#]);
}

#[rstest]
#[case::empty_id(json_response(json!({"file_id": ""})))]
#[case::missing_id(json_response(json!({})))]
#[case::null_id(json_response(json!({"file_id": null})))]
#[case::upload_failure(status_response(503, json!({"error": "unavailable"})))]
#[tokio::test]
async fn a_failed_upload_stops_before_parse(#[case] upload: ResponseTemplate) {
    let upstream = upstream([upload]).await;

    let result = perform(ocr_request("reducto/parse-v3", &upstream.uri(), json!({}))).await;

    assert!(result.is_err());
    assert_eq!(received(&upstream).await.len(), 1);
}

#[rstest]
#[case::remote_url("https://example.com/a.pdf", Error::ReductoSource)]
#[case::empty_file_id("reducto://", Error::RequestField { path: "document file id".into() })]
#[case::data_uri_without_payload("data:application/pdf;base64", Error::InvalidDataUri)]
#[case::invalid_base64("data:application/pdf;base64,INVALID!", Error::InvalidDataUri)]
#[tokio::test]
async fn invalid_document_sources_are_rejected_before_sending(
    #[case] source: &str,
    #[case] expected: Error,
) {
    let upstream = upstream([json_response(json!({}))]).await;

    let result = perform(with_source(
        ocr_request("reducto/parse-v3", &upstream.uri(), json!({})),
        source,
    ))
    .await;

    assert!(
        received(&upstream).await.is_empty(),
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

#[tokio::test]
async fn a_forwarded_authorization_wins_and_the_native_response_is_omitted_by_default() {
    let upstream = upstream([json_response(
        json!({"job_id": "job-1", "result": {"chunks": []}}),
    )])
    .await;
    let request = with_headers(
        with_source(
            ocr_request("reducto/parse-v3", &upstream.uri(), json!({})),
            "reducto://ready.pdf",
        ),
        &[("authorization", "Bearer existing")],
    );

    let response = perform(request).await.unwrap();

    assert_eq!(response.provider_native_response, None);
    assert_eq!(
        only_request(&upstream).await.header_values("authorization"),
        ["Bearer existing"]
    );
}

#[tokio::test]
async fn native_format_retains_the_provider_response() {
    let raw = json!({
        "result": {"chunks": [{"content": "native OCR response"}]},
        "usage": {"num_pages": 1}
    });
    let upstream = upstream([json_response(raw.clone())]).await;

    let response = perform(with_source(
        ocr_request(
            "reducto/parse-v3",
            &upstream.uri(),
            json!({"req_format": "native"}),
        ),
        "reducto://ready.pdf",
    ))
    .await
    .unwrap();

    assert_eq!(response.pages[0].markdown, "native OCR response");
    assert_eq!(
        response.provider_native_response.map(Value::Object),
        Some(raw)
    );
}

#[tokio::test]
async fn an_unknown_model_reaches_parse_and_keeps_its_name() {
    let upstream = upstream([chunks_response(
        json!([{"content": "future model response"}]),
    )])
    .await;

    let response = perform(with_source(
        ocr_request("reducto/future-parse-model", &upstream.uri(), json!({})),
        "reducto://ready.pdf",
    ))
    .await
    .unwrap();

    assert_eq!(response.model, "future-parse-model");
    assert_eq!(response.pages[0].markdown, "future model response");
    let sent = only_request(&upstream).await;
    assert_eq!(sent.url.path(), "/parse");
    assert_eq!(sent.json(), json!({"input": "reducto://ready.pdf"}));
}

#[tokio::test]
async fn a_guardrail_can_replace_the_document_before_upload() {
    let upstream = upstream([chunks_response(json!([]))]).await;
    let host = LocalOcrHost::new(ocr_request("reducto/parse-v3", &upstream.uri(), json!({})))
        .with_before_send(|wire, _| {
            assert_eq!(wire.body["document_url"], INLINE_PDF);
            Ok(WireRequest {
                body: json!({"type": "document_url", "document_url": "reducto://guarded.pdf"}),
                ..wire
            })
        });

    perform_with(host).await.unwrap();

    let sent = only_request(&upstream).await;
    assert_eq!(sent.url.path(), "/parse");
    assert_eq!(sent.json(), json!({"input": "reducto://guarded.pdf"}));
}

#[rstest]
#[tokio::test]
async fn guardrail_headers_reach_both_upload_and_parse(
    #[values("reducto/parse-v3", "reducto/parse-legacy")] model: &str,
) {
    let upstream = upstream([upload_response(), chunks_response(json!([]))]).await;
    let request = with_headers(
        ocr_request(model, &upstream.uri(), json!({})),
        &[("authorization", "Bearer original")],
    );
    let host = LocalOcrHost::new(request).with_before_send(|wire, _| {
        Ok(WireRequest {
            headers: vec![("authorization".into(), "Bearer guarded".into())],
            ..wire
        })
    });

    perform_with(host).await.unwrap();

    let requests = received(&upstream).await;
    let paths: Vec<&str> = requests.iter().map(|request| request.url.path()).collect();
    assert_eq!(paths, ["/upload", "/parse"]);
    for request in &requests {
        assert_eq!(request.header_values("authorization"), ["Bearer guarded"]);
    }
}
