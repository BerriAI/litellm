use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use litellm_auth::{InputSource, Sourced};
use litellm_host::{interceptors::WireRequest, lifecycle::ExecutionEvent};
use rstest::rstest;
use wiremock::Request;

use super::*;

const MARKDOWN: &str =
    "--- Page 1 ---\nInvoice SYN-01\n\n--- Page 2 ---\nTerms\n\n--- Page 3 ---\nDelivery note";
const PUBLIC_PDF: &str = "https://docs.example/files/report.pdf";

fn job(status: &str, source: Value, result: Value) -> Value {
    json!({
        "job_id": "job-1",
        "kind": "parse",
        "status": status,
        "source": source,
        "result": result,
        "error": null,
        "units": if status == "succeeded" { json!(3) } else { Value::Null },
        "cost": {"currency": "USD", "total_usd": "0.0000", "lines": []}
    })
}

fn url_source() -> Value {
    json!({"type": "url", "url": PUBLIC_PDF, "file_name": "report.pdf"})
}

fn workspace_source() -> Value {
    json!({"type": "workspace_file", "workspace_id": "workspace-1", "file_id": "file-1"})
}

fn parsed(markdown: &str, extra: Value) -> Value {
    let mut result = json!({
        "result_type": "parse",
        "lane": "pdf",
        "document": {
            "page_count": 3,
            "markdown": markdown,
            "content_url": null,
            "blocks": [{
                "id": "blk_2_1",
                "type": "text",
                "content": "Terms",
                "location": {"kind": "page_region", "page": 2, "polygons": []}
            }]
        },
        "units": 3,
        "disclosures": []
    });
    merge(&mut result, extra);
    result
}

fn merge(target: &mut Value, patch: Value) {
    match (target, patch) {
        (Value::Object(target), Value::Object(patch)) => {
            for (key, value) in patch {
                merge(target.entry(key).or_insert(Value::Null), value);
            }
        }
        (target, patch) => *target = patch,
    }
}

fn succeeded(source: Value) -> ResponseTemplate {
    json_response(job("succeeded", source, parsed(MARKDOWN, json!({}))))
}

fn grant(upload_url: &str) -> ResponseTemplate {
    json_response(json!({
        "workspace_id": "workspace-1",
        "upload_url": upload_url,
        "transient": false,
        "token": "upload-token",
        "expires_at": "2026-01-01T00:10:00Z",
        "max_bytes": 1048576
    }))
}

fn uploaded() -> ResponseTemplate {
    json_response(json!({
        "job_id": "upload-job-1",
        "kind": "upload",
        "status": "succeeded",
        "result": {
            "file": {"workspace_id": "workspace-1", "file_id": "file-1", "path": "litellm/file.pdf"},
            "deduplicated": false
        }
    }))
}

fn deleted() -> ResponseTemplate {
    json_response(json!({
        "object": "file_deletion",
        "workspace_id": "workspace-1",
        "file_id": "file-1",
        "status": "deleted"
    }))
}

fn job_deleted() -> ResponseTemplate {
    ResponseTemplate::new(204)
}

fn nace_request(model: &str, base: &str, options: Value) -> LiteLLMOcrRequest {
    with_headers(
        ocr_request(model, base, options),
        &[("X-Trace", "nace-test")],
    )
}

fn calls(requests: &[Request]) -> Vec<String> {
    requests
        .iter()
        .map(|request| format!("{} {}", request.method, request.url.path()))
        .collect()
}

/// Scripts an upload of the inline PDF followed by `rest` on one upstream.
async fn upload_upstream(rest: impl IntoIterator<Item = ResponseTemplate>) -> MockServer {
    let server = MockServer::start().await;
    let upload_url = format!("{}/upload/file", server.uri());
    respond_in_order(
        &server,
        [grant(&upload_url), uploaded()].into_iter().chain(rest),
    )
    .await;
    server
}

#[rstest]
#[case::low("nace/parse-low", "low")]
#[case::medium("nace/parse-medium", "medium")]
#[case::high("nace/parse-high", "high")]
#[tokio::test]
async fn a_public_link_is_parsed_by_url_in_one_call(#[case] model: &str, #[case] mode: &str) {
    let upstream = upstream([succeeded(url_source()), job_deleted()]).await;

    let response = perform(with_source(
        nace_request(
            model,
            &upstream.uri(),
            json!({"page_ranges": [{"start": 1, "end": 3}]}),
        ),
        PUBLIC_PDF,
    ))
    .await
    .unwrap();

    let requests = received(&upstream).await;
    assert_eq!(
        calls(&requests),
        [
            "POST /v1/documents/parse",
            "DELETE /v1/documents/jobs/job-1"
        ]
    );
    assert_eq!(requests[1].header("authorization"), Some("Bearer test-key"));
    assert_eq!(requests[1].header("x-trace"), None);
    let sent = &requests[0];
    assert_eq!(sent.target(), "/v1/documents/parse?wait_seconds=1");
    assert_eq!(sent.header("authorization"), Some("Bearer test-key"));
    assert_eq!(
        sent.json(),
        json!({
            "source": url_source(),
            "parse_mode": mode,
            "output": {"include_page_markers": true},
            "page_ranges": [{"start": 1, "end": 3}]
        })
    );
    let pages: Vec<(i64, &str)> = response
        .pages
        .iter()
        .map(|page| (page.index, page.markdown.as_str()))
        .collect();
    assert_eq!(
        pages,
        [(0, "Invoice SYN-01"), (1, "Terms"), (2, "Delivery note")]
    );
    assert_eq!(response.pages[1].extra_fields["blocks"][0]["id"], "blk_2_1");
    assert_eq!(response.usage_info.unwrap().pages_processed, Some(3));
    assert_eq!(response.model, model.split_once('/').unwrap().1);
}

#[tokio::test]
async fn an_inline_document_is_uploaded_parsed_by_file_and_deleted() {
    let upstream = upload_upstream([succeeded(workspace_source()), job_deleted(), deleted()]).await;

    let response = perform(nace_request("nace/parse-low", &upstream.uri(), json!({})))
        .await
        .unwrap();

    assert_eq!(response.pages.len(), 3);
    let requests = received(&upstream).await;
    assert_eq!(
        calls(&requests),
        [
            "POST /v1/documents/upload-grants",
            "POST /upload/file",
            "POST /v1/documents/parse",
            "DELETE /v1/documents/jobs/job-1",
            "DELETE /v1/documents/workspaces/workspace-1/files/file-1",
        ]
    );
    let [grant, upload, parse, _, delete] = requests.as_slice() else {
        unreachable!()
    };
    let path = grant.json()["path"].as_str().unwrap().to_owned();
    assert!(
        path.starts_with("litellm/") && path.ends_with(".pdf"),
        "{path}"
    );
    assert_eq!(grant.json()["ttl_seconds"], 600);
    assert_eq!(upload.header("x-upload-token"), Some("upload-token"));
    let multipart = upload.body_text();
    assert!(
        multipart.contains("Content-Type: application/pdf\r\n"),
        "{multipart}"
    );
    assert!(multipart.contains("\r\n\r\nabc\r\n--"), "{multipart}");
    assert!(
        multipart.contains(&json!({"path": path, "total_size_bytes": 3}).to_string()),
        "{multipart}"
    );
    assert_eq!(parse.json()["source"], workspace_source());
    assert_eq!(
        [grant, parse].map(|request| (request.header("authorization"), request.header("x-trace"))),
        [(Some("Bearer test-key"), Some("nace-test")); 2]
    );
    assert_eq!(delete.header("authorization"), Some("Bearer test-key"));
    assert_eq!(delete.header("x-trace"), None);
}

#[rstest]
#[case::boolean_false(json!({"include_image_base64": false}))]
#[case::dropped(json!({"include_image_base64": true, "drop_params": true}))]
#[tokio::test]
async fn an_open_webui_request_is_parsed_without_include_image_base64(#[case] options: Value) {
    let upstream = upload_upstream([succeeded(workspace_source()), job_deleted(), deleted()]).await;

    let response = perform(nace_request("nace/parse-low", &upstream.uri(), options))
        .await
        .unwrap();

    assert_eq!(
        page_texts(&response),
        [(0, "Invoice SYN-01"), (1, "Terms"), (2, "Delivery note")]
    );
    let requests = received(&upstream).await;
    assert_eq!(requests[2].url.path(), "/v1/documents/parse");
    assert_eq!(requests[2].json().get("include_image_base64"), None);
    assert_eq!(requests[2].json().get("drop_params"), None);
}

#[rstest]
#[case::base64(json!({"include_image_base64": true}), "include_image_base64")]
#[case::links(json!({"output": {"include_images": true}}), "output.include_images")]
#[tokio::test]
async fn an_image_request_is_refused_before_anything_is_uploaded(
    #[case] options: Value,
    #[case] refused: &str,
) {
    let upstream = upstream([]).await;

    let error = perform(nace_request("nace/parse-low", &upstream.uri(), options))
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains(&format!("unsupported OCR param {refused}")),
        "{error}"
    );
    assert!(received(&upstream).await.is_empty());
}

#[tokio::test]
async fn the_upload_carries_only_the_grant_token() {
    let upstream = upload_upstream([succeeded(workspace_source()), job_deleted(), deleted()]).await;

    perform(nace_request("nace/parse-low", &upstream.uri(), json!({})))
        .await
        .unwrap();

    let requests = received(&upstream).await;
    let upload = &requests[1];
    assert_eq!(upload.url.path(), "/upload/file");
    assert_eq!(upload.header("authorization"), None);
    assert_eq!(upload.header("x-trace"), None);
    assert_eq!(upload.header("user-agent"), Some("litellm"));
    assert!(
        !upload.body_text().contains("test-key"),
        "the API key reached the upload"
    );
}

#[rstest]
#[case::plain_http_host("http://uploads.example/file")]
#[case::other_local_port("{other}/file")]
#[case::credentials("https://user:pass@uploads.example/file")]
#[tokio::test]
async fn an_upload_url_off_the_api_base_is_refused_before_the_file_is_sent(
    #[case] upload_url: &str,
) {
    let other = upstream([uploaded()]).await;
    let upload_url = upload_url.replace("{other}", &other.uri());
    let upstream = upstream([grant(&upload_url)]).await;

    let error = perform(nace_request("nace/parse-low", &upstream.uri(), json!({})))
        .await
        .unwrap_err();

    assert!(error.to_string().contains("upload_url"), "{error}");
    assert!(received(&other).await.is_empty());
    assert_eq!(
        calls(&received(&upstream).await),
        ["POST /v1/documents/upload-grants"]
    );
}

#[rstest]
#[case::private_network("https://10.1.2.3/file")]
#[case::link_local("https://169.254.169.254/latest/file")]
#[case::loopback("https://127.0.0.1:9/file")]
#[case::loopback_name("https://localhost:9/file")]
#[tokio::test]
async fn an_upload_url_on_an_internal_address_is_refused_by_the_url_policy(
    #[case] upload_url: &str,
) {
    let upstream = upstream([grant(upload_url)]).await;
    let route = build_ocr_route(
        &resources(),
        &http_config(),
        litellm_http::media::UrlPolicy::default(),
        OcrSettings::default(),
        no_secrets(),
    );

    let error = route
        .execute(
            nace_request("nace/parse-low", &upstream.uri(), json!({})),
            &(),
            None,
        )
        .await
        .unwrap_err();

    assert!(error.to_string().contains("upload_url"), "{error}");
    assert_eq!(
        calls(&received(&upstream).await),
        ["POST /v1/documents/upload-grants"]
    );
}

#[tokio::test]
async fn a_configured_user_agent_reaches_the_upload_unchanged() {
    let upstream = upload_upstream([succeeded(workspace_source()), job_deleted(), deleted()]).await;
    let route = build_ocr_route(
        &resources(),
        &litellm_http::HttpClientConfig {
            user_agent: Some("operator-gateway/1.0".into()),
            ..http_config()
        },
        litellm_http::media::UrlPolicy {
            validate: false,
            allowed_hosts: Vec::new(),
        },
        OcrSettings::default(),
        no_secrets(),
    );

    route
        .execute(
            nace_request("nace/parse-low", &upstream.uri(), json!({})),
            &(),
            None,
        )
        .await
        .unwrap();

    let requests = received(&upstream).await;
    assert_eq!(requests[1].url.path(), "/upload/file");
    assert_eq!(
        requests[1].header("user-agent"),
        Some("operator-gateway/1.0")
    );
}

#[tokio::test]
async fn a_queued_job_is_polled_until_it_ends() {
    let running = job("running", url_source(), Value::Null);
    let done = job("succeeded", url_source(), parsed(MARKDOWN, json!({})));
    let upstream = upstream([
        ResponseTemplate::new(202).set_body_json(job("queued", url_source(), Value::Null)),
        json_response(running).insert_header("Retry-After", "0"),
        json_response(done.clone()),
        job_deleted(),
    ])
    .await;
    let observed = Arc::new(Mutex::new(Vec::new()));
    let recorder = observed.clone();
    let host = LocalOcrHost::new(with_source(
        nace_request(
            "nace/parse-low",
            &upstream.uri(),
            json!({"req_format": "native"}),
        ),
        PUBLIC_PDF,
    ))
    .with_observer(move |event| {
        if let CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { raw }) = event {
            recorder.lock().unwrap().push(raw.body.clone());
        }
    });

    let response = perform_with(host).await.unwrap();

    let requests = received(&upstream).await;
    assert_eq!(
        calls(&requests),
        [
            "POST /v1/documents/parse",
            "GET /v1/documents/jobs/job-1",
            "GET /v1/documents/jobs/job-1",
            "DELETE /v1/documents/jobs/job-1",
        ]
    );
    assert_eq!(
        requests[1..]
            .iter()
            .map(|call| (call.header("authorization"), call.header("x-trace")))
            .collect::<Vec<_>>(),
        [(Some("Bearer test-key"), None); 3]
    );
    assert_eq!(response.pages.len(), 3);
    assert_eq!(
        response.provider_native_response.map(Value::Object),
        Some(done)
    );
    let observed = observed.lock().unwrap();
    assert_eq!(observed.len(), 2, "the create and the finished poll");
    assert!(observed[1].contains("\"succeeded\""), "{}", observed[1]);
    assert!(!observed[0].contains(PUBLIC_PDF), "{}", observed[0]);
    assert!(!observed[1].contains(PUBLIC_PDF), "{}", observed[1]);
}

#[tokio::test]
async fn a_poll_timeout_cancels_the_job_and_deletes_the_uploaded_file() {
    let upstream = upload_upstream([
        ResponseTemplate::new(202).set_body_json(job("queued", workspace_source(), Value::Null)),
        job_deleted(),
        deleted(),
    ])
    .await;
    let route = ocr_route_with(OcrSettings {
        poll_timeout: Duration::from_millis(100),
        ..OcrSettings::default()
    });

    let error = tokio::time::timeout(
        Duration::from_secs(1),
        route.execute(
            nace_request("nace/parse-low", &upstream.uri(), json!({})),
            &(),
            None,
        ),
    )
    .await
    .expect("the deadline cuts the poll delay short")
    .unwrap_err();

    assert!(error.to_string().contains("timed out"), "{error}");
    assert_eq!(error.http_status_code(), None);
    assert_eq!(
        calls(&received(&upstream).await)[3..],
        [
            "DELETE /v1/documents/jobs/job-1",
            "DELETE /v1/documents/workspaces/workspace-1/files/file-1"
        ]
    );
}

#[tokio::test]
async fn a_failed_job_is_a_bad_request_and_the_uploaded_file_is_deleted() {
    let mut failed = job("failed", workspace_source(), Value::Null);
    failed["error"] = json!({
        "code": "corrupt_file",
        "message": "The uploaded file is not a readable PDF.",
        "detail": null,
        "retryable": false,
        "request_id": null
    });
    let upstream = upload_upstream([json_response(failed), job_deleted(), deleted()]).await;

    let error = perform(nace_request("nace/parse-low", &upstream.uri(), json!({})))
        .await
        .unwrap_err();

    assert_eq!(error.http_status_code(), Some(400));
    assert!(
        error
            .to_string()
            .contains("corrupt_file: The uploaded file is not a readable PDF."),
        "{error}"
    );
    assert_eq!(
        calls(&received(&upstream).await)[3..],
        [
            "DELETE /v1/documents/jobs/job-1",
            "DELETE /v1/documents/workspaces/workspace-1/files/file-1"
        ]
    );
}

#[tokio::test]
async fn a_failed_delete_does_not_fail_the_parse() {
    let upstream = upload_upstream([
        succeeded(workspace_source()),
        status_response(503, json!({"error": {"code": "unavailable"}})),
        status_response(503, json!({"error": {"code": "unavailable"}})),
    ])
    .await;

    let response = perform(nace_request("nace/parse-low", &upstream.uri(), json!({})))
        .await
        .unwrap();

    assert_eq!(response.pages.len(), 3);
    assert_eq!(received(&upstream).await.len(), 5);
}

#[rstest]
#[case::invalid_key(
    401,
    json!({"error": {"type": "authentication_error", "code": "invalid_api_key", "message": "Invalid API key.", "retryable": false}, "request_id": "request-1"}),
    "Invalid API key."
)]
#[case::invalid_request(
    422,
    json!({"error": {"type": "invalid_request_error", "code": "unsupported_file_type", "message": "Unsupported file type.", "retryable": false}, "request_id": "request-1"}),
    "unsupported_file_type"
)]
#[tokio::test]
async fn nace_error_envelopes_keep_their_status_and_message(
    #[case] status: u16,
    #[case] body: Value,
    #[case] message: &str,
) {
    let upstream = upstream([status_response(status, body)]).await;

    let error = perform(with_source(
        nace_request("nace/parse-low", &upstream.uri(), json!({})),
        PUBLIC_PDF,
    ))
    .await
    .unwrap_err();

    assert_eq!(error.http_status_code(), Some(status));
    let report = litellm_error::Report::new(&error).to_string();
    assert!(report.contains(message), "{report}");
}

#[rstest]
#[case::litellm(json!({}), false)]
#[case::native(json!({"req_format": "native"}), true)]
#[tokio::test]
async fn a_truncated_result_is_read_in_full_from_the_content_url(
    #[case] options: Value,
    #[case] native: bool,
) {
    let server = MockServer::start().await;
    let content_url = format!(
        "{}/v1/documents/jobs/job-1/document-content/full",
        server.uri()
    );
    let preview = job(
        "succeeded",
        url_source(),
        parsed(
            "--- Page 1 ---\npreview",
            json!({
                "document": {"content_url": content_url},
                "disclosures": [{
                    "code": "content_truncated",
                    "target": "document_content",
                    "unit": "characters",
                    "original_size": 40,
                    "retained_size": 20
                }]
            }),
        ),
    );
    respond_in_order(
        &server,
        [
            json_response(preview),
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/markdown; charset=utf-8")
                .set_body_string(
                    "--- Page 1 ---\nfull one\n\n--- Page 2 ---\nfull two\n\n--- Page 3 ---\nfull three",
                ),
            job_deleted(),
        ],
    )
    .await;

    let response = perform(with_source(
        nace_request("nace/parse-low", &server.uri(), options),
        PUBLIC_PDF,
    ))
    .await
    .unwrap();

    let document = response
        .provider_native_response
        .as_ref()
        .map(|job| &job["result"]["document"]);
    assert_eq!(
        document.map(|document| document["markdown"].as_str()),
        native.then_some(Some(
            "--- Page 1 ---\nfull one\n\n--- Page 2 ---\nfull two\n\n--- Page 3 ---\nfull three"
        ))
    );
    assert_eq!(
        document.and_then(|document| document.get("content_url")),
        None
    );
    let markdown: Vec<&str> = response
        .pages
        .iter()
        .map(|page| page.markdown.as_str())
        .collect();
    assert_eq!(markdown, ["full one", "full two", "full three"]);
    let requests = received(&server).await;
    assert_eq!(
        calls(&requests)[1],
        "GET /v1/documents/jobs/job-1/document-content/full"
    );
    assert_eq!(requests[1].header("authorization"), Some("Bearer test-key"));
}

#[rstest]
#[case::other_origin("{other}/v1/documents/jobs/job-1/document-content/full")]
#[case::other_path_on_the_api_base("{base}/v1/documents/workspaces/workspace-1/files/file-1")]
#[case::other_job_on_the_api_base("{base}/v1/documents/jobs/job-2/document-content/full")]
#[tokio::test]
async fn a_content_url_outside_the_jobs_files_is_not_fetched(#[case] content_url: &str) {
    let other = upstream([json_response(json!({}))]).await;
    let server = MockServer::start().await;
    let content_url = content_url
        .replace("{other}", &other.uri())
        .replace("{base}", &server.uri());
    respond_in_order(
        &server,
        [
            json_response(job(
                "succeeded",
                url_source(),
                parsed(
                    "--- Page 1 ---\npreview",
                    json!({
                        "document": {"content_url": content_url},
                        "disclosures": [{"code": "content_truncated", "target": "document_content"}]
                    }),
                ),
            )),
            job_deleted(),
        ],
    )
    .await;

    let error = perform(with_source(
        nace_request("nace/parse-low", &server.uri(), json!({})),
        PUBLIC_PDF,
    ))
    .await
    .unwrap_err();

    assert!(error.to_string().contains("content_url"), "{error}");
    assert!(received(&other).await.is_empty());
    assert_eq!(
        calls(&received(&server).await),
        [
            "POST /v1/documents/parse",
            "DELETE /v1/documents/jobs/job-1"
        ]
    );
}

#[tokio::test]
async fn a_request_controlled_api_base_does_not_receive_the_operator_key() {
    let upstream = upstream([succeeded(url_source())]).await;
    let mut request = with_source(
        nace_request("nace/parse-low", &upstream.uri(), json!({})),
        PUBLIC_PDF,
    );
    request.credentials.api_key = Some(Sourced::new(
        litellm_secret_value::SecretValue::new("operator-key"),
        InputSource::Deployment,
    ));
    request.credentials.api_base = Some(Sourced::new(upstream.uri(), InputSource::Request));

    let error = perform(request).await.unwrap_err();

    let report = litellm_error::Report::new(&error).to_string();
    assert!(
        report.contains("request-controlled Nace endpoint"),
        "{report}"
    );
    assert!(received(&upstream).await.is_empty());
}

#[rstest]
#[case::top_level(json!({"parse_mode": "high"}))]
#[case::extra_body(json!({"extra_body": {"parse_mode": "high"}}))]
#[case::extra_body_output(json!({"extra_body": {"output": {"include_page_markers": false}}}))]
#[case::extra_body_not_an_object(json!({"extra_body": "page_ranges=1"}))]
#[tokio::test]
async fn billing_and_page_split_settings_are_refused_before_any_upload(#[case] options: Value) {
    let upstream = upstream([grant("https://uploads.example/file")]).await;

    let error = perform(nace_request("nace/parse-low", &upstream.uri(), options))
        .await
        .unwrap_err();

    assert_eq!(error.http_status_code(), Some(400));
    assert!(received(&upstream).await.is_empty());
}

#[tokio::test]
async fn an_unknown_effort_is_refused() {
    let error = perform(nace_request("nace/parse-max", UNREACHABLE_BASE, json!({})))
        .await
        .unwrap_err();

    assert!(
        matches!(
            error,
            Error::InvalidModel {
                provider: "nace",
                ..
            }
        ),
        "{error}"
    );
}

#[tokio::test]
async fn guardrails_see_the_source_document_before_it_is_uploaded() {
    let upstream = upstream([succeeded(url_source()), job_deleted()]).await;
    let host = LocalOcrHost::new(nace_request("nace/parse-low", &upstream.uri(), json!({})))
        .with_before_send(|wire, _| {
            assert_eq!(wire.body["document_url"], INLINE_PDF);
            Ok(WireRequest {
                body: json!({"type": "document_url", "document_url": PUBLIC_PDF}),
                ..wire
            })
        });

    perform_with(host).await.unwrap();

    let requests = received(&upstream).await;
    let sent = &requests[0];
    assert_eq!(sent.url.path(), "/v1/documents/parse");
    assert_eq!(sent.json()["source"], url_source());
}

#[rstest]
#[case::error_status(status_response(
    402,
    json!({"error": {"type": "billing_error", "code": "insufficient_credit", "message": "No spendable credit.", "retryable": false}})
))]
#[case::timeout(succeeded(workspace_source()).set_delay(Duration::from_secs(5)))]
#[tokio::test]
async fn a_create_that_fails_still_deletes_the_uploaded_file(#[case] create: ResponseTemplate) {
    let upstream = upload_upstream([create, deleted()]).await;

    perform(nace_request("nace/parse-low", &upstream.uri(), json!({})))
        .await
        .unwrap_err();

    assert_eq!(
        calls(&received(&upstream).await),
        [
            "POST /v1/documents/upload-grants",
            "POST /upload/file",
            "POST /v1/documents/parse",
            "DELETE /v1/documents/workspaces/workspace-1/files/file-1",
        ]
    );
}

#[tokio::test]
async fn a_result_that_zero_retention_did_not_keep_is_a_bad_request() {
    let mut not_retained = job("succeeded", url_source(), Value::Null);
    not_retained["result_state"] = json!("not_retained");
    let upstream = upstream([
        ResponseTemplate::new(202).set_body_json(job("queued", url_source(), Value::Null)),
        json_response(not_retained),
        job_deleted(),
    ])
    .await;

    let error = perform(with_source(
        nace_request("nace/parse-low", &upstream.uri(), json!({})),
        PUBLIC_PDF,
    ))
    .await
    .unwrap_err();

    assert_eq!(error.http_status_code(), Some(400));
    let report = litellm_error::Report::new(&error).to_string();
    assert!(report.contains("zero retention"), "{report}");
}

#[tokio::test]
async fn nace_error_text_drops_url_queries() {
    let upstream = upstream([status_response(
        422,
        json!({"error": {
            "code": "fetch_failed",
            "message": "Could not read https://bucket.example/a.pdf?X-Amz-Signature=abc123 in time.",
            "retryable": false
        }}),
    )])
    .await;

    let error = perform(with_source(
        nace_request("nace/parse-low", &upstream.uri(), json!({})),
        PUBLIC_PDF,
    ))
    .await
    .unwrap_err();

    let report = litellm_error::Report::new(&error).to_string();
    assert!(
        report.contains("https://bucket.example/a.pdf in time."),
        "{report}"
    );
    assert!(!report.contains("X-Amz-Signature"), "{report}");
}

#[tokio::test]
async fn a_text_file_that_quotes_page_markers_stays_one_page() {
    let text = "notes\n--- Page 1 ---\n--- Page 2 ---\nend";
    let mut result = parsed(text, json!({"document": {"page_count": 1, "blocks": []}}));
    result["units"] = json!(1);
    let mut done = job("succeeded", url_source(), result);
    done["units"] = json!(1);
    let upstream = upstream([json_response(done), job_deleted()]).await;

    let response = perform(with_source(
        nace_request("nace/parse-low", &upstream.uri(), json!({})),
        PUBLIC_PDF,
    ))
    .await
    .unwrap();

    assert_eq!(response.pages.len(), 1);
    assert_eq!(response.pages[0].markdown, text);
}

fn page_texts(response: &LiteLLMOcrResponse) -> Vec<(i64, &str)> {
    response
        .pages
        .iter()
        .map(|page| (page.index, page.markdown.as_str()))
        .collect()
}

#[rstest]
#[case::param(json!({"page_ranges": [{"start": 1, "end": 1}, {"start": 4, "end": 5}]}))]
#[case::extra_body(
    json!({"extra_body": {"page_ranges": [{"start": 1, "end": 1}, {"start": 4, "end": 5}]}})
)]
#[case::pages(json!({"pages": [4, 0, 3, 0]}))]
#[tokio::test]
async fn selected_pages_keep_their_document_page_numbers(#[case] options: Value) {
    let result = parsed(
        "--- Page 1 ---\nCover\n\n--- Page 4 ---\nTerms\n\n--- Page 5 ---\nDelivery note",
        json!({"document": {"blocks": [{
            "id": "blk_4_1",
            "type": "text",
            "content": "Terms",
            "location": {"kind": "page_region", "page": 4, "polygons": []}
        }]}}),
    );
    let upstream = upstream([
        json_response(job("succeeded", url_source(), result)),
        job_deleted(),
    ])
    .await;

    let response = perform(with_source(
        nace_request("nace/parse-low", &upstream.uri(), options),
        PUBLIC_PDF,
    ))
    .await
    .unwrap();

    let requests = received(&upstream).await;
    assert_eq!(requests[0].url.path(), "/v1/documents/parse");
    let body = requests[0].json();
    assert_eq!(
        body["page_ranges"],
        json!([{"start": 1, "end": 1}, {"start": 4, "end": 5}])
    );
    assert_eq!(body.get("pages"), None);
    assert_eq!(
        page_texts(&response),
        [(0, "Cover"), (3, "Terms"), (4, "Delivery note")]
    );
    assert_eq!(response.pages[1].extra_fields["blocks"][0]["id"], "blk_4_1");
    assert_eq!(response.usage_info.unwrap().pages_processed, Some(3));
}

#[rstest]
#[tokio::test]
async fn an_empty_page_list_lets_nace_parse_the_whole_document() {
    let upstream = upstream([succeeded(url_source()), job_deleted()]).await;

    let response = perform(with_source(
        nace_request("nace/parse-low", &upstream.uri(), json!({"pages": []})),
        PUBLIC_PDF,
    ))
    .await
    .unwrap();

    let requests = received(&upstream).await;
    assert_eq!(requests[0].url.path(), "/v1/documents/parse");
    let body = requests[0].json();
    assert_eq!(body.get("pages"), None);
    assert_eq!(body.get("page_ranges"), None);
    assert_eq!(response.usage_info.unwrap().pages_processed, Some(3));
}

#[rstest]
#[case::markdown(json!(["markdown"]), json!(["markdown"]))]
#[case::text(json!(["text"]), json!(["text", "markdown"]))]
#[tokio::test]
async fn an_output_format_list_still_gives_pages_and_usage(
    #[case] formats: Value,
    #[case] sent: Value,
) {
    let result = parsed(
        MARKDOWN,
        json!({"document": {"blocks": null, "text": null}, "disclosures": null}),
    );
    let upstream = upstream([
        json_response(job("succeeded", url_source(), result)),
        job_deleted(),
    ])
    .await;

    let response = perform(with_source(
        nace_request(
            "nace/parse-low",
            &upstream.uri(),
            json!({"output": {"formats": formats}}),
        ),
        PUBLIC_PDF,
    ))
    .await
    .unwrap();

    assert_eq!(
        received(&upstream).await[0].json()["output"],
        json!({"formats": sent, "include_page_markers": true})
    );
    assert_eq!(
        page_texts(&response),
        [(0, "Invoice SYN-01"), (1, "Terms"), (2, "Delivery note")]
    );
    assert_eq!(response.usage_info.unwrap().pages_processed, Some(3));
}

#[tokio::test]
async fn a_create_response_that_does_not_decode_still_deletes_its_job() {
    let upstream = upstream([
        json_response(json!({"job_id": "job-1", "status": "running", "units": "invalid"})),
        job_deleted(),
    ])
    .await;

    perform(with_source(
        nace_request("nace/parse-low", &upstream.uri(), json!({})),
        PUBLIC_PDF,
    ))
    .await
    .unwrap_err();

    assert_eq!(
        calls(&received(&upstream).await),
        [
            "POST /v1/documents/parse",
            "DELETE /v1/documents/jobs/job-1"
        ]
    );
}

#[tokio::test]
async fn a_grant_error_drops_url_queries_and_is_capped() {
    let body = format!(
        r#"{{"error": {{"message": "see https:\/\/bucket.example\/a.pdf?sig=secret and https://bucket.example/b?sig=secret {}"}}}}"#,
        "x".repeat(5000)
    );
    let upstream = upstream([ResponseTemplate::new(403).set_body_string(body)]).await;

    let error = perform(nace_request("nace/parse-low", &upstream.uri(), json!({})))
        .await
        .unwrap_err();

    assert_eq!(error.http_status_code(), Some(403));
    let report = litellm_error::Report::new(&error).to_string();
    assert!(!report.contains("sig=secret"), "{report}");
    assert!(
        report.contains("see https://bucket.example/a.pdf and https://bucket.example/b "),
        "{report}"
    );
    assert!(!report.contains(&"x".repeat(1000)), "{report}");
}

#[tokio::test]
async fn a_poll_that_names_another_job_is_refused_and_the_created_job_deleted() {
    let server = MockServer::start().await;
    let mut other = job(
        "succeeded",
        url_source(),
        parsed(
            "preview",
            json!({
                "document": {
                    "content_url": format!("{}/v1/documents/jobs/job-2/document-content/full", server.uri())
                },
                "disclosures": [{"code": "content_truncated", "target": "document_content"}]
            }),
        ),
    );
    other["job_id"] = json!("job-2");
    respond_in_order(
        &server,
        [
            ResponseTemplate::new(202).set_body_json(job("queued", url_source(), Value::Null)),
            json_response(other),
            job_deleted(),
        ],
    )
    .await;

    let error = perform(with_source(
        nace_request("nace/parse-low", &server.uri(), json!({})),
        PUBLIC_PDF,
    ))
    .await
    .unwrap_err();

    assert!(error.to_string().contains("job_id"), "{error}");
    assert_eq!(
        calls(&received(&server).await),
        [
            "POST /v1/documents/parse",
            "GET /v1/documents/jobs/job-1",
            "DELETE /v1/documents/jobs/job-1",
        ]
    );
}

#[tokio::test]
async fn dropped_image_links_are_not_requested_and_the_job_is_deleted() {
    let upstream = upload_upstream([succeeded(workspace_source()), job_deleted(), deleted()]).await;

    perform(nace_request(
        "nace/parse-low",
        &upstream.uri(),
        json!({"output": {"include_images": true}, "drop_params": true}),
    ))
    .await
    .unwrap();

    let requests = received(&upstream).await;
    assert_eq!(requests[2].json()["output"].get("include_images"), None);
    assert_eq!(
        calls(&requests)[3..],
        [
            "DELETE /v1/documents/jobs/job-1",
            "DELETE /v1/documents/workspaces/workspace-1/files/file-1"
        ]
    );
}

#[tokio::test]
async fn a_grant_redirect_is_not_followed() {
    let server = MockServer::start().await;
    respond_in_order(
        &server,
        [ResponseTemplate::new(307)
            .insert_header("Location", format!("{}/elsewhere", server.uri()).as_str())],
    )
    .await;

    perform(nace_request("nace/parse-low", &server.uri(), json!({})))
        .await
        .unwrap_err();

    assert_eq!(
        calls(&received(&server).await),
        ["POST /v1/documents/upload-grants"]
    );
}

#[rstest]
#[case::refused(status_response(
    400,
    json!({"error": {"message": "upload rejected for upload-token, try again"}}),
))]
#[case::undecodable(json_response(json!({"result": {"file": "upload-token"}})))]
#[tokio::test]
async fn an_upload_answer_never_reaches_the_caller(#[case] upload: ResponseTemplate) {
    let server = MockServer::start().await;
    respond_in_order(
        &server,
        [grant(&format!("{}/upload/file", server.uri())), upload],
    )
    .await;

    let error = perform(nace_request("nace/parse-low", &server.uri(), json!({})))
        .await
        .unwrap_err();

    let report = litellm_error::Report::new(&error).to_string();
    assert!(!report.contains("upload-token"), "{report}");
    assert!(!report.contains("upload rejected"), "{report}");
}

#[rstest]
#[case::empty("")]
#[case::blank("  ")]
#[tokio::test]
async fn a_job_without_an_id_is_refused_and_nothing_is_deleted(#[case] job_id: &str) {
    let mut done = job("succeeded", url_source(), parsed("text", json!({})));
    done["job_id"] = json!(job_id);
    let upstream = upstream([json_response(done)]).await;

    let error = perform(with_source(
        nace_request("nace/parse-low", &upstream.uri(), json!({})),
        PUBLIC_PDF,
    ))
    .await
    .unwrap_err();

    assert!(error.to_string().contains("job_id"), "{error}");
    assert_eq!(
        calls(&received(&upstream).await),
        ["POST /v1/documents/parse"]
    );
}

/// The grant error path masks the exact API key. A parse request error is masked by pattern only,
/// since the shared handler builds it without the key.
#[tokio::test]
async fn an_upload_grant_error_never_shows_the_api_key() {
    let upstream = upstream([status_response(
        403,
        json!({"error": {"code": "forbidden", "message": "key test-key and Bearer abc.def are not allowed"}, "echo": "test-key"}),
    )])
    .await;

    let error = perform(nace_request("nace/parse-low", &upstream.uri(), json!({})))
        .await
        .unwrap_err();

    let report = litellm_error::Report::new(&error).to_string();
    assert!(
        report.contains("forbidden: key **** and **** are not allowed"),
        "{report}"
    );
    assert!(!report.contains("echo"), "{report}");
}
