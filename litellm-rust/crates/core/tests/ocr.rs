use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::Command;
use std::thread;
use std::time::Duration;

use litellm_core::Error;
use litellm_core::lifecycle::CallLifecycleContext;
use litellm_core::ocr::request::build_pre_call_request;
use litellm_core::ocr::types::{OcrDocument, OcrDocumentProjection};
use litellm_core::ocr::{DefaultOcrServices, OcrAdmissionRequest as OcrRequest, OcrPreCallRequest};
use serde_json::{Value, json};

fn request() -> OcrRequest {
    OcrRequest {
        model: "mistral/mistral-ocr-latest".into(),
        custom_llm_provider: None,
        api_key: Some(" test-key ".into()),
        api_base: None,
        extra_headers: vec![],
        timeout_seconds: 2.0,
        request_format: None,
        document: OcrDocument::DocumentUrl {
            document_url: "data:application/pdf;base64,cGRm".into(),
        },
        credentials: Default::default(),
        vertex_project: Some("test-project".into()),
        vertex_location: Some("us-central1".into()),
        stream: false,
    }
}

fn body(built: &OcrPreCallRequest) -> Value {
    let mut body = built.body.structured_callback().unwrap().clone();
    body.insert(
        "document".into(),
        json!({"type": "document_url", "document_url": "data:application/pdf;base64,cGRm"}),
    );
    body.insert("include_image_base64".into(), json!(true));
    body.insert("pages".into(), json!([0, 2]));
    Value::Object(body)
}

async fn ocr(
    built: OcrPreCallRequest,
    headers: Vec<(String, String)>,
    body: Value,
) -> Result<litellm_core::ocr::OcrResponseData, Error> {
    let model = built.endpoint.model().to_string();
    let provider = built.endpoint.custom_llm_provider().to_string();
    let response = litellm_core::ocr::ocr(
        &DefaultOcrServices,
        built.endpoint.settle(
            headers,
            litellm_core::lifecycle::PreCallBody::StructuredAtSend { callback: body },
        )?,
        Default::default(),
        CallLifecycleContext::new("ocr", model, provider, "test-call"),
    )
    .await
    .into_result()?;
    serde_json::from_value(response).map_err(|error| Error::InvalidResponse(error.to_string()))
}

#[test]
fn builds_provider_template_auth_and_url() {
    let built = build_pre_call_request(OcrRequest {
        api_base: Some(" https://ocr.example/v1/ ".into()),
        extra_headers: vec![("X-Request-Id".into(), "request-1".into())],
        request_format: Some("litellm".into()),
        ..request()
    })
    .unwrap();
    assert_eq!(built.endpoint.model(), "mistral-ocr-latest");
    assert_eq!(built.endpoint.custom_llm_provider(), "mistral");
    assert_eq!(built.endpoint.url(), "https://ocr.example/v1/ocr");
    assert_eq!(built.endpoint.timeout_seconds(), 2.0);
    assert_eq!(
        built.request_body_policy(),
        litellm_core::lifecycle::RequestBodyPolicy::StructuredAtSend
    );
    assert_eq!(
        built.document_projection,
        OcrDocumentProjection::RetainedDocument
    );
    assert_eq!(
        Value::Object(built.body.structured_callback().unwrap().clone()),
        json!({"model": "mistral-ocr-latest", "document": {"type": "document_url", "document_url": "data:application/pdf;base64,cGRm"}})
    );
    assert_eq!(
        built.parameter_fields,
        litellm_core::providers::mistral::ocr::transformation::supported_ocr_params()
    );
    assert_eq!(
        built.headers,
        vec![
            ("Authorization".into(), "Bearer test-key".into()),
            ("X-Request-Id".into(), "request-1".into()),
        ]
    );
    let explicit = build_pre_call_request(OcrRequest {
        model: "mistral-ocr-latest".into(),
        custom_llm_provider: Some("mistral".into()),
        api_key: None,
        extra_headers: vec![("aUtHoRiZaTiOn".into(), "Bearer explicit".into())],
        ..request()
    })
    .unwrap();
    assert_eq!(explicit.endpoint.model(), "mistral-ocr-latest");
    assert_eq!(explicit.endpoint.url(), "https://api.mistral.ai/v1/ocr");
    assert_eq!(
        explicit.headers,
        vec![("aUtHoRiZaTiOn".into(), "Bearer explicit".into())]
    );
}

#[test]
fn build_request_environment_credentials() {
    if let Ok(case) = std::env::var("LITELLM_OCR_ENV_TEST") {
        let result = build_pre_call_request(OcrRequest {
            api_key: Some(" ".into()),
            ..request()
        });
        if case == "present" {
            assert_eq!(
                result.unwrap().headers,
                vec![("Authorization".into(), "Bearer env-key".into())]
            );
            assert_eq!(
                build_pre_call_request(request()).unwrap().headers,
                vec![("Authorization".into(), "Bearer test-key".into())]
            );
        } else {
            assert!(matches!(result, Err(Error::Auth(_))));
        }
        return;
    }
    for (case, key) in [
        ("present", Some("env-key")),
        ("absent", None),
        ("blank", Some(" ")),
    ] {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "build_request_environment_credentials"])
            .env("LITELLM_OCR_ENV_TEST", case)
            .env_remove("MISTRAL_API_KEY");
        if let Some(key) = key {
            command.env("MISTRAL_API_KEY", key);
        }
        assert!(command.status().unwrap().success());
    }
}

#[test]
fn build_request_rejects_unsupported_providers_formats_and_invalid_metadata() {
    for (provider, capability) in [("reducto", "OCR provider"), ("openai", "OCR provider")] {
        let result = build_pre_call_request(OcrRequest {
            custom_llm_provider: Some(provider.into()),
            api_key: None,
            api_base: Some("not a URL".into()),
            ..request()
        });
        assert!(matches!(result, Err(Error::Unsupported(message)) if message.contains(capability)));
    }
    for format in ["native", "json", "", "LiteLLM"] {
        assert!(matches!(
            build_pre_call_request(OcrRequest {
                request_format: Some(format.into()),
                ..request()
            }),
            Err(Error::Unsupported(_))
        ));
    }
    for timeout_seconds in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MAX] {
        assert!(matches!(
            build_pre_call_request(OcrRequest {
                timeout_seconds,
                ..request()
            }),
            Err(Error::InvalidRequest(_))
        ));
    }
    assert!(matches!(
        build_pre_call_request(OcrRequest {
            model: "mistral-ocr-latest".into(),
            ..request()
        }),
        Err(Error::InvalidProvider(_))
    ));
    assert!(matches!(
        build_pre_call_request(OcrRequest {
            api_base: Some("file:///secret".into()),
            ..request()
        }),
        Err(Error::InvalidRequest(_))
    ));
}

#[test]
fn cloud_capabilities_fail_only_when_required() {
    for model in [
        "azure_ai/mistral-ocr-latest",
        "vertex_ai/mistral-ocr-latest",
    ] {
        assert!(matches!(
            build_pre_call_request(OcrRequest {
                model: model.into(),
                api_base: Some("http://127.0.0.1:1".into()),
                document: OcrDocument::ImageUrl {
                    image_url: "https://example.test/image.png".into()
                },
                ..request()
            }),
            Err(Error::Unsupported(
                "OCR HTTP document URL to data URI conversion"
            ))
        ));
    }
    for model in [
        "azure_ai/doc-intelligence/prebuilt-read",
        "azure_ai/documentintelligence/prebuilt-layout",
    ] {
        assert!(matches!(
            build_pre_call_request(OcrRequest {
                model: model.into(),
                api_base: Some("http://127.0.0.1:1".into()),
                ..request()
            }),
            Err(Error::Unsupported(
                "Azure Document Intelligence OCR polling"
            ))
        ));
    }
    for model in [
        "azure_ai/cohere/parse-v5.0",
        "vertex_ai/cohere/parse-v5.0",
        "cohere/parse-v5.0",
    ] {
        assert!(matches!(
            build_pre_call_request(OcrRequest {
                model: model.into(),
                ..request()
            }),
            Err(Error::Unsupported(_))
        ));
    }
}

#[test]
fn cloud_credentials_use_native_keys_headers_or_narrow_acquisition_stub() {
    if std::env::var_os("LITELLM_CLOUD_OCR_ENV_TEST").is_some() {
        for (model, operation, header, key) in [
            (
                "azure_ai/mistral-ocr-latest",
                "Azure OCR credential acquisition",
                "Authorization",
                "Bearer env-azure",
            ),
            (
                "vertex_ai/mistral-ocr-latest",
                "Vertex OCR credential acquisition",
                "Authorization",
                "Bearer env-vertex",
            ),
        ] {
            let make_request = || OcrRequest {
                model: model.into(),
                api_key: None,
                api_base: Some("http://127.0.0.1:1".into()),
                ..request()
            };
            let result = build_pre_call_request(make_request());
            if std::env::var("LITELLM_CLOUD_OCR_ENV_TEST").unwrap() == "present" {
                assert_eq!(result.unwrap().headers, vec![(header.into(), key.into())]);
            } else {
                assert!(matches!(result, Err(Error::Unsupported(message)) if message == operation));
            }
            let result = build_pre_call_request(OcrRequest {
                extra_headers: vec![("aUtHoRiZaTiOn".into(), "Bearer supplied".into())],
                ..make_request()
            });
            if model.starts_with("azure_ai/") {
                if std::env::var("LITELLM_CLOUD_OCR_ENV_TEST").unwrap() == "present" {
                    assert_eq!(
                        result.unwrap().headers,
                        vec![
                            ("Authorization".into(), "Bearer env-azure".into()),
                            ("aUtHoRiZaTiOn".into(), "Bearer supplied".into()),
                        ]
                    );
                } else {
                    assert!(matches!(result, Err(Error::InvalidRequest(_))));
                }
                continue;
            }
            assert_eq!(
                result.unwrap().headers,
                vec![("aUtHoRiZaTiOn".into(), "Bearer supplied".into())]
            );
        }
        return;
    }
    for case in ["present", "absent"] {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "cloud_credentials_use_native_keys_headers_or_narrow_acquisition_stub",
            ])
            .env("LITELLM_CLOUD_OCR_ENV_TEST", case)
            .env_remove("AZURE_AI_API_KEY")
            .env_remove("VERTEX_AI_API_KEY")
            .env_remove("VERTEXAI_API_KEY");
        if case == "present" {
            command
                .env("AZURE_AI_API_KEY", "env-azure")
                .env("VERTEX_AI_API_KEY", "env-vertex");
        }
        assert!(command.status().unwrap().success());
    }
}

#[tokio::test]
async fn cloud_providers_use_existing_auth_urls_and_request_response_transforms() {
    for (model, path, auth_name, auth_value, deepseek) in [
        (
            "azure_ai/mistral-ocr-latest",
            "/providers/mistral/azure/ocr",
            "authorization",
            "Bearer  test-key ",
            false,
        ),
        (
            "vertex_ai/mistral-ocr-latest",
            "/v1/projects/test-project/locations/us-central1/publishers/mistralai/models/mistral-ocr-latest:rawPredict",
            "authorization",
            "Bearer test-key",
            false,
        ),
        (
            "vertex_ai/deepseek-ai/deepseek-ocr-maas",
            "/v1/projects/test-project/locations/us-central1/endpoints/openapi/chat/completions",
            "authorization",
            "Bearer test-key",
            true,
        ),
    ] {
        let response = if deepseek {
            json!({"choices": [{"message": {"content": "proof"}}], "usage": {"pages_processed": 1}})
        } else {
            json!({"pages": [{"index": 0, "markdown": "proof"}], "usage_info": {"pages_processed": 1}})
        };
        let (base, handle) = server(200, "", &response.to_string(), Duration::ZERO);
        let built = build_pre_call_request(OcrRequest {
            model: model.into(),
            api_base: Some(base),
            ..request()
        })
        .unwrap();
        let mut body = Value::Object(built.body.structured_callback().unwrap().clone());
        body[if deepseek {
            "temperature"
        } else {
            "include_image_base64"
        }] = if deepseek { json!(0.1) } else { json!(true) };
        assert_eq!(
            built.headers,
            vec![(
                if auth_name == "api-key" {
                    "Api-Key"
                } else {
                    "Authorization"
                }
                .into(),
                auth_value.into()
            )]
        );
        let headers = built.headers.clone();
        let response = ocr(built, headers, body.clone()).await.unwrap();
        let (headers, sent) = handle.join().unwrap();
        assert!(headers.starts_with(&format!("POST {path} HTTP/1.1\r\n")));
        assert!(headers.contains(&format!("{auth_name}: {auth_value}\r\n")));
        assert_eq!(sent, body);
        if deepseek {
            assert_eq!(sent["model"], "deepseek-ai/deepseek-ocr-maas");
            assert_eq!(
                sent["messages"][0]["content"][0],
                json!({"type": "image_url", "image_url": "data:application/pdf;base64,cGRm"})
            );
            assert!(sent.get("document").is_none());
        } else {
            assert_eq!(
                sent["document"]["document_url"],
                "data:application/pdf;base64,cGRm"
            );
        }
        assert_eq!(response.pages[0]["markdown"], "proof");
    }
}

fn server(
    status: u16,
    headers: &str,
    response_body: &str,
    delay: Duration,
) -> (String, thread::JoinHandle<(String, Value)>) {
    let response = format!(
        "HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nConnection: close\r\n{headers}\r\n{response_body}",
        response_body.len()
    );
    let (base, handle) = raw_server(response, delay);
    (
        base,
        thread::spawn(move || {
            let (headers, body) = handle.join().unwrap();
            (headers, serde_json::from_slice(&body).unwrap())
        }),
    )
}

fn raw_server(
    response: String,
    delay: Duration,
) -> (String, thread::JoinHandle<(String, Vec<u8>)>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let handle = thread::spawn(move || {
        listener.set_nonblocking(true).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let mut stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "no OCR request received"
                    );
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("{error}"),
            }
        };
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut received = Vec::new();
        let mut buffer = [0; 4096];
        let header_end = loop {
            let count = stream.read(&mut buffer).unwrap();
            assert_ne!(count, 0);
            received.extend_from_slice(&buffer[..count]);
            if let Some(end) = received.windows(4).position(|part| part == b"\r\n\r\n") {
                break end + 4;
            }
        };
        let headers = String::from_utf8(received[..header_end].to_vec()).unwrap();
        let length: usize = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse().unwrap())
            })
            .unwrap();
        while received.len() < header_end + length {
            let count = stream.read(&mut buffer).unwrap();
            assert_ne!(count, 0);
            received.extend_from_slice(&buffer[..count]);
        }
        let body = received[header_end..header_end + length].to_vec();
        thread::sleep(delay);
        let _ = stream.write_all(response.as_bytes());
        (headers, body)
    });
    (base, handle)
}

#[tokio::test]
async fn posts_filled_body_and_normalizes_provider_response() {
    let response_json = json!({
        "pages": [{"index": 0, "markdown": "result", "header": "title", "blocks": []}],
        "model": "provider-model",
        "document_annotation": {"title": "result"},
        "usage_info": {"pages_processed": 1},
        "private_provider_field": "not forwarded"
    });
    let (base, handle) = server(200, "", &response_json.to_string(), Duration::ZERO);
    let built = build_pre_call_request(OcrRequest {
        api_base: Some(base),
        ..request()
    })
    .unwrap();
    let body = body(&built);
    let headers = built
        .headers
        .iter()
        .cloned()
        .chain([("X-Retained".into(), "header".into())])
        .collect();
    let response = ocr(built, headers, body.clone()).await.unwrap().into_json();
    let (headers, sent_body) = handle.join().unwrap();
    let headers = headers.to_ascii_lowercase();
    assert!(headers.starts_with("post /v1/ocr http/1.1\r\n"));
    assert!(headers.contains("authorization: bearer test-key\r\n"));
    assert!(headers.contains("accept-encoding: identity\r\n"));
    assert!(!headers.contains("gzip"));
    assert!(headers.contains("content-type: application/json\r\n"));
    assert!(headers.contains("x-retained: header\r\n"));
    assert_eq!(sent_body, body);
    assert_eq!(
        response,
        json!({
            "pages": response_json["pages"],
            "model": "provider-model",
            "document_annotation": response_json["document_annotation"],
            "usage_info": response_json["usage_info"],
            "object": "ocr"
        })
    );
}

#[tokio::test]
async fn preserves_callback_body_and_header_changes() {
    let (base, handle) = server(200, "", "{}", Duration::ZERO);
    let built = build_pre_call_request(OcrRequest {
        api_base: Some(base),
        ..request()
    })
    .unwrap();
    let mut body = body(&built);
    body["model"] = json!("callback-model");
    body["custom_provider_field"] = json!({"nested": [1, true, null]});
    let headers = built
        .headers
        .iter()
        .cloned()
        .chain([
            ("cOnTeNt-TyPe".into(), "application/vnd.ocr+json".into()),
            ("aCcEpT-EnCoDiNg".into(), "gzip, br".into()),
        ])
        .collect();
    ocr(built, headers, body.clone()).await.unwrap();
    let (headers, sent_body) = handle.join().unwrap();
    let headers = headers.to_ascii_lowercase();
    assert_eq!(sent_body, body);
    assert_eq!(
        headers
            .lines()
            .filter(|line| line.starts_with("content-type:"))
            .collect::<Vec<_>>(),
        vec!["content-type: application/vnd.ocr+json"]
    );
    assert_eq!(
        headers
            .lines()
            .filter(|line| line.starts_with("accept-encoding:"))
            .collect::<Vec<_>>(),
        vec!["accept-encoding: gzip, br"]
    );
}

#[tokio::test]
async fn settled_headers_are_the_only_headers_sent() {
    let (base, handle) = server(200, "", "{}", Duration::ZERO);
    let built = build_pre_call_request(OcrRequest {
        api_base: Some(base),
        ..request()
    })
    .unwrap();
    let body = body(&built);
    ocr(built, Vec::new(), body).await.unwrap();
    let (headers, _) = handle.join().unwrap();
    assert!(!headers.to_ascii_lowercase().contains("authorization:"));
}

#[tokio::test]
async fn rejects_unsupported_inputs_before_io() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    for case in ["file", "local", "compression"] {
        let built = build_pre_call_request(OcrRequest {
            api_base: Some(base.clone()),
            ..request()
        })
        .unwrap();
        let mut body = body(&built);
        let mut headers = built.headers.clone();
        match case {
            "file" => body["document"] = json!({"type": "file", "file": "private"}),
            "local" => body["document"]["document_url"] = json!("file:///private.pdf"),
            "compression" => headers.push(("Content-Encoding".into(), "gzip".into())),
            _ => unreachable!(),
        }
        assert!(matches!(
            ocr(built, headers, body).await,
            Err(Error::Unsupported(_))
        ));
        assert_eq!(
            listener.accept().unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock
        );
    }
}

#[tokio::test]
async fn handles_errors_compression_and_timeout_without_exposing_payloads() {
    for (status, headers, response_body, delay, expected) in [
        (
            401,
            "",
            "private-document api-key",
            Duration::ZERO,
            Error::Http {
                status: 401,
                body: "OCR provider request failed".into(),
            },
        ),
        (
            302,
            "Location: http://127.0.0.1:1/private\r\n",
            "secret",
            Duration::ZERO,
            Error::Http {
                status: 302,
                body: "OCR provider request failed".into(),
            },
        ),
        (
            200,
            "",
            "private-document invalid JSON",
            Duration::ZERO,
            Error::InvalidResponse("invalid OCR JSON response".into()),
        ),
        (
            200,
            "Content-Encoding: gzip\r\n",
            "{}",
            Duration::ZERO,
            Error::Unsupported("compressed OCR response"),
        ),
        (
            200,
            "Content-Encoding: identity\r\nContent-Encoding: br\r\n",
            "{}",
            Duration::ZERO,
            Error::Unsupported("compressed OCR response"),
        ),
        (
            200,
            "Content-Encoding: identity, br\r\n",
            "{}",
            Duration::ZERO,
            Error::Unsupported("compressed OCR response"),
        ),
        (
            200,
            "",
            "{}",
            Duration::from_millis(200),
            Error::Network("transport failed".into()),
        ),
    ] {
        let (base, handle) = server(status, headers, response_body, delay);
        let built = build_pre_call_request(OcrRequest {
            api_base: Some(base),
            timeout_seconds: if delay.is_zero() { 2.0 } else { 0.05 },
            ..request()
        })
        .unwrap();
        let body = body(&built);
        let headers = built.headers.clone();
        assert_eq!(ocr(built, headers, body).await.unwrap_err(), expected);
        handle.join().unwrap();
    }
}

#[tokio::test]
async fn normalizes_missing_fields_and_accepts_identity_response() {
    let (base, handle) = server(200, "Content-Encoding: Identity\r\n", "{}", Duration::ZERO);
    let built = build_pre_call_request(OcrRequest {
        api_base: Some(base),
        ..request()
    })
    .unwrap();
    let body = body(&built);
    let headers = built.headers.clone();
    let response = ocr(built, headers, body).await.unwrap().into_json();
    handle.join().unwrap();
    assert_eq!(
        response,
        json!({
            "pages": [], "model": "mistral-ocr-latest", "document_annotation": null,
            "usage_info": null, "object": "ocr"
        })
    );
}

#[tokio::test]
async fn invalid_headers_and_timeouts_never_reach_the_server() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    for timeout_seconds in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MAX] {
        let result = build_pre_call_request(OcrRequest {
            api_base: Some(base.clone()),
            timeout_seconds,
            ..request()
        });
        assert!(matches!(result, Err(Error::InvalidRequest(message))
            if message == "timeout must be positive and finite"));
    }
    for (name, value, expected) in [
        ("private\nname", "secret", "invalid header name"),
        ("x-proof", "private\nvalue", "invalid header value"),
    ] {
        let built = build_pre_call_request(OcrRequest {
            api_base: Some(base.clone()),
            ..request()
        })
        .unwrap();
        let body = body(&built);
        let headers = vec![(name.into(), value.into())];
        assert_eq!(
            ocr(built, headers, body).await.unwrap_err(),
            Error::InvalidRequest(expected.into())
        );
    }
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[tokio::test]
async fn body_read_failure_precedes_status_and_encoding_errors() {
    for status in [200, 401] {
        let (base, handle) = raw_server(
            format!(
                "HTTP/1.1 {status} Test\r\nContent-Length: 100\r\nContent-Encoding: gzip\r\nConnection: close\r\n\r\nprivate"
            ),
            Duration::ZERO,
        );
        let built = build_pre_call_request(OcrRequest {
            api_base: Some(base),
            ..request()
        })
        .unwrap();
        let body = body(&built);
        let headers = built.headers.clone();
        assert_eq!(
            ocr(built, headers, body).await.unwrap_err(),
            Error::Network("could not read response".into())
        );
        handle.join().unwrap();
    }
}

#[tokio::test]
async fn sends_exact_settled_json_bytes_and_preserves_duplicate_headers() {
    let (base, handle) = raw_server(
        "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}".into(),
        Duration::ZERO,
    );
    let built = build_pre_call_request(OcrRequest {
        api_base: Some(base),
        ..request()
    })
    .unwrap();
    let body = body(&built);
    let expected = serde_json::to_vec(&body).unwrap();
    let headers = vec![
        ("x-proof".into(), "one".into()),
        ("X-Proof".into(), "two".into()),
    ];
    ocr(built, headers, body).await.unwrap();
    let (headers, sent) = handle.join().unwrap();
    assert_eq!(sent, expected);
    assert_eq!(
        headers
            .lines()
            .filter(|line| line.starts_with("x-proof:"))
            .collect::<Vec<_>>(),
        vec!["x-proof: one", "x-proof: two"]
    );
}
