use super::*;

#[test]
fn parses_document_url_and_flattens_provider_params() {
    let call = parse_json_body(
        br#"{"model":"rust-ocr","document":{"type":"document_url","document_url":"https://x/doc.pdf"},"include_image_base64":true}"#,
    )
    .expect("valid body parses");

    assert_eq!(call.model, "rust-ocr");
    assert_eq!(call.document["type"], "document_url");
    assert_eq!(call.document["document_url"], "https://x/doc.pdf");
    assert_eq!(
        call.optional_params["include_image_base64"],
        Value::Bool(true)
    );
    assert!(call.timeout.is_none());
}

#[test]
fn parses_image_url_document() {
    let call = parse_json_body(
        br#"{"model":"m","document":{"type":"image_url","image_url":"https://x/i.png"}}"#,
    )
    .expect("valid body parses");
    assert_eq!(call.document["type"], "image_url");
    assert_eq!(call.document["image_url"], "https://x/i.png");
}

#[test]
fn extracts_timeout_and_keeps_it_out_of_provider_params() {
    let call = parse_json_body(
        br#"{"model":"m","document":{"type":"document_url","document_url":"https://x"},"timeout":12.5}"#,
    )
    .expect("valid body parses");
    assert_eq!(call.timeout, Some(Duration::from_secs_f64(12.5)));
    assert!(!call.optional_params.contains_key("timeout"));
}

#[test]
fn rejects_file_document_over_json() {
    let err = parse_json_body(br#"{"model":"m","document":{"type":"file","file":"/etc/passwd"}}"#)
        .expect_err("file type rejected");
    match err {
        CoreError::InvalidRequest(message) => assert!(message.contains("multipart/form-data")),
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn preserves_reducto_file_id_over_json() {
    let call = parse_json_body(
        br#"{"model":"m","document":{"type":"document_url","document_url":"reducto://abc123"}}"#,
    )
    .expect("reducto id preserved");
    assert_eq!(call.document["type"], "document_url");
    assert_eq!(call.document["document_url"], "reducto://abc123");
}

#[test]
fn empty_body_is_rejected() {
    assert!(matches!(
        parse_json_body(b""),
        Err(CoreError::InvalidRequest(_))
    ));
}

#[test]
fn malformed_json_error_is_generic_and_never_echoes_attacker_text() {
    let markers = [
        (
            "query",
            br#"{"model":"m","document":"https://x?secret=QUERYMARKERZZZ"}"#.to_vec(),
        ),
        (
            "base64",
            br#"{"model":"m","document":{"type":"document_url","document_url":"https://x"},"timeout":"BASE64MARKERzz=="}"#.to_vec(),
        ),
        (
            "credential-path",
            br#"{"model":"m","document":{"type":"/etc/creds/CREDMARKER.json"}}"#.to_vec(),
        ),
        (
            "unknown-type",
            br#"{"model":"m","document":{"type":"UNKNOWNTYPEZZZ"}}"#.to_vec(),
        ),
    ];
    for (label, body) in markers {
        let err = parse_json_body(&body).expect_err("malformed body rejected");
        let CoreError::InvalidRequest(message) = err else {
            panic!("{label}: expected InvalidRequest, got {err:?}");
        };
        for needle in [
            "QUERYMARKERZZZ",
            "BASE64MARKERzz==",
            "CREDMARKER",
            "UNKNOWNTYPEZZZ",
            "/etc/creds",
        ] {
            assert!(
                !message.contains(needle),
                "{label}: parse error must not echo attacker text ({needle}): {message}"
            );
        }
        assert_eq!(
            message,
            "request body is not a valid OCR request; expected JSON with 'model' and 'document'",
            "{label}: parse errors must be a single fixed message"
        );
    }
}

#[test]
fn json_rejects_blank_model() {
    for model in ["", " ", "\t\n"] {
        let body = format!(
            r#"{{"model":{model:?},"document":{{"type":"document_url","document_url":"https://x"}}}}"#
        );
        let err = parse_json_body(body.as_bytes()).expect_err("blank model rejected");
        match err {
            CoreError::InvalidRequest(message) => assert!(message.contains("model")),
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }
}

#[test]
fn upload_prefers_declared_content_type() {
    let document = build_upload_document(
        b"%PDF-1.4".to_vec(),
        Some("scan.bin"),
        Some("application/pdf"),
    )
    .expect("builds document");
    assert_eq!(document["type"], "document_url");
    assert!(document["document_url"]
        .as_str()
        .expect("data uri")
        .starts_with("data:application/pdf;base64,"));
}

#[test]
fn upload_normalizes_declared_mime_case_before_image_classification() {
    let document = build_upload_document(
        vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A],
        Some("scan.bin"),
        Some("Image/PNG"),
    )
    .expect("builds document");
    assert_eq!(document["type"], "image_url");
    assert!(document["image_url"]
        .as_str()
        .expect("data uri")
        .starts_with("data:image/png;base64,"));
}

#[test]
fn upload_treats_generic_with_parameters_as_generic() {
    let document = build_upload_document(
        b"%PDF-1.7 minimal".to_vec(),
        None,
        Some("application/octet-stream; charset=binary"),
    )
    .expect("builds document");
    assert_eq!(document["type"], "document_url");
    assert!(document["document_url"]
        .as_str()
        .expect("data uri")
        .starts_with("data:application/pdf;base64,"));
}

#[test]
fn upload_sniffs_pdf_from_bytes_when_unnamed_octet_stream() {
    let document = build_upload_document(
        b"%PDF-1.7 minimal".to_vec(),
        None,
        Some("application/octet-stream"),
    )
    .expect("builds document");
    assert_eq!(document["type"], "document_url");
    assert!(document["document_url"]
        .as_str()
        .expect("data uri")
        .starts_with("data:application/pdf;base64,"));
}

#[test]
fn upload_sniffs_png_from_bytes_when_unnamed_and_no_content_type() {
    let document = build_upload_document(
        vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x00],
        None,
        None,
    )
    .expect("builds document");
    assert_eq!(document["type"], "image_url");
    assert!(document["image_url"]
        .as_str()
        .expect("data uri")
        .starts_with("data:image/png;base64,"));
}

#[test]
fn upload_infers_mime_from_filename_when_octet_stream() {
    let document = build_upload_document(
        vec![0x89, b'P', b'N', b'G'],
        Some("photo.PNG"),
        Some("application/octet-stream"),
    )
    .expect("builds document");
    assert_eq!(document["type"], "image_url");
    assert!(document["image_url"]
        .as_str()
        .expect("data uri")
        .starts_with("data:image/png;base64,"));
}

#[test]
fn upload_falls_back_to_octet_stream() {
    let document =
        build_upload_document(vec![1, 2, 3], Some("data.unknown"), None).expect("builds document");
    assert_eq!(document["type"], "document_url");
    assert!(document["document_url"]
        .as_str()
        .expect("data uri")
        .starts_with("data:application/octet-stream;base64,"));
}

#[test]
fn empty_upload_is_rejected() {
    assert!(matches!(
        build_upload_document(Vec::new(), Some("a.pdf"), Some("application/pdf")),
        Err(CoreError::InvalidRequest(_))
    ));
}

#[test]
fn multipart_extracts_model_timeout_and_json_params() {
    let document =
        json!({"type": "document_url", "document_url": "data:application/pdf;base64,AA=="});
    let fields = vec![
        ("model".to_string(), "rust-ocr".to_string()),
        ("timeout".to_string(), "30".to_string()),
        ("pages".to_string(), "[0,1,2]".to_string()),
        ("id".to_string(), "abc".to_string()),
    ];
    let call = assemble_multipart_call(document, &fields).expect("assembles call");

    assert_eq!(call.model, "rust-ocr");
    assert_eq!(call.timeout, Some(Duration::from_secs(30)));
    assert_eq!(call.optional_params["pages"], json!([0, 1, 2]));
    assert_eq!(call.optional_params["id"], Value::String("abc".to_string()));
    assert!(!call.optional_params.contains_key("timeout"));
    assert!(!call.optional_params.contains_key("model"));
}

#[test]
fn multipart_requires_model() {
    let document = json!({"type": "document_url", "document_url": "data:x"});
    let err = assemble_multipart_call(document, &[]).expect_err("model required");
    assert!(matches!(err, CoreError::InvalidRequest(_)));
}

#[test]
fn multipart_rejects_blank_model() {
    let document = json!({"type": "document_url", "document_url": "data:x"});
    let fields = vec![("model".to_string(), "   ".to_string())];
    let err = assemble_multipart_call(document, &fields).expect_err("blank model rejected");
    match err {
        CoreError::InvalidRequest(message) => assert!(message.contains("model")),
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn multipart_rejects_document_form_field() {
    let document = json!({"type": "document_url", "document_url": "data:x"});
    let fields = vec![
        ("model".to_string(), "m".to_string()),
        (
            "document".to_string(),
            r#"{"type":"document_url","document_url":"reducto://smuggled"}"#.to_string(),
        ),
    ];
    let err = assemble_multipart_call(document, &fields).expect_err("document field rejected");
    match err {
        CoreError::InvalidRequest(message) => {
            assert!(message.contains("document"));
            assert!(
                !message.contains("smuggled"),
                "must not echo the attacker value: {message}"
            );
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn json_rejects_reserved_control_params() {
    for reserved in [
        "api_key",
        "api_base",
        "custom_llm_provider",
        "extra_headers",
        "vertex_credentials",
        "vertex_ai_credentials",
        "vertex_project",
        "vertex_ai_project",
        "vertex_location",
        "vertex_ai_location",
    ] {
        let body = format!(
            r#"{{"model":"m","document":{{"type":"document_url","document_url":"https://x"}},"{reserved}":"attacker"}}"#
        );
        let err =
            parse_json_body(body.as_bytes()).expect_err("reserved control param must be rejected");
        match err {
            CoreError::InvalidRequest(message) => {
                assert!(
                    message.contains(reserved),
                    "names the rejected key: {message}"
                );
                assert!(
                    !message.contains("attacker"),
                    "must not echo the attacker value: {message}"
                );
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }
}

#[test]
fn multipart_rejects_reserved_control_params() {
    let document = json!({"type": "document_url", "document_url": "data:x"});
    let fields = vec![
        ("model".to_string(), "m".to_string()),
        (
            "vertex_credentials".to_string(),
            "/etc/gcp/service-account.json".to_string(),
        ),
    ];
    let err = assemble_multipart_call(document, &fields).expect_err("reserved param rejected");
    match err {
        CoreError::InvalidRequest(message) => {
            assert!(message.contains("vertex_credentials"));
            assert!(
                !message.contains("service-account"),
                "must not echo the attacker value: {message}"
            );
        }
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn json_rejects_non_positive_timeout() {
    for timeout in ["0", "-1", "-0.5", "1e300"] {
        let body = format!(
            r#"{{"model":"m","document":{{"type":"document_url","document_url":"https://x"}},"timeout":{timeout}}}"#
        );
        assert!(
            matches!(
                parse_json_body(body.as_bytes()),
                Err(CoreError::InvalidRequest(_))
            ),
            "timeout {timeout} must be rejected"
        );
    }
}

#[test]
fn multipart_rejects_non_positive_and_non_finite_timeout() {
    let document = json!({"type": "document_url", "document_url": "data:x"});
    for timeout in ["0", "-3", "inf", "-inf", "NaN", "1e300"] {
        let fields = vec![
            ("model".to_string(), "m".to_string()),
            ("timeout".to_string(), timeout.to_string()),
        ];
        assert!(
            matches!(
                assemble_multipart_call(document.clone(), &fields),
                Err(CoreError::InvalidRequest(_))
            ),
            "timeout {timeout} must be rejected"
        );
    }
}

#[test]
fn multipart_rejects_non_numeric_timeout_without_echoing_value() {
    let document = json!({"type": "document_url", "document_url": "data:x"});
    let fields = vec![
        ("model".to_string(), "m".to_string()),
        ("timeout".to_string(), "not-a-number".to_string()),
    ];
    let err = assemble_multipart_call(document, &fields).expect_err("non-numeric timeout rejected");
    match err {
        CoreError::InvalidRequest(message) => assert!(
            !message.contains("not-a-number"),
            "must not echo the attacker value: {message}"
        ),
        other => panic!("expected InvalidRequest, got {other:?}"),
    }
}

#[test]
fn multipart_rejects_duplicate_model_field() {
    let document = json!({"type": "document_url", "document_url": "data:x"});
    let fields = vec![
        ("model".to_string(), "first".to_string()),
        ("model".to_string(), "second".to_string()),
    ];
    let err = assemble_multipart_call(document, &fields).expect_err("duplicate model rejected");
    assert!(matches!(err, CoreError::InvalidRequest(_)));
}

#[test]
fn multipart_rejects_duplicate_timeout_field() {
    let document = json!({"type": "document_url", "document_url": "data:x"});
    let fields = vec![
        ("model".to_string(), "m".to_string()),
        ("timeout".to_string(), "10".to_string()),
        ("timeout".to_string(), "20".to_string()),
    ];
    let err = assemble_multipart_call(document, &fields).expect_err("duplicate timeout rejected");
    assert!(matches!(err, CoreError::InvalidRequest(_)));
}

#[test]
fn upload_treats_binary_octet_stream_as_generic() {
    let document = build_upload_document(
        b"%PDF-1.7 minimal".to_vec(),
        None,
        Some("Binary/Octet-Stream"),
    )
    .expect("builds document");
    assert!(document["document_url"]
        .as_str()
        .expect("data uri")
        .starts_with("data:application/pdf;base64,"));
}
