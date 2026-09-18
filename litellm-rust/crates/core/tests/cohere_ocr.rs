mod transformation {
    use litellm_llms::{
        base_llm::ocr::{
            error::Error,
            transformation::{BaseOcrConfig, OcrDocument, OcrResponseFormat},
        },
        cohere::ocr::transformation::*,
    };
    use rstest::rstest;
    use serde_json::{Value, json};

    #[tokio::test]
    async fn composed_body_preserves_native_document_fields_and_untyped_overrides() {
        let request = crate::ocr::test_support::wire_request(
            "cohere/parse",
            "https://example.com",
            json!({
                "output_format":"markdown", "timeout":30,
                "extra_body":{
                    "output_format": {"future":true},
                    "document":{"type":"image_url","image_url":"https://example.com/a.png",
                        "provider_options":{"nested":[false,0,null]}}
                }
            }),
        );
        let request = request.with_document(
            serde_json::from_value(json!({
                "type":"image_url","image_url":"https://example.com/original.png"
            }))
            .unwrap(),
        );
        let request = crate::ocr::prepare::prepare_request_for_test(request);
        let http = CohereParseConfig
            .prepare_request(
                &request,
                &crate::ocr::test_support::ocr_client(),
                &crate::ocr::test_support::NoHooks,
            )
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(http.body().unwrap().as_bytes().unwrap()).unwrap();
        assert_eq!(
            body,
            json!({
                "model":"parse", "output_format":{"future":true},
                "document":{"type":"image_url","image_url":"https://example.com/a.png",
                    "provider_options":{"nested":[false,0,null]}}
            })
        );
    }

    #[tokio::test]
    async fn explicit_null_options_use_defaults_before_http() {
        let request = crate::ocr::test_support::wire_request(
            "cohere/parse",
            "https://example.com",
            json!({"output_format":null,"req_format":null}),
        );
        let request = request.with_document(
            serde_json::from_value(
                json!({"type":"image_url","image_url":"https://example.com/a.png"}),
            )
            .unwrap(),
        );
        assert_eq!(
            request.response_format().unwrap(),
            OcrResponseFormat::Litellm
        );
        let request = crate::ocr::prepare::prepare_request_for_test(request);
        let http = CohereParseConfig
            .prepare_request(
                &request,
                &crate::ocr::test_support::ocr_client(),
                &crate::ocr::test_support::NoHooks,
            )
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(http.body().unwrap().as_bytes().unwrap()).unwrap();
        assert_eq!(body["output_format"], "markdown");
        assert!(body.get("req_format").is_none());
    }

    #[rstest]
    #[case::cohere("cohere/parse-v5.0", "POST /v2/parse ")]
    #[case::azure_ai("azure_ai/Cohere-parse-v5.0", "POST /providers/cohere/v2/parse ")]
    #[tokio::test]
    async fn route_sends_image_to_its_parse_endpoint_with_the_bearer_key(
        #[case] model: &str,
        #[case] request_line: &str,
    ) {
        use crate::ocr::test_support::{MockResponse, header, mock_server, perform_ocr};

        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
        let request = crate::ocr::test_support::wire_request(model, &base, json!({}))
            .with_document(
                serde_json::from_value::<OcrDocument>(
                    json!({"type":"image_url","image_url":"data:image/png;base64,YWJj"}),
                )
                .unwrap()
                .into(),
            );

        perform_ocr(request).await.unwrap();
        server.await.unwrap();

        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with(request_line), "{}", requests[0]);
        assert_eq!(
            header(&requests[0], "authorization"),
            Some("Bearer test-key")
        );
    }

    #[rstest]
    #[tokio::test]
    async fn route_rejects_non_image_document_without_a_request(
        #[values("cohere/parse-v5.0", "azure_ai/Cohere-parse-v5.0")] model: &str,
    ) {
        use crate::ocr::test_support::{MockResponse, mock_server, perform_ocr};

        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;

        let error = perform_ocr(crate::ocr::test_support::wire_request(
            model,
            &base,
            json!({}),
        ))
        .await
        .unwrap_err();
        server.abort();

        assert!(matches!(error, Error::CohereImageOnly), "{error:?}");
        assert!(seen.lock().unwrap().is_empty());
    }
}
