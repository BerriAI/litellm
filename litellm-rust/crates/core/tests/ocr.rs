use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use super::OcrClient;
use super::hooks::{OcrHookFuture, OcrHooks, OcrLogFuture, OcrPreCallRequest, OcrPreparedRequest};
use super::test_support::{MockResponse, mock_server, perform_ocr, wire_request};
use super::wire::{OcrWireRequest, decode_request};
use crate::call_lifecycle::{CallLifecycleContext, CallLifecycleTiming};

#[test]
fn request_boundary_selects_mistral_and_rejects_unknown_providers() {
    let request = OcrWireRequest {
        model: "mistral/model".into(),
        document: json!({"type":"document_url","document_url":"https://example.com/doc.pdf"}),
        api_key: Some("key".into()),
        api_base: None,
        custom_llm_provider: None,
        extra_headers: None,
        optional_params: json!({"extract_header":true,"unknown":42})
            .as_object()
            .unwrap()
            .clone(),
        input_sources: Default::default(),
        timeout_seconds: None,
    };
    assert!(decode_request(request).is_ok());
    assert!(
        decode_request(OcrWireRequest {
            model: "model".into(),
            document: json!({"type":"document_url","document_url":"https://example.com/doc.pdf"}),
            api_key: Some("key".into()),
            api_base: None,
            custom_llm_provider: Some("unknown".into()),
            extra_headers: None,
            optional_params: serde_json::Map::new(),
            input_sources: Default::default(),
            timeout_seconds: None,
        })
        .is_err()
    );
}

#[tokio::test]
async fn facade_executes_direct_mistral_once() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
        "pages":[{"index":0,"markdown":"hello","custom":"preserved"}],
        "usage_info":{"pages_processed":1}
    }))])
    .await;
    let result = perform_ocr(wire_request(
        "mistral/model",
        &base,
        json!({"extract_header":true,"unknown":"ignored"}),
    ))
    .await
    .unwrap();
    server.await.unwrap();
    assert_eq!(result.pages[0]["markdown"], "hello");
    assert_eq!(result.pages[0]["custom"], "preserved");
    let requests = seen.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].starts_with("POST /v1/ocr "));
    assert!(
        requests[0]
            .to_ascii_lowercase()
            .contains("authorization: bearer test-key\r\n")
    );
    let body: Value = serde_json::from_str(requests[0].split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(
        body,
        json!({
            "model":"model",
            "document":{"type":"document_url","document_url":"data:application/pdf;base64,YWJj"},
            "extract_header":true
        })
    );
}

#[tokio::test]
async fn facade_retains_native_response_when_requested() {
    let provider_response = json!({
        "pages":[{"index":0,"markdown":"hello"}],
        "usage_info":{"pages_processed":1},
        "provider_only":"preserved"
    });
    let (base, _, server) = mock_server(vec![MockResponse::json(provider_response.clone())]).await;
    let response = perform_ocr(wire_request(
        "mistral/model",
        &base,
        json!({"req_format":"native"}),
    ))
    .await
    .unwrap();

    server.await.unwrap();
    assert_eq!(response.provider_native_response, Some(provider_response));
}

#[tokio::test]
async fn facade_uses_the_injected_http_client() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
    let mut default_headers = reqwest::header::HeaderMap::new();
    default_headers.insert(
        "x-transport-owner",
        reqwest::header::HeaderValue::from_static("host"),
    );
    let provider_http = reqwest::Client::builder()
        .default_headers(default_headers)
        .build()
        .unwrap();
    OcrClient::new(provider_http)
        .unwrap()
        .perform(wire_request("mistral/model", &base, json!({})))
        .await
        .unwrap();
    server.await.unwrap();
    assert!(seen.lock().unwrap()[0].contains("x-transport-owner: host"));
}

struct RecordingHooks {
    events: Arc<Mutex<Vec<&'static str>>>,
    block: bool,
}

struct EditPreparedRequest;

impl OcrHooks for EditPreparedRequest {
    fn prepared_request(
        &self,
        mut request: OcrPreparedRequest,
    ) -> OcrHookFuture<'_, OcrPreparedRequest> {
        Box::pin(async move {
            assert_eq!(request.model, "model");
            assert!(request.url.ends_with("/v1/ocr"));
            assert!(
                request
                    .headers
                    .contains(&("Authorization".into(), "Bearer test-key".into()))
            );
            request.body["include_image_base64"] = json!(true);
            request
                .headers
                .push(("x-host-hook".into(), "called".into()));
            Ok(request)
        })
    }
}

#[tokio::test]
async fn prepared_request_hook_edits_wire_body_and_headers_without_guardrails() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
    let request = wire_request(
        "mistral/model",
        &base,
        json!({"include_image_base64":false}),
    )
    .with_host_hooks(Arc::new(EditPreparedRequest), None);
    perform_ocr(request).await.unwrap();
    server.await.unwrap();
    let requests = seen.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].contains("x-host-hook: called\r\n"));
    let body: Value = serde_json::from_str(requests[0].split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(body["include_image_base64"], true);
}

impl OcrHooks for RecordingHooks {
    fn prepared_request(
        &self,
        request: OcrPreparedRequest,
    ) -> OcrHookFuture<'_, OcrPreparedRequest> {
        Box::pin(async move {
            self.events.lock().unwrap().push("prepared");
            Ok(request)
        })
    }

    fn pre_call(&self, request: OcrPreCallRequest) -> OcrHookFuture<'_, OcrPreCallRequest> {
        Box::pin(async move {
            self.events.lock().unwrap().push("pre");
            if self.block {
                return Err(crate::Error::InvalidRequest("blocked".into()));
            }
            Ok(request)
        })
    }

    fn during_call(
        &self,
        request: super::hooks::OcrDuringCallRequest,
    ) -> OcrHookFuture<'_, super::hooks::OcrDuringCallRequest> {
        Box::pin(async move {
            self.events.lock().unwrap().push("during");
            Ok(request)
        })
    }

    fn success<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        _response: &'a super::LiteLLMOcrResponse,
        _timing: &'a CallLifecycleTiming,
    ) -> OcrLogFuture<'a> {
        Box::pin(async move {
            self.events.lock().unwrap().push("success");
        })
    }

    fn failure<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        _error: &'a crate::Error,
        _timing: &'a CallLifecycleTiming,
    ) -> OcrLogFuture<'a> {
        Box::pin(async move {
            self.events.lock().unwrap().push("failure");
        })
    }
}

#[tokio::test]
async fn lifecycle_orders_hooks_and_emits_one_success() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let request = wire_request("mistral/model", &base, json!({}));
    let request = super::LiteLLMOcrRequest {
        hooks: Arc::new(RecordingHooks {
            events: events.clone(),
            block: false,
        }),
        ..request
    };
    perform_ocr(request).await.unwrap();
    server.await.unwrap();
    assert_eq!(
        *events.lock().unwrap(),
        ["pre", "during", "prepared", "success"]
    );
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn lifecycle_blocking_prevents_execution_and_emits_one_failure() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let request = wire_request("mistral/model", "http://127.0.0.1:1", json!({}));
    let request = super::LiteLLMOcrRequest {
        hooks: Arc::new(RecordingHooks {
            events: events.clone(),
            block: true,
        }),
        ..request
    };
    let error = perform_ocr(request).await.unwrap_err();
    assert!(matches!(error, crate::Error::InvalidRequest(_)));
    assert_eq!(*events.lock().unwrap(), ["pre", "failure"]);
}

#[tokio::test]
async fn upstream_failure_emits_one_terminal_failure() {
    let (base, seen, server) = mock_server(vec![MockResponse {
        status: 500,
        headers: vec![],
        body: json!({"error":"failed"}),
    }])
    .await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let request = wire_request("mistral/model", &base, json!({}));
    let request = super::LiteLLMOcrRequest {
        hooks: Arc::new(RecordingHooks {
            events: events.clone(),
            block: false,
        }),
        ..request
    };
    assert!(perform_ocr(request).await.is_err());
    server.await.unwrap();
    assert_eq!(
        *events.lock().unwrap(),
        ["pre", "during", "prepared", "failure"]
    );
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn every_adapter_runs_the_complete_lifecycle() {
    use super::registry::OcrAdapterKind;

    macro_rules! adapter_kinds {
        ($( $variant:ident, $adapter:ty, $instance:expr, $provider:ident; )+) => {
            [$(OcrAdapterKind::$variant,)+]
        };
    }
    for adapter in super::adapters::for_each_ocr_adapter!(adapter_kinds) {
        let (model, response) = match adapter {
            OcrAdapterKind::Mistral => ("mistral/model", json!({"pages":[]})),
            OcrAdapterKind::AzureMistral => ("azure_ai/model", json!({"pages":[]})),
            OcrAdapterKind::AzureDocumentIntelligence => (
                "azure_ai/documentintelligence/prebuilt-read",
                json!({"status":"succeeded", "analyzeResult":{"pages":[]}}),
            ),
            OcrAdapterKind::ReductoLegacy => {
                ("reducto/parse-legacy", json!({"result":{"chunks":[]}}))
            }
            OcrAdapterKind::ReductoV3 => ("reducto/parse-v3", json!({"result":{"chunks":[]}})),
            OcrAdapterKind::VertexMistral => ("vertex_ai/mistral-ocr", json!({"pages":[]})),
            OcrAdapterKind::VertexDeepSeek => (
                "vertex_ai/deepseek-ocr",
                json!({"choices":[{"message":{"content":"text"}}]}),
            ),
        };
        let (base, seen, server) = mock_server(vec![MockResponse::json(response)]).await;
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut request = wire_request(
            model,
            &base,
            json!({"vertex_project":"project", "vertex_location":"us-central1"}),
        )
        .with_host_hooks(
            Arc::new(RecordingHooks {
                events: events.clone(),
                block: false,
            }),
            Some("call-id".into()),
        );
        if matches!(
            adapter,
            OcrAdapterKind::ReductoLegacy | OcrAdapterKind::ReductoV3
        ) {
            request.document = request.document.with_source("reducto://ready.pdf".into());
        }
        perform_ocr(request).await.unwrap();
        server.await.unwrap();
        assert_eq!(
            *events.lock().unwrap(),
            ["pre", "during", "prepared", "success"],
            "{model}"
        );
        assert_eq!(seen.lock().unwrap().len(), 1, "{model}");
    }
}

#[derive(Clone, Copy, Debug)]
enum FailureStage {
    During,
    Prepared,
    InvalidBody,
    Preparation,
    Response,
}

struct FailingHooks {
    recording: RecordingHooks,
    stage: FailureStage,
}

impl OcrHooks for FailingHooks {
    fn pre_call(&self, request: OcrPreCallRequest) -> OcrHookFuture<'_, OcrPreCallRequest> {
        self.recording.pre_call(request)
    }

    fn during_call(
        &self,
        request: super::hooks::OcrDuringCallRequest,
    ) -> OcrHookFuture<'_, super::hooks::OcrDuringCallRequest> {
        Box::pin(async move {
            let request = self.recording.during_call(request).await?;
            if matches!(self.stage, FailureStage::During) {
                return Err(crate::Error::InvalidRequest("blocked during call".into()));
            }
            Ok(request)
        })
    }

    fn prepared_request(
        &self,
        request: OcrPreparedRequest,
    ) -> OcrHookFuture<'_, OcrPreparedRequest> {
        Box::pin(async move {
            let request = self.recording.prepared_request(request).await?;
            match self.stage {
                FailureStage::Prepared => Err(crate::Error::InvalidRequest(
                    "blocked prepared request".into(),
                )),
                FailureStage::InvalidBody => Ok(OcrPreparedRequest {
                    body: json!({"document":null}),
                    ..request
                }),
                _ => Ok(request),
            }
        })
    }

    fn success<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        response: &'a super::LiteLLMOcrResponse,
        timing: &'a CallLifecycleTiming,
    ) -> OcrLogFuture<'a> {
        self.recording.success(context, response, timing)
    }

    fn failure<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        error: &'a crate::Error,
        timing: &'a CallLifecycleTiming,
    ) -> OcrLogFuture<'a> {
        Box::pin(async move {
            assert_eq!(context.litellm_call_id, "call-id");
            assert_eq!(timing.phases.len(), 3);
            self.recording.failure(context, error, timing).await;
        })
    }
}

#[rstest::rstest]
#[case(FailureStage::During)]
#[case(FailureStage::Prepared)]
#[case(FailureStage::InvalidBody)]
#[case(FailureStage::Preparation)]
#[case(FailureStage::Response)]
#[tokio::test]
async fn lifecycle_reports_failures_once_at_each_boundary(#[case] stage: FailureStage) {
    let (base, seen, server) = mock_server(if matches!(stage, FailureStage::Response) {
        vec![MockResponse::json(json!({"pages":"invalid"}))]
    } else {
        vec![]
    })
    .await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let request = wire_request(
        "mistral/model",
        &base,
        if matches!(stage, FailureStage::Preparation) {
            json!({"pages":"invalid"})
        } else {
            json!({})
        },
    )
    .with_host_hooks(
        Arc::new(FailingHooks {
            recording: RecordingHooks {
                events: events.clone(),
                block: false,
            },
            stage,
        }),
        Some("call-id".into()),
    );
    let result = perform_ocr(request).await;
    assert!(result.is_err());
    server.await.unwrap();
    let expected = match stage {
        FailureStage::Preparation => vec!["pre", "failure"],
        FailureStage::During => vec!["pre", "during", "failure"],
        _ => vec!["pre", "during", "prepared", "failure"],
    };
    assert_eq!(*events.lock().unwrap(), expected);
    assert_eq!(
        seen.lock().unwrap().len(),
        usize::from(matches!(stage, FailureStage::Response))
    );
}
