use litellm_host::interceptors::RawResponse;
use litellm_host::lifecycle::ExecutionEvent;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use litellm_core::ocr::{
    route::{Ocr, OcrCall, OcrOp},
    types::OcrDocumentInput,
};
use litellm_host::{
    interceptors::{RequestContext, WireRequest},
    lifecycle::CallEvent,
};
use rstest::rstest;

use super::*;

pub(crate) fn event_name(event: &CallEvent) -> &'static str {
    match event {
        CallEvent::Started { .. } => "started",
        CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { .. }) => "response",
        CallEvent::Succeeded { .. } => "success",
        CallEvent::Failed { .. } => "failure",
        CallEvent::Cancelled { .. } => "cancelled",
    }
}

fn recording_host(
    request: LiteLLMOcrRequest,
    events: Arc<Mutex<Vec<&'static str>>>,
    interceptions: Arc<AtomicUsize>,
    block: bool,
) -> LocalOcrHost {
    LocalOcrHost::new(request)
        .with_before_send(move |wire, _| {
            interceptions.fetch_add(1, Ordering::SeqCst);
            match block {
                true => Err(Error::InvalidRequest("blocked".into())),
                false => Ok(wire),
            }
        })
        .with_observer(move |event| events.lock().unwrap().push(event_name(event)))
}

#[rstest::rstest]
#[tokio::test]
async fn interception_runs_once_and_observers_receive_ordered_success_events() {
    let upstream = upstream([pages_response()]).await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let interceptions = Arc::new(AtomicUsize::new(0));

    perform_with(recording_host(
        ocr_request("mistral/model", &upstream.uri(), json!({})),
        events.clone(),
        interceptions.clone(),
        false,
    ))
    .await
    .unwrap();

    assert_eq!(interceptions.load(Ordering::SeqCst), 1);
    assert_eq!(*events.lock().unwrap(), ["started", "response", "success"]);
    assert_eq!(received(&upstream).await.len(), 1);
}

#[rstest::rstest]
#[tokio::test]
async fn a_blocking_before_send_prevents_the_call_and_emits_one_failure() {
    let upstream = upstream([pages_response()]).await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let interceptions = Arc::new(AtomicUsize::new(0));

    let error = perform_with(recording_host(
        ocr_request("mistral/model", &upstream.uri(), json!({})),
        events.clone(),
        interceptions.clone(),
        true,
    ))
    .await
    .unwrap_err();

    assert!(
        matches!(&error, Error::InvalidRequest(message) if message == "blocked"),
        "{error:?}"
    );
    assert_eq!(interceptions.load(Ordering::SeqCst), 1);
    assert_eq!(*events.lock().unwrap(), ["started", "failure"]);
    assert!(received(&upstream).await.is_empty());
}

#[rstest::rstest]
#[tokio::test]
async fn an_upstream_failure_emits_one_terminal_failure() {
    let upstream = upstream([status_response(500, json!({"error": "failed"}))]).await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let interceptions = Arc::new(AtomicUsize::new(0));

    let result = perform_with(recording_host(
        ocr_request("mistral/model", &upstream.uri(), json!({})),
        events.clone(),
        interceptions.clone(),
        false,
    ))
    .await;

    assert!(result.is_err());
    assert_eq!(interceptions.load(Ordering::SeqCst), 1);
    assert_eq!(*events.lock().unwrap(), ["started", "failure"]);
    assert_eq!(received(&upstream).await.len(), 1);
}

#[rstest::rstest]
#[tokio::test]
async fn an_invalid_provider_response_is_observed_before_normalization_fails() {
    let upstream = upstream([json_response(json!({"pages": "invalid"}))]).await;
    let observed = Arc::new(Mutex::new(Vec::new()));
    let recorder = observed.clone();
    let host = LocalOcrHost::new(ocr_request("mistral/model", &upstream.uri(), json!({})))
        .with_observer(move |event| {
            if let CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { raw }) = event {
                recorder.lock().unwrap().push(raw.body.clone());
            }
        });

    let error = perform_with(host).await.unwrap_err();

    assert!(matches!(error, Error::ResponseField { .. }), "{error:?}");
    assert_eq!(*observed.lock().unwrap(), [r#"{"pages":"invalid"}"#]);
}

#[rstest::rstest]
#[tokio::test]
async fn headers_returned_by_before_send_are_sent() {
    let upstream = upstream([pages_response()]).await;
    let host = LocalOcrHost::new(ocr_request("mistral/model", &upstream.uri(), json!({})))
        .with_before_send(|mut wire, _| {
            wire.headers
                .push(("x-core-callback".into(), "edited".into()));
            Ok(wire)
        });

    perform_with(host).await.unwrap();

    assert_eq!(
        only_request(&upstream).await.header("x-core-callback"),
        Some("edited")
    );
}

async fn before_send_context(request: LiteLLMOcrRequest) -> (WireRequest, RequestContext) {
    let observed = Arc::new(Mutex::new(None));
    let captured = observed.clone();
    let host = LocalOcrHost::new(request).with_before_send(move |wire, context| {
        *captured.lock().unwrap() = Some((wire.clone(), context.clone()));
        Ok(wire)
    });
    perform_with(host).await.unwrap();
    let context = observed.lock().unwrap().take();
    context.expect("before_provider_request ran")
}

#[rstest::rstest]
#[tokio::test]
async fn before_send_sees_the_route_its_params_and_the_body() {
    let upstream = upstream([pages_response()]).await;

    let (wire, context) = before_send_context(ocr_request(
        "mistral/model",
        &upstream.uri(),
        json!({"pages": [0], "req_format": "native"}),
    ))
    .await;

    assert_eq!(context.custom_llm_provider, "mistral");
    assert_eq!(context.model, "model");
    assert_eq!(context.optional_params["req_format"], "native");
    assert!(context.secret_fields.is_empty());
    assert_eq!(wire.body["pages"], json!([0]));
}

#[rstest]
#[case::client_secret(json!({"client_secret": "shh", "tenant_id": "t"}), &["client_secret"])]
#[case::no_secrets(json!({"tenant_id": "t"}), &[])]
#[tokio::test]
async fn before_send_names_the_secret_params(#[case] options: Value, #[case] secrets: &[&str]) {
    let upstream = upstream([pages_response()]).await;
    let request = ocr_request("azure_ai/model", &upstream.uri(), options).with_document(
        OcrDocumentInput::Bytes {
            bytes: b"abc".as_slice().into(),
            file_name: None,
            mime_type: Some("application/pdf".into()),
        },
    );

    let (_, context) = before_send_context(request).await;

    assert_eq!(context.secret_fields, secrets);
}

/// Hands the route a caller-owned Azure token and rewrites the bearer in `before_provider_request`.
struct CallerTokenHost {
    request: Mutex<Option<LiteLLMOcrRequest>>,
    trace: Mutex<Vec<String>>,
}

impl CallerTokenHost {
    pub fn request(&self) -> Result<OcrCall, Error> {
        self.trace.lock().unwrap().push("project".into());
        Ok(OcrCall {
            request: self.request.lock().unwrap().take().unwrap(),
            caller_token: true,
        })
    }
    pub fn runtime(&self) -> litellm_host_native::in_process::Host<'_, Self, Self, ()> {
        litellm_host_native::in_process::Host {
            services: self,
            interceptors: self,
            stream: &(),
            observers: None,
        }
    }
}
impl litellm_host_native::services::HostCallHandler<Ocr> for CallerTokenHost {
    async fn handle_host_call(&self, op: OcrOp) -> Result<(), Error> {
        match op {
            OcrOp::AcquireAzureAdToken(reply) => {
                self.trace.lock().unwrap().push("token".into());
                reply.send(litellm_auth::ResolvedCredential::Static(
                    litellm_auth::SecretValue::new("caller-token"),
                ));
                Ok(())
            }
        }
    }
}

impl litellm_host::lifecycle::CallObserver for CallerTokenHost {
    fn observe(&self, _: litellm_host::lifecycle::CallEvent) {}
}
impl litellm_host::interceptors::Interceptors<<Ocr as litellm_host::protocol::Protocol>::Error>
    for CallerTokenHost
{
    async fn before_provider_request(
        &self,
        wire: WireRequest,
        _: RequestContext,
    ) -> Result<WireRequest, Error> {
        let is_authorization = |name: &str| name.eq_ignore_ascii_case("authorization");
        let authorization = wire
            .headers
            .iter()
            .find(|(name, _)| is_authorization(name))
            .map(|(_, value)| value.clone())
            .unwrap_or_default();
        self.trace
            .lock()
            .unwrap()
            .push(format!("before_provider_request:{authorization}"));
        let headers = wire
            .headers
            .into_iter()
            .map(|(name, value)| match is_authorization(&name) {
                true => (name, "Bearer edited".to_string()),
                false => (name, value),
            })
            .collect();
        Ok(WireRequest { headers, ..wire })
    }
    async fn after_provider_response(
        &self,
        raw: litellm_host::interceptors::RawResponse,
    ) -> Result<(), <Ocr as litellm_host::protocol::Protocol>::Error> {
        litellm_host::lifecycle::CallObserver::observe(
            self,
            litellm_host::lifecycle::CallEvent::Execution(
                litellm_host::lifecycle::ExecutionEvent::ProviderResponseReceived { raw },
            ),
        );
        Ok(())
    }
}

#[rstest::rstest]
#[tokio::test]
async fn the_callers_azure_token_is_acquired_before_before_send_which_can_still_replace_it() {
    let upstream = upstream([pages_response()]).await;
    let host = CallerTokenHost {
        request: Mutex::new(Some(without_api_key(ocr_request(
            "azure_ai/model",
            &upstream.uri(),
            json!({}),
        )))),
        trace: Mutex::new(Vec::new()),
    };

    litellm_host_native::in_process::run_hosted(
        ocr_route().machine(host.request().unwrap(), None),
        host.runtime(),
    )
    .await
    .unwrap();

    assert_eq!(
        *host.trace.lock().unwrap(),
        [
            "project",
            "token",
            "before_provider_request:Bearer caller-token"
        ]
    );
    assert_eq!(
        only_request(&upstream).await.header_values("authorization"),
        ["Bearer edited"]
    );
}

#[rstest]
#[tokio::test]
async fn direct_execution_uses_hooks_without_a_machine() {
    use litellm_host::interceptors::Interceptors;

    struct Hooks;

    impl Interceptors<Error> for Hooks {
        async fn before_provider_request(
            &self,
            wire: WireRequest,
            _: RequestContext,
        ) -> Result<WireRequest, Error> {
            Ok(WireRequest {
                headers: wire
                    .headers
                    .into_iter()
                    .chain([("x-direct-hook".into(), "called".into())])
                    .collect(),
                ..wire
            })
        }

        async fn after_provider_response(&self, _: RawResponse) -> Result<(), Error> {
            Ok(())
        }
    }

    let upstream = upstream([json_response(
        json!({"pages":[{"index":0,"markdown":"direct"}]}),
    )])
    .await;
    let events = Arc::new(super::support::CallEvents::default());
    let route = ocr_route();
    let interceptors = Hooks;
    let builder = route.execute(
        ocr_request("mistral/model", &upstream.uri(), json!({})),
        &interceptors,
        Some(events.0.sender.clone()),
    );
    assert!(events.0.lock().unwrap().is_empty());
    assert!(received(&upstream).await.is_empty());
    let result = builder.await.unwrap();
    assert_eq!(result.pages[0].markdown, "direct");
    assert_eq!(
        only_request(&upstream).await.header("x-direct-hook"),
        Some("called")
    );
    assert!(matches!(
        &events.0.lock().unwrap()[..],
        [
            CallEvent::Started { .. },
            CallEvent::Execution(_),
            CallEvent::Succeeded { .. }
        ]
    ));
}

#[rstest]
#[case::native(false)]
#[case::hosted(true)]
#[tokio::test]
async fn ocr_records_one_route_summary_across_both_execution_paths(
    traces: TraceCapture,
    #[case] hosted: bool,
) {
    let upstream = upstream([pages_response()]).await;
    let request = ocr_request("mistral/model", &upstream.uri(), json!({}));
    let model = request.model.clone();
    traces
        .logger()
        .instrument(async {
            if hosted {
                perform_with(LocalOcrHost::new(request)).await
            } else {
                perform(request).await
            }
        })
        .await
        .unwrap();
    let summaries = traces.summaries("litellm.route");
    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0]["route"], "ocr");
    assert_eq!(summaries[0]["model"], model);
    assert_eq!(summaries[0]["provider"], "mistral");
    assert_eq!(summaries[0]["outcome"], "success");
    assert_eq!(summaries[0]["stream"], false);
}
