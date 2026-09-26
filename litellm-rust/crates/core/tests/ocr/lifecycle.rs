use std::sync::{Arc, Mutex};

use litellm_core::ocr::{
    route::{Ocr, OcrOp, OcrProjection, ocr_machine},
    types::OcrDocumentInput,
};
use litellm_host::{
    event::{CallEvent, MachineEvent, RequestContext, WireRequest},
    host::Host,
};
use rstest::rstest;

use super::*;

pub(crate) fn event_name(event: &CallEvent) -> &'static str {
    match event {
        CallEvent::Started { .. } => "started",
        CallEvent::Machine(MachineEvent::ResponseReceived { .. }) => "response",
        CallEvent::Succeeded { .. } => "success",
        CallEvent::Failed { .. } => "failure",
    }
}

fn recording_host(
    request: LiteLLMOcrRequest,
    events: Arc<Mutex<Vec<&'static str>>>,
    block: bool,
) -> LocalOcrHost {
    let before_send_events = events.clone();
    LocalOcrHost::new(request)
        .with_before_send(move |wire, _| {
            before_send_events.lock().unwrap().push("before_send");
            match block {
                true => Err(Error::InvalidRequest("blocked".into())),
                false => Ok(wire),
            }
        })
        .with_observer(move |event| events.lock().unwrap().push(event_name(event)))
}

#[tokio::test]
async fn hooks_run_in_order_and_one_success_is_emitted() {
    let upstream = upstream([pages_response()]).await;
    let events = Arc::new(Mutex::new(Vec::new()));

    perform_with(recording_host(
        ocr_request("mistral/model", &upstream.uri(), json!({})),
        events.clone(),
        false,
    ))
    .await
    .unwrap();

    assert_eq!(
        *events.lock().unwrap(),
        ["started", "before_send", "response", "success"]
    );
    assert_eq!(received(&upstream).await.len(), 1);
}

#[tokio::test]
async fn a_blocking_before_send_prevents_the_call_and_emits_one_failure() {
    let upstream = upstream([pages_response()]).await;
    let events = Arc::new(Mutex::new(Vec::new()));

    let error = perform_with(recording_host(
        ocr_request("mistral/model", &upstream.uri(), json!({})),
        events.clone(),
        true,
    ))
    .await
    .unwrap_err();

    assert!(
        matches!(&error, Error::InvalidRequest(message) if message == "blocked"),
        "{error:?}"
    );
    assert_eq!(
        *events.lock().unwrap(),
        ["started", "before_send", "failure"]
    );
    assert!(received(&upstream).await.is_empty());
}

#[tokio::test]
async fn an_upstream_failure_emits_one_terminal_failure() {
    let upstream = upstream([status_response(500, json!({"error": "failed"}))]).await;
    let events = Arc::new(Mutex::new(Vec::new()));

    let result = perform_with(recording_host(
        ocr_request("mistral/model", &upstream.uri(), json!({})),
        events.clone(),
        false,
    ))
    .await;

    assert!(result.is_err());
    assert_eq!(
        *events.lock().unwrap(),
        ["started", "before_send", "failure"]
    );
    assert_eq!(received(&upstream).await.len(), 1);
}

#[tokio::test]
async fn an_invalid_provider_response_is_observed_before_normalization_fails() {
    let upstream = upstream([json_response(json!({"pages": "invalid"}))]).await;
    let observed = Arc::new(Mutex::new(Vec::new()));
    let recorder = observed.clone();
    let host = LocalOcrHost::new(ocr_request("mistral/model", &upstream.uri(), json!({})))
        .with_observer(move |event| {
            if let CallEvent::Machine(MachineEvent::ResponseReceived { raw }) = event {
                recorder.lock().unwrap().push(raw.body.clone());
            }
        });

    let error = perform_with(host).await.unwrap_err();

    assert!(matches!(error, Error::ResponseField { .. }), "{error:?}");
    assert_eq!(*observed.lock().unwrap(), [r#"{"pages":"invalid"}"#]);
}

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
    context.expect("before_send ran")
}

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

/// Hands the route a caller-owned Azure token and rewrites the bearer in `before_send`.
struct CallerTokenHost {
    request: Mutex<Option<LiteLLMOcrRequest>>,
    trace: Mutex<Vec<String>>,
}

impl Host<Ocr> for CallerTokenHost {
    async fn project(&self) -> Result<OcrProjection, Error> {
        self.trace.lock().unwrap().push("project".into());
        Ok(OcrProjection {
            request: self.request.lock().unwrap().take().unwrap(),
            caller_token: true,
        })
    }

    async fn custom_op(&self, op: OcrOp) -> Result<(), Error> {
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

    async fn before_send(
        &self,
        wire: WireRequest,
        _: &RequestContext,
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
            .push(format!("before_send:{authorization}"));
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
}

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

    litellm_host::run::run(ocr_machine(ocr_client()), &host)
        .await
        .unwrap();

    assert_eq!(
        *host.trace.lock().unwrap(),
        ["project", "token", "before_send:Bearer caller-token"]
    );
    assert_eq!(
        only_request(&upstream).await.header_values("authorization"),
        ["Bearer edited"]
    );
}
