use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;

use serde_json::{Map, json};

use super::types::AudioTranscriptionRequest;
use super::{AudioRoute, AudioRouteRequest, DefaultAudioServices, audio_transcription};
use crate::Error;
use crate::integrations::custom_guardrail::{
    CustomGuardrail, GuardrailContext, GuardrailDecision, GuardrailEventHook, GuardrailFuture,
    GuardrailRequest,
};
use crate::lifecycle::{ExecutedCall, RouteProjection};

#[tokio::test]
async fn bedrock_request_is_signed_and_contains_audio() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    let address = listener.local_addr().expect("address");
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("connection");
        let mut request = Vec::new();
        let mut buffer = [0_u8; 16_384];
        let count = stream.read(&mut buffer).expect("request");
        request.extend_from_slice(&buffer[..count]);
        let request = String::from_utf8_lossy(&request);
        assert!(request.contains("POST /model/mistral.voxtral-mini-3b-2507/converse"));
        assert!(request.contains("authorization: AWS4-HMAC-SHA256"));
        assert!(request.contains("x-amz-date:"));
        assert!(request.contains("\"bytes\":\"AQI=\""));
        assert!(request.contains("Transcribe the audio. Respond with only the transcript."));
        let response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 53\r\nConnection: close\r\n\r\n{\"output\":{\"message\":{\"content\":[{\"text\":\"hello\"}]}}}";
        stream.write_all(response).expect("response");
    });

    let optional_params = Map::from_iter([
        ("aws_access_key_id".to_string(), json!("access-key")),
        ("aws_secret_access_key".to_string(), json!("secret-key")),
        ("aws_region_name".to_string(), json!("us-east-1")),
    ]);
    let api_base = format!("http://{address}");
    let response = audio_transcription(AudioTranscriptionRequest {
        model: "mistral.voxtral-mini-3b-2507",
        audio: json!({"data": "AQI=", "format": "wav", "filename": "audio.wav"}),
        api_key: None,
        api_base: Some(&api_base),
        custom_llm_provider: Some("bedrock"),
        extra_headers: None,
        optional_params,
        timeout: None,
    })
    .await
    .expect("transcription");
    assert_eq!(response, json!({"text": "hello"}));
    server.join().expect("server");
}

struct ReplacingGuardrail {
    calls: Mutex<Vec<&'static str>>,
}

impl CustomGuardrail for ReplacingGuardrail {
    fn guardrail_name(&self) -> &str {
        "audio-test"
    }

    fn supported_event_hooks(&self) -> &[GuardrailEventHook] {
        &[GuardrailEventHook::PreCall, GuardrailEventHook::DuringCall]
    }

    fn async_pre_call_hook<'a>(
        &'a self,
        _: &'a GuardrailContext,
        mut request: GuardrailRequest,
    ) -> GuardrailFuture<'a> {
        Box::pin(async move {
            self.calls.lock().unwrap().push("pre");
            request.data["audio"]["data"] = json!("AwQ=");
            Ok(GuardrailDecision::Mask(request))
        })
    }

    fn async_moderation_hook<'a>(
        &'a self,
        _: &'a GuardrailContext,
        mut request: GuardrailRequest,
    ) -> GuardrailFuture<'a> {
        Box::pin(async move {
            self.calls.lock().unwrap().push("during");
            request.data["body"]["messages"][0]["content"][0]["text"] =
                json!("Guarded transcription prompt");
            Ok(GuardrailDecision::Mask(request))
        })
    }
}

#[tokio::test]
async fn route_owns_guardrail_provider_and_terminal_sequence() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    let address = listener.local_addr().expect("address");
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("connection");
        let mut buffer = [0_u8; 16_384];
        let count = stream.read(&mut buffer).expect("request");
        let request = String::from_utf8_lossy(&buffer[..count]);
        assert!(request.contains("\"bytes\":\"AwQ=\""));
        assert!(request.contains("Guarded transcription prompt"));
        let response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 53\r\nConnection: close\r\n\r\n{\"output\":{\"message\":{\"content\":[{\"text\":\"hello\"}]}}}";
        stream.write_all(response).expect("response");
    });
    let guardrail = Arc::new(ReplacingGuardrail {
        calls: Mutex::new(Vec::new()),
    });
    let services = DefaultAudioServices::new(Vec::new(), vec![guardrail.clone()]);
    let api_base = format!("http://{address}");
    let executed = AudioRoute::execute(
        &services,
        AudioRouteRequest {
            model: "bedrock/mistral.voxtral-mini-3b-2507",
            audio: json!({"data": "AQI=", "format": "wav", "filename": "audio.wav"}),
            api_key: None,
            api_base: Some(&api_base),
            custom_llm_provider: None,
            extra_headers: None,
            optional_params: Map::from_iter([
                ("aws_access_key_id".to_string(), json!("access-key")),
                ("aws_secret_access_key".to_string(), json!("secret-key")),
                ("aws_region_name".to_string(), json!("us-east-1")),
            ]),
            timeout: None,
            request_metadata: Default::default(),
            litellm_call_id: Some("audio-call-1"),
        },
    )
    .await;

    assert_eq!(*guardrail.calls.lock().unwrap(), vec!["pre", "during"]);
    assert!(matches!(
        executed,
        ExecutedCall::Success {
            response,
            terminal,
        } if response == json!({"text": "hello"})
            && terminal.call_id == "audio-call-1"
            && matches!(terminal.projection, RouteProjection::Audio { ref value } if value == &response)
    ));
    server.join().expect("server");
}

#[tokio::test]
async fn route_returns_preparation_failure_with_audio_terminal() {
    let services = DefaultAudioServices::new(Vec::new(), Vec::new());
    let executed = AudioRoute::execute(
        &services,
        AudioRouteRequest {
            model: "unsupported/model",
            audio: json!({"data": "AQI="}),
            api_key: None,
            api_base: None,
            custom_llm_provider: Some("unsupported"),
            extra_headers: None,
            optional_params: Map::new(),
            timeout: None,
            request_metadata: Default::default(),
            litellm_call_id: Some("audio-call-failure"),
        },
    )
    .await;

    assert!(matches!(
        executed,
        ExecutedCall::Failure {
            error: Error::InvalidProvider(_),
            terminal,
        } if terminal.call_id == "audio-call-failure"
            && matches!(terminal.projection, RouteProjection::Audio { .. })
    ));
}
