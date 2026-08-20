use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::thread;

use serde_json::{Map, json};

use super::{AudioTranscriptionRequest, audio_transcription};
use crate::integrations::custom_guardrail::{
    CustomGuardrail, GuardrailContext, GuardrailDecision, GuardrailEventHook, GuardrailFuture,
    GuardrailRequest,
};

struct DuringCallAudioGuardrail;

impl CustomGuardrail for DuringCallAudioGuardrail {
    fn guardrail_name(&self) -> &str {
        "audio-during-call-test"
    }

    fn supported_event_hooks(&self) -> &[GuardrailEventHook] {
        static HOOKS: [GuardrailEventHook; 1] = [GuardrailEventHook::DuringCall];

        &HOOKS
    }

    fn async_moderation_hook<'a>(
        &'a self,
        _context: &'a GuardrailContext,
        mut request: GuardrailRequest,
    ) -> GuardrailFuture<'a> {
        Box::pin(async move {
            request.data["body"]["guarded_during"] = json!("added-after-provider-transform");

            Ok(GuardrailDecision::Mask(request))
        })
    }
}

fn bedrock_optional_params() -> Map<String, serde_json::Value> {
    Map::from_iter([
        ("aws_access_key_id".to_string(), json!("access-key")),
        ("aws_secret_access_key".to_string(), json!("secret-key")),
        ("aws_region_name".to_string(), json!("us-east-1")),
    ])
}

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

        let response = b"HTTP/1.1 200 OK\r\n\
Content-Type: application/json\r\n\
Content-Length: 53\r\n\
Connection: close\r\n\
\r\n\
{\"output\":{\"message\":{\"content\":[{\"text\":\"hello\"}]}}}";

        stream.write_all(response).expect("response");
    });

    let api_base = format!("http://{address}");

    let response = audio_transcription(AudioTranscriptionRequest {
        model: "mistral.voxtral-mini-3b-2507",
        audio: json!({
            "data": "AQI=",
            "format": "wav",
            "filename": "audio.wav"
        }),
        api_key: None,
        api_base: Some(&api_base),
        custom_llm_provider: Some("bedrock"),
        extra_headers: None,
        optional_params: bedrock_optional_params(),
        timeout: None,
        callbacks: Vec::new(),
        guardrails: Vec::new(),
        request_metadata: Default::default(),
        litellm_call_id: None,
    })
    .await
    .expect("transcription");

    assert_eq!(response, json!({"text": "hello"}));

    server.join().expect("server");
}

#[tokio::test]
async fn during_call_guardrail_mutation_is_sent_after_provider_transform() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    let address = listener.local_addr().expect("address");

    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("connection");

        let mut request = Vec::new();
        let mut buffer = [0_u8; 16_384];

        let count = stream.read(&mut buffer).expect("request");

        request.extend_from_slice(&buffer[..count]);

        let request = String::from_utf8_lossy(&request);

        // Provider transformation happened before the during-call hook.
        assert!(
            request.contains("\"bytes\":\"AQI=\""),
            "provider-transformed audio should remain in final request:\n{request}"
        );

        // The during-call guardrail changed the transformed provider body.
        assert!(
            request.contains("\"guarded_during\":\"added-after-provider-transform\""),
            "guardrail mutation should be present in final provider body:\n{request}"
        );

        // Core still finalized provider authentication after the mutation.
        assert!(
            request.contains("authorization: AWS4-HMAC-SHA256"),
            "final request should contain SigV4 authorization:\n{request}"
        );

        assert!(
            request.contains("x-amz-date:"),
            "final request should contain SigV4 date:\n{request}"
        );

        let response = b"HTTP/1.1 200 OK\r\n\
Content-Type: application/json\r\n\
Content-Length: 53\r\n\
Connection: close\r\n\
\r\n\
{\"output\":{\"message\":{\"content\":[{\"text\":\"hello\"}]}}}";

        stream.write_all(response).expect("response");
    });

    let api_base = format!("http://{address}");

    let response = audio_transcription(AudioTranscriptionRequest {
        model: "mistral.voxtral-mini-3b-2507",
        audio: json!({
            "data": "AQI=",
            "format": "wav",
            "filename": "audio.wav"
        }),
        api_key: None,
        api_base: Some(&api_base),
        custom_llm_provider: Some("bedrock"),
        extra_headers: None,
        optional_params: bedrock_optional_params(),
        timeout: None,
        callbacks: Vec::new(),
        guardrails: vec![Arc::new(DuringCallAudioGuardrail)],
        request_metadata: Default::default(),
        litellm_call_id: Some("audio-guardrail-signing-regression"),
    })
    .await
    .expect("transcription with during-call guardrail");

    assert_eq!(response, json!({"text": "hello"}));

    server.join().expect("server");
}
