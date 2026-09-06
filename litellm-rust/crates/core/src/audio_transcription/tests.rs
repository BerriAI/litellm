#![cfg(feature = "bedrock-auth")]

use crate::request_context::LiteLlmRequestContext;
use crate::request_options::{BedrockOptions, RequestOptions};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use serde_json::{Map, json};

use super::audio_transcription;
use super::types::AudioTranscriptionRequest;

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

    let bedrock = BedrockOptions {
        aws_access_key_id: Some("access-key".to_string()),
        aws_secret_access_key: Some("secret-key".to_string()),
        aws_region_name: Some("us-east-1".to_string()),
        ..Default::default()
    };
    let api_base = format!("http://{address}");
    let response = audio_transcription(
        AudioTranscriptionRequest {
            model: "mistral.voxtral-mini-3b-2507",
            audio: json!({"data": "AQI=", "format": "wav", "filename": "audio.wav"}),
            optional_params: Map::new(),
        },
        &RequestOptions {
            bedrock: Some(bedrock),
            api_key: None,
            api_base: (Some(&api_base)).map(|value| value.to_string()),
            custom_llm_provider: (Some("bedrock")).map(|value| value.to_string()),
            extra_headers: None,
            timeout: None,
            ..Default::default()
        },
        &LiteLlmRequestContext {
            ..Default::default()
        },
    )
    .await
    .expect("transcription");
    assert_eq!(response, json!({"text": "hello"}));
    server.join().expect("server");
}
