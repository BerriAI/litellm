use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use serde_json::{Map, json};

use super::{AudioTranscriptionRequest, audio_transcription};

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
