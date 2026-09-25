use std::{
    io::{Read, Write},
    net::TcpListener,
    thread,
};

use litellm_core::audio_transcription::{audio_transcription, types::AudioTranscriptionRequest};
use serde_json::{Map, json};

#[tokio::test]
async fn bedrock_request_is_signed_and_contains_audio() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    let address = listener.local_addr().expect("address");
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("connection");
        let mut request = Vec::new();
        let mut buffer = [0_u8; 16_384];
        loop {
            let count = stream.read(&mut buffer).expect("request");
            assert!(
                count > 0,
                "request stream closed before the full body arrived"
            );
            request.extend_from_slice(&buffer[..count]);
            let Some(headers_end) = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|position| position + 4)
            else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..headers_end]).to_ascii_uppercase();
            let content_length: usize = headers
                .lines()
                .find_map(|line| line.strip_prefix("CONTENT-LENGTH:"))
                .expect("content-length header")
                .trim()
                .parse()
                .expect("content-length");
            if request.len() >= headers_end + content_length {
                break;
            }
        }
        let request = String::from_utf8_lossy(&request);
        assert!(request.contains("POST /model/mistral.voxtral-mini-3b-2507/converse"));
        assert!(request.contains("authorization: AWS4-HMAC-SHA256"));
        assert!(request.contains("x-amz-date:"));
        assert!(request.contains("\"bytes\":\"AQI=\""));
        assert!(request.contains("Transcribe the audio. Respond with only the transcript."));
        assert!(!request.contains("\"system\""));
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
