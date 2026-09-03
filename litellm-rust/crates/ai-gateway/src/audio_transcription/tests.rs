use std::io::{BufRead, BufReader, Read, Write};
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
        let mut reader = BufReader::new(&mut stream);
        let mut headers = String::new();
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).expect("headers");
            if line == "\r\n" {
                break;
            }
            headers.push_str(&line);
        }
        let length = headers
            .lines()
            .find_map(|line| line.strip_prefix("content-length: "))
            .expect("content length")
            .parse::<usize>()
            .unwrap();
        let mut body = vec![0; length];
        reader.read_exact(&mut body).expect("complete request body");
        let request = headers + &String::from_utf8(body).unwrap();
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
