use serde_json::{Map, Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use super::{AudioTranscriptionRequest, audio_transcription};

fn optional_params() -> Map<String, Value> {
    let mut params = Map::new();

    params.insert("aws_access_key_id".to_string(), json!("test-access-key"));
    params.insert(
        "aws_secret_access_key".to_string(),
        json!("test-secret-key"),
    );
    params.insert("aws_region_name".to_string(), json!("us-east-1"));

    params
}

#[tokio::test]
async fn bedrock_request_is_signed_and_contains_audio() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test listener");

    let address = listener.local_addr().expect("read listener address");

    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept request");

        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];

        loop {
            let read = socket.read(&mut buffer).await.expect("read request");

            if read == 0 {
                break;
            }

            request.extend_from_slice(&buffer[..read]);

            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                let request_text = String::from_utf8_lossy(&request);

                let content_length = request_text
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;

                        if name.eq_ignore_ascii_case("content-length") {
                            value.trim().parse::<usize>().ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);

                let header_end = request
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .expect("header terminator")
                    + 4;

                if request.len() >= header_end + content_length {
                    break;
                }
            }
        }

        let request_text = String::from_utf8_lossy(&request);

        assert!(
            request_text.contains("Authorization: AWS4-HMAC-SHA256")
                || request_text.contains("authorization: AWS4-HMAC-SHA256"),
            "request should contain AWS SigV4 authorization header:\n{request_text}"
        );

        assert!(
            request_text.contains("test-audio"),
            "request should contain encoded audio payload:\n{request_text}"
        );

        assert!(
            request_text.contains("transcribe this"),
            "request should preserve instruction:\n{request_text}"
        );

        let response_body = json!({
            "output": {
                "message": {
                    "content": [
                        {
                            "text": "hello"
                        }
                    ]
                }
            }
        })
        .to_string();

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response_body.len(),
            response_body,
        );

        socket
            .write_all(response.as_bytes())
            .await
            .expect("write response");
    });

    let mut params = optional_params();

    params.insert("prompt".to_string(), json!("transcribe this"));

    let api_base = format!("http://{address}");

    let response = audio_transcription(AudioTranscriptionRequest {
        model: "amazon.nova-2-sonic-v1:0",
        audio: json!({
            "data": "test-audio",
            "format": "wav",
            "filename": "sample.wav"
        }),
        api_key: None,
        api_base: Some(&api_base),
        custom_llm_provider: Some("bedrock"),
        extra_headers: None,
        optional_params: params,
        timeout: None,
    })
    .await
    .expect("audio transcription succeeds");

    server.await.expect("server completes");

    assert_eq!(response.text, "hello");
}
