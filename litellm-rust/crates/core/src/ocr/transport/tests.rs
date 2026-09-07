use super::*;

fn request() -> Request {
    Request {
        url: "unknown://private-document?secret=credential".into(),
        headers: vec![],
        body: vec![0, 255],
        timeout_seconds: 1.0,
    }
}

#[tokio::test]
async fn rejects_invalid_timeouts_and_headers_without_echoing_wire_values() {
    for timeout_seconds in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MAX] {
        let result = send(Request {
            timeout_seconds,
            ..request()
        })
        .await;
        assert!(
            matches!(result, Err(Error::InvalidRequest(message)) if message == "OCR timeout must be positive and finite")
        );
    }
    for (headers, expected) in [
        (
            vec![(b"private\nname".to_vec(), b"secret".to_vec())],
            "invalid OCR header name",
        ),
        (
            vec![(b"x-proof".to_vec(), b"private\nvalue".to_vec())],
            "invalid OCR header value",
        ),
    ] {
        let result = send(Request {
            headers,
            ..request()
        })
        .await;
        assert!(matches!(result, Err(Error::InvalidRequest(message)) if message == expected));
    }
    assert!(
        matches!(send(request()).await, Err(Error::Network(message)) if message == "OCR transport failed")
    );
}
