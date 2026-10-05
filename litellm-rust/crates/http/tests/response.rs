use http::{HeaderMap, HeaderValue, StatusCode};
use litellm_http::response::{Response, ResponseHead};

#[rstest::rstest]
fn body_mapping_preserves_status_and_every_header_value() {
    let mut headers = HeaderMap::new();
    headers.append("x-repeat", HeaderValue::from_static("first"));
    headers.append("x-repeat", HeaderValue::from_static("second"));
    headers.insert("x-opaque", HeaderValue::from_bytes(b"\x80\xff").unwrap());
    let head = ResponseHead {
        status: StatusCode::CREATED,
        headers,
    };
    let response = Response {
        head: head.clone(),
        body: "provider body",
    };

    let mapped = response.map(str::len);

    assert_eq!(mapped.body, "provider body".len());
    assert_eq!(mapped.head, head);
}
