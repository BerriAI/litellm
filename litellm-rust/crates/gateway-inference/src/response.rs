use axum::http::{self, HeaderMap, HeaderName};
use axum::{Json, response::IntoResponse};
use litellm_http::response::{Response, ResponseHead};
use serde::Serialize;

pub(crate) fn json<T: Serialize>(response: Response<T>) -> axum::response::Response {
    let mut output = Json(response.body).into_response();
    *output.status_mut() = response.head.status;
    append_provider_headers(output.headers_mut(), &response.head.headers);
    output
}

pub(crate) fn stream_head(head: ResponseHead) -> http::Response<()> {
    let mut response = http::Response::new(());
    *response.status_mut() = head.status;
    append_provider_headers(response.headers_mut(), &head.headers);
    response
}

pub(crate) fn append_provider_headers(output: &mut HeaderMap, provider: &HeaderMap) {
    for (name, value) in provider {
        let Ok(name) = HeaderName::from_bytes(format!("llm_provider-{name}").as_bytes()) else {
            continue;
        };
        output.append(name, value.clone());
    }
}
