use axum::{
    Json,
    response::{IntoResponse, Response},
};
use litellm_http::response::{ProviderResponse, append_provider_headers};
use serde::Serialize;

pub(crate) fn json<T: Serialize>(provider: ProviderResponse<T>) -> Response {
    let mut response = Json(provider.body).into_response();
    append_provider_headers(response.headers_mut(), &provider.headers);
    response
}
