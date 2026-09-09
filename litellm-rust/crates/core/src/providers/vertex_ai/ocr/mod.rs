use crate::auth::AuthError;
use crate::constants::VERTEX_OCR_DEFAULT_LOCATION;
use crate::ocr::error::OcrRequestError;
use crate::ocr::types::OcrConnection;
use crate::providers::vertex_ai::auth;

pub(crate) fn project(connection: &OcrConnection) -> Result<&str, OcrRequestError> {
    connection
        .vertex
        .project
        .as_deref()
        .ok_or(OcrRequestError::MissingField("vertex_project"))
}
pub(crate) fn location(connection: &OcrConnection) -> &str {
    connection
        .vertex
        .location
        .as_deref()
        .unwrap_or(VERTEX_OCR_DEFAULT_LOCATION)
}

pub(crate) async fn authenticate_vertex(
    connection: &OcrConnection,
) -> Result<Vec<(String, String)>, AuthError> {
    auth::authenticate(
        connection.extra_headers.clone(),
        connection.api_key.as_deref(),
        &connection.vertex_auth,
        connection.vertex.project.clone(),
        &|name| std::env::var(name).ok(),
    )
    .await
    .map(|authentication| authentication.headers)
}
