use axum::{
    body::Bytes,
    extract::{FromRequest, Multipart, Request},
    http::Uri,
    response::Response,
};
use serde_json::{Map, Value};

use crate::{Deployment, Error, Gateway};

pub(crate) const MAX_FILE_BYTES: usize = 50 * 1024 * 1024;
pub(crate) const MAX_BODY_BYTES: usize = MAX_FILE_BYTES + 1024 * 1024;

pub(crate) struct Upload {
    pub bytes: Bytes,
    pub file_name: Option<String>,
    pub mime_type: Option<String>,
}

pub struct JsonObject(pub Map<String, Value>);

impl<S: Send + Sync> FromRequest<S> for JsonObject {
    type Rejection = Error;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        let body = Bytes::from_request(request, state).await.map_err(|error| {
            if error.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
                Error::BodyTooLarge
            } else {
                Error::InvalidBody(error.to_string())
            }
        })?;
        object(&body).map(Self)
    }
}

pub(crate) struct InferenceBody {
    pub fields: Map<String, Value>,
    pub upload: Option<Upload>,
}

impl<S: Send + Sync> FromRequest<S> for InferenceBody {
    type Rejection = Error;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        let multipart = request
            .headers()
            .get("content-type")
            .and_then(|header| header.to_str().ok())
            .is_some_and(|value| {
                value
                    .to_ascii_lowercase()
                    .starts_with("multipart/form-data")
            });
        if !multipart {
            let JsonObject(fields) = JsonObject::from_request(request, state).await?;
            return Ok(Self {
                fields,
                upload: None,
            });
        }
        let multipart = Multipart::from_request(request, state)
            .await
            .map_err(|error| Error::InvalidBody(error.to_string()))?;
        parse_multipart(multipart).await
    }
}

fn object(body: &[u8]) -> Result<Map<String, Value>, Error> {
    match serde_json::from_slice(body) {
        Ok(Value::Object(body)) => Ok(body),
        Ok(_) => Err(Error::InvalidBody("expected a JSON object".into())),
        Err(error) => Err(Error::InvalidBody(error.to_string())),
    }
}

pub(crate) fn resolve_deployment<'a>(
    gateway: &'a Gateway,
    body: &Map<String, Value>,
) -> Result<&'a Deployment, Error> {
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::InvalidBody("model is required".into()))?;
    gateway
        .models
        .get(model)
        .ok_or_else(|| Error::UnknownModel(model.to_owned()))
}

async fn parse_multipart(mut multipart: Multipart) -> Result<InferenceBody, Error> {
    let mut fields = Map::new();
    let mut upload = None;
    while let Some(field) = multipart.next_field().await.map_err(multipart_error)? {
        let name = field.name().unwrap_or_default().to_owned();
        if name == "file" {
            let file_name = field.file_name().map(str::to_owned);
            let mime_type = field
                .content_type()
                .and_then(|value| value.split(';').next())
                .map(str::trim)
                .filter(|value| *value != "application/octet-stream")
                .map(str::to_owned);
            let bytes = field.bytes().await.map_err(multipart_error)?;
            if bytes.len() > MAX_FILE_BYTES {
                return Err(Error::BodyTooLarge);
            }
            if bytes.is_empty() {
                return Err(Error::InvalidBody("uploaded file is empty".into()));
            }
            upload = Some(Upload {
                bytes,
                file_name,
                mime_type,
            });
        } else if name != "document" {
            let text = field.text().await.map_err(multipart_error)?;
            let value = serde_json::from_str(&text).unwrap_or(Value::String(text));
            fields.insert(name, value);
        }
    }
    if upload.is_none() {
        return Err(Error::InvalidBody(
            "multipart request requires a file field".into(),
        ));
    }
    Ok(InferenceBody { fields, upload })
}

fn multipart_error(error: axum::extract::multipart::MultipartError) -> Error {
    if error.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
        return Error::BodyTooLarge;
    }
    Error::InvalidBody(error.to_string())
}

pub(crate) async fn unsupported(uri: Uri) -> Response {
    Error::Unsupported(uri.path().to_owned()).openai_response()
}

#[cfg(test)]
mod tests {
    use axum::{
        Router,
        body::{Body, to_bytes},
        extract::DefaultBodyLimit,
        routing::post,
    };
    use rstest::{fixture, rstest};
    use tower::ServiceExt;

    use super::*;

    #[fixture]
    fn limited_uploads() -> Router {
        Router::new()
            .route("/", post(|_: InferenceBody| async {}))
            .layer(DefaultBodyLimit::max(64))
    }

    #[rstest]
    #[case::json("application/json", format!("{{\"text\":\"{}\"}}", "x".repeat(64)))]
    #[case::multipart(
        "multipart/form-data; boundary=test",
        format!("--test\r\nContent-Disposition: form-data; name=\"file\"; filename=\"file.pdf\"\r\n\r\n{}\r\n--test--\r\n", "x".repeat(64)),
    )]
    #[tokio::test]
    async fn uploads_respect_the_configured_body_limit(
        limited_uploads: Router,
        #[case] content_type: &str,
        #[case] payload: String,
    ) {
        let response = limited_uploads
            .oneshot(
                Request::post("/")
                    .header("content-type", content_type)
                    .body(Body::from(payload))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 413);
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(body["error"]["type"], "request_too_large");
        assert_eq!(body["error"]["code"], 413);
    }
}
