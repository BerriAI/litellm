use axum::{
    body::{Bytes, to_bytes},
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

pub(crate) fn object(body: &[u8]) -> Result<Map<String, Value>, Error> {
    match serde_json::from_slice(body) {
        Ok(Value::Object(body)) => Ok(body),
        Ok(_) => Err(Error::InvalidBody("expected a JSON object".into())),
        Err(error) => Err(Error::InvalidBody(error.to_string())),
    }
}

pub(crate) fn deployment<'a>(
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

pub(crate) async fn parse(request: Request) -> Result<(Map<String, Value>, Option<Upload>), Error> {
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
        let body = to_bytes(request.into_body(), MAX_BODY_BYTES)
            .await
            .map_err(|_| Error::BodyTooLarge)?;
        return Ok((object(&body)?, None));
    }
    let mut multipart = Multipart::from_request(request, &())
        .await
        .map_err(|error| Error::InvalidBody(error.to_string()))?;
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
    Ok((fields, upload))
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
