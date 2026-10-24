use axum::{
    body::Bytes,
    extract::{FromRequest, FromRequestParts, Multipart, Request},
    http::request::Parts,
};
use serde_json::{Map, Value};

use crate::{BodyError, Deployment, Error, Gateway};

pub(crate) const MAX_FILE_BYTES: usize = 50 * 1024 * 1024;
pub const MAX_BODY_BYTES: usize = MAX_FILE_BYTES + 1024 * 1024;

pub(crate) struct Upload {
    pub bytes: Bytes,
    pub file_name: Option<String>,
    pub mime_type: Option<String>,
}

pub struct JsonObject(pub Map<String, Value>);

pub struct RequestId(pub Option<String>);

pub(crate) struct Path<T>(pub T);

impl<T: serde::de::DeserializeOwned + Send, S: Send + Sync> FromRequestParts<S> for Path<T> {
    type Rejection = Error;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        axum::extract::Path::<T>::from_request_parts(parts, state)
            .await
            .map(|axum::extract::Path(value)| Self(value))
            .map_err(Error::InvalidPath)
    }
}

impl<S: Send + Sync> FromRequestParts<S> for RequestId {
    type Rejection = std::convert::Infallible;

    async fn from_request_parts(parts: &mut Parts, _: &S) -> Result<Self, Self::Rejection> {
        Ok(Self(
            parts
                .headers
                .get("x-request-id")
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned),
        ))
    }
}

impl<S: Send + Sync> FromRequest<S> for JsonObject {
    type Rejection = Error;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        let body = Bytes::from_request(request, state).await.map_err(|error| {
            if error.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
                Error::BodyTooLarge
            } else {
                BodyError::Read(error).into()
            }
        })?;
        Ok(Self(object(&body)?))
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
            .map_err(BodyError::Multipart)?;
        parse_multipart(multipart).await
    }
}

fn object(body: &[u8]) -> Result<Map<String, Value>, BodyError> {
    match serde_json::from_slice(body) {
        Ok(Value::Object(body)) => Ok(body),
        Ok(_) => Err(BodyError::NotAnObject),
        Err(error) => Err(BodyError::Json(error)),
    }
}

pub(crate) async fn authorize_model(
    identity: &litellm_gateway_auth::AuthenticatedRequest,
    deployment: &Deployment,
    body: &Map<String, Value>,
) -> Result<(), Error> {
    let name = body
        .get("model")
        .and_then(Value::as_str)
        .ok_or(BodyError::MissingModel)?;
    let access = litellm_gateway_auth::AccessRequest::Model {
        name: name.into(),
        deployment: deployment.model.clone(),
    };
    identity
        .authorize(access.clone())
        .await?
        .consume(identity.caller(), &access)?;
    Ok(())
}

pub(crate) fn resolve_deployment<'a>(
    gateway: &'a Gateway,
    route: &'static str,
    body: &Map<String, Value>,
) -> Result<&'a Deployment, Error> {
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .ok_or(BodyError::MissingModel)?;
    gateway
        .models
        .get(model)
        .ok_or_else(|| Error::UnknownModel {
            route,
            model: model.to_owned(),
        })
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
    Ok(InferenceBody { fields, upload })
}

fn multipart_error(error: axum::extract::multipart::MultipartError) -> Error {
    if error.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
        return Error::BodyTooLarge;
    }
    BodyError::MultipartField(error).into()
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
