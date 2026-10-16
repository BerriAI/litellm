use axum::{
    body::{Body, Bytes, HttpBody},
    extract::{FromRequest, FromRequestParts, Multipart, OriginalUri, Request},
    http::request::Parts,
    response::{IntoResponse, Response},
};
use futures_util::StreamExt;
use litellm_router::{ModelGroup, Routed};
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

/// A JSON object body and the bytes it was parsed from, for a handler that parses it again.
pub struct JsonBody {
    pub object: Map<String, Value>,
    pub raw: Bytes,
}

pub struct RequestId(pub Option<String>);

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
        let body = JsonBody::from_request(request, state).await?;
        Ok(Self(body.object))
    }
}

impl<S: Send + Sync> FromRequest<S> for JsonBody {
    type Rejection = Error;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        let raw = Bytes::from_request(request, state).await.map_err(|error| {
            if error.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
                Error::BodyTooLarge
            } else {
                Error::InvalidBody(error.to_string())
            }
        })?;
        Ok(Self {
            object: object(&raw)?,
            raw,
        })
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

pub(crate) fn object(body: &[u8]) -> Result<Map<String, Value>, Error> {
    match serde_json::from_slice(body) {
        Ok(Value::Object(body)) => Ok(body),
        Ok(_) => Err(Error::InvalidBody("expected a JSON object".into())),
        Err(error) => Err(Error::InvalidBody(error.to_string())),
    }
}

pub(crate) async fn authorize_model(
    identity: &litellm_gateway_auth::AuthenticatedRequest,
    deployment: &Deployment,
    body: &Map<String, Value>,
) -> Result<(), Error> {
    authorize(identity, model_name(body)?, deployment).await
}

/// Checks that the caller may use the public model `name` served by `deployment`.
pub(crate) async fn authorize(
    identity: &litellm_gateway_auth::AuthenticatedRequest,
    name: &str,
    deployment: &Deployment,
) -> Result<(), Error> {
    let access = litellm_gateway_auth::AccessRequest::Model {
        name: name.into(),
        deployment: deployment.model.clone(),
        team_id: deployment.team_id.clone(),
    };
    identity
        .authorize(access.clone())
        .await?
        .consume(identity.caller(), &access)?;
    Ok(())
}

pub(crate) fn model_name(body: &Map<String, Value>) -> Result<&str, Error> {
    body.get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::InvalidBody("model is required".into()))
}

/// The model group a request names.
pub(crate) fn resolve_group<'a>(
    gateway: &'a Gateway,
    body: &Map<String, Value>,
) -> Result<&'a ModelGroup, Error> {
    let model = model_name(body)?;
    gateway
        .models
        .group(model)
        .ok_or_else(|| Error::UnknownModel(model.to_owned()))
}

/// Whether routing may pick `deployment` for the caller: a deployment owned by a team
/// (`model_info.team_id`) serves only that team's virtual keys, as the proxy's
/// `filter_team_based_models` does.
pub(crate) fn usable(
    identity: &litellm_gateway_auth::AuthenticatedRequest,
) -> impl Fn(&Deployment) -> bool + '_ {
    |deployment| identity.may_use_deployment(deployment.team_id.as_deref())
}

/// The error for a group none of whose deployments the caller may use.
pub(crate) fn no_usable_deployment(body: &Map<String, Value>) -> Error {
    Error::UnknownModel(model_name(body).unwrap_or_default().to_owned())
}

/// One deployment the caller may use of the group a request names, for endpoints served
/// without retries.
pub(crate) fn resolve_deployment<'a>(
    gateway: &'a Gateway,
    identity: &litellm_gateway_auth::AuthenticatedRequest,
    body: &Map<String, Value>,
) -> Result<&'a Deployment, Error> {
    let model = model_name(body)?;
    let deployment = gateway
        .models
        .select_where(resolve_group(gateway, body)?, usable(identity))
        .ok_or_else(|| no_usable_deployment(body))?;
    crate::observe::record(model, deployment);
    Ok(deployment)
}

/// A routed response that keeps its deployment marked in flight until its body is done.
/// A body of known length is already complete, so its mark is released at once.
pub(crate) fn hold(routed: Routed<Response>) -> Response {
    let Routed { value, in_flight } = routed;
    if value.body().size_hint().exact().is_some() {
        return value;
    }
    value.map(|body| {
        Body::from_stream(body.into_data_stream().map(move |chunk| {
            let _held = &in_flight;
            chunk
        }))
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
    Error::InvalidBody(error.to_string())
}

pub(crate) async fn unsupported(OriginalUri(uri): OriginalUri) -> impl IntoResponse {
    Error::Unsupported(uri.path().to_owned())
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
