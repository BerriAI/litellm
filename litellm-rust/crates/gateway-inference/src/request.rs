use axum::{
    body::{Body, Bytes, HttpBody},
    extract::{FromRequest, FromRequestParts, OriginalUri, Request},
    http::request::Parts,
    response::{IntoResponse, Response},
};
use futures_util::StreamExt;
use litellm_router::{ModelGroup, Routed};
use serde_json::{Map, Value};

use crate::{Deployment, Error, Gateway};

pub(crate) const MAX_FILE_BYTES: usize = 50 * 1024 * 1024;
pub(crate) const MAX_BODY_BYTES: usize = MAX_FILE_BYTES + 1024 * 1024;

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

pub(crate) fn object(body: &[u8]) -> Result<Map<String, Value>, Error> {
    match serde_json::from_slice(body) {
        Ok(Value::Object(body)) => Ok(body),
        Ok(_) => Err(Error::InvalidBody("expected a JSON object".into())),
        Err(error) => Err(Error::InvalidBody(error.to_string())),
    }
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
    fn limited_bodies() -> Router {
        Router::new()
            .route("/", post(|_: JsonBody| async {}))
            .layer(DefaultBodyLimit::max(64))
    }

    #[rstest]
    #[tokio::test]
    async fn bodies_respect_the_configured_limit(limited_bodies: Router) {
        let payload = format!("{{\"text\":\"{}\"}}", "x".repeat(64));
        let response = limited_bodies
            .oneshot(
                Request::post("/")
                    .header("content-type", "application/json")
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
