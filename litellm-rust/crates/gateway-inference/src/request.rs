use axum::{
    body::{Body, Bytes, HttpBody},
    extract::{FromRequest, FromRequestParts, OriginalUri, Request},
    http::request::Parts,
    response::{IntoResponse, Response},
};
use futures_util::StreamExt;
use litellm_router::{ModelGroup, Routed};
use serde_json::{Map, Value};

use crate::{Deployment, Error, Gateway, MalformedMessages};

/// The names the proxy's `ProxyMissingRequiredParamError` gives each route, as
/// `ROUTE_ENDPOINT_MAPPING` maps them.
const CHAT_COMPLETIONS: &str = "/chat/completions";
const MESSAGES: &str = "anthropic_messages";

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

/// Refuses a Chat Completions body whose `messages` no provider could take, before a hook counts
/// the request or a provider sees it. As in the proxy, `messages` must be present and not null,
/// then as Python's `completion` reads it, an array of objects. A message may leave out its
/// `role`, which `validate_and_fix_openai_messages` fills in as `assistant`, but a role it names
/// must be a string. An empty array is refused here rather than by every provider.
pub(crate) fn check_chat_messages(body: &Map<String, Value>) -> Result<(), Error> {
    let messages = match body.get("messages") {
        None | Some(Value::Null) => {
            return Err(Error::MissingParam {
                route: CHAT_COMPLETIONS,
                param: "messages",
            });
        }
        Some(messages) => messages,
    };
    each_message(messages, |message| match message.get("role") {
        None | Some(Value::Null | Value::String(_)) => Ok(()),
        Some(_) => Err(MalformedMessages::Role),
    })?;
    match messages.as_array().is_some_and(Vec::is_empty) {
        true => Err(MalformedMessages::Empty.into()),
        false => Ok(()),
    }
}

/// Refuses a `/v1/messages` body whose `messages` no provider could take, before a hook counts
/// the request or a provider sees it: the proxy requires the key, and Anthropic's shape needs
/// each message to have a string `role` and a string or array `content`.
pub(crate) fn check_messages(body: &Map<String, Value>) -> Result<(), Error> {
    let messages = body.get("messages").ok_or(Error::MissingParam {
        route: MESSAGES,
        param: "messages",
    })?;
    each_message(messages, |message| {
        match message.get("role") {
            Some(Value::String(_)) => {}
            _ => return Err(MalformedMessages::Role),
        }
        match message.get("content") {
            Some(Value::String(_) | Value::Array(_)) => Ok(()),
            _ => Err(MalformedMessages::Content),
        }
    })
}

/// Refuses a `/v1/messages` body without a `max_tokens` key, as the proxy does once its hooks
/// have run, since a hook may set one, as the rate limits hook does for a request it caps. A
/// `max_tokens` that some deployment the caller may use sets stands in for the caller's.
pub(crate) fn check_max_tokens(
    body: &Map<String, Value>,
    deployment_sets_max_tokens: impl FnOnce() -> bool,
) -> Result<(), Error> {
    match body.contains_key("max_tokens") || deployment_sets_max_tokens() {
        true => Ok(()),
        false => Err(Error::MissingParam {
            route: MESSAGES,
            param: "max_tokens",
        }),
    }
}

/// Whether `deployment` names a `max_tokens` of its own in `litellm_params`.
pub(crate) fn sets_max_tokens(deployment: &Deployment) -> bool {
    deployment
        .litellm_params
        .extra
        .get("max_tokens")
        .is_some_and(|value| !value.is_null())
}

/// Runs `check` on each message of a `messages` array, refusing anything else in its place.
fn each_message(
    messages: &Value,
    check: impl Fn(&Map<String, Value>) -> Result<(), fn(usize) -> MalformedMessages>,
) -> Result<(), Error> {
    let Value::Array(messages) = messages else {
        return Err(MalformedMessages::NotAnArray.into());
    };
    messages
        .iter()
        .enumerate()
        .try_for_each(|(index, message)| match message {
            Value::Object(message) => check(message).map_err(|fault| fault(index)),
            _ => Err(MalformedMessages::NotAnObject(index)),
        })
        .map_err(Error::from)
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
    use serde_json::json;
    use tower::ServiceExt;

    use super::*;

    fn body(value: Value) -> Map<String, Value> {
        let Value::Object(fields) = value else {
            unreachable!("test bodies are objects")
        };
        fields
    }

    fn missing(route: &'static str, param: &'static str) -> Option<String> {
        Some(Error::MissingParam { route, param }.to_string())
    }

    fn malformed(fault: MalformedMessages) -> Option<String> {
        Some(Error::from(fault).to_string())
    }

    #[rstest]
    #[case::text(json!([{"role": "user", "content": "hi"}]), None)]
    #[case::parts_and_tools(json!([{"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text", "text": "hi"}]}, {"role": "tool", "tool_call_id": "t", "content": "x"}]), None)]
    #[case::role_left_out(json!([{"content": "hi"}]), None)]
    #[case::null_role(json!([{"role": null, "content": "hi"}]), None)]
    #[case::absent(Value::Null, missing("/chat/completions", "messages"))]
    #[case::string(json!("invalid"), malformed(MalformedMessages::NotAnArray))]
    #[case::object(json!({"role": "user", "content": "hi"}), malformed(MalformedMessages::NotAnArray))]
    #[case::empty(json!([]), malformed(MalformedMessages::Empty))]
    #[case::string_element(json!([{"role": "user", "content": "hi"}, "hi"]), malformed(MalformedMessages::NotAnObject(1)))]
    #[case::array_element(json!([[]]), malformed(MalformedMessages::NotAnObject(0)))]
    #[case::numeric_role(json!([{"role": 1, "content": "hi"}]), malformed(MalformedMessages::Role(0)))]
    #[case::object_role(json!([{"role": "user", "content": "a"}, {"role": {}, "content": "b"}]), malformed(MalformedMessages::Role(1)))]
    fn chat_messages_must_be_a_list_of_messages(
        #[case] messages: Value,
        #[case] refusal: Option<String>,
    ) {
        let result = check_chat_messages(&body(json!({"model": "m", "messages": messages})));
        assert_eq!(result.err().map(|error| error.to_string()), refusal);
    }

    #[rstest]
    fn chat_messages_left_out_are_missing() {
        let error = check_chat_messages(&body(json!({"model": "m"}))).unwrap_err();
        assert_eq!(
            error.to_string(),
            "/chat/completions: Missing required parameter: 'messages'."
        );
    }

    #[rstest]
    #[case::text(json!({"messages": [{"role": "user", "content": "hi"}]}), None)]
    #[case::blocks(json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}), None)]
    #[case::empty_list_is_left_to_the_provider(json!({"messages": []}), None)]
    #[case::no_messages(json!({"max_tokens": 8}), missing("anthropic_messages", "messages"))]
    #[case::string(json!({"messages": "invalid"}), malformed(MalformedMessages::NotAnArray))]
    #[case::null(json!({"messages": null}), malformed(MalformedMessages::NotAnArray))]
    #[case::string_element(json!({"messages": ["hi"]}), malformed(MalformedMessages::NotAnObject(0)))]
    #[case::role_left_out(json!({"messages": [{"content": "hi"}]}), malformed(MalformedMessages::Role(0)))]
    #[case::numeric_role(json!({"messages": [{"role": 1, "content": "hi"}]}), malformed(MalformedMessages::Role(0)))]
    #[case::content_left_out(json!({"messages": [{"role": "user", "content": "a"}, {"role": "user"}]}), malformed(MalformedMessages::Content(1)))]
    #[case::null_content(json!({"messages": [{"role": "user", "content": null}]}), malformed(MalformedMessages::Content(0)))]
    fn anthropic_messages_need_a_role_and_content(
        #[case] request: Value,
        #[case] refusal: Option<String>,
    ) {
        let result = check_messages(&body(request));
        assert_eq!(result.err().map(|error| error.to_string()), refusal);
    }

    #[rstest]
    #[case::set(json!({"max_tokens": 8}), false, None)]
    #[case::null_is_present(json!({"max_tokens": null}), false, None)]
    #[case::deployment_sets_it(json!({}), true, None)]
    #[case::unset(json!({}), false, missing("anthropic_messages", "max_tokens"))]
    fn anthropic_messages_need_max_tokens_from_the_caller_or_a_deployment(
        #[case] request: Value,
        #[case] deployment_sets_max_tokens: bool,
        #[case] refusal: Option<String>,
    ) {
        let result = check_max_tokens(&body(request), || deployment_sets_max_tokens);
        assert_eq!(result.err().map(|error| error.to_string()), refusal);
    }

    #[rstest]
    #[case::set(json!({"model": "anthropic/x", "max_tokens": 8}), true)]
    #[case::null(json!({"model": "anthropic/x", "max_tokens": null}), false)]
    #[case::unset(json!({"model": "anthropic/x"}), false)]
    fn deployments_set_max_tokens_in_their_litellm_params(
        #[case] params: Value,
        #[case] sets: bool,
    ) {
        let deployment = Deployment::from_params(&serde_json::from_value(params).unwrap()).unwrap();
        assert_eq!(sets_max_tokens(&deployment), sets);
    }

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
