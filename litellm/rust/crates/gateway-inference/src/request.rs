use axum::{
    body::Bytes,
    extract::{FromRequest, FromRequestParts, Multipart, Request},
    http::request::Parts,
};
use litellm_llms_types::as_sent::AsSent;
use serde_json::{Map, Value};

use crate::{BodyError, Error};

const MODEL: &str = "model";
const STREAM: &str = "stream";

pub(crate) const MAX_FILE_BYTES: usize = 50 * 1024 * 1024;
/// The largest request body an inference route reads.
pub const MAX_BODY_BYTES: usize = MAX_FILE_BYTES + 1024 * 1024;

pub(crate) struct Upload {
    pub bytes: Bytes,
    pub file_name: Option<String>,
    pub mime_type: Option<String>,
}

/// A request's JSON object body. One already in the request's extensions is taken as parsed, so
/// whoever read the body before the route does not make the route parse it again.
#[derive(Clone)]
pub struct JsonObject(pub RequestBody);

/// A request body as the client sent it, with the fields every route reads typed. It turns back
/// into what was sent, explicit nulls and values of the wrong type included, with `model` and
/// `stream` first and every other field in the order sent.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RequestBody {
    model: Option<AsSent<String>>,
    stream: Option<AsSent<bool>>,
    others: Map<String, Value>,
}

impl RequestBody {
    /// The public model name the request asks for.
    pub fn model(&self) -> Option<&str> {
        self.model.as_ref()?.read().map(String::as_str)
    }

    /// Whether the client asked for a stream.
    pub fn streamed(&self) -> bool {
        self.stream.as_ref().and_then(AsSent::read) == Some(&true)
    }

    /// The fields no route reads typed yet, in the order sent.
    pub fn others(&self) -> &Map<String, Value> {
        &self.others
    }

    pub fn set_model(&mut self, model: &str) {
        self.model = Some(AsSent::new(Value::from(model)));
    }

    /// The body with `fields` written over it, as a dict's `update` writes them.
    #[must_use]
    pub fn merged(self, fields: impl IntoIterator<Item = (String, Value)>) -> Self {
        Map::from(self)
            .into_iter()
            .chain(fields)
            .collect::<Map<_, _>>()
            .into()
    }

    /// The body without the fields `names` names, which the route consumed.
    #[must_use]
    pub(crate) fn without(self, names: &[&str]) -> Self {
        Self {
            others: self
                .others
                .into_iter()
                .filter(|(name, _)| !names.contains(&name.as_str()))
                .collect(),
            ..self
        }
    }
}

impl From<Map<String, Value>> for RequestBody {
    fn from(mut sent: Map<String, Value>) -> Self {
        Self {
            model: sent.shift_remove(MODEL).map(AsSent::new),
            stream: sent.shift_remove(STREAM).map(AsSent::new),
            others: sent,
        }
    }
}

impl From<RequestBody> for Map<String, Value> {
    fn from(body: RequestBody) -> Self {
        let model = body
            .model
            .map(|model| (MODEL.to_owned(), model.into_sent()));
        let stream = body
            .stream
            .map(|stream| (STREAM.to_owned(), stream.into_sent()));
        model.into_iter().chain(stream).chain(body.others).collect()
    }
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

    async fn from_request(mut request: Request, state: &S) -> Result<Self, Self::Rejection> {
        let parsed = request.extensions_mut().remove::<Self>();
        let body = Bytes::from_request(request, state).await.map_err(|error| {
            if error.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
                Error::BodyTooLarge
            } else {
                BodyError::Read(error).into()
            }
        })?;
        match parsed {
            Some(parsed) => Ok(parsed),
            None => Ok(Self(object(&body)?.into())),
        }
    }
}

pub(crate) struct InferenceBody {
    pub fields: RequestBody,
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
    Ok(InferenceBody {
        fields: fields.into(),
        upload,
    })
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
    use serde_json::json;
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

    fn sent(body: &str) -> RequestBody {
        serde_json::from_str::<Map<String, Value>>(body)
            .unwrap()
            .into()
    }

    fn written(body: RequestBody) -> String {
        Value::Object(body.into()).to_string()
    }

    #[rstest]
    #[case::typed_fields_first(
        r#"{"max_tokens":5,"stream":true,"model":"claude","metadata":null}"#,
        r#"{"model":"claude","stream":true,"max_tokens":5,"metadata":null}"#
    )]
    #[case::values_of_the_wrong_type(
        r#"{"model":7,"stream":"yes","tools":[]}"#,
        r#"{"model":7,"stream":"yes","tools":[]}"#
    )]
    #[case::explicit_nulls(r#"{"model":null,"stream":null}"#, r#"{"model":null,"stream":null}"#)]
    #[case::neither_typed_field(r#"{"messages":[]}"#, r#"{"messages":[]}"#)]
    fn a_body_writes_back_what_was_sent(#[case] body: &str, #[case] expected: &str) {
        assert_eq!(written(sent(body)), expected);
    }

    #[rstest]
    #[case::a_name(r#"{"model":"claude"}"#, Some("claude"), false)]
    #[case::a_stream(r#"{"model":"claude","stream":true}"#, Some("claude"), true)]
    #[case::no_stream(r#"{"stream":false}"#, None, false)]
    #[case::not_a_name_or_a_flag(r#"{"model":7,"stream":"true"}"#, None, false)]
    #[case::nulls(r#"{"model":null,"stream":null}"#, None, false)]
    #[case::neither(r#"{}"#, None, false)]
    fn reads_the_model_and_the_stream_flag_only_when_typed(
        #[case] body: &str,
        #[case] model: Option<&str>,
        #[case] streamed: bool,
    ) {
        let body = sent(body);

        assert_eq!((body.model(), body.streamed()), (model, streamed));
    }

    #[rstest]
    fn merged_fields_write_over_the_body_and_are_read_typed() {
        let fields = [
            ("model".to_owned(), json!("fallback")),
            ("stream".to_owned(), json!(true)),
            ("max_tokens".to_owned(), json!(9)),
            ("top_k".to_owned(), json!(3)),
        ];

        let merged = sent(r#"{"model":"claude","max_tokens":5,"system":"terse"}"#).merged(fields);

        assert_eq!(
            (merged.model(), merged.streamed(), written(merged.clone())),
            (
                Some("fallback"),
                true,
                r#"{"model":"fallback","stream":true,"max_tokens":9,"system":"terse","top_k":3}"#
                    .to_owned()
            )
        );
    }

    fn already_parsed(sent: &'static str) -> Request {
        let parsed = serde_json::from_str::<Map<String, Value>>(r#"{"model":"parsed"}"#).unwrap();
        let mut request = Request::post("/").body(Body::from(sent)).unwrap();
        request.extensions_mut().insert(JsonObject(parsed.into()));
        request
    }

    #[rstest]
    #[tokio::test]
    async fn a_body_already_parsed_is_not_parsed_again() {
        let JsonObject(body) = JsonObject::from_request(already_parsed(r#"{"model":"sent"}"#), &())
            .await
            .ok()
            .unwrap();

        assert_eq!(body.model(), Some("parsed"));
    }

    #[rstest]
    #[tokio::test]
    async fn a_body_already_parsed_is_still_held_to_the_body_limit() {
        let limited = Router::new()
            .route("/", post(|_: JsonObject| async {}))
            .layer(DefaultBodyLimit::max(8));

        let response = limited
            .oneshot(already_parsed(r#"{"model":"sent"}"#))
            .await
            .unwrap();

        assert_eq!(response.status(), 413);
    }
}
