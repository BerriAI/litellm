use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicU64, Ordering},
};

use axum::{
    BoxError, Extension, Json,
    body::{Body, Bytes},
    extract::{Request, State},
    http::{HeaderValue, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use futures_util::{StreamExt, stream};
use http_body_util::{BodyExt, LengthLimitError, Limited};
use litellm_config::Environment;
use litellm_license::Entitlements;
use serde_json::{Value, json};
use tower::{ServiceExt, util::MapRequest};
use tower_http::decompression::{DecompressionBody, RequestDecompression};

use super::body_limit::{Overflow, SizeLimits, bytes, content_length, too_large};

const SIGNATURE_LENGTH: usize = 4;

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    max_decompressed_request_body_size_mb: f64,
    max_request_body_decompression_ratio: u64,
    max_decompressed_request_bodies_in_flight_mb: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            max_decompressed_request_body_size_mb: 50.0,
            max_request_body_decompression_ratio: 200,
            max_decompressed_request_bodies_in_flight_mb: 200.0,
        }
    }
}

impl Settings {
    pub fn from_environment(environment: &Environment) -> Self {
        let defaults = Self::default();
        Self {
            max_decompressed_request_body_size_mb: environment
                .max_decompressed_request_body_size_mb
                .unwrap_or(defaults.max_decompressed_request_body_size_mb),
            max_request_body_decompression_ratio: environment
                .max_request_body_decompression_ratio
                .unwrap_or(defaults.max_request_body_decompression_ratio),
            max_decompressed_request_bodies_in_flight_mb: environment
                .max_decompressed_request_bodies_in_flight_mb
                .unwrap_or(defaults.max_decompressed_request_bodies_in_flight_mb),
        }
    }
}

#[derive(Clone, Copy, strum::IntoStaticStr)]
enum Coding {
    #[strum(serialize = "gzip")]
    Gzip,
    #[strum(serialize = "zstd")]
    Zstd,
}

impl Coding {
    /// `CODING_SIGNATURES`, where zstd also starts with a skippable frame.
    fn starts(self, body: &[u8]) -> bool {
        match self {
            Self::Gzip => body.starts_with(&[0x1f, 0x8b]),
            Self::Zstd => {
                body.starts_with(&[0x28, 0xb5, 0x2f, 0xfd])
                    || matches!(body, [first, 0x2a, 0x4d, 0x18, ..] if first & 0xf0 == 0x50)
            }
        }
    }
}

struct Peeked {
    signature: Vec<u8>,
    failed: bool,
    body: Body,
}

/// `_receive_prefix`: reads until the body shows whether it starts like its coding, and hands it
/// back whole.
async fn peeked(body: Body) -> Peeked {
    let mut rest = body.into_data_stream();
    let mut head = Vec::new();
    let mut seen = 0;
    let mut failed = false;
    while seen < SIGNATURE_LENGTH && !failed {
        let Some(chunk) = rest.next().await else {
            break;
        };
        failed = chunk.is_err();
        seen += chunk.as_ref().map_or(0, Bytes::len);
        head.push(chunk);
    }
    let signature = head
        .iter()
        .flatten()
        .flatten()
        .copied()
        .take(SIGNATURE_LENGTH)
        .collect();
    Peeked {
        signature,
        failed,
        body: Body::from_stream(stream::iter(head).chain(rest)),
    }
}

/// `_DecodedBodyBudget`. Python reserves the size a body could decode to before decoding it, then
/// holds its real size until the response is sent. A stream never allocates the first, so a body
/// holds what it has decoded so far.
struct Budget {
    max_bytes: u64,
    held: AtomicU64,
}

impl Budget {
    /// A body that is alone in flight always fits.
    fn hold(&self, already: u64, more: u64) -> bool {
        self.held
            .try_update(Ordering::AcqRel, Ordering::Acquire, |held| {
                (held == already || held + more <= self.max_bytes).then_some(held + more)
            })
            .is_ok()
    }
}

/// `_coding_for`: exactly one `Content-Encoding` naming a coding Python recognizes.
fn coding(request: &Request) -> Option<Coding> {
    let mut encodings = request.headers().get_all(header::CONTENT_ENCODING).iter();
    let (Some(encoding), None) = (encodings.next(), encodings.next()) else {
        return None;
    };
    match encoding.to_str().ok()?.trim().to_ascii_lowercase().as_str() {
        "gzip" | "x-gzip" => Some(Coding::Gzip),
        "zstd" => Some(Coding::Zstd),
        _ => None,
    }
}

#[derive(Debug)]
enum Rejection {
    TooLarge,
    DecodedTooLarge,
    OverRatio,
    Undecodable(String),
    Busy,
}

#[derive(Debug, thiserror::Error)]
#[error("request body was rejected while decoding")]
struct Rejected;

struct Decoding {
    coding: Coding,
    max: f64,
    ratio: u64,
    compressed: AtomicU64,
    decoded: AtomicU64,
    budget: Arc<Budget>,
    held: AtomicU64,
    rejection: OnceLock<Rejection>,
}

impl Drop for Decoding {
    fn drop(&mut self) {
        self.budget
            .held
            .fetch_sub(*self.held.get_mut(), Ordering::AcqRel);
    }
}

/// A stream that stops inside its coding: `not decompressor.eof` in Python.
fn ends_early(error: &axum::Error) -> bool {
    std::iter::successors(Some(error as &(dyn std::error::Error + 'static)), |error| {
        error.source()
    })
    .filter_map(|error| error.downcast_ref::<std::io::Error>())
    .any(|error| error.kind() == std::io::ErrorKind::UnexpectedEof)
}

impl Decoding {
    fn reject(&self, rejection: Rejection) -> Rejected {
        let _ = self.rejection.set(rejection);
        Rejected
    }

    fn compressed_body(self: Arc<Self>, body: Body) -> Body {
        let max_bytes = usize::try_from(bytes(self.max)).unwrap_or(usize::MAX);
        let counter = self.clone();
        Body::new(
            Limited::new(body, max_bytes)
                .map_frame(move |frame| {
                    if let Some(data) = frame.data_ref() {
                        counter
                            .compressed
                            .fetch_add(data.len() as u64, Ordering::AcqRel);
                    }
                    frame
                })
                .map_err(move |error| -> BoxError {
                    if error.is::<LengthLimitError>() {
                        return self.reject(Rejection::TooLarge).into();
                    }
                    error
                }),
        )
    }

    fn decoded_body(self: Arc<Self>, body: DecompressionBody<Body>) -> Body {
        Body::from_stream(
            Body::new(body)
                .into_data_stream()
                .map(move |chunk| self.check(chunk)),
        )
    }

    /// `_decode`'s checks, applied as the decoded stream grows.
    fn check(&self, chunk: Result<Bytes, axum::Error>) -> Result<Bytes, Rejected> {
        let chunk = chunk.map_err(|error| {
            let coding = <&str>::from(self.coding);
            self.reject(Rejection::Undecodable(if ends_early(&error) {
                format!("Request body ends before the end of its {coding} stream")
            } else {
                format!("Request body is not valid {coding} data: {error}")
            }))
        })?;
        let decoded =
            self.decoded.fetch_add(chunk.len() as u64, Ordering::AcqRel) + chunk.len() as u64;
        if decoded > bytes(self.max) {
            return Err(self.reject(Rejection::DecodedTooLarge));
        }
        if decoded
            > self
                .compressed
                .load(Ordering::Acquire)
                .saturating_mul(self.ratio)
        {
            return Err(self.reject(Rejection::OverRatio));
        }
        let held = self.held.load(Ordering::Acquire);
        if !self.budget.hold(held, chunk.len() as u64) {
            return Err(self.reject(Rejection::Busy));
        }
        self.held.fetch_add(chunk.len() as u64, Ordering::AcqRel);
        Ok(chunk)
    }

    fn rejection(&self) -> Option<Response> {
        let invalid = |status, message: String| {
            proxy_exception(status, &message, "invalid_request_error").into_response()
        };
        self.rejection.get().map(|rejection| match rejection {
            Rejection::TooLarge => too_large(self.max),
            Rejection::DecodedTooLarge => invalid(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("Request body is larger than {} MB when decoded", self.max),
            ),
            Rejection::OverRatio => invalid(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!(
                    "Request body expands more than {} times when decoded",
                    self.ratio
                ),
            ),
            Rejection::Undecodable(message) => invalid(StatusCode::BAD_REQUEST, message.clone()),
            Rejection::Busy => (
                [(header::RETRY_AFTER, "1")],
                proxy_exception(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "The proxy is decoding too many compressed request bodies right now. \
                     Retry later, or send this body uncompressed",
                    "overloaded_error",
                ),
            )
                .into_response(),
        })
    }
}

/// `ProxyException.to_dict()`, whose `code` is the status as text.
fn proxy_exception(
    status: StatusCode,
    message: &str,
    error_type: &str,
) -> (StatusCode, Json<Value>) {
    let body = json!({"error": {
        "message": message,
        "type": error_type,
        "param": null,
        "code": status.as_u16().to_string(),
    }});
    (status, Json(body))
}

pub(super) struct Decompression {
    settings: Settings,
    limits: Arc<SizeLimits>,
    budget: Arc<Budget>,
}

impl Decompression {
    pub(super) fn new(settings: Settings, limits: Arc<SizeLimits>) -> Self {
        Self {
            settings,
            limits,
            budget: Arc::new(Budget {
                max_bytes: bytes(settings.max_decompressed_request_bodies_in_flight_mb),
                held: AtomicU64::new(0),
            }),
        }
    }
}

/// `RequestBodyDecompressionMiddleware` over `tower_http`'s streaming gzip and zstd decoders. The
/// caps are checked as the decoded stream grows, and a body that breaks one before the response
/// starts is answered with the rejection. A body that does not start like its coding is sent on as
/// it came, without the header that named the coding.
///
/// `tower_http` configures its decoders itself, which leaves three differences from Python. Its
/// gzip decoder ignores bytes after the stream and its zstd decoder reads frame after frame, where
/// Python refuses any data after the first. Python refuses a zstd frame whose window is over 8 MB
/// (RFC 9659), where zstd's own 128 MB default applies here. Python reads the size a body
/// declares to cap its decoding early, where the decoders here only check it at the end. Python's
/// `RAW_BODY_ROUTE_PREFIXES` are not ported because those routes have no native route and are
/// always served by Python.
pub(super) async fn decompress(
    State(decompression): State<Arc<Decompression>>,
    Extension(entitlements): Extension<Arc<Entitlements>>,
    overflow: Option<Extension<Overflow>>,
    request: Request,
    next: Next,
) -> Response {
    let Some(coding) = coding(&request) else {
        return next.run(request).await;
    };
    let cap = decompression.settings.max_decompressed_request_body_size_mb;
    let max = decompression
        .limits
        .megabytes(&entitlements)
        .map_or(cap, |limit| cap.min(limit));
    let too_long = content_length(&request).is_some_and(|length| length > bytes(max));
    let (mut parts, body) = request.into_parts();
    let Peeked {
        signature,
        failed,
        body,
    } = peeked(body).await;
    let decodes = coding.starts(&signature);
    if (decodes || failed)
        && let Some(Extension(overflow)) = &overflow
    {
        overflow.read_by_middleware();
    }
    if !decodes {
        parts.headers.remove(header::CONTENT_ENCODING);
        return next.run(Request::from_parts(parts, body)).await;
    }
    if too_long {
        return too_large(max);
    }
    parts.headers.insert(
        header::CONTENT_ENCODING,
        HeaderValue::from_static(coding.into()),
    );
    let decoding = Arc::new(Decoding {
        coding,
        max,
        ratio: decompression.settings.max_request_body_decompression_ratio,
        compressed: AtomicU64::new(0),
        decoded: AtomicU64::new(0),
        budget: decompression.budget.clone(),
        held: AtomicU64::new(0),
        rejection: OnceLock::new(),
    });
    let request = Request::from_parts(parts, decoding.clone().compressed_body(body));
    let decoded = decoding.clone();
    let Ok(response) = RequestDecompression::new(MapRequest::new(
        next,
        move |request: axum::http::Request<DecompressionBody<Body>>| {
            request.map(|body| decoded.clone().decoded_body(body))
        },
    ))
    .oneshot(request)
    .await;
    decoding.rejection().unwrap_or_else(|| {
        response.map(|body| {
            Body::new(body.map_frame(move |frame| {
                let _held = &decoding;
                frame
            }))
        })
    })
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use axum::{Router, http::HeaderMap, middleware::from_fn_with_state, routing::post};
    use flate2::{Compression, write::GzEncoder};
    use rstest::rstest;

    use super::*;
    use crate::middleware::body_limit;

    fn premium() -> Extension<Arc<Entitlements>> {
        Extension(Arc::new(Entitlements::ServerVerified))
    }

    fn gzip(data: &[u8]) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    fn app(settings: Settings, configured_mb: Option<f64>) -> Router {
        let limits = body_limit::SizeLimits::new(&body_limit::Settings {
            max_request_size_mb: configured_mb,
        });
        let echo = |headers: HeaderMap, body: Bytes| async move {
            json!({
                "content_encoding": headers.get(header::CONTENT_ENCODING).map(|value| value.to_str().unwrap()),
                "body": String::from_utf8_lossy(&body),
            })
            .to_string()
        };
        Router::new()
            .route("/v1/messages", post(echo))
            .layer(from_fn_with_state(
                Arc::new(Decompression::new(settings, Arc::new(limits))),
                decompress,
            ))
            .layer(premium())
    }

    async fn send(app: Router, path: &str, encodings: &[&str], body: Vec<u8>) -> Response {
        let request = encodings
            .iter()
            .fold(Request::post(path), |request, encoding| {
                request.header(header::CONTENT_ENCODING, *encoding)
            })
            .header(header::CONTENT_LENGTH, body.len())
            .body(Body::from(body))
            .unwrap();
        app.oneshot(request).await.unwrap()
    }

    async fn json_body(response: Response) -> Value {
        serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), 1 << 20)
                .await
                .unwrap(),
        )
        .unwrap()
    }

    fn zstd(data: &[u8]) -> Vec<u8> {
        zstd::encode_all(data, 0).unwrap()
    }

    #[rstest]
    #[case::gzip("gzip", gzip)]
    #[case::gzip_alias("x-gzip", gzip)]
    #[case::gzip_spelled_loosely(" GZip ", gzip)]
    #[case::zstd("zstd", zstd)]
    #[case::zstd_spelled_loosely(" ZSTD ", zstd)]
    #[tokio::test]
    async fn an_encoded_body_reaches_the_route_decoded(
        #[case] encoding: &str,
        #[case] encode: fn(&[u8]) -> Vec<u8>,
    ) {
        let response = send(
            app(Settings::default(), None),
            "/v1/messages",
            &[encoding],
            encode(br#"{"model":"m"}"#),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            json_body(response).await,
            json!({"content_encoding": null, "body": r#"{"model":"m"}"#})
        );
    }

    #[rstest]
    #[case::two_encodings(&["gzip", "gzip"], "gzip")]
    #[case::unknown_coding(&["br"], "br")]
    #[tokio::test]
    async fn other_bodies_pass_through_untouched(
        #[case] encodings: &[&str],
        #[case] first_encoding: &str,
    ) {
        let response = send(
            app(Settings::default(), None),
            "/v1/messages",
            encodings,
            b"raw".to_vec(),
        )
        .await;
        assert_eq!(
            json_body(response).await,
            json!({"content_encoding": first_encoding, "body": "raw"})
        );
    }

    #[rstest]
    #[case::gzip("gzip", b"raw")]
    #[case::zstd("zstd", b"raw")]
    #[case::shorter_than_a_signature("gzip", b"\x1f")]
    #[case::empty("gzip", b"")]
    #[tokio::test]
    async fn a_body_that_does_not_start_like_its_coding_is_sent_on_as_it_came(
        #[case] encoding: &str,
        #[case] body: &[u8],
    ) {
        let response = send(
            app(Settings::default(), None),
            "/v1/messages",
            &[encoding],
            body.to_vec(),
        )
        .await;
        assert_eq!(
            json_body(response).await,
            json!({"content_encoding": null, "body": String::from_utf8_lossy(body)})
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_signature_split_across_chunks_is_still_decoded() {
        let chunks = gzip(br#"{"model":"m"}"#)
            .into_iter()
            .map(|byte| Ok::<_, std::convert::Infallible>(Bytes::from(vec![byte])));
        let request = Request::post("/v1/messages")
            .header(header::CONTENT_ENCODING, "gzip")
            .body(Body::from_stream(stream::iter(chunks)))
            .unwrap();
        let response = app(Settings::default(), None)
            .oneshot(request)
            .await
            .unwrap();
        assert_eq!(
            json_body(response).await,
            json!({"content_encoding": null, "body": r#"{"model":"m"}"#})
        );
    }

    #[rstest]
    #[case::gzip("gzip", gzip)]
    #[case::zstd("zstd", zstd)]
    #[tokio::test]
    async fn a_body_that_stops_inside_its_coding_is_a_bad_request(
        #[case] encoding: &str,
        #[case] encode: fn(&[u8]) -> Vec<u8>,
    ) {
        let whole = encode(&incompressible(4096));
        let response = send(
            app(Settings::default(), None),
            "/v1/messages",
            &[encoding],
            whole[..whole.len() / 2].to_vec(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            json_body(response).await["error"]["message"],
            format!("Request body ends before the end of its {encoding} stream")
        );
    }

    fn in_flight(megabytes: f64) -> Settings {
        Settings {
            max_request_body_decompression_ratio: 100_000,
            max_decompressed_request_bodies_in_flight_mb: megabytes,
            ..Settings::default()
        }
    }

    #[rstest]
    #[tokio::test]
    async fn decoded_bodies_in_flight_share_a_budget_until_their_responses_are_done() {
        let app = app(in_flight(1.0), None);
        let body = || gzip(&vec![b'a'; 600 << 10]);
        let unfinished = send(app.clone(), "/v1/messages", &["gzip"], body()).await;
        assert_eq!(unfinished.status(), StatusCode::OK);

        let shed = send(app.clone(), "/v1/messages", &["gzip"], body()).await;
        assert_eq!(shed.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(shed.headers()[header::RETRY_AFTER], "1");
        assert_eq!(
            json_body(shed).await,
            json!({"error": {
                "message": "The proxy is decoding too many compressed request bodies right now. \
                            Retry later, or send this body uncompressed",
                "type": "overloaded_error",
                "param": null,
                "code": "503",
            }})
        );

        drop(unfinished);
        let admitted = send(app, "/v1/messages", &["gzip"], body()).await;
        assert_eq!(admitted.status(), StatusCode::OK);
    }

    #[rstest]
    #[tokio::test]
    async fn a_body_alone_in_flight_fits_whatever_the_budget() {
        let response = send(
            app(in_flight(1.0), None),
            "/v1/messages",
            &["gzip"],
            gzip(&vec![b'a'; 2 << 20]),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    fn small(cap_mb: f64, ratio: u64) -> Settings {
        Settings {
            max_decompressed_request_body_size_mb: cap_mb,
            max_request_body_decompression_ratio: ratio,
            ..Settings::default()
        }
    }

    fn incompressible(bytes: usize) -> Vec<u8> {
        std::iter::successors(Some(1_u32), |state| {
            Some(state.wrapping_mul(1_103_515_245).wrapping_add(12_345))
        })
        .map(|state| (state >> 16) as u8)
        .take(bytes)
        .collect()
    }

    #[rstest]
    #[case::compressed_over_the_cap(
        small(1.0, 200), None, "gzip", gzip(&incompressible(2 << 20)),
        StatusCode::PAYLOAD_TOO_LARGE, "/error",
        "Request size is too large. Max size is 1 MB"
    )]
    #[case::compressed_over_the_operator_limit(
        small(50.0, 200), Some(1.0), "gzip", gzip(&incompressible(2 << 20)),
        StatusCode::PAYLOAD_TOO_LARGE, "/error",
        "Request size is too large. Max size is 1 MB"
    )]
    #[case::decoded_over_the_cap(
        small(1.0, 100_000), None, "gzip", gzip(&vec![b'a'; 2 << 20]),
        StatusCode::PAYLOAD_TOO_LARGE, "/error/message",
        "Request body is larger than 1 MB when decoded"
    )]
    #[case::over_the_ratio(
        small(50.0, 10), None, "gzip", gzip(&vec![b'a'; 1 << 20]),
        StatusCode::PAYLOAD_TOO_LARGE, "/error/message",
        "Request body expands more than 10 times when decoded"
    )]
    #[case::zstd_decoded_over_the_cap(
        small(1.0, 100_000), None, "zstd", zstd(&vec![b'a'; 2 << 20]),
        StatusCode::PAYLOAD_TOO_LARGE, "/error/message",
        "Request body is larger than 1 MB when decoded"
    )]
    #[case::zstd_over_the_ratio(
        small(50.0, 10), None, "zstd", zstd(&vec![b'a'; 1 << 20]),
        StatusCode::PAYLOAD_TOO_LARGE, "/error/message",
        "Request body expands more than 10 times when decoded"
    )]
    #[tokio::test]
    async fn a_body_over_a_cap_is_rejected(
        #[case] settings: Settings,
        #[case] configured_mb: Option<f64>,
        #[case] encoding: &str,
        #[case] body: Vec<u8>,
        #[case] status: StatusCode,
        #[case] message_at: &str,
        #[case] message: &str,
    ) {
        let response = send(
            app(settings, configured_mb),
            "/v1/messages",
            &[encoding],
            body,
        )
        .await;
        assert_eq!(response.status(), status);
        assert_eq!(
            json_body(response).await.pointer(message_at),
            Some(&Value::from(message))
        );
    }

    #[rstest]
    #[case::gzip("gzip", b"\x1f\x8bnot gzip")]
    #[case::zstd("zstd", b"\x28\xb5\x2f\xfdnot zstd")]
    #[tokio::test]
    async fn an_undecodable_body_is_a_bad_request(#[case] encoding: &str, #[case] body: &[u8]) {
        let response = send(
            app(Settings::default(), None),
            "/v1/messages",
            &[encoding],
            body.to_vec(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "400");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .starts_with(&format!("Request body is not valid {encoding} data: ")),
            "{body}"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_compressed_body_past_the_request_limit_gets_the_middleware_rejection() {
        let limits = Arc::new(body_limit::SizeLimits::new(&body_limit::Settings {
            max_request_size_mb: Some(1.0),
        }));
        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(|body: Bytes| async move { body.len().to_string() }),
            )
            .layer(from_fn_with_state(
                Arc::new(Decompression::new(Settings::default(), limits.clone())),
                decompress,
            ))
            .layer(from_fn_with_state(limits, body_limit::limit_body))
            .layer(premium());
        let request = Request::post("/v1/chat/completions")
            .header(header::CONTENT_ENCODING, "gzip")
            .body(Body::from(gzip(&incompressible(2 << 20))))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            json_body(response).await,
            json!({"error": "Request size is too large. Max size is 1 MB"})
        );
    }
}
