use std::time::Instant;

use axum::{
    body::{Body, Bytes},
    extract::Request,
    http::{Method, Uri},
    middleware::Next,
    response::Response,
};
use http_body_util::BodyExt;
use litellm_tracing::ByteChunk;

use super::call_id::CallId;

pub(super) async fn log_request(
    CallId(request_id): CallId,
    method: Method,
    uri: Uri,
    request: Request,
    next: Next,
) -> Response {
    let log_body_chunks = tracing::enabled!(tracing::Level::DEBUG);
    let started = Instant::now();
    let request = if log_body_chunks {
        request.map(|body| logged_body(body, request_id.clone(), "input"))
    } else {
        request
    };
    let response = next.run(request).await;
    tracing::info!(
        request_id = request_id.as_deref(),
        %method,
        path = %uri.path(),
        status = response.status().as_u16(),
        time_to_headers_ms = started.elapsed().as_secs_f64() * 1000.0,
        "response headers"
    );
    if log_body_chunks {
        response.map(|body| logged_body(body, request_id, "output"))
    } else {
        response
    }
}

fn logged_body(body: Body, request_id: Option<String>, direction: &'static str) -> Body {
    Body::new(body.map_frame(move |frame| {
        if let Some(data) = frame.data_ref() {
            log_chunk(request_id.as_deref(), direction, data);
        }
        frame
    }))
}

fn log_chunk(request_id: Option<&str>, direction: &str, data: &Bytes) {
    let chunk = ByteChunk::new(data);
    tracing::debug!(request_id, direction, encoding = chunk.encoding(), chunk = %chunk, "body chunk");
}

#[cfg(test)]
mod tests {
    use std::{convert::Infallible, sync::mpsc};

    use axum::{Router, body::to_bytes, http::StatusCode, routing::post};
    use futures_util::stream;
    use litellm_tracing::{Logger, Metadata, Record, Sink};
    use rstest::rstest;
    use serde_json::{Value, json};
    use tower::{ServiceBuilder, ServiceExt};

    use super::*;

    struct LogSink(mpsc::Sender<Value>);

    impl Sink for LogSink {
        fn enabled(&self, _: &Metadata<'_>) -> bool {
            true
        }

        fn emit(&self, record: &Record) {
            self.0
                .send(json!({"message": record.message, "fields": record.fields}))
                .unwrap();
        }
    }

    #[rstest]
    #[tokio::test]
    async fn logs_each_body_chunk_under_the_call_id_without_changing_streamed_bytes() {
        let app = Router::new()
            .route(
                "/stream",
                post(|_: Bytes| async {
                    (
                        StatusCode::OK,
                        Body::from_stream(stream::iter([
                            Ok::<_, Infallible>(Bytes::from_static(b"event: first\n\n")),
                            Ok(Bytes::from_static(b"event: second\n\n")),
                        ])),
                    )
                }),
            )
            .layer(
                ServiceBuilder::new()
                    .layer(super::super::call_id::layer())
                    .layer(axum::middleware::from_fn(log_request)),
            );
        let request_chunks = [
            Ok::<_, Infallible>(Bytes::from_static(b"hello")),
            Ok(Bytes::from_static(b" world")),
        ];
        let request = Request::post("/stream")
            .header("x-litellm-call-id", "call-1")
            .body(Body::from_stream(stream::iter(request_chunks)))
            .unwrap();
        let (sender, receiver) = mpsc::channel();
        let logger = Logger::new(LogSink(sender));

        let output = logger
            .instrument(async {
                let response = app.oneshot(request).await.unwrap();
                to_bytes(response.into_body(), 1024).await.unwrap()
            })
            .await;

        assert_eq!(output, "event: first\n\nevent: second\n\n");
        let records: Vec<Value> = receiver.try_iter().collect();
        let chunks: Vec<&Value> = records
            .iter()
            .map(|record| &record["fields"]["chunk"])
            .collect();
        assert_eq!(
            chunks,
            [
                &json!("hello"),
                &json!(" world"),
                &Value::Null,
                &json!("event: first\n\n"),
                &json!("event: second\n\n"),
            ]
        );
        assert_eq!(records[2]["fields"]["status"], 200);
        let request_ids: Vec<&Value> = records
            .iter()
            .map(|record| &record["fields"]["request_id"])
            .collect();
        assert_eq!(request_ids, [&json!("call-1"); 5]);
    }
}
