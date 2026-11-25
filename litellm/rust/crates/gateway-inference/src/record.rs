//! The record of one inference call: what the route, the callbacks and every later reader know of
//! it so far. [`scoped`] makes one for each request on the inference routes and runs the handler,
//! every layer inside it and the response body with it in a task-local, so code reaches the call's
//! record through [`read`] and [`write`] and never takes it as a parameter. The record lives in
//! `litellm_call_scope`'s task-local, so a task any crate spawns for the call through
//! `litellm_call_scope::spawn_in_call` keeps it.

mod entries;
mod trace;

use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use axum::{
    body::{Body, Bytes, HttpBody},
    extract::Request,
    middleware::Next,
    response::Response,
};
use http_body::{Frame, SizeHint};
use litellm_call_scope::Call;
use parking_lot::Mutex;
use serde_json::Value;
use tokio::task::futures::TaskLocalFuture;

pub use self::entries::{CallState, Charged, Ended, unix};

/// One call as it stands. Every access is one short lock, never held across an await.
pub struct CallRecord(Mutex<CallState>);

impl CallRecord {
    pub fn new() -> Arc<Self> {
        Self::traced(uuid::Uuid::new_v4().to_string())
    }

    fn traced(trace_id: String) -> Arc<Self> {
        Arc::new(Self(Mutex::new(CallState::new(trace_id))))
    }
}

/// Runs `future` as part of the call `record` keeps.
pub fn scope<F: Future>(record: Arc<CallRecord>, future: F) -> TaskLocalFuture<Call, F> {
    litellm_call_scope::scope(record, future)
}

/// Changes the record of the call this code runs in. `None` outside a call, which is a bug.
pub fn write<R>(change: impl FnOnce(&mut CallState) -> R) -> Option<R> {
    litellm_call_scope::with(|record: &CallRecord| change(&mut record.0.lock()))
        .map_err(missed)
        .ok()
}

/// Reads the record of the call this code runs in. `None` outside a call, which is a bug.
pub fn read<R>(look: impl FnOnce(&CallState) -> R) -> Option<R> {
    litellm_call_scope::with(|record: &CallRecord| look(&record.0.lock()))
        .map_err(missed)
        .ok()
}

/// Adds a header to the answer of the call this code runs in.
pub fn add_header(name: impl Into<String>, value: impl Into<Value>) {
    write(|entries| entries.add_header(name.into(), value.into()));
}

/// A read or write with no record in scope: a spawn that skipped `litellm_call_scope::spawn_in_call`,
/// or code run outside a call.
fn missed(error: litellm_call_scope::Error) {
    if cfg!(debug_assertions) {
        panic!("no call record in scope: {error}");
    }
    tracing::error!(%error, "no call record in scope");
}

/// The layer outermost on the inference routes. It makes the request's record, under the trace
/// its headers name, and runs the rest of the request with it, the response body included, since
/// hyper polls the body after the handler returned.
pub(crate) async fn scoped(request: Request, next: Next) -> Response {
    let record = CallRecord::traced(trace::of(request.headers()));
    let response = scope(record.clone(), next.run(request)).await;
    response.map(|body| {
        Body::new(Scoped {
            record,
            body: Some(body),
        })
    })
}

/// A response body polled, and dropped, as part of its call, so a stream's end and a stream the
/// client dropped still reach the record.
struct Scoped {
    record: Arc<CallRecord>,
    body: Option<Body>,
}

impl HttpBody for Scoped {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        let this = &mut *self;
        let Some(body) = this.body.as_mut() else {
            return Poll::Ready(None);
        };
        litellm_call_scope::sync_scope(this.record.clone(), || Pin::new(body).poll_frame(context))
    }

    fn is_end_stream(&self) -> bool {
        self.body.as_ref().is_none_or(HttpBody::is_end_stream)
    }

    fn size_hint(&self) -> SizeHint {
        self.body
            .as_ref()
            .map_or_else(|| SizeHint::with_exact(0), HttpBody::size_hint)
    }
}

impl Drop for Scoped {
    fn drop(&mut self) {
        let body = self.body.take();
        litellm_call_scope::sync_scope(self.record.clone(), || drop(body));
    }
}

#[cfg(test)]
mod tests {
    use axum::{Router, routing::get};
    use futures_util::StreamExt;
    use rstest::rstest;
    use serde_json::json;
    use tower::ServiceExt;

    use super::*;

    fn added() -> Option<Value> {
        read(|entries| json!(entries.response_headers()))
    }

    fn router() -> Router {
        Router::new()
            .route(
                "/",
                get(|| async {
                    add_header("x-handler", "yes");
                    let body = futures_util::stream::once(async {
                        add_header("x-body", "yes");
                        Ok::<_, std::convert::Infallible>(Bytes::from(added().unwrap().to_string()))
                    });
                    Body::from_stream(body)
                }),
            )
            .layer(axum::middleware::from_fn(scoped))
    }

    #[rstest]
    #[tokio::test]
    async fn the_handler_and_the_body_write_the_same_record() {
        let response = router()
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap(),
            json!({"x-handler": "yes", "x-body": "yes"})
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_call_is_traced_under_the_session_its_caller_names() {
        let session = "e96634a3-fa28-4083-b354-55542e2dca01";
        let router = Router::new()
            .route(
                "/",
                get(|| async { read(|entries| entries.trace_id().to_owned()).unwrap() }),
            )
            .layer(axum::middleware::from_fn(scoped));
        let request = Request::get("/")
            .header("x-claude-code-session-id", session)
            .body(Body::empty())
            .unwrap();
        let body = axum::body::to_bytes(
            router.oneshot(request).await.unwrap().into_body(),
            usize::MAX,
        )
        .await
        .unwrap();
        assert_eq!(body, session);
    }

    #[rstest]
    #[tokio::test]
    async fn a_task_spawned_in_a_call_writes_its_record() {
        let record = CallRecord::new();
        scope(record.clone(), async {
            litellm_call_scope::spawn_in_call(async { add_header("x-spawned", "yes") })
                .await
                .unwrap();
        })
        .await;
        assert_eq!(
            scope(record, async { added() }).await,
            Some(json!({"x-spawned": "yes"}))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_body_dropped_before_its_end_is_dropped_in_its_call() {
        struct Dropped;
        impl Drop for Dropped {
            fn drop(&mut self) {
                add_header("x-dropped", "yes");
            }
        }
        let record = CallRecord::new();
        let guard = Dropped;
        let stream = futures_util::stream::pending::<Result<Bytes, std::convert::Infallible>>()
            .map(move |frame| {
                let _ = &guard;
                frame
            });
        drop(Scoped {
            record: record.clone(),
            body: Some(Body::from_stream(stream)),
        });
        assert_eq!(
            scope(record, async { added() }).await,
            Some(json!({"x-dropped": "yes"}))
        );
    }
}
