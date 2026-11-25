use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use axum::{
    Json,
    body::Body,
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use http_body_util::BodyExt;
use litellm_config::GeneralSettings;
use serde_json::json;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    max_in_flight: usize,
    max_queued: usize,
    queue_timeout: Duration,
}

impl Settings {
    /// `get_admission_control_settings`: off unless `max_in_flight_requests_per_worker` is set,
    /// and off with an error when a value is out of range. The whole gateway is one worker.
    pub fn from_general(general: &GeneralSettings) -> Option<Self> {
        let max_in_flight = general.max_in_flight_requests_per_worker?;
        let queue_timeout = Duration::try_from_secs_f64(general.admission_queue_timeout_seconds)
            .ok()
            .filter(|timeout| !timeout.is_zero());
        let (1.., Some(queue_timeout)) = (max_in_flight, queue_timeout) else {
            tracing::error!(
                max_in_flight_requests_per_worker = max_in_flight,
                admission_queue_timeout_seconds = general.admission_queue_timeout_seconds,
                "ignoring invalid admission control settings, admission control is disabled"
            );
            return None;
        };
        let permits = |count: u64| {
            usize::try_from(count)
                .unwrap_or(usize::MAX)
                .min(Semaphore::MAX_PERMITS)
        };
        Some(Self {
            max_in_flight: permits(max_in_flight),
            max_queued: permits(
                general
                    .max_queued_requests_per_worker
                    .unwrap_or(max_in_flight),
            ),
            queue_timeout,
        })
    }
}

pub(super) struct Admission {
    settings: Settings,
    in_flight: Arc<Semaphore>,
    queued: AtomicUsize,
}

struct Queued<'a>(&'a AtomicUsize);

impl Drop for Queued<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Admission {
    pub(super) fn new(settings: Settings) -> Self {
        Self {
            settings,
            in_flight: Arc::new(Semaphore::new(settings.max_in_flight)),
            queued: AtomicUsize::new(0),
        }
    }

    /// A free slot admits at once. Otherwise the request waits in the queue while it has room,
    /// for at most the queue timeout.
    async fn slot(&self) -> Option<OwnedSemaphorePermit> {
        if let Ok(slot) = self.in_flight.clone().try_acquire_owned() {
            return Some(slot);
        }
        let _queued = self.queue()?;
        tokio::time::timeout(
            self.settings.queue_timeout,
            self.in_flight.clone().acquire_owned(),
        )
        .await
        .ok()?
        .ok()
    }

    fn queue(&self) -> Option<Queued<'_>> {
        self.queued
            .try_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
                (queued < self.settings.max_queued).then_some(queued + 1)
            })
            .ok()?;
        Some(Queued(&self.queued))
    }

    /// `_overloaded_response`.
    fn overloaded(&self) -> Response {
        let admitted = self.settings.max_in_flight - self.in_flight.available_permits();
        let queued = self.queued.load(Ordering::Acquire);
        let message = format!(
            "Worker at capacity: {admitted} in-flight, {queued} queued requests. Retry later."
        );
        (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::RETRY_AFTER, "1")],
            Json(json!({"error": {"message": message, "type": "overloaded_error", "code": "503"}})),
        )
            .into_response()
    }
}

/// `AdmissionControlMiddleware`: a request holds its slot until its response body is dropped,
/// as Python holds it until the response is sent. Of Python's `_EXEMPT_PATHS` only
/// `/health/readiness` is exempt here, because it is the only one with a native route.
pub(super) async fn admit(
    State(admission): State<Arc<Admission>>,
    request: Request,
    next: Next,
) -> Response {
    if request.uri().path() == "/health/readiness" {
        return next.run(request).await;
    }
    let Some(slot) = admission.slot().await else {
        return admission.overloaded();
    };
    next.run(request).await.map(|body| {
        Body::new(body.map_frame(move |frame| {
            let _held = &slot;
            frame
        }))
    })
}

#[cfg(test)]
mod tests {
    use axum::{Router, middleware::from_fn_with_state, routing::get};
    use rstest::rstest;
    use serde_json::Value;
    use tower::ServiceExt;

    use super::*;

    fn general(
        max_in_flight: Option<u64>,
        max_queued: Option<u64>,
        queue_timeout: f64,
    ) -> GeneralSettings {
        GeneralSettings {
            max_in_flight_requests_per_worker: max_in_flight,
            max_queued_requests_per_worker: max_queued,
            admission_queue_timeout_seconds: queue_timeout,
            ..GeneralSettings::default()
        }
    }

    fn settings(max_in_flight: usize, max_queued: usize, queue_timeout: Duration) -> Settings {
        Settings {
            max_in_flight,
            max_queued,
            queue_timeout,
        }
    }

    #[rstest]
    #[case::unset(general(None, Some(1), 1.0), None)]
    #[case::queue_defaults_to_the_cap(
        general(Some(2), None, 1.0),
        Some(settings(2, 2, Duration::from_secs(1)))
    )]
    #[case::no_queue(
        general(Some(2), Some(0), 0.5),
        Some(settings(2, 0, Duration::from_millis(500)))
    )]
    #[case::zero_cap(general(Some(0), None, 1.0), None)]
    #[case::zero_timeout(general(Some(2), None, 0.0), None)]
    #[case::negative_timeout(general(Some(2), None, -1.0), None)]
    fn settings_come_from_general_settings(
        #[case] general: GeneralSettings,
        #[case] expected: Option<Settings>,
    ) {
        assert_eq!(Settings::from_general(&general), expected);
    }

    fn app(max_queued: usize, queue_timeout: Duration) -> Router {
        Router::new()
            .route("/v1/messages", get(|| async { "ok" }))
            .route("/health/readiness", get(|| async { "ok" }))
            .layer(from_fn_with_state(
                Arc::new(Admission::new(settings(1, max_queued, queue_timeout))),
                admit,
            ))
    }

    async fn send(app: &Router, path: &str) -> Response {
        app.clone()
            .oneshot(Request::get(path).body(Body::empty()).unwrap())
            .await
            .unwrap()
    }

    async fn assert_overloaded(response: Response, message: &str) {
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()[header::RETRY_AFTER], "1");
        let body = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap(),
            json!({"error": {"message": message, "type": "overloaded_error", "code": "503"}})
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_request_holds_its_slot_until_its_response_body_is_dropped() {
        let app = app(0, Duration::from_secs(60));
        let unfinished = send(&app, "/v1/messages").await;
        assert_eq!(unfinished.status(), StatusCode::OK);

        assert_overloaded(
            send(&app, "/v1/messages").await,
            "Worker at capacity: 1 in-flight, 0 queued requests. Retry later.",
        )
        .await;

        drop(unfinished);
        assert_eq!(send(&app, "/v1/messages").await.status(), StatusCode::OK);
    }

    #[rstest]
    #[tokio::test]
    async fn a_queued_request_is_admitted_when_a_slot_frees() {
        let app = app(1, Duration::from_secs(60));
        let unfinished = send(&app, "/v1/messages").await;
        let mut queued = std::pin::pin!(send(&app, "/v1/messages"));
        assert!(futures_util::poll!(&mut queued).is_pending());

        assert_overloaded(
            send(&app, "/v1/messages").await,
            "Worker at capacity: 1 in-flight, 1 queued requests. Retry later.",
        )
        .await;

        drop(unfinished);
        assert_eq!(queued.await.status(), StatusCode::OK);
    }

    #[rstest]
    #[tokio::test]
    async fn a_request_queued_past_the_timeout_is_shed() {
        let app = app(1, Duration::from_millis(10));
        let _unfinished = send(&app, "/v1/messages").await;
        assert_overloaded(
            send(&app, "/v1/messages").await,
            "Worker at capacity: 1 in-flight, 0 queued requests. Retry later.",
        )
        .await;
    }

    #[rstest]
    #[tokio::test]
    async fn a_health_probe_is_served_at_capacity() {
        let app = app(0, Duration::from_secs(60));
        let _unfinished = send(&app, "/v1/messages").await;
        assert_eq!(
            send(&app, "/health/readiness").await.status(),
            StatusCode::OK
        );
    }
}
