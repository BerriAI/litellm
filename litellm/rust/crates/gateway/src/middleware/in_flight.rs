pub use tower_http::metrics::in_flight_requests::InFlightRequestsCounter as InFlight;
use tower_http::metrics::in_flight_requests::InFlightRequestsLayer;

/// `InFlightRequestsMiddleware`: a request counts until its response body is done or dropped.
pub(super) fn layer(in_flight: &InFlight) -> InFlightRequestsLayer {
    InFlightRequestsLayer::new(in_flight.clone())
}

#[cfg(test)]
mod tests {
    use axum::{Router, body::Body, extract::Request, response::Response, routing::get};
    use http_body_util::BodyExt;
    use rstest::rstest;
    use tower::ServiceExt;

    use super::*;

    async fn finish_by_reading(response: Response) {
        assert_eq!(
            response.into_body().collect().await.unwrap().to_bytes(),
            "chunk"
        );
    }

    async fn finish_by_dropping(response: Response) {
        drop(response);
    }

    #[rstest]
    #[case::read(finish_by_reading)]
    #[case::dropped(finish_by_dropping)]
    #[tokio::test]
    async fn a_request_counts_until_its_response_body_is_finished<F: Future<Output = ()>>(
        #[case] finish: fn(Response) -> F,
    ) {
        let in_flight = InFlight::new();
        let observed = in_flight.clone();
        let app = Router::new()
            .route(
                "/",
                get(move || async move {
                    assert_eq!(observed.get(), 1);
                    "chunk"
                }),
            )
            .layer(layer(&in_flight));

        let response = app
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(in_flight.get(), 1);

        finish(response).await;
        assert_eq!(in_flight.get(), 0);
    }
}
