use std::fmt::Display;
use std::future::Future;

use litellm_core::observability::{FunctionTrace, FunctionTraceEvent};
use serde::Serialize;
use tracing::instrument::WithSubscriber;

#[derive(Serialize)]
pub(crate) struct TracedResponse<T> {
    #[serde(skip_serializing_if = "Option::is_none")]
    response: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    trace: Vec<FunctionTraceEvent>,
}

pub(crate) async fn capture<T, E>(
    future: impl Future<Output = Result<T, E>>,
) -> Result<TracedResponse<T>, E>
where
    E: Display,
{
    let trace = FunctionTrace::default();
    let result = future.with_subscriber(trace.dispatcher()).await;
    let events = trace.events();
    Ok(match result {
        Ok(response) => TracedResponse {
            response: Some(response),
            error: None,
            trace: events,
        },
        Err(error) => TracedResponse {
            response: None,
            error: Some(error.to_string()),
            trace: events,
        },
    })
}
