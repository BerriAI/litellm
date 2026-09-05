use std::future::Future;

use litellm_core::observability::{FunctionTrace, FunctionTraceEvent};
use serde::Serialize;
use tracing::instrument::WithSubscriber;

#[derive(Serialize)]
pub(crate) struct TracedResponse<T> {
    response: T,
    trace: Vec<FunctionTraceEvent>,
}

pub(crate) async fn capture<T, E>(
    future: impl Future<Output = Result<T, E>>,
) -> Result<TracedResponse<T>, E> {
    let trace = FunctionTrace::default();
    let response = future.with_subscriber(trace.dispatcher()).await?;
    Ok(TracedResponse {
        response,
        trace: trace.events(),
    })
}
