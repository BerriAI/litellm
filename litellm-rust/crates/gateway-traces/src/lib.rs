mod error;
mod runs;

use std::sync::Arc;

use axum::{Router, routing::get};
pub use error::ReadFailure;
use litellm_traces_cache::{TraceReader, TraceStore};

pub struct Traces<S> {
    pub reader: TraceReader,
    pub store: S,
}

pub fn router<S>(traces: Arc<Traces<S>>) -> Router
where
    S: TraceStore + Send + 'static,
{
    Router::new()
        .route("/v1/traces", get(runs::list::<S>))
        .route("/v1/traces/histogram", get(runs::histogram::<S>))
        .route("/v1/traces/values/{field}", get(runs::values::<S>))
        .with_state(traces)
}
