//! What the gateway does with a request it does not serve natively.
//!
//! A mounted router declines by putting a [`Decline`] in its response's extensions. Nothing
//! else crosses this boundary: only this module records requests and talks to Python.

mod forward;
mod replay;

pub use forward::{Forward, InvalidUpstream};

use axum::{Router, extract::Request};
use litellm_core_utils::decline::Decline;

const NOT_NATIVE: Decline = Decline::new("route not native");

#[derive(Clone)]
pub enum Fallback {
    /// Standalone: an unserved path is a 404, a decline is the mounted router's own error.
    Reject,
    /// Next to the Python proxy: both go to Python unchanged.
    Forward(Forward),
}

/// Forwards declined requests on the routes `served` holds so far. `record_limit` must cover
/// the largest body those routes accept, or a declined request that size keeps its error.
pub(crate) fn replay_declines(served: Router, fallback: &Fallback, record_limit: usize) -> Router {
    match fallback {
        Fallback::Reject => served,
        Fallback::Forward(forward) => served.route_layer(axum::middleware::from_fn_with_state(
            replay::Replay {
                forward: forward.clone(),
                record_limit,
            },
            replay::replay,
        )),
    }
}

pub(crate) fn forward_unmatched(app: Router, fallback: Fallback) -> Router {
    match fallback {
        Fallback::Reject => app,
        Fallback::Forward(forward) => {
            let unmatched = forward.clone();
            app.fallback(move |request: Request| async move {
                unmatched.send(request, NOT_NATIVE).await
            })
            .method_not_allowed_fallback(move |request: Request| async move {
                forward.send(request, NOT_NATIVE).await
            })
        }
    }
}
