use std::sync::Arc;

use axum::{
    extract::{MatchedPath, Request, State},
    http::{HeaderName, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use litellm_gateway_auth::Authenticated;

use crate::{
    callbacks::parallel_request_limiter::{Caller, KeyLimiter, rate_limited},
    host::NativeRows,
};

const KEY_RPM_LIMIT: HeaderName = HeaderName::from_static("x-litellm-key-rpm-limit");
const KEY_TPM_LIMIT: HeaderName = HeaderName::from_static("x-litellm-key-tpm-limit");

/// `x-litellm-key-rpm-limit` and `x-litellm-key-tpm-limit`, which the proxy's custom headers
/// carry on every answer of a route's handler, a failed one included.
fn with_key_limits(mut response: Response, caller: &Caller) -> Response {
    for (name, limit) in [
        (KEY_RPM_LIMIT, caller.limits.rpm),
        (KEY_TPM_LIMIT, caller.limits.tpm),
    ] {
        if let Some(limit) = limit {
            response
                .headers_mut()
                .insert(name, HeaderValue::from(limit));
        }
    }
    response
}

/// `raise_if_over_limit_before_body`, which the proxy's authentication runs for a key it has
/// cached: a caller already over a limit that needs no body, its `rpm_limit` or its
/// `max_parallel_requests`, is refused before the body is read and before any callback counts the
/// call. The refusal is the proxy's `ProxyRateLimitError` in the route's envelope with the
/// limit's headers. Counters are only read here. The limiter's pre-call hook, which the route
/// runs through the callback stage, counts an admitted call and refuses the limits that need the
/// body. Every answer of the handler names the key's own limits in its headers.
///
/// Accepted difference: the proxy skips this for a key it reads for the first time, and the
/// gateway never does.
pub(super) async fn before_body(
    State(limiter): State<Arc<KeyLimiter>>,
    identity: Authenticated<NativeRows>,
    path: MatchedPath,
    request: Request,
    next: Next,
) -> Response {
    let caller = Caller::of(&identity);
    let Some(abort) = limiter.before_body(&caller).await else {
        return with_key_limits(next.run(request).await, &caller);
    };
    match rate_limited(&abort) {
        Some(limited) => limited.response(path.as_str()),
        None => {
            let status =
                StatusCode::from_u16(abort.status).unwrap_or(StatusCode::SERVICE_UNAVAILABLE);
            (status, abort.message).into_response()
        }
    }
}
