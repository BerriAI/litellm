use std::sync::Arc;

use axum::{
    extract::{MatchedPath, Request, State},
    http::{HeaderName, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::Credential;
use litellm_python_compat::repr::float_repr;

use crate::{
    callbacks::parallel_request_limiter::{Caller, KeyLimiter, rate_limited},
    host::NativeRows,
};

const KEY_RPM_LIMIT: HeaderName = HeaderName::from_static("x-litellm-key-rpm-limit");
const KEY_TPM_LIMIT: HeaderName = HeaderName::from_static("x-litellm-key-tpm-limit");
const KEY_MAX_BUDGET: HeaderName = HeaderName::from_static("x-litellm-key-max-budget");

/// `x-litellm-key-rpm-limit`, `x-litellm-key-tpm-limit` and `x-litellm-key-max-budget`, which the
/// proxy's custom headers carry on every answer of a route's handler, a failed one included. The
/// budget is written as Python's `str(float)`, and the master key has none.
fn with_key_headers(
    mut response: Response,
    caller: &Caller,
    identity: &Authenticated<NativeRows>,
) -> Response {
    let max_budget = match &identity.credential {
        Credential::Key { key } => key.budget().max,
        Credential::MasterKey => None,
    };
    let named = [
        (KEY_RPM_LIMIT, caller.limits.rpm.map(HeaderValue::from)),
        (KEY_TPM_LIMIT, caller.limits.tpm.map(HeaderValue::from)),
        (
            KEY_MAX_BUDGET,
            max_budget.and_then(|max| HeaderValue::try_from(float_repr(max)).ok()),
        ),
    ];
    for (name, value) in named {
        if let Some(value) = value {
            response.headers_mut().insert(name, value);
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
/// body. Every answer of the handler names the key's own limits and budget in its headers.
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
        return with_key_headers(next.run(request).await, &caller, &identity);
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

#[cfg(test)]
mod tests {
    use litellm_gateway_auth_core::{Belongs, record::Budget, record::EndUserRecord};
    use rstest::rstest;

    use super::*;
    use crate::host::Key;

    fn holding(max: Option<f64>) -> Authenticated<NativeRows> {
        Authenticated::from(litellm_gateway_auth_core::Caller {
            credential: Credential::Key {
                key: Key::stored("hash", None).with_budget(Budget { max, spend: 3.0 }),
            },
            belongs: Belongs::nobody(),
            end_user: EndUserRecord::default(),
        })
    }

    #[rstest]
    #[case::a_whole_budget(holding(Some(10.0)), Some("10.0"))]
    #[case::a_fraction(holding(Some(0.5)), Some("0.5"))]
    #[case::a_tiny_budget(holding(Some(1e-05)), Some("1e-05"))]
    #[case::a_huge_budget(holding(Some(1e16)), Some("1e+16"))]
    #[case::a_zero_budget(holding(Some(0.0)), Some("0.0"))]
    #[case::no_budget(holding(None), None)]
    #[case::the_master_key(crate::test_support::master(), None)]
    fn an_answer_names_the_keys_max_budget_as_python_prints_it(
        #[case] identity: Authenticated<NativeRows>,
        #[case] expected: Option<&str>,
    ) {
        let answered = with_key_headers(Response::default(), &Caller::of(&identity), &identity);

        assert_eq!(
            answered
                .headers()
                .get("x-litellm-key-max-budget")
                .map(|value| value.to_str().unwrap()),
            expected
        );
    }
}
