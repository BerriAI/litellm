use std::{convert::Infallible, fmt::Display};

use axum::{
    extract::{Request, State},
    http::{HeaderName, HeaderValue},
    middleware::{Next, from_fn_with_state},
    response::Response,
};
use tower::{Layer, Service, ServiceExt};
use tower_http::set_header::SetResponseHeader;

use crate::{forward::Forward, rewrite::ANSWERED_BY_HEADER};

/// Runs `admit` on each request, then sends it to `native` if admitted, otherwise forwards it to
/// Python as `admit` gave it back.
async fn answer<A, Admitted, Why>(
    State((admit, python)): State<(A, Forward)>,
    request: Request,
    native: Next,
) -> Result<Response, Infallible>
where
    A: Fn(Request) -> Admitted,
    Admitted: Future<Output = Result<Request, (Box<Request>, Why)>>,
    Why: Display + Into<&'static str>,
{
    match admit(request).await {
        Ok(request) => Ok(native.run(request).await),
        Err((request, why)) => {
            tracing::debug!(%why, "python answers a request the native app refused");
            python.because(why.into()).oneshot(*request).await
        }
    }
}

impl Forward {
    /// `native` in front of Python. `admit` sees every request before `native` does, and either
    /// admits it or refuses it, giving it back as the client sent it, with why. Python answers
    /// every refused request, and none of them pass through `native` or a layer it has. `admit` is
    /// the only thing that sends a request to Python, so a request it admits gets `native`'s
    /// answer even when that is a 404. Each response names who answered it in
    /// [`ANSWERED_BY_HEADER`].
    pub(crate) fn behind<S, A, Admitted, Why>(
        self,
        native: S,
        admit: A,
    ) -> impl Service<Request, Response = Response, Error = Infallible, Future: Send + 'static>
    + Clone
    + Send
    + 'static
    where
        S: Service<Request, Response = Response, Error = Infallible>
            + Clone
            + Send
            + Sync
            + 'static,
        S::Future: Send + 'static,
        A: Fn(Request) -> Admitted + Clone + Send + Sync + 'static,
        Admitted: Future<Output = Result<Request, (Box<Request>, Why)>> + Send + 'static,
        Why: Display + Into<&'static str> + Send + 'static,
    {
        let native = SetResponseHeader::overriding(
            native,
            HeaderName::from_static(ANSWERED_BY_HEADER),
            HeaderValue::from_static("native"),
        );
        from_fn_with_state((admit, self), answer::<A, Admitted, Why>).layer(native)
    }

    /// Python's answer naming Python and `why` in [`ANSWERED_BY_HEADER`], which replaces any copy
    /// the answer carried.
    fn because(self, why: &str) -> SetResponseHeader<Self, HeaderValue> {
        SetResponseHeader::overriding(
            self,
            HeaderName::from_static(ANSWERED_BY_HEADER),
            answered_by_python(why),
        )
    }
}

fn answered_by_python(why: &str) -> HeaderValue {
    const PYTHON: HeaderValue = HeaderValue::from_static("python");
    HeaderValue::try_from(format!("python; reason={why}")).unwrap_or(PYTHON)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::code("body", "python; reason=body")]
    #[case::unprintable_code("a\nb", "python")]
    fn a_forwarded_answer_names_python_and_why(#[case] why: &str, #[case] header: &str) {
        assert_eq!(answered_by_python(why), header);
    }
}
