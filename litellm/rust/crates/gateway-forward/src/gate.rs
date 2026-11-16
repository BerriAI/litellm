use std::fmt::Display;

use axum::{
    Router,
    extract::Request,
    http::HeaderValue,
    middleware::{self, Next},
    response::Response,
};

use crate::{Forward, rewrite};

/// Why Python answers a request the gateway received.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reason {
    /// The app serves nothing natively for this deployment.
    Configuration,
    /// No native route takes this path and method.
    Route,
    /// A native route takes it, and the app handed this request back under its own code.
    Refused(&'static str),
}

impl Reason {
    fn code(self) -> &'static str {
        match self {
            Self::Configuration => "configuration",
            Self::Route => "route",
            Self::Refused(code) => code,
        }
    }
}

/// Which of the two servers answers a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    Native,
    Python(Reason),
}

impl Verdict {
    fn header(self) -> HeaderValue {
        const PYTHON: HeaderValue = HeaderValue::from_static("python");
        match self {
            Self::Native => HeaderValue::from_static("native"),
            Self::Python(reason) => {
                HeaderValue::try_from(format!("python; reason={}", reason.code())).unwrap_or(PYTHON)
            }
        }
    }
}

impl Forward {
    /// `native` in front of Python. A request on one of its routes goes through `admit`, which
    /// either lets it into `native` or hands it back as the client sent it, with why. Python
    /// answers every request handed back and every request no route of `native` takes. Each
    /// response names who answered it in [`VERDICT_HEADER`](crate::VERDICT_HEADER). `native` must
    /// have at least one route.
    pub fn behind<A, Admitted, Why>(self, native: Router, admit: A) -> Router
    where
        A: Fn(Request) -> Admitted + Clone + Send + Sync + 'static,
        Admitted: Future<Output = Result<Request, (Box<Request>, Why)>> + Send + 'static,
        Why: Display + Send + 'static,
        for<'a> &'a Why: Into<&'static str>,
    {
        let refused = self.clone();
        let wrong_method = self.clone();
        let gate = move |request: Request, next: Next| {
            let admitted = admit(request);
            let python = refused.clone();
            async move {
                match admitted.await {
                    Ok(request) => {
                        rewrite::answered_by(next.run(request).await, Verdict::Native.header())
                    }
                    Err((request, why)) => {
                        let reason = Reason::Refused((&why).into());
                        tracing::debug!(reason = reason.code(), %why, "python answers a request the native app handed back");
                        python.answer(*request, reason).await
                    }
                }
            }
        };
        native
            .route_layer(middleware::from_fn(gate))
            .method_not_allowed_fallback(move |request: Request| async move {
                wrong_method.answer(request, Reason::Route).await
            })
            .fallback(
                move |request: Request| async move { self.answer(request, Reason::Route).await },
            )
    }

    /// Python answers every request, because the app serves nothing natively for this deployment.
    /// Each response still says so in [`VERDICT_HEADER`](crate::VERDICT_HEADER).
    pub fn without_native(self) -> Router {
        Router::new().fallback(move |request: Request| async move {
            self.answer(request, Reason::Configuration).await
        })
    }

    async fn answer(&self, request: Request, reason: Reason) -> Response {
        rewrite::answered_by(self.relay(request).await, Verdict::Python(reason).header())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::native(Verdict::Native, "native")]
    #[case::configuration(Verdict::Python(Reason::Configuration), "python; reason=configuration")]
    #[case::route(Verdict::Python(Reason::Route), "python; reason=route")]
    #[case::refused(Verdict::Python(Reason::Refused("body")), "python; reason=body")]
    #[case::unprintable_code(Verdict::Python(Reason::Refused("a\nb")), "python")]
    fn a_verdict_names_who_answers_and_why(#[case] verdict: Verdict, #[case] header: &str) {
        assert_eq!(verdict.header(), header);
    }
}
