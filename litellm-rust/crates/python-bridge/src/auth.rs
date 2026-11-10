//! The native auth core, driven a step at a time by Python. Each facet call is its own class with
//! one `answer_*` method per host call, so the stub types the answer Python gives every step.
//!
//! A call owns its host, so the host parks each host call in the call's own exchange and the
//! facet future stays pending until Python answers it.

use std::{
    future::{Future, poll_fn},
    pin::Pin,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
    task::{Context, Poll, Waker},
};

use litellm_error::Report;
use litellm_gateway_auth_core::{
    AuthCore, CustomAuthCore, CustomAuthHost, CustomAuthResult, Introspection, NativeCore,
    OAuth2Core, OAuth2Denial, OAuth2Identity, OAuth2IntrospectionHost, OAuth2IntrospectionSettings,
    RequestFacts,
};
use litellm_host_python::{from_py_argument, to_py};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, strum::IntoStaticStr)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
enum HostCall {
    Introspect { token: String },
    CustomAuth,
}

#[derive(Default)]
struct Exchange {
    call: Option<HostCall>,
    introspection: Option<Introspection>,
    custom_auth: Option<CustomAuthResult>,
}

type Shared = Arc<Mutex<Exchange>>;

fn lock(exchange: &Mutex<Exchange>) -> MutexGuard<'_, Exchange> {
    exchange.lock().unwrap_or_else(PoisonError::into_inner)
}

type Slot<T> = fn(&mut Exchange) -> &mut Option<T>;

struct StepHost {
    exchange: Shared,
}

impl StepHost {
    fn new(exchange: &Shared) -> Self {
        Self {
            exchange: exchange.clone(),
        }
    }

    async fn ask<T>(&self, call: HostCall, slot: Slot<T>) -> T {
        let mut call = Some(call);
        poll_fn(move |_| {
            let mut exchange = lock(&self.exchange);
            if let Some(call) = call.take() {
                exchange.call = Some(call);
                return Poll::Pending;
            }
            slot(&mut exchange)
                .take()
                .map_or(Poll::Pending, Poll::Ready)
        })
        .await
    }
}

impl OAuth2IntrospectionHost for StepHost {
    type Error = std::convert::Infallible;

    async fn introspect(&self, token: &str) -> Result<Introspection, Self::Error> {
        let call = HostCall::Introspect {
            token: token.to_owned(),
        };
        Ok(self.ask(call, |exchange| &mut exchange.introspection).await)
    }
}

impl CustomAuthHost for StepHost {
    type Error = std::convert::Infallible;

    async fn custom_auth(&self, _facts: &RequestFacts) -> Result<CustomAuthResult, Self::Error> {
        Ok(self
            .ask(HostCall::CustomAuth, |exchange| &mut exchange.custom_auth)
            .await)
    }
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum OAuth2Verdict {
    Identity(Box<OAuth2Identity>),
    InactiveToken,
    IntrospectionNotConfigured,
}

impl From<Result<OAuth2Identity, OAuth2Denial>> for OAuth2Verdict {
    fn from(outcome: Result<OAuth2Identity, OAuth2Denial>) -> Self {
        match outcome {
            Ok(identity) => Self::Identity(Box::new(identity)),
            Err(OAuth2Denial::InactiveToken) => Self::InactiveToken,
            Err(OAuth2Denial::IntrospectionNotConfigured) => Self::IntrospectionNotConfigured,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename = "done")]
struct Finished<T> {
    outcome: T,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(untagged)]
enum Step<T> {
    Ask(HostCall),
    Done(Finished<T>),
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
enum DriveError {
    #[error("the auth call already started")]
    Started,
    #[error("the auth call has not started")]
    NotStarted,
    #[error("the auth call already finished")]
    Finished,
    #[error("the auth call is not waiting for a `{0}` answer")]
    NotAsked(&'static str),
    #[error("the auth call is pending on something other than a host call")]
    Stalled,
}

impl From<DriveError> for PyErr {
    fn from(error: DriveError) -> Self {
        PyValueError::new_err(Report::new(&error).to_string())
    }
}

type Facet<T> = Pin<Box<dyn Future<Output = T> + Send>>;

struct Driver<T> {
    future: Option<Facet<T>>,
    exchange: Shared,
    started: bool,
}

impl<T> Driver<T> {
    fn new(exchange: Shared, future: impl Future<Output = T> + Send + 'static) -> Self {
        Self {
            future: Some(Box::pin(future)),
            exchange,
            started: false,
        }
    }

    fn start(&mut self) -> Result<Step<T>, DriveError> {
        if self.started {
            return Err(DriveError::Started);
        }
        self.started = true;
        self.poll()
    }

    fn answer<A>(
        &mut self,
        kind: &'static str,
        slot: Slot<A>,
        answer: A,
    ) -> Result<Step<T>, DriveError> {
        if !self.started {
            return Err(DriveError::NotStarted);
        }
        if self.future.is_none() {
            return Err(DriveError::Finished);
        }
        {
            let mut exchange = lock(&self.exchange);
            match &exchange.call {
                Some(call) if <&str>::from(call) == kind => {}
                _ => return Err(DriveError::NotAsked(kind)),
            }
            exchange.call = None;
            *slot(&mut exchange) = Some(answer);
        }
        self.poll()
    }

    fn poll(&mut self) -> Result<Step<T>, DriveError> {
        let Some(future) = self.future.as_mut() else {
            return Err(DriveError::Finished);
        };
        if let Poll::Ready(outcome) = future
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
        {
            self.future = None;
            return Ok(Step::Done(Finished { outcome }));
        }
        let Some(call) = lock(&self.exchange).call.clone() else {
            self.future = None;
            return Err(DriveError::Stalled);
        };
        Ok(Step::Ask(call))
    }
}

fn oauth2_driver(
    core: NativeCore,
    token: String,
    settings: OAuth2IntrospectionSettings,
) -> Driver<OAuth2Verdict> {
    let exchange = Shared::default();
    let host = StepHost::new(&exchange);
    Driver::new(exchange, async move {
        let Ok(outcome) = core.oauth2().introspect(&host, &token, &settings).await;
        OAuth2Verdict::from(outcome)
    })
}

fn custom_driver(core: NativeCore, facts: RequestFacts) -> Driver<CustomAuthResult> {
    let exchange = Shared::default();
    let host = StepHost::new(&exchange);
    Driver::new(exchange, async move {
        let Ok(result) = core.custom().authenticate(&host, &facts).await;
        result
    })
}

fn lock_driver<T>(driver: &Mutex<Driver<T>>) -> MutexGuard<'_, Driver<T>> {
    driver.lock().unwrap_or_else(PoisonError::into_inner)
}

fn step_to_py<T: Serialize>(
    py: Python<'_>,
    step: Result<Step<T>, DriveError>,
) -> PyResult<Py<PyAny>> {
    to_py(py, &step?)
}

/// The native auth core. Each facet method returns an unstarted call.
#[pyclass(frozen)]
pub(crate) struct AuthCoreHandle {
    core: NativeCore,
}

#[pymethods]
impl AuthCoreHandle {
    #[new]
    fn new() -> Self {
        Self {
            core: NativeCore::default(),
        }
    }

    fn oauth2(&self, token: String, settings: &Bound<'_, PyAny>) -> PyResult<OAuth2Call> {
        Ok(OAuth2Call(Mutex::new(oauth2_driver(
            self.core,
            token,
            from_py_argument(settings)?,
        ))))
    }

    fn custom(&self, facts: &Bound<'_, PyAny>) -> PyResult<CustomAuthCall> {
        Ok(CustomAuthCall(Mutex::new(custom_driver(
            self.core,
            from_py_argument(facts)?,
        ))))
    }
}

/// One OAuth2 introspection call.
#[pyclass(frozen)]
pub(crate) struct OAuth2Call(Mutex<Driver<OAuth2Verdict>>);

#[pymethods]
impl OAuth2Call {
    fn start(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        step_to_py(py, lock_driver(&self.0).start())
    }

    fn answer_introspection(
        &self,
        py: Python<'_>,
        introspection: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let introspection = from_py_argument(introspection)?;
        step_to_py(
            py,
            lock_driver(&self.0).answer(
                "introspect",
                |exchange| &mut exchange.introspection,
                introspection,
            ),
        )
    }
}

/// One custom auth call.
#[pyclass(frozen)]
pub(crate) struct CustomAuthCall(Mutex<Driver<CustomAuthResult>>);

#[pymethods]
impl CustomAuthCall {
    fn start(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        step_to_py(py, lock_driver(&self.0).start())
    }

    fn answer_custom_auth(&self, py: Python<'_>, result: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let result = from_py_argument(result)?;
        step_to_py(
            py,
            lock_driver(&self.0).answer(
                "custom_auth",
                |exchange| &mut exchange.custom_auth,
                result,
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use litellm_gateway_auth_core::CustomIdentity;
    use litellm_secret_value::SecretValue;
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    fn facts() -> RequestFacts {
        RequestFacts {
            route: "/chat/completions".to_owned(),
            token: Some(SecretValue::from("tok-1")),
            is_llm_api_route: true,
        }
    }

    fn oauth2(endpoint_configured: bool) -> Driver<OAuth2Verdict> {
        let settings = OAuth2IntrospectionSettings {
            endpoint_configured,
            requires_active: true,
        };
        oauth2_driver(NativeCore::default(), "tok-1".to_owned(), settings)
    }

    fn introspection(active: bool) -> Introspection {
        Introspection {
            active,
            user_id: json!("user-1"),
            role: json!("internal_user"),
            team_id: Value::Null,
        }
    }

    fn answer_introspection(
        driver: &mut Driver<OAuth2Verdict>,
        introspection: Introspection,
    ) -> Result<Step<OAuth2Verdict>, DriveError> {
        driver.answer(
            "introspect",
            |exchange| &mut exchange.introspection,
            introspection,
        )
    }

    fn answer_custom_auth(
        driver: &mut Driver<CustomAuthResult>,
        result: CustomAuthResult,
    ) -> Result<Step<CustomAuthResult>, DriveError> {
        driver.answer("custom_auth", |exchange| &mut exchange.custom_auth, result)
    }

    fn wire<T: Serialize>(step: &Step<T>) -> Value {
        serde_json::to_value(step).unwrap()
    }

    #[test]
    fn oauth2_asks_python_to_introspect_the_presented_token() {
        let step = oauth2(true).start().unwrap();
        assert_eq!(wire(&step), json!({"kind": "introspect", "token": "tok-1"}));
    }

    #[rstest]
    #[case::active(
        true,
        json!({"kind": "done", "outcome": {"kind": "identity", "token": "tok-1", "user_id": "user-1", "role": "internal_user", "team_id": null}})
    )]
    #[case::inactive(false, json!({"kind": "done", "outcome": {"kind": "inactive_token"}}))]
    fn oauth2_finishes_with_the_verdict_on_python_s_introspection(
        #[case] active: bool,
        #[case] expected: Value,
    ) {
        let mut driver = oauth2(true);
        driver.start().unwrap();
        let step = answer_introspection(&mut driver, introspection(active)).unwrap();
        assert_eq!(wire(&step), expected);
    }

    #[test]
    fn oauth2_without_an_endpoint_finishes_without_a_host_call() {
        let step = oauth2(false).start().unwrap();
        assert_eq!(
            wire(&step),
            json!({"kind": "done", "outcome": {"kind": "introspection_not_configured"}})
        );
    }

    #[rstest]
    #[case::identity(
        CustomAuthResult::Identity(CustomIdentity {
            user_id: Some("user-1".to_owned()),
            team_id: Some("team-1".to_owned()),
            hashed_token: None,
        }),
        json!({"kind": "identity", "user_id": "user-1", "team_id": "team-1", "hashed_token": null})
    )]
    #[case::replaced_key(
        CustomAuthResult::ReplacedKey { token: "sk-new".to_owned() },
        json!({"kind": "replaced_key", "token": "sk-new"})
    )]
    #[case::no_identity(CustomAuthResult::NoIdentity, json!({"kind": "no_identity"}))]
    fn custom_auth_finishes_with_what_python_resolved(
        #[case] result: CustomAuthResult,
        #[case] outcome: Value,
    ) {
        let mut driver = custom_driver(NativeCore::default(), facts());
        assert_eq!(
            wire(&driver.start().unwrap()),
            json!({"kind": "custom_auth"})
        );
        let step = answer_custom_auth(&mut driver, result).unwrap();
        assert_eq!(wire(&step), json!({"kind": "done", "outcome": outcome}));
    }

    #[test]
    fn an_answer_before_start_is_refused() {
        let mut driver = oauth2(true);
        assert_eq!(
            answer_introspection(&mut driver, introspection(true)),
            Err(DriveError::NotStarted)
        );
    }

    #[test]
    fn a_second_start_is_refused() {
        let mut driver = oauth2(true);
        driver.start().unwrap();
        assert_eq!(driver.start(), Err(DriveError::Started));
    }

    #[test]
    fn an_answer_to_a_call_that_was_not_made_is_refused() {
        let mut driver = custom_driver(NativeCore::default(), facts());
        assert_eq!(
            driver.start().map(|step| wire(&step)),
            Ok(json!({"kind": "custom_auth"}))
        );
        let mut oauth2 = oauth2(true);
        oauth2.start().unwrap();
        assert_eq!(
            oauth2.answer(
                "custom_auth",
                |exchange| &mut exchange.custom_auth,
                CustomAuthResult::NoIdentity
            ),
            Err(DriveError::NotAsked("custom_auth"))
        );
    }

    #[test]
    fn an_answer_after_the_call_finished_is_refused() {
        let mut driver = oauth2(false);
        driver.start().unwrap();
        assert_eq!(
            answer_introspection(&mut driver, introspection(true)),
            Err(DriveError::Finished)
        );
    }
}
