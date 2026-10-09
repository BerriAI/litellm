//! The caller's public call as the legacy `Logging` contract sees it. Legacy callbacks
//! receive these exact objects and may mutate them, so the call keeps them for its whole
//! lifetime. No other callback host has that obligation, which is why nothing outside
//! this crate holds them, and why the keyword layer and the signature base are merged
//! here and nowhere else.

use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyDict, PyTuple},
};

pub struct PublicCall {
    args: Py<PyTuple>,
    kwargs: Py<PyDict>,
    bound: Py<PyDict>,
}

impl PublicCall {
    /// Copies the keyword arguments once, so the legacy path's rewrites never reach the
    /// caller's own dict while every value keeps its identity.
    pub fn capture(
        bound: &Bound<'_, PyDict>,
        args: &Bound<'_, PyTuple>,
        kwargs: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        Ok(Self {
            args: args.clone().unbind(),
            kwargs: kwargs.copy()?.unbind(),
            bound: bound.clone().unbind(),
        })
    }

    pub fn arguments(&self, py: Python<'_>) -> Py<PyDict> {
        self.kwargs.clone_ref(py)
    }

    pub(crate) fn base(&self, py: Python<'_>) -> Py<PyDict> {
        self.bound.clone_ref(py)
    }

    pub(crate) fn args(&self) -> &Py<PyTuple> {
        &self.args
    }

    /// The keyword view the legacy path currently reads: the caller's copy until
    /// `function_setup`, then each rewrite (setup, deployment hook, the driver's preflight)
    /// in turn.
    pub(crate) fn kwargs(&self) -> &Py<PyDict> {
        &self.kwargs
    }

    pub(crate) fn set_kwargs(&mut self, kwargs: Py<PyDict>) {
        self.kwargs = kwargs;
    }

    pub(crate) fn lookup<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        match self.kwargs.bind(py).get_item(name)? {
            Some(value) => Ok(Some(value)),
            None => self.bound.bind(py).get_item(name),
        }
    }

    #[cfg(test)]
    pub(crate) fn resolved<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        resolved(self.bound.bind(py), self.kwargs.bind(py))
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.args)?;
        visit.call(&self.kwargs)?;
        visit.call(&self.bound)
    }
}

/// What `original_function(*args, **kwargs)` sees: the signature base with the keyword
/// layer laid over it, so a rewritten keyword wins and a deleted keyword falls back to the
/// signature default. Shallow, like the Python path: nested values stay shared with the
/// caller.
pub(crate) fn resolved<'py>(
    bound: &Bound<'py, PyDict>,
    kwargs: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let merged = bound.copy()?;
    merged.update(kwargs.as_mapping())?;
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::test_support::{local, local_dict};

    fn capture<'py>(py: Python<'py>, source: &std::ffi::CStr) -> (PublicCall, Bound<'py, PyDict>) {
        let locals = PyDict::new(py);
        py.run(source, Some(&locals), Some(&locals)).unwrap();
        let call = PublicCall::capture(
            &local_dict(&locals, "bound"),
            &PyTuple::empty(py),
            &local_dict(&locals, "kwargs"),
        )
        .unwrap();
        (call, locals)
    }

    #[rstest::rstest]
    fn capture_copies_the_keyword_dict_without_copying_its_values() {
        Python::initialize();
        Python::attach(|py| {
            let (call, locals) = capture(
                py,
                c"
pages = [0]
bound = {'pages': [1]}
kwargs = {'pages': pages}
",
            );
            let caller = local_dict(&locals, "kwargs");
            call.kwargs()
                .bind(py)
                .set_item("litellm_call_id", "call")
                .unwrap();
            assert!(!caller.contains("litellm_call_id").unwrap());
            assert!(
                call.lookup(py, "pages")
                    .unwrap()
                    .unwrap()
                    .is(local(&locals, "pages"))
            );
        });
    }

    #[rstest]
    #[case::keyword_wins(
        "{'api_key': 'keyword'}",
        "{'api_key': 'bound'}",
        Some(Some("keyword"))
    )]
    #[case::explicit_none_wins("{'api_key': None}", "{'api_key': 'bound'}", Some(None))]
    #[case::bound_fallback("{}", "{'api_key': 'bound'}", Some(Some("bound")))]
    #[case::missing("{}", "{}", None)]
    fn lookup_prefers_the_keyword_and_falls_back_to_bound(
        #[case] kwargs: &str,
        #[case] bound: &str,
        #[case] expected: Option<Option<&str>>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let source = format!("kwargs = {kwargs}\nbound = {bound}");
            let (call, _) = capture(py, &std::ffi::CString::new(source).unwrap());
            let value = call
                .lookup(py, "api_key")
                .unwrap()
                .map(|value| value.extract::<Option<String>>().unwrap());
            assert_eq!(value, expected.map(|value| value.map(str::to_owned)));
        });
    }

    #[rstest]
    #[case::keyword_wins(
        "{'api_key': 'keyword'}",
        "{'api_key': 'bound'}",
        Some(Some("keyword"))
    )]
    #[case::explicit_none_wins("{'api_key': None}", "{'api_key': 'bound'}", Some(None))]
    #[case::bound_default("{}", "{'api_key': 'bound'}", Some(Some("bound")))]
    #[case::keyword_only("{'api_key': 'keyword'}", "{}", Some(Some("keyword")))]
    #[case::missing("{}", "{}", None)]
    fn resolved_lays_the_keywords_over_the_base(
        #[case] kwargs: &str,
        #[case] bound: &str,
        #[case] expected: Option<Option<&str>>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let source = format!("kwargs = {kwargs}\nbound = {bound}");
            let (call, _) = capture(py, &std::ffi::CString::new(source).unwrap());
            let value = call
                .resolved(py)
                .unwrap()
                .get_item("api_key")
                .unwrap()
                .map(|value| value.extract::<Option<String>>().unwrap());
            assert_eq!(value, expected.map(|value| value.map(str::to_owned)));
        });
    }

    #[rstest]
    fn resolved_is_a_fresh_dict_whose_values_are_the_callers_objects() {
        Python::initialize();
        Python::attach(|py| {
            let (call, locals) = capture(
                py,
                c"
document = {'type': 'file'}
bound = {'document': document, 'timeout': 600}
kwargs = {'pages': [0]}
",
            );
            let resolved = call.resolved(py).unwrap();
            assert!(
                resolved
                    .get_item("document")
                    .unwrap()
                    .unwrap()
                    .is(local(&locals, "document"))
            );
            resolved.set_item("timeout", 1).unwrap();
            assert_eq!(
                local_dict(&locals, "bound")
                    .get_item("timeout")
                    .unwrap()
                    .unwrap()
                    .extract::<i64>()
                    .unwrap(),
                600
            );
            assert!(!call.kwargs().bind(py).contains("timeout").unwrap());
        });
    }
}
