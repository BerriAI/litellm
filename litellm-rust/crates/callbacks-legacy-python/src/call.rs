//! The caller's public call as the legacy `Logging` contract sees it. Legacy callbacks
//! receive these exact objects and may mutate them, so the call keeps them for its whole
//! lifetime. No other callback host has that obligation, which is why nothing outside
//! this crate holds them.

use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyDict, PyTuple},
};

pub struct PublicCall {
    args: Py<PyTuple>,
    kwargs: Py<PyDict>,
}

impl PublicCall {
    /// Copies the keyword arguments once, so the legacy path's rewrites never reach the
    /// caller's own dict while every value keeps its identity.
    pub fn capture(args: &Bound<'_, PyTuple>, kwargs: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            args: args.clone().unbind(),
            kwargs: kwargs.copy()?.unbind(),
        })
    }

    pub fn arguments(&self, py: Python<'_>) -> Py<PyDict> {
        self.kwargs.clone_ref(py)
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

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.args)?;
        visit.call(&self.kwargs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{local, local_dict};

    fn capture<'py>(py: Python<'py>, source: &std::ffi::CStr) -> (PublicCall, Bound<'py, PyDict>) {
        let locals = PyDict::new(py);
        py.run(source, Some(&locals), Some(&locals)).unwrap();
        let call =
            PublicCall::capture(&PyTuple::empty(py), &local_dict(&locals, "kwargs")).unwrap();
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
                call.kwargs()
                    .bind(py)
                    .get_item("pages")
                    .unwrap()
                    .unwrap()
                    .is(local(&locals, "pages"))
            );
        });
    }
}
