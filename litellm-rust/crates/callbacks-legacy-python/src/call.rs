//! The caller's public call as the legacy `Logging` contract sees it. Legacy callbacks
//! receive these exact objects and may mutate them, so the call keeps them for its whole
//! lifetime. No other callback host has that obligation, which is why nothing outside
//! this crate holds them.

use litellm_host::{machine::Machine, protocol::Protocol};
use litellm_host_python::{Preflight, ProtocolHost, lookup, run_call};
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyDict, PyTuple},
};

use crate::{LegacyLogging, LegacySurface};

pub struct PublicCall {
    args: Py<PyTuple>,
    kwargs: Py<PyDict>,
    request: Py<PyAny>,
}

impl PublicCall {
    /// Copies the keyword arguments once, so the legacy path's rewrites never reach the
    /// caller's own dict while every value keeps its identity.
    pub fn capture(
        request: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        Ok(Self {
            args: args.clone().unbind(),
            kwargs: kwargs.copy()?.unbind(),
            request: request.clone().unbind(),
        })
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
        lookup(self.kwargs.bind(py), self.request.bind(py), name)
    }

    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.args)?;
        visit.call(&self.kwargs)?;
        visit.call(&self.request)
    }
}

/// Runs one native call under the legacy `Logging` contract: the protocol host projects from
/// the keyword view the contract prepares and `preflight` rewrites, and the contract
/// observes the call.
pub fn run_legacy_call<H, M>(
    py: Python<'_>,
    surface: LegacySurface,
    call: PublicCall,
    machine: M,
    host: H,
    preflight: Preflight,
    asynchronous: bool,
) -> PyResult<Py<PyAny>>
where
    H: ProtocolHost + 'static,
    M: Machine<Protocol = H::Protocol, Complete = <H::Protocol as Protocol>::Response> + 'static,
{
    let arguments = call.kwargs.clone_ref(py);
    run_call(
        py,
        machine,
        host,
        Box::new(LegacyLogging::new(py, surface, call, asynchronous)),
        preflight,
        arguments,
        asynchronous,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture<'py>(py: Python<'py>, source: &std::ffi::CStr) -> (PublicCall, Bound<'py, PyDict>) {
        let locals = PyDict::new(py);
        py.run(source, Some(&locals), Some(&locals)).unwrap();
        let request = locals.get_item("request").unwrap().unwrap();
        let kwargs = locals
            .get_item("kwargs")
            .unwrap()
            .unwrap()
            .cast_into::<PyDict>()
            .unwrap();
        let call = PublicCall::capture(&request, &PyTuple::empty(py), &kwargs).unwrap();
        (call, locals)
    }

    #[test]
    fn capture_copies_the_keyword_dict_without_copying_its_values() {
        Python::initialize();
        Python::attach(|py| {
            let (call, locals) = capture(
                py,
                c"
pages = [0]
class Request:
    pass
request = Request()
kwargs = {'pages': pages}
",
            );
            let caller = locals
                .get_item("kwargs")
                .unwrap()
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            call.kwargs()
                .bind(py)
                .set_item("litellm_call_id", "call")
                .unwrap();
            assert!(!caller.contains("litellm_call_id").unwrap());
            let pages = locals.get_item("pages").unwrap().unwrap();
            assert!(call.lookup(py, "pages").unwrap().unwrap().is(&pages));
        });
    }
}
