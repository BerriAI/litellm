use crate::{InvokeError, PythonOwned};
use litellm_host::protocol::Protocol;
use pyo3::prelude::*;

pub trait PythonHostCalls<P: Protocol>: PythonOwned {
    fn handle_host_call(
        &mut self,
        py: Python<'_>,
        call: P::HostCall,
    ) -> Result<(), InvokeError<P::Error>>;

    fn begin_host_call(
        &mut self,
        py: Python<'_>,
        call: P::HostCall,
    ) -> Result<Option<Py<PyAny>>, InvokeError<P::Error>> {
        self.handle_host_call(py, call).map(|()| None)
    }

    fn resume_host_call(
        &mut self,
        _: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> Result<Option<Py<PyAny>>, InvokeError<P::Error>> {
        result.map(|_| None).map_err(InvokeError::Python)
    }
}
