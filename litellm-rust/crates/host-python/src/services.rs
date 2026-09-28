use crate::{InvokeError, PythonOwned};
use litellm_host::protocol::Protocol;
use pyo3::prelude::*;

pub trait PythonHostCalls<P: Protocol>: PythonOwned {
    fn handle_host_call(
        &mut self,
        py: Python<'_>,
        call: P::HostCall,
    ) -> Result<(), InvokeError<P::Error>>;
}
