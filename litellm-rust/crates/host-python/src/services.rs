use std::future::Future;

use crate::{InvokeError, PythonOwned, PythonTask};
use litellm_host::protocol::Protocol;

/// Answers the host calls a route yields, on the caller's task. An answer is sent through
/// the call's own reply; a Python exception interrupts the call as raised, a native error
/// resumes the route with it.
pub trait PythonHostCalls<P: Protocol>: PythonOwned {
    fn handle_host_call(
        &mut self,
        task: &PythonTask,
        call: P::HostCall,
    ) -> impl Future<Output = Result<(), InvokeError<P::Error>>> + Send;
}
