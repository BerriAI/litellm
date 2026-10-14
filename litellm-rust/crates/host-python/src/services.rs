use std::future::Future;

use crate::{Awaiter, InvokeError, PythonOwned};
use litellm_host::protocol::Protocol;

pub trait PythonHostCalls<P: Protocol>: PythonOwned {
    fn handle_host_call(
        &self,
        task: &dyn Awaiter,
        call: P::HostCall,
    ) -> impl Future<Output = Result<(), InvokeError<P::Error>>> + Send;
}
