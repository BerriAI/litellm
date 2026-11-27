use std::future::Future;

use litellm_host::protocol::Protocol;

pub trait HostCallHandler<P: Protocol>: Send + Sync {
    fn handle_host_call(
        &self,
        call: P::HostCall,
    ) -> impl Future<Output = Result<(), P::Error>> + Send;
}

impl<P: Protocol, T: HostCallHandler<P> + ?Sized> HostCallHandler<P> for &T {
    fn handle_host_call(
        &self,
        call: P::HostCall,
    ) -> impl Future<Output = Result<(), P::Error>> + Send {
        (**self).handle_host_call(call)
    }
}

/// A host that serves no host calls. It drops each one unanswered, which the call sees as
/// abandoned.
impl<P: Protocol> HostCallHandler<P> for () {
    async fn handle_host_call(&self, call: P::HostCall) -> Result<(), P::Error> {
        drop(call);
        Ok(())
    }
}
