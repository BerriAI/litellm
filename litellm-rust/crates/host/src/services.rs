use crate::protocol::Protocol;
use std::{convert::Infallible, future::Future};

pub trait HostCallHandler<P: Protocol>: Send + Sync {
    fn handle_host_call(
        &self,
        call: P::HostCall,
    ) -> impl Future<Output = Result<(), P::Error>> + Send;
}

impl<P: Protocol<HostCall = Infallible>> HostCallHandler<P> for () {
    async fn handle_host_call(&self, call: Infallible) -> Result<(), P::Error> {
        match call {}
    }
}
