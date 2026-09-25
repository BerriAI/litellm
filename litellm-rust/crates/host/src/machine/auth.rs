use std::sync::Arc;

use super::HostChannel;
use crate::MachineFault;
use crate::{host::Reply, protocol::Protocol};
use litellm_auth::{Error, ResolvedCredential, TokenFuture, TokenProvider, TokenProviderHandle};

/// A protocol whose host can mint credentials on the call's behalf.
pub trait TokenProtocol: Protocol {
    fn acquire_token_op(reply: Reply<ResolvedCredential>) -> Self::Op;
}

/// A [`TokenProvider`] that asks the host for each credential through the call's own
/// operation channel, so the host answers it on the caller's thread and context.
pub struct HostTokenProvider<R: Protocol> {
    channel: HostChannel<R>,
}

impl<R: Protocol> std::fmt::Debug for HostTokenProvider<R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("HostTokenProvider")
    }
}

impl<R> HostTokenProvider<R>
where
    R: TokenProtocol,
    R::Error: From<MachineFault> + std::fmt::Display,
{
    pub fn handle(channel: HostChannel<R>) -> TokenProviderHandle {
        TokenProviderHandle::new(Arc::new(Self { channel }))
    }
}

impl<R> TokenProvider for HostTokenProvider<R>
where
    R: TokenProtocol,
    R::Error: From<MachineFault> + std::fmt::Display,
{
    fn acquire(&self) -> TokenFuture<'_> {
        Box::pin(async move {
            self.channel
                .custom_op(R::acquire_token_op)
                .await
                .map_err(|error| Error::AzureTokenAcquisition(error.to_string()))
        })
    }
}
