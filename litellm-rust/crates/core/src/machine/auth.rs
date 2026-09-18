use std::sync::Arc;

use litellm_auth::{Error, ResolvedCredential, TokenFuture, TokenProvider, TokenProviderHandle};
use litellm_host::route::Route;

use super::{HostChannel, MachineFault};

/// A route whose host can mint credentials on the call's behalf.
pub trait TokenRoute: Route {
    fn acquire_token_op() -> Self::Op;
    fn token_credential(result: Self::OpResult) -> Option<ResolvedCredential>;
}

/// A [`TokenProvider`] that asks the host for each credential through the call's own
/// operation channel, so the host answers it on the caller's thread and context.
pub struct HostTokenProvider<R: Route> {
    channel: HostChannel<R>,
}

impl<R: Route> std::fmt::Debug for HostTokenProvider<R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("HostTokenProvider")
    }
}

impl<R> HostTokenProvider<R>
where
    R: TokenRoute,
    R::Error: From<MachineFault> + std::fmt::Display,
{
    pub fn handle(channel: HostChannel<R>) -> TokenProviderHandle {
        TokenProviderHandle::new(Arc::new(Self { channel }))
    }
}

impl<R> TokenProvider for HostTokenProvider<R>
where
    R: TokenRoute,
    R::Error: From<MachineFault> + std::fmt::Display,
{
    fn acquire(&self) -> TokenFuture<'_> {
        Box::pin(async move {
            let result = self
                .channel
                .route(R::acquire_token_op())
                .await
                .map_err(|error| Error::AzureTokenAcquisition(error.to_string()))?;
            R::token_credential(result).ok_or_else(|| {
                Error::AzureTokenAcquisition("invalid token provider host result".into())
            })
        })
    }
}
