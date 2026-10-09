use std::sync::{Mutex, PoisonError};

use crate::{
    error::HookError,
    hooks::NativeHooks,
    interceptors::{Interceptors, RawResponse, RequestContext, WireRequest},
};

/// One call's built-in hooks as an onion: the first hook added is the outermost, so it sees
/// the wire request first. Runs as the call's interceptors under a driver with no
/// runtime of its own.
pub struct NativeChain(Mutex<Vec<Box<dyn NativeHooks>>>);

impl NativeChain {
    pub fn new(hooks: impl IntoIterator<Item = Box<dyn NativeHooks>>) -> Self {
        Self(Mutex::new(hooks.into_iter().collect()))
    }
}

impl<E: From<HookError>> Interceptors<E> for NativeChain {
    async fn before_provider_request(
        &self,
        wire: WireRequest,
        context: RequestContext,
    ) -> Result<WireRequest, E> {
        let mut hooks = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        hooks
            .iter_mut()
            .try_fold(Box::new(wire), |wire, hooks| {
                hooks.before_provider_request(wire, &context)
            })
            .map(|wire| *wire)
            .map_err(E::from)
    }

    async fn after_provider_response(&self, _: RawResponse) -> Result<(), E> {
        Ok(())
    }
}
