use futures_util::future::BoxFuture;

use crate::{
    host::{Host, HostCallError},
    input::{FieldError, HookInput, RequestState},
    manifest::Manifest,
    verdict::Verdict,
};

#[derive(Debug, thiserror::Error)]
pub enum CallbackError {
    #[error(transparent)]
    Field(#[from] FieldError),
    #[error(transparent)]
    Host(#[from] HostCallError),
    #[error("{0}")]
    Failed(String),
}

pub type HookOutput = Result<Option<Verdict>, CallbackError>;

pub trait Callback: Send + Sync + 'static {
    fn manifest(&self) -> &Manifest;

    fn request_state(&self) -> Option<RequestState> {
        None
    }

    fn handle<'a>(&'a self, input: &'a HookInput) -> BoxFuture<'a, HookOutput>;

    fn handle_hosted<'a>(&'a self, input: &'a HookInput, host: Host) -> BoxFuture<'a, HookOutput> {
        drop(host);
        self.handle(input)
    }

    fn flush(&self) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }
}
