mod auth;
mod call_machine;

use std::future::Future;
use std::pin::Pin;

pub use auth::{HostTokenProvider, TokenProtocol};
pub use call_machine::{CallMachine, ExecuteFuture, HostChannel, MachineFault};

use crate::host::HostOp;
use crate::protocol::Protocol;

pub enum MachineStep<R: Protocol, C> {
    Host(HostOp<R>),
    Complete(C),
}

pub type Step<'a, M> = Pin<
    Box<
        dyn Future<
                Output = Result<
                    MachineStep<<M as Machine>::Protocol, <M as Machine>::Complete>,
                    <<M as Machine>::Protocol as Protocol>::Error,
                >,
            > + Send
            + 'a,
    >,
>;

pub type Interrupted<'a, M> = Pin<
    Box<
        dyn Future<
                Output = Result<
                    <M as Machine>::Complete,
                    <<M as Machine>::Protocol as Protocol>::Error,
                >,
            > + Send
            + 'a,
    >,
>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostFailure<E> {
    Error(E),
    Cancelled(E),
}

impl<E> HostFailure<E> {
    pub fn into_error(self) -> E {
        match self {
            Self::Error(error) | Self::Cancelled(error) => error,
        }
    }
}

/// A resumable call. Core implements it per route; a host drives it. Every suspension
/// point is an op the host performs and answers through the op's own reply before it
/// resumes the call again.
pub trait Machine: Send {
    type Protocol: Protocol;
    type Complete: Send + 'static;

    fn resume(&mut self) -> Step<'_, Self>;

    /// The host failed to perform the pending op, or the caller cancelled. The call
    /// yields no further ops.
    fn interrupt(
        &mut self,
        failure: HostFailure<<Self::Protocol as Protocol>::Error>,
    ) -> Interrupted<'_, Self>;
}
