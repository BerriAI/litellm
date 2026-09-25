mod auth;
mod route_machine;

use std::future::Future;
use std::pin::Pin;

pub use auth::{HostTokenProvider, TokenRoute};
pub use route_machine::{ExecuteFuture, HostChannel, MachineFault, RouteMachine};

use crate::host::{HostOp, HostResult};
use crate::route::Route;

pub enum MachineStep<R: Route, C> {
    Host(HostOp<R>),
    Complete(C),
}

pub type Step<'a, M> = Pin<
    Box<
        dyn Future<
                Output = Result<
                    MachineStep<<M as Machine>::Route, <M as Machine>::Complete>,
                    <<M as Machine>::Route as Route>::Error,
                >,
            > + Send
            + 'a,
    >,
>;

pub type Interrupted<'a, M> = Pin<
    Box<
        dyn Future<
                Output = Result<<M as Machine>::Complete, <<M as Machine>::Route as Route>::Error>,
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
/// point is an op the host performs and answers with a result.
pub trait Machine: Send {
    type Route: Route;
    type Complete: Send + 'static;

    /// `None` on the first call and whenever the previous step completed without
    /// yielding an op; otherwise the result of the op last yielded.
    fn resume(&mut self, result: Option<HostResult<Self::Route>>) -> Step<'_, Self>;

    /// The host failed to perform the pending op, or the caller cancelled. The call
    /// yields no further ops.
    fn interrupt(
        &mut self,
        failure: HostFailure<<Self::Route as Route>::Error>,
    ) -> Interrupted<'_, Self>;
}
