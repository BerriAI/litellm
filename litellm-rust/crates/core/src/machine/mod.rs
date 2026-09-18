//! The one machine every route runs on: it owns the route's provider future, polls it in
//! place, and turns the host operations that future requests into [`Machine`] steps. No
//! task is spawned; dropping the machine drops the in-flight call.

mod auth;

use std::{future::Future, pin::Pin};

pub use auth::{HostTokenProvider, TokenRoute};
use litellm_callbacks::{
    event::{CallEvent, RequestContext, WireRequest},
    host::{HostOp, HostResult},
    machine::{HostFailure, Interrupted, Machine, MachineStep, Step},
    route::Route,
};
use tokio::sync::{mpsc, oneshot};

/// The machine's own failures, distinct from anything the provider call reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MachineFault {
    /// The host driver went away while the call was waiting on it.
    Abandoned,
    /// The host answered out of turn: a result with nothing pending, or nothing when a
    /// result was pending.
    Protocol(&'static str),
    /// The host answered a route operation with the wrong result variant.
    Mismatch,
}

pub type ExecuteFuture<R> =
    Pin<Box<dyn Future<Output = Result<<R as Route>::Response, <R as Route>::Error>> + Send>>;

struct PendingOp<R: Route> {
    op: HostOp<R>,
    reply: oneshot::Sender<HostResult<R>>,
}

/// The provider side of the machine: how the in-flight call reaches its host.
pub struct HostChannel<R: Route> {
    ops: mpsc::UnboundedSender<PendingOp<R>>,
}

impl<R: Route> Clone for HostChannel<R> {
    fn clone(&self) -> Self {
        Self {
            ops: self.ops.clone(),
        }
    }
}

impl<R: Route> HostChannel<R>
where
    R::Error: From<MachineFault>,
{
    async fn invoke(&self, op: HostOp<R>) -> Result<HostResult<R>, R::Error> {
        let (reply, answer) = oneshot::channel();
        self.ops
            .send(PendingOp { op, reply })
            .map_err(|_| MachineFault::Abandoned)?;
        answer.await.map_err(|_| MachineFault::Abandoned.into())
    }

    pub async fn route(&self, op: R::Op) -> Result<R::OpResult, R::Error> {
        match self.invoke(HostOp::Route(op)).await? {
            HostResult::Route(result) => Ok(result),
            _ => Err(MachineFault::Mismatch.into()),
        }
    }

    pub async fn before_send(
        &self,
        wire: WireRequest,
        context: RequestContext,
    ) -> Result<WireRequest, R::Error> {
        let op = HostOp::BeforeSend {
            wire: Box::new(wire),
            context: Box::new(context),
        };
        match self.invoke(op).await? {
            HostResult::BeforeSend(wire) => Ok(*wire),
            _ => Err(MachineFault::Mismatch.into()),
        }
    }

    pub async fn emit(&self, event: CallEvent) -> Result<(), R::Error> {
        match self.invoke(HostOp::Emit(event)).await? {
            HostResult::Emitted => Ok(()),
            _ => Err(MachineFault::Mismatch.into()),
        }
    }
}

enum Execution<R: Route> {
    Unstarted(Box<dyn FnOnce(HostChannel<R>) -> ExecuteFuture<R> + Send>),
    Running(ExecuteFuture<R>),
    Done,
}

pub struct RouteMachine<R: Route> {
    execution: Execution<R>,
    ops: mpsc::UnboundedReceiver<PendingOp<R>>,
    channel: HostChannel<R>,
    reply: Option<oneshot::Sender<HostResult<R>>>,
}

impl<R: Route> RouteMachine<R>
where
    R::Error: From<MachineFault>,
{
    pub fn new(execute: impl FnOnce(HostChannel<R>) -> ExecuteFuture<R> + Send + 'static) -> Self {
        let (ops_tx, ops) = mpsc::unbounded_channel();
        Self {
            execution: Execution::Unstarted(Box::new(execute)),
            ops,
            channel: HostChannel { ops: ops_tx },
            reply: None,
        }
    }

    async fn step(
        &mut self,
        result: Option<HostResult<R>>,
    ) -> Result<MachineStep<R, R::Response>, R::Error> {
        match (self.reply.take(), result) {
            (Some(reply), Some(result)) => {
                reply
                    .send(result)
                    .map_err(|_| MachineFault::Protocol("the call stopped waiting on the host"))?;
            }
            (None, None) if matches!(self.execution, Execution::Unstarted(_)) => {}
            (Some(reply), None) => {
                self.reply = Some(reply);
                return Err(MachineFault::Protocol("host operation result is required").into());
            }
            (None, Some(_)) => {
                return Err(MachineFault::Protocol("unexpected host operation result").into());
            }
            (None, None) => {
                return Err(
                    MachineFault::Protocol("call cannot be resumed after completion").into(),
                );
            }
        }
        if let Execution::Unstarted(_) = self.execution {
            let Execution::Unstarted(start) =
                std::mem::replace(&mut self.execution, Execution::Done)
            else {
                unreachable!()
            };
            self.execution = Execution::Running(start(self.channel.clone()));
        }
        let Execution::Running(future) = &mut self.execution else {
            return Err(MachineFault::Protocol("call cannot be resumed after completion").into());
        };
        tokio::select! {
            biased;
            pending = self.ops.recv() => {
                let pending = pending.ok_or(MachineFault::Abandoned)?;
                self.reply = Some(pending.reply);
                Ok(MachineStep::Host(pending.op))
            }
            outcome = future => {
                self.execution = Execution::Done;
                outcome.map(MachineStep::Complete)
            }
        }
    }
}

impl<R: Route> Machine for RouteMachine<R>
where
    R::Error: From<MachineFault>,
{
    type Route = R;
    type Complete = R::Response;

    fn resume(&mut self, result: Option<HostResult<R>>) -> Step<'_, Self> {
        Box::pin(self.step(result))
    }

    fn interrupt(&mut self, failure: HostFailure<R::Error>) -> Interrupted<'_, Self> {
        self.reply = None;
        self.execution = Execution::Done;
        Box::pin(async move { Err(failure.into_error()) })
    }
}
