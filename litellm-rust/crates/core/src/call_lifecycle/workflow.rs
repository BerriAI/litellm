use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::execution::{HostExchange, HostExecution};
use super::host::{HostCall, HostCallFuture, HostCallStep, HostFailure, HostLifecycle, HostPhase};
use super::{CallLifecycleContext, CallLifecycleTiming};
use crate::Error;

pub type WorkflowFuture<T> = Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'static>>;

#[derive(Clone, Debug)]
pub enum LifecycleOperation<T> {
    ProjectRequest,
    Phase(HostPhase),
    ConstructResponse(Arc<T>),
    ConstructCachedResponse(serde_json::Value),
    MapFailure(Error),
    Success {
        context: CallLifecycleContext,
        response: Arc<T>,
        timing: CallLifecycleTiming,
    },
    Failure {
        context: CallLifecycleContext,
        error: Error,
        timing: CallLifecycleTiming,
    },
}

pub enum WorkflowReply<Q, R> {
    Request(Result<Q, Error>),
    Lifecycle(Result<(), HostFailure>),
    Operation(R),
}

pub trait Workflow: Send + Sync {
    type Request: Send + 'static;
    type Operation: Clone + Send + Sync + 'static;
    type Reply: Send + Sync + 'static;
    type Response: Clone + Send + Sync + 'static;

    fn operation(operation: LifecycleOperation<Self::Response>) -> Self::Operation;
    fn accepts(operation: &Self::Operation, reply: &Self::Reply) -> bool;
    fn reply(&mut self, reply: Self::Reply) -> WorkflowReply<Self::Request, Self::Reply>;
    fn start(
        &mut self,
        request: Self::Request,
        host: HostExchange<Self::Operation, Self::Reply>,
    ) -> WorkflowFuture<Self::Response>;
    fn terminal(&self) -> Option<(CallLifecycleContext, CallLifecycleTiming)> {
        None
    }
    fn cache_lookup(&mut self) -> WorkflowFuture<Option<Self::Response>> {
        Box::pin(async { Ok(None) })
    }
    fn cached_public_response(&self) -> Option<serde_json::Value> {
        None
    }
    fn flush_cache(&mut self) -> WorkflowFuture<()> {
        Box::pin(async { Ok(()) })
    }
}

pub struct LifecycleCall<W: Workflow> {
    workflow: W,
    lifecycle: HostLifecycle,
    execution: HostExecution<W::Operation, W::Reply, W::Response>,
    pending: Option<W::Operation>,
    response: Option<Arc<W::Response>>,
    error: Option<Error>,
    completed: bool,
}

impl<W: Workflow> LifecycleCall<W> {
    pub fn new(workflow: W, asynchronous: bool) -> Self {
        Self {
            workflow,
            lifecycle: HostLifecycle::new(asynchronous),
            execution: HostExecution::new(W::accepts),
            pending: None,
            response: None,
            error: None,
            completed: false,
        }
    }

    pub async fn resume(
        &mut self,
        result: Option<W::Reply>,
    ) -> Result<HostCallStep<W::Operation, W::Response>, Error> {
        if self.completed {
            return Err(Error::InvalidRequest(
                "call cannot be resumed after completion".into(),
            ));
        }
        match (&self.pending, &result) {
            (Some(operation), Some(reply)) if W::accepts(operation, reply) => {}
            (None, None) => {}
            _ => {
                return Err(Error::InvalidRequest(
                    "host reply does not match pending operation".into(),
                ));
            }
        }
        self.pending = None;
        let provider_reply = match result.map(|reply| self.workflow.reply(reply)) {
            Some(WorkflowReply::Request(Ok(request))) => {
                let future = self.workflow.start(request, self.execution.exchange());
                self.execution.start(future)?;
                None
            }
            Some(WorkflowReply::Request(Err(error))) => {
                self.accept(Err(HostFailure::Error(error)));
                None
            }
            Some(WorkflowReply::Lifecycle(result)) => {
                self.accept(result);
                None
            }
            Some(WorkflowReply::Operation(reply)) => Some(reply),
            None => None,
        };
        self.workflow.flush_cache().await?;
        if self.lifecycle.phase() == HostPhase::CacheLookup {
            match self.workflow.cache_lookup().await {
                Ok(Some(response)) => {
                    self.response = Some(Arc::new(response));
                    self.lifecycle.cache_hit();
                }
                Ok(None) => self.accept(Ok(())),
                Err(error) => self.accept(Err(HostFailure::Error(error))),
            }
        }
        if self.lifecycle.phase() == HostPhase::Execute {
            if !self.execution.started() {
                return Ok(self.host_step(W::operation(LifecycleOperation::ProjectRequest)));
            }
            match self.execution.resume(provider_reply).await {
                Ok(HostCallStep::Host(operation)) => return Ok(self.host_step(operation)),
                Ok(HostCallStep::Complete(response)) => {
                    self.response = Some(Arc::new(response));
                    self.accept(Ok(()));
                }
                Err(error) => self.accept(Err(HostFailure::Error(error))),
            }
        }
        if self.error.is_some() {
            self.execution.stop().await;
        }
        let operation = match self.lifecycle.phase() {
            HostPhase::Complete => {
                self.completed = true;
                return match self.error.take() {
                    Some(error) => Err(error),
                    None => self
                        .response
                        .take()
                        .map(Arc::unwrap_or_clone)
                        .map(HostCallStep::Complete)
                        .ok_or_else(|| {
                            Error::InvalidRequest("call completed without a response".into())
                        }),
                };
            }
            HostPhase::ConstructResponse => match self.workflow.cached_public_response() {
                Some(response) => LifecycleOperation::ConstructCachedResponse(response),
                None => LifecycleOperation::ConstructResponse(self.response()?),
            },
            HostPhase::MapFailure => LifecycleOperation::MapFailure(self.error()?),
            phase => match (phase, self.workflow.terminal()) {
                (HostPhase::Success, Some((context, timing))) => LifecycleOperation::Success {
                    context,
                    response: self.response()?,
                    timing,
                },
                (HostPhase::Failure, Some((context, timing))) => LifecycleOperation::Failure {
                    context,
                    error: self.error()?,
                    timing,
                },
                _ => LifecycleOperation::Phase(phase),
            },
        };
        Ok(self.host_step(W::operation(operation)))
    }

    fn response(&self) -> Result<Arc<W::Response>, Error> {
        self.response
            .clone()
            .ok_or_else(|| Error::InvalidRequest("missing response".into()))
    }

    fn error(&self) -> Result<Error, Error> {
        self.error
            .clone()
            .ok_or_else(|| Error::InvalidRequest("missing failure".into()))
    }

    fn host_step(&mut self, operation: W::Operation) -> HostCallStep<W::Operation, W::Response> {
        self.pending = Some(operation.clone());
        HostCallStep::Host(operation)
    }

    fn accept(&mut self, result: Result<(), HostFailure>) {
        let cancelled = matches!(&result, Err(HostFailure::Cancelled(_)));
        if let Some(error) = self.lifecycle.accept(result) {
            if cancelled {
                self.error = Some(error);
            } else {
                self.error.get_or_insert(error);
            }
            self.execution.cancel();
        }
    }

    pub async fn interrupt(
        &mut self,
        failure: HostFailure,
    ) -> Result<HostCallStep<W::Operation, W::Response>, Error> {
        if self.completed {
            return Err(Error::InvalidRequest(
                "call cannot be interrupted after completion".into(),
            ));
        }
        self.pending = None;
        self.accept(Err(failure));
        self.resume(None).await
    }
}

impl<W: Workflow> HostCall for LifecycleCall<W> {
    type Operation = W::Operation;
    type Result = W::Reply;
    type Complete = W::Response;

    fn resume(
        &mut self,
        result: Option<Self::Result>,
    ) -> HostCallFuture<'_, Self::Operation, Self::Complete> {
        Box::pin(LifecycleCall::resume(self, result))
    }

    fn interrupt(
        &mut self,
        failure: HostFailure,
    ) -> HostCallFuture<'_, Self::Operation, Self::Complete> {
        Box::pin(LifecycleCall::interrupt(self, failure))
    }
}
