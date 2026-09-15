use std::future::Future;

use tokio::sync::{mpsc, oneshot};

use super::host::HostCallStep;
use crate::Error;

struct PendingOperation<O, R> {
    operation: O,
    reply: oneshot::Sender<R>,
}

pub struct HostExchange<O, R> {
    sender: mpsc::UnboundedSender<PendingOperation<O, R>>,
}

impl<O, R> Clone for HostExchange<O, R> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
        }
    }
}

impl<O, R> std::fmt::Debug for HostExchange<O, R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HostExchange")
            .finish_non_exhaustive()
    }
}

impl<O, R> HostExchange<O, R> {
    pub async fn invoke(&self, operation: O) -> Result<R, Error> {
        let (reply, receiver) = oneshot::channel();
        self.sender
            .send(PendingOperation { operation, reply })
            .map_err(|_| Error::InvalidRequest("host driver was abandoned".into()))?;
        receiver
            .await
            .map_err(|_| Error::InvalidRequest("host operation was abandoned".into()))
    }
}

pub struct HostExecution<O, R, T> {
    exchange: HostExchange<O, R>,
    operations: mpsc::UnboundedReceiver<PendingOperation<O, R>>,
    pending: Option<PendingOperation<O, R>>,
    task: Option<tokio::task::JoinHandle<Result<T, Error>>>,
    completed: bool,
    accepts: fn(&O, &R) -> bool,
}

impl<O: Clone, R, T: Send + 'static> HostExecution<O, R, T> {
    pub fn new(accepts: fn(&O, &R) -> bool) -> Self {
        let (sender, operations) = mpsc::unbounded_channel();
        Self {
            exchange: HostExchange { sender },
            operations,
            pending: None,
            task: None,
            completed: false,
            accepts,
        }
    }

    pub fn exchange(&self) -> HostExchange<O, R> {
        self.exchange.clone()
    }

    pub fn started(&self) -> bool {
        self.task.is_some() || self.completed
    }

    pub fn start(
        &mut self,
        future: impl Future<Output = Result<T, Error>> + Send + 'static,
    ) -> Result<(), Error> {
        if self.started() {
            return Err(Error::InvalidRequest(
                "host execution already started".into(),
            ));
        }
        self.task = Some(tokio::spawn(future));
        Ok(())
    }

    pub async fn resume(&mut self, result: Option<R>) -> Result<HostCallStep<O, T>, Error> {
        if self.completed {
            return Err(Error::InvalidRequest(
                "call cannot be resumed after completion".into(),
            ));
        }
        match (&self.pending, &result) {
            (Some(pending), Some(reply)) if (self.accepts)(&pending.operation, reply) => {}
            (None, None) => {}
            _ => {
                return Err(Error::InvalidRequest(
                    "host reply does not match pending operation".into(),
                ));
            }
        }
        if let (Some(pending), Some(reply)) = (self.pending.take(), result) {
            pending
                .reply
                .send(reply)
                .map_err(|_| Error::InvalidRequest("host operation was abandoned".into()))?;
        }
        let task = self
            .task
            .as_mut()
            .ok_or_else(|| Error::InvalidRequest("host execution has not started".into()))?;
        tokio::select! {
            operation = self.operations.recv() => {
                let pending = operation.ok_or_else(|| Error::InvalidRequest("host operation channel closed".into()))?;
                let operation = pending.operation.clone();
                self.pending = Some(pending);
                Ok(HostCallStep::Host(operation))
            }
            result = task => {
                self.task = None;
                self.completed = true;
                result.map_err(|error| Error::Network(format!("execution task failed: {error}")))?.map(HostCallStep::Complete)
            }
        }
    }

    pub fn cancel(&mut self) {
        self.pending = None;
        if let Some(task) = &self.task {
            task.abort();
        }
    }

    pub async fn stop(&mut self) {
        self.cancel();
        if let Some(task) = self.task.as_mut() {
            let _ = task.await;
        }
        self.task = None;
        self.completed = true;
    }
}

impl<O, R, T> Drop for HostExecution<O, R, T> {
    fn drop(&mut self) {
        if let Some(task) = &self.task {
            task.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    #[tokio::test]
    async fn wrong_and_missing_replies_preserve_the_pending_exchange() {
        let mut execution =
            HostExecution::new(|operation: &u32, reply: &u32| *reply == *operation + 1);
        let exchange = execution.exchange();
        execution
            .start(async move { exchange.invoke(40).await })
            .unwrap();
        assert!(matches!(
            execution.resume(None).await.unwrap(),
            HostCallStep::Host(40)
        ));
        assert!(execution.resume(Some(99)).await.is_err());
        assert!(execution.resume(None).await.is_err());
        assert!(matches!(
            execution.resume(Some(41)).await.unwrap(),
            HostCallStep::Complete(41)
        ));
        assert!(execution.resume(None).await.is_err());
    }

    struct Capture(Arc<AtomicBool>);
    impl Drop for Capture {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn stop_waits_for_provider_captures_to_drop() {
        let dropped = Arc::new(AtomicBool::new(false));
        let capture = Capture(dropped.clone());
        let mut execution = HostExecution::<u32, u32, u32>::new(|_, _| true);
        let exchange = execution.exchange();
        execution
            .start(async move {
                let _capture = capture;
                exchange.invoke(1).await
            })
            .unwrap();
        assert!(matches!(
            execution.resume(None).await.unwrap(),
            HostCallStep::Host(1)
        ));
        execution.stop().await;
        assert!(dropped.load(Ordering::SeqCst));
        assert!(execution.resume(Some(2)).await.is_err());
    }

    #[tokio::test]
    async fn provider_panics_are_terminal_errors() {
        let mut execution = HostExecution::<u32, u32, u32>::new(|_, _| true);
        execution.start(async { panic!("provider panic") }).unwrap();
        assert!(matches!(
            execution.resume(None).await,
            Err(Error::Network(_))
        ));
        assert!(execution.resume(None).await.is_err());
    }
}
