use std::{
    future::{Future, poll_fn},
    pin::Pin,
    sync::Weak,
    task::{Context, Poll},
};

use tokio::sync::mpsc;

use crate::{Co, ResumeError, co::Request};

/// What one `resume` produced, as in [`std::ops::CoroutineState`].
#[derive(Debug, PartialEq, Eq)]
pub enum CoroutineState<Y, C> {
    Yielded(Y),
    Complete(C),
}

type Body<C> = Pin<Box<dyn Future<Output = C> + Send>>;

enum Step<Y, C> {
    Yielded(Request<Y>),
    Complete(C),
}

fn queued<Y>(
    yields: &mut mpsc::UnboundedReceiver<Request<Y>>,
    context: &mut Context<'_>,
) -> Option<Request<Y>> {
    match yields.poll_recv(context) {
        Poll::Ready(request) => request,
        Poll::Pending => None,
    }
}

pub struct Coroutine<Y, C> {
    body: Option<Body<C>>,
    yields: mpsc::UnboundedReceiver<Request<Y>>,
    outstanding: Weak<()>,
}

impl<Y, C> Coroutine<Y, C> {
    /// Builds the body from `producer`. Nothing runs until the first `resume`.
    pub fn new<F>(producer: impl FnOnce(Co<Y>) -> F) -> Self
    where
        F: Future<Output = C> + Send + 'static,
    {
        let (sender, yields) = mpsc::unbounded_channel();
        Self {
            body: Some(Box::pin(producer(Co::new(sender)))),
            yields,
            outstanding: Weak::new(),
        }
    }

    pub async fn resume(&mut self) -> Result<CoroutineState<Y, C>, ResumeError> {
        let Some(body) = self.body.as_mut() else {
            return Err(ResumeError::Finished);
        };
        if self.outstanding.strong_count() > 0 {
            return Err(ResumeError::Unanswered);
        }
        let yields = &mut self.yields;
        let step = poll_fn(|context| {
            if let Some(request) = queued(yields, context) {
                return Poll::Ready(Step::Yielded(request));
            }
            if let Poll::Ready(output) = body.as_mut().poll(context) {
                return Poll::Ready(Step::Complete(output));
            }
            queued(yields, context)
                .map_or(Poll::Pending, |request| Poll::Ready(Step::Yielded(request)))
        })
        .await;
        match step {
            Step::Yielded(Request { value, outstanding }) => {
                self.outstanding = outstanding;
                Ok(CoroutineState::Yielded(value))
            }
            Step::Complete(output) => {
                self.cancel();
                Ok(CoroutineState::Complete(output))
            }
        }
    }

    /// Drops the body and fails every yield still waiting, or yet to be made, with
    /// [`Abandoned`](crate::Abandoned).
    pub fn cancel(&mut self) {
        self.body = None;
        self.yields.close();
        while self.yields.try_recv().is_ok() {}
    }
}
