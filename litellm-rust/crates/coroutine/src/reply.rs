use std::{
    fmt,
    future::Future,
    pin::Pin,
    sync::{Arc, Weak},
    task::{Context, Poll},
};

use tokio::sync::oneshot;

use crate::Abandoned;

/// The one way to answer a yield. Sending or dropping it settles the yield.
pub struct Reply<A> {
    slot: oneshot::Sender<A>,
    outstanding: Arc<()>,
}

impl<A> Reply<A> {
    /// An answer the yield no longer awaits is discarded.
    pub fn send(self, answer: A) {
        let _ = self.slot.send(answer);
    }

    /// Alive until this reply is sent or dropped.
    pub(crate) fn outstanding(&self) -> Weak<()> {
        Arc::downgrade(&self.outstanding)
    }
}

impl<A> fmt::Debug for Reply<A> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Reply")
    }
}

/// The waiting end of a [`Reply`].
pub struct Answer<A> {
    slot: oneshot::Receiver<A>,
}

impl<A> Future for Answer<A> {
    type Output = Result<A, Abandoned>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.slot)
            .poll(context)
            .map(|answer| answer.map_err(|_| Abandoned))
    }
}

/// A reply outside any coroutine, for answering a host operation directly.
pub fn reply<A>() -> (Reply<A>, Answer<A>) {
    let (slot, answer) = oneshot::channel();
    let reply = Reply {
        slot,
        outstanding: Arc::new(()),
    };
    (reply, Answer { slot: answer })
}
