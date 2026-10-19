use std::sync::Weak;

use tokio::sync::mpsc;

use crate::{Abandoned, Reply, reply};

pub(crate) struct Request<Y> {
    pub(crate) value: Y,
    pub(crate) outstanding: Weak<()>,
}

/// The body's handle for yielding, `genawaiter`'s `Co`.
pub struct Co<Y> {
    yields: mpsc::UnboundedSender<Request<Y>>,
}

impl<Y> Clone for Co<Y> {
    fn clone(&self) -> Self {
        Self {
            yields: self.yields.clone(),
        }
    }
}

impl<Y> Co<Y> {
    pub(crate) fn new(yields: mpsc::UnboundedSender<Request<Y>>) -> Self {
        Self { yields }
    }

    /// Yields the value `ask` builds around a fresh [`Reply`] and waits for its answer.
    pub async fn yield_<A>(&self, ask: impl FnOnce(Reply<A>) -> Y) -> Result<A, Abandoned> {
        let (reply, answer) = reply();
        let outstanding = reply.outstanding();
        self.yields
            .send(Request {
                value: ask(reply),
                outstanding,
            })
            .map_err(|_| Abandoned)?;
        answer.await
    }
}
