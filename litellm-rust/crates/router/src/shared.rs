use std::{future::Future, pin::Pin, time::Duration};

use crate::Failure;

pub type CountFuture<'a> = Pin<Box<dyn Future<Output = Option<u32>> + Send + 'a>>;

/// Health shared with the other replicas of a gateway, so a deployment one replica finds
/// failing cools down on all of them. The router never waits on it for a call that succeeds.
pub trait SharedHealth: Send + Sync {
    /// Tells the other replicas `deployment` cooled down for `cooldown`. Must not block: the
    /// router calls it between a failed attempt and its retry.
    fn cooled(&self, deployment: &str, cooldown: Duration, failure: Failure);

    /// Adds one failure to `deployment`'s count across replicas for `allowed_fails`, in a
    /// window of `window`, and returns the new count; `None` when the count is unavailable, and
    /// the router then counts this replica's failures alone.
    fn count_failure<'a>(&'a self, deployment: &'a str, window: Duration) -> CountFuture<'a>;
}
