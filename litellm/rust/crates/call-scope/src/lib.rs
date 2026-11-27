//! The call a task runs for, kept in one task-local that every crate can reach, so a crate below
//! the gateway can spawn work for the call without depending on the gateway.
//!
//! The gateway scopes each call's record with [`scope`] and reads it back with [`with`]. Every task
//! spawned in production code goes through one of two functions, and clippy refuses a bare
//! `tokio::spawn`:
//!
//! - [`spawn_in_call`] for work done for the call that spawns it, which keeps the call
//! - [`spawn_detached`] for a task that outlives any call, such as a process-wide listener or a
//!   loop started lazily by the first request. Carrying the call into one would keep that call's
//!   record alive for the life of the process and let the task write into a call long finished

mod error;

use std::{any::Any, future::Future, sync::Arc};

use tokio::{
    runtime::Handle,
    task::{JoinHandle, futures::TaskLocalFuture},
};

pub use self::error::Error;

/// What a call keeps. The gateway keeps its `CallRecord` here.
pub type Call = Arc<dyn Any + Send + Sync>;

tokio::task_local! {
    static CALL: Call;
}

/// Runs `future` as part of `call`.
pub fn scope<F: Future>(call: Call, future: F) -> TaskLocalFuture<Call, F> {
    CALL.scope(call, future)
}

/// Runs `run` as part of `call`, for a poll or a drop outside the call's task.
pub fn sync_scope<R>(call: Call, run: impl FnOnce() -> R) -> R {
    CALL.sync_scope(call, run)
}

/// Runs `look` on the call this code runs in, which must hold a `T`.
pub fn with<T: Any, R>(look: impl FnOnce(&T) -> R) -> Result<R, Error> {
    CALL.try_with(|call| {
        call.downcast_ref::<T>().map(look).ok_or(Error::OtherKind {
            expected: std::any::type_name::<T>(),
        })
    })
    .map_err(|_| Error::Outside)?
}

/// Spawns `future` on the current runtime as part of the call this code runs in. Outside a call it
/// is spawned as it is.
pub fn spawn_in_call<F>(future: F) -> JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    spawn_in_call_on(&Handle::current(), future)
}

/// [`spawn_in_call`] on `runtime`, for code that may run off the runtime's threads.
pub fn spawn_in_call_on<F>(runtime: &Handle, future: F) -> JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    match CALL.try_get() {
        #[expect(
            clippy::disallowed_methods,
            reason = "a sanctioned spawn, which carries the call"
        )]
        Ok(call) => runtime.spawn(CALL.scope(call, future)),
        #[expect(
            clippy::disallowed_methods,
            reason = "a sanctioned spawn, outside a call there is nothing to carry"
        )]
        Err(_) => runtime.spawn(future),
    }
}

/// Spawns `future` on the current runtime outside any call, for a task that outlives the call that
/// may have started it.
pub fn spawn_detached<F>(future: F) -> JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    spawn_detached_on(&Handle::current(), future)
}

/// [`spawn_detached`] on `runtime`, for code that may run off the runtime's threads.
pub fn spawn_detached_on<F>(runtime: &Handle, future: F) -> JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    #[expect(
        clippy::disallowed_methods,
        reason = "a sanctioned spawn, which leaves the call behind on purpose"
    )]
    runtime.spawn(future)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;

    use super::*;

    fn call(name: &'static str) -> Call {
        Arc::new(name)
    }

    fn name() -> Result<&'static str, Error> {
        with(|name: &&'static str| *name)
    }

    #[rstest]
    #[tokio::test]
    async fn a_task_spawned_in_a_call_keeps_it() {
        let seen = scope(call("first"), async {
            spawn_in_call(async { name() }).await
        })
        .await
        .unwrap();
        assert_eq!(seen.unwrap(), "first");
    }

    #[rstest]
    #[tokio::test]
    async fn a_detached_task_leaves_the_call_behind() {
        let seen = scope(call("first"), async {
            spawn_detached(async { name() }).await
        })
        .await
        .unwrap();
        assert!(matches!(seen, Err(Error::Outside)));
    }

    #[rstest]
    #[tokio::test]
    async fn a_task_spawned_outside_a_call_has_none() {
        let seen = spawn_in_call(async { name() }).await.unwrap();
        assert!(matches!(seen, Err(Error::Outside)));
    }

    #[rstest]
    #[tokio::test]
    async fn a_call_holding_another_type_is_refused() {
        let seen = scope(Arc::new(7_u8), async { name() }).await;
        assert!(matches!(seen, Err(Error::OtherKind { .. })));
    }
}
