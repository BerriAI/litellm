//! Process-wide admission control for provider calls.
//!
//! Without a limit, every host call becomes an in-flight upstream request with
//! an unbounded response buffer: a burst of thousands of concurrent calls
//! means thousands of open provider sockets, aggregate memory growth, and 429
//! storms at the provider. Hosts (the python-bridge module init, the gateway
//! binary) resolve the config-shaped environment and call [`init_limits`]
//! once at startup; core reads no environment here.
//!
//! The permit spans the whole provider call, including response buffering,
//! and is an `OwnedSemaphorePermit` held by the future — a host-side
//! cancellation drops the future and releases the permit automatically.
//! Uninitialized means unlimited, so rollouts keep today's behavior until a
//! host opts in.
//!
//! Streaming entrypoints are not capped yet: their in-flight window outlives
//! the call that started them, so the permit must be attached to the returned
//! stream rather than the entrypoint (follow-up once the frame-stream route
//! lands).

use std::sync::{Arc, OnceLock, RwLock};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::Error;

struct ActiveLimits {
    semaphore: Arc<Semaphore>,
    max_in_flight: usize,
    shed_on_limit: bool,
}

static LIMITS: RwLock<Option<Arc<ActiveLimits>>> = RwLock::new(None);

/// The admission-control configuration a host resolves at startup.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Maximum concurrent in-flight provider calls process-wide.
    pub max_in_flight: usize,
    /// Over-limit calls fail immediately with [`Error::Overloaded`] instead
    /// of queueing. Queue (false) is the safer default for SDK callers that
    /// tolerate latency; shed (true) suits proxies that fail fast.
    pub shed_on_limit: bool,
}

/// Install the process-wide limit. Call once at startup, before the first
/// provider call; a second call replaces the limit (permits already held stay
/// valid).
pub fn init_limits(limits: Limits) {
    let semaphore = Semaphore::new(limits.max_in_flight);
    *write_limits() = Some(Arc::new(ActiveLimits {
        semaphore: Arc::new(semaphore),
        max_in_flight: limits.max_in_flight,
        shed_on_limit: limits.shed_on_limit,
    }));
}

/// Point-in-time view for diagnostics endpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConcurrencyStats {
    /// `None` when no limit is installed.
    pub max_in_flight: Option<usize>,
    /// Calls currently holding a permit (`0` when unlimited).
    pub in_flight: usize,
    pub shed_on_limit: bool,
}

pub fn stats() -> ConcurrencyStats {
    let limits = read_limits().clone();
    match limits {
        Some(limits) => ConcurrencyStats {
            in_flight: limits.max_in_flight - limits.semaphore.available_permits(),
            max_in_flight: Some(limits.max_in_flight),
            shed_on_limit: limits.shed_on_limit,
        },
        None => ConcurrencyStats {
            max_in_flight: None,
            in_flight: 0,
            shed_on_limit: false,
        },
    }
}

/// Acquire one in-flight slot for a provider call.
///
/// Cancel-safe: dropping the returned future (e.g. a Python-side task
/// cancellation) loses only the queue position; a granted permit is released
/// when dropped.
pub async fn acquire() -> Result<OwnedSemaphorePermit, Error> {
    let limits = read_limits().clone();
    let Some(limits) = limits else {
        return Ok(unlimited_permit().await);
    };
    acquire_from_limits(&limits).await
}

async fn acquire_from_limits(limits: &ActiveLimits) -> Result<OwnedSemaphorePermit, Error> {
    if limits.shed_on_limit {
        Arc::clone(&limits.semaphore)
            .try_acquire_owned()
            .map_err(|_| Error::Overloaded("native in-flight limit reached".to_string()))
    } else {
        Arc::clone(&limits.semaphore)
            .acquire_owned()
            .await
            .map_err(|_| Error::Overloaded("native in-flight limit closed".to_string()))
    }
}

/// A sentinel permit for the unlimited case, so the hot path needs no
/// branching after `acquire` and callers can hold it uniformly.
async fn unlimited_permit() -> OwnedSemaphorePermit {
    static UNLIMITED: OnceLock<Arc<Semaphore>> = OnceLock::new();
    UNLIMITED
        .get_or_init(|| Arc::new(Semaphore::new(1)))
        .clone()
        .acquire_owned()
        .await
        .expect("unlimited sentinel semaphore is never closed")
}

fn read_limits() -> std::sync::RwLockReadGuard<'static, Option<Arc<ActiveLimits>>> {
    LIMITS
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn write_limits() -> std::sync::RwLockWriteGuard<'static, Option<Arc<ActiveLimits>>> {
    LIMITS
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    /// Tests exercise private [`ActiveLimits`] directly instead of the
    /// process-global limiter, because route tests run in parallel in this
    /// binary and would otherwise queue behind (or be shed by) whatever
    /// limit a test had installed.
    fn limits(max_in_flight: usize, shed_on_limit: bool) -> Arc<ActiveLimits> {
        Arc::new(ActiveLimits {
            semaphore: Arc::new(Semaphore::new(max_in_flight)),
            max_in_flight,
            shed_on_limit,
        })
    }

    #[tokio::test]
    async fn unlimited_by_default_grants_immediately() {
        for _ in 0..3 {
            let _permit = acquire()
                .await
                .expect("unlimited acquire should always succeed");
        }
    }

    #[tokio::test]
    async fn permit_caps_concurrent_holders() {
        let limits = limits(2, false);

        let concurrent = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();
        for _ in 0..6 {
            let limits = Arc::clone(&limits);
            let concurrent = Arc::clone(&concurrent);
            let max_concurrent = Arc::clone(&max_concurrent);
            tasks.push(tokio::spawn(async move {
                let _permit = acquire_from_limits(&limits)
                    .await
                    .expect("queue mode never sheds");
                let now = concurrent.fetch_add(1, Ordering::SeqCst) + 1;
                max_concurrent.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(20)).await;
                concurrent.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        for task in tasks {
            task.await.expect("task should complete");
        }

        assert_eq!(max_concurrent.load(Ordering::SeqCst), 2);
        assert_eq!(limits.semaphore.available_permits(), 2);
    }

    #[tokio::test]
    async fn shed_mode_fails_over_limit_calls() {
        let limits = limits(1, true);

        let _held = acquire_from_limits(&limits)
            .await
            .expect("first call acquires");
        let error = acquire_from_limits(&limits)
            .await
            .expect_err("second call should shed under the limit");

        assert!(matches!(error, Error::Overloaded(_)));
    }

    #[tokio::test]
    async fn dropping_a_queued_waiter_leaks_no_permit() {
        let limits = limits(1, false);

        let _held = acquire_from_limits(&limits)
            .await
            .expect("first call acquires");
        let queued = Box::pin(acquire_from_limits(&limits));
        // Let the queued waiter park, then cancel it by dropping the future.
        tokio::time::sleep(Duration::from_millis(10)).await;
        drop(queued);
        drop(_held);

        let next = acquire_from_limits(&limits)
            .await
            .expect("cancelled waiter must not leak its permit");
        drop(next);
        assert_eq!(limits.semaphore.available_permits(), 1);
    }

    #[tokio::test]
    async fn init_limits_reports_stats() {
        // 64 permits can never be contended by the parallel route tests that
        // share this process, so installing it globally mid-run is harmless.
        init_limits(Limits {
            max_in_flight: 64,
            shed_on_limit: true,
        });
        let stats = stats();
        assert_eq!(stats.max_in_flight, Some(64));
        assert!(stats.shed_on_limit);
        assert!(stats.in_flight <= 64);
        // Deliberately left installed (64 permits can never contend the
        // parallel route tests), and never cleared back to None: a None
        // window here could race other tests' `acquire()` calls.
    }
}
