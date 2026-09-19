use std::sync::atomic::{AtomicU32, Ordering};

const UNSET: u32 = 0;

/// Decides which process may use the Tokio runtime. Its worker threads do not survive
/// `fork()`: a child forked after they started hangs on its first native call. The gate turns
/// both halves of that hazard into errors, keyed by pid so a fork needs no hook to be seen:
/// a process reserved for forking can never start the runtime, and a child of a process that
/// did start it is refused instead of hanging.
pub(crate) struct ForkGate {
    runtime_pid: AtomicU32,
    fork_only_pid: AtomicU32,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Refused {
    ReservedForForking,
    ForkedAfterStart,
}

#[derive(Debug, PartialEq, Eq)]
pub struct RuntimeAlreadyStarted;

impl ForkGate {
    pub(crate) const fn new() -> Self {
        Self {
            runtime_pid: AtomicU32::new(UNSET),
            fork_only_pid: AtomicU32::new(UNSET),
        }
    }

    /// Claims the runtime for `pid`. Claim first, then look for a reservation: `reserve` does
    /// the mirror image, so when the two race at least one of them sees the other.
    pub(crate) fn enter(&self, pid: u32) -> Result<(), Refused> {
        match self
            .runtime_pid
            .compare_exchange(UNSET, pid, Ordering::SeqCst, Ordering::SeqCst)
        {
            Err(owner) if owner != pid => return Err(Refused::ForkedAfterStart),
            _ => {}
        }

        if self.fork_only_pid.load(Ordering::SeqCst) == pid {
            // Nothing was started, so the workers forked from here must still find it unclaimed.
            let _ =
                self.runtime_pid
                    .compare_exchange(pid, UNSET, Ordering::SeqCst, Ordering::SeqCst);
            return Err(Refused::ReservedForForking);
        }

        Ok(())
    }

    /// Reserves `pid` for forking. Reserve first, then look for a started runtime: `enter` does
    /// the mirror image, so when the two race at least one of them sees the other. A refused
    /// reservation leaves the gate exactly as it was, so a process already running the runtime
    /// keeps refusing the children it forks.
    pub(crate) fn reserve(&self, pid: u32) -> Result<(), RuntimeAlreadyStarted> {
        self.fork_only_pid.store(pid, Ordering::SeqCst);
        if self.runtime_pid.load(Ordering::SeqCst) == pid {
            // Nothing may change for a process that already runs the runtime: its children
            // must still be refused.
            let _ =
                self.fork_only_pid
                    .compare_exchange(pid, UNSET, Ordering::SeqCst, Ordering::SeqCst);
            return Err(RuntimeAlreadyStarted);
        }
        Ok(())
    }

    pub(crate) fn started(&self, pid: u32) -> bool {
        self.runtime_pid.load(Ordering::SeqCst) == pid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MASTER: u32 = 100;
    const WORKER: u32 = 101;

    #[test]
    fn unreserved_process_starts_the_runtime_and_stays_started() {
        let gate = ForkGate::new();

        assert!(!gate.started(MASTER));
        assert_eq!(gate.enter(MASTER), Ok(()));
        assert_eq!(gate.enter(MASTER), Ok(()));
        assert!(gate.started(MASTER));
    }

    #[test]
    fn reserved_process_can_never_start_the_runtime() {
        let gate = ForkGate::new();

        assert_eq!(gate.reserve(MASTER), Ok(()));
        assert_eq!(gate.enter(MASTER), Err(Refused::ReservedForForking));
        assert_eq!(gate.enter(MASTER), Err(Refused::ReservedForForking));
        assert!(!gate.started(MASTER));
    }

    #[test]
    fn workers_forked_from_a_reserved_process_start_their_own_runtime() {
        let gate = ForkGate::new();
        gate.reserve(MASTER).unwrap();
        gate.enter(MASTER).unwrap_err();

        assert_eq!(gate.enter(WORKER), Ok(()));
        assert!(gate.started(WORKER));
    }

    #[test]
    fn reserving_after_the_runtime_started_is_refused() {
        let gate = ForkGate::new();
        gate.enter(MASTER).unwrap();

        assert_eq!(gate.reserve(MASTER), Err(RuntimeAlreadyStarted));
    }

    #[test]
    fn a_refused_reservation_leaves_the_runtime_claimed_and_its_children_refused() {
        let gate = ForkGate::new();
        gate.enter(MASTER).unwrap();

        assert_eq!(gate.reserve(MASTER), Err(RuntimeAlreadyStarted));
        assert_eq!(gate.enter(MASTER), Ok(()));
        assert!(gate.started(MASTER));
        assert_eq!(gate.enter(WORKER), Err(Refused::ForkedAfterStart));
    }

    #[test]
    fn child_forked_after_the_runtime_started_is_refused_instead_of_hanging() {
        let gate = ForkGate::new();
        gate.enter(MASTER).unwrap();

        assert_eq!(gate.enter(WORKER), Err(Refused::ForkedAfterStart));
        assert!(!gate.started(WORKER));
        assert_eq!(gate.enter(MASTER), Ok(()));
    }
}
