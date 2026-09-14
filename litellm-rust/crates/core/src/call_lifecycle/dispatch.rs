#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsyncSuccessDelivery {
    Skip,
    Bookkeeping,
    Background,
    Deferred,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SuccessDispatch {
    SyncBookkeeping,
    SyncWorker,
    Async(AsyncSuccessDelivery),
}

pub trait SuccessFacts {
    type Error;

    fn has_fallbacks(&self) -> Result<bool, Self::Error>;
    fn callbacks_needed(&self, asynchronous: bool) -> Result<bool, Self::Error>;
    fn defers_async_logging(&self) -> Result<bool, Self::Error>;
}

pub fn success_dispatch<F: SuccessFacts>(
    asynchronous: bool,
    internal: bool,
    facts: &F,
) -> Result<SuccessDispatch, F::Error> {
    if !asynchronous {
        return Ok(if facts.callbacks_needed(false)? {
            SuccessDispatch::SyncWorker
        } else {
            SuccessDispatch::SyncBookkeeping
        });
    }
    let delivery = if internal || facts.has_fallbacks()? {
        AsyncSuccessDelivery::Skip
    } else if !facts.callbacks_needed(true)? {
        AsyncSuccessDelivery::Bookkeeping
    } else if facts.defers_async_logging()? {
        AsyncSuccessDelivery::Deferred
    } else {
        AsyncSuccessDelivery::Background
    };
    Ok(SuccessDispatch::Async(delivery))
}

pub fn failure_dispatch(asynchronous: bool, internal: bool, logger_available: bool) -> bool {
    logger_available && !(asynchronous && internal)
}

pub struct DeferredSuccess {
    pending: bool,
}

impl Default for DeferredSuccess {
    fn default() -> Self {
        Self { pending: true }
    }
}

impl DeferredSuccess {
    pub fn resolve(&mut self, accepted: bool) -> bool {
        std::mem::replace(&mut self.pending, false) && accepted
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    struct Facts {
        reads: RefCell<Vec<&'static str>>,
        fallbacks: Result<bool, &'static str>,
        callbacks: Result<bool, &'static str>,
        deferred: Result<bool, &'static str>,
    }

    impl SuccessFacts for Facts {
        type Error = &'static str;

        fn has_fallbacks(&self) -> Result<bool, Self::Error> {
            self.reads.borrow_mut().push("fallbacks");
            self.fallbacks
        }

        fn callbacks_needed(&self, asynchronous: bool) -> Result<bool, Self::Error> {
            self.reads
                .borrow_mut()
                .push(if asynchronous { "async" } else { "sync" });
            self.callbacks
        }

        fn defers_async_logging(&self) -> Result<bool, Self::Error> {
            self.reads.borrow_mut().push("deferred");
            self.deferred
        }
    }

    #[test]
    fn dispatch_reads_only_the_facts_needed_for_the_selected_delivery() {
        use AsyncSuccessDelivery as Async;
        for (asynchronous, internal, fallbacks, callbacks, deferred, expected, reads) in [
            (
                false,
                false,
                Err("unused"),
                Ok(false),
                Err("unused"),
                SuccessDispatch::SyncBookkeeping,
                vec!["sync"],
            ),
            (
                false,
                true,
                Err("unused"),
                Ok(true),
                Err("unused"),
                SuccessDispatch::SyncWorker,
                vec!["sync"],
            ),
            (
                true,
                true,
                Err("unused"),
                Err("unused"),
                Err("unused"),
                SuccessDispatch::Async(Async::Skip),
                vec![],
            ),
            (
                true,
                false,
                Ok(true),
                Err("unused"),
                Err("unused"),
                SuccessDispatch::Async(Async::Skip),
                vec!["fallbacks"],
            ),
            (
                true,
                false,
                Ok(false),
                Ok(false),
                Err("unused"),
                SuccessDispatch::Async(Async::Bookkeeping),
                vec!["fallbacks", "async"],
            ),
            (
                true,
                false,
                Ok(false),
                Ok(true),
                Ok(false),
                SuccessDispatch::Async(Async::Background),
                vec!["fallbacks", "async", "deferred"],
            ),
            (
                true,
                false,
                Ok(false),
                Ok(true),
                Ok(true),
                SuccessDispatch::Async(Async::Deferred),
                vec!["fallbacks", "async", "deferred"],
            ),
        ] {
            let facts = Facts {
                reads: RefCell::default(),
                fallbacks,
                callbacks,
                deferred,
            };
            assert_eq!(
                success_dispatch(asynchronous, internal, &facts),
                Ok(expected)
            );
            assert_eq!(*facts.reads.borrow(), reads);
        }
    }

    #[test]
    fn failed_observations_stop_dispatch_without_reading_later_facts() {
        for (fallbacks, callbacks, deferred, reads) in [
            (
                Err("failure"),
                Err("unused"),
                Err("unused"),
                vec!["fallbacks"],
            ),
            (
                Ok(false),
                Err("failure"),
                Err("unused"),
                vec!["fallbacks", "async"],
            ),
            (
                Ok(false),
                Ok(true),
                Err("failure"),
                vec!["fallbacks", "async", "deferred"],
            ),
        ] {
            let facts = Facts {
                reads: RefCell::default(),
                fallbacks,
                callbacks,
                deferred,
            };
            assert_eq!(success_dispatch(true, false, &facts), Err("failure"));
            assert_eq!(*facts.reads.borrow(), reads);
        }
    }

    #[test]
    fn failure_dispatch_preserves_sync_internal_calls_and_skips_async_internal_calls() {
        for asynchronous in [false, true] {
            for internal in [false, true] {
                assert!(!failure_dispatch(asynchronous, internal, false));
            }
        }
        assert!(failure_dispatch(false, false, true));
        assert!(failure_dispatch(false, true, true));
        assert!(failure_dispatch(true, false, true));
        assert!(!failure_dispatch(true, true, true));
    }

    #[test]
    fn deferred_success_can_be_accepted_or_rejected_only_once() {
        for accepted in [false, true] {
            let mut gate = DeferredSuccess::default();
            assert_eq!(gate.resolve(accepted), accepted);
            assert!(!gate.resolve(true));
            assert!(!gate.resolve(false));
        }
    }
}
