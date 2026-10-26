use std::sync::{
    OnceLock,
    atomic::{AtomicPtr, Ordering},
};

pub(crate) struct ProcessLocal<T> {
    entry: AtomicPtr<Entry<T>>,
}

struct Entry<T> {
    pid: u32,
    value: OnceLock<T>,
}

impl<T> ProcessLocal<T> {
    pub(crate) fn new() -> Self {
        Self {
            entry: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    pub(crate) fn get_or_init(&self, init: impl FnOnce() -> T) -> &T {
        let pid = std::process::id();
        let mut pointer = self.entry.load(Ordering::Acquire);
        loop {
            // Published entries remain alive until Drop. Inherited entries are never touched or freed.
            if !pointer.is_null() && unsafe { (*pointer).pid } == pid {
                return unsafe { &(*pointer).value }.get_or_init(init);
            }
            let fresh = Box::into_raw(Box::new(Entry {
                pid,
                value: OnceLock::new(),
            }));
            match self
                .entry
                .compare_exchange(pointer, fresh, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return unsafe { &(*fresh).value }.get_or_init(init),
                Err(current) => {
                    // This candidate was never published, so no other thread can reference it.
                    drop(unsafe { Box::from_raw(fresh) });
                    pointer = current;
                }
            }
        }
    }
}

// The atomic pointer publishes only process-local OnceLocks; their values must be shareable.
unsafe impl<T: Send + Sync> Send for ProcessLocal<T> {}
unsafe impl<T: Send + Sync> Sync for ProcessLocal<T> {}

impl<T> Drop for ProcessLocal<T> {
    fn drop(&mut self) {
        let pointer = *self.entry.get_mut();
        // Exclusive access rules out live readers. Parent locks and runtime handles must be leaked.
        if !pointer.is_null() && unsafe { (*pointer).pid } == std::process::id() {
            drop(unsafe { Box::from_raw(pointer) });
        }
    }
}
