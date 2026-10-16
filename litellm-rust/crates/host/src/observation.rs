use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use tokio::sync::mpsc;

use crate::lifecycle::CallEvent;

#[derive(Clone)]
pub struct ObservationSender {
    sender: mpsc::Sender<CallEvent>,
    dropped: Arc<AtomicU64>,
}

impl ObservationSender {
    pub fn emit(&self, event: CallEvent) {
        if self.sender.try_send(event).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn dropped_events(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

pub fn observation_channel(
    capacity: NonZeroUsize,
) -> (ObservationSender, mpsc::Receiver<CallEvent>) {
    let (sender, receiver) = mpsc::channel(capacity.get());
    (
        ObservationSender {
            sender,
            dropped: Arc::new(AtomicU64::new(0)),
        },
        receiver,
    )
}
