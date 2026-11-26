//! The bounded queue that takes records off runtime threads and delivers them from the blocking pool.

use std::{
    iter,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use litellm_tracing::Record;
use serde_json::Value;
use tokio::{runtime::Handle, sync::mpsc};

const BATCH: usize = 64;

/// Where background records end up. Both methods run on the blocking pool, so they may wait for the GIL and do I/O.
pub(super) trait Delivery: Send + Sync + 'static {
    fn record(&self, record: &Record);
    fn lost(&self, count: u64);
}

/// A record with the text bytes it holds, counted once so they can be given back when it is delivered.
struct Held {
    record: Record,
    bytes: usize,
}

/// The bytes of text a record holds: its message and its fields' names and strings. A debug record can carry a whole
/// request body, so the queue is bounded by this as well as by a count.
fn text_bytes(record: &Record) -> usize {
    record.message.len()
        + record
            .fields
            .iter()
            .map(|(name, value)| name.len() + value.as_str().map_or(size_of::<Value>(), str::len))
            .sum::<usize>()
}

/// Records from runtime threads, handed to one task that delivers them in order off the async workers. The queue
/// holds at most `capacity` records and `byte_budget` bytes of their text. A record past either is dropped and
/// counted, so a slow log handler costs lines, never a worker or unbounded memory.
pub(super) struct Background<D> {
    delivery: Arc<D>,
    capacity: usize,
    byte_budget: usize,
    sender: OnceLock<mpsc::Sender<Held>>,
    held_bytes: Arc<AtomicUsize>,
    lost: Arc<AtomicU64>,
}

impl<D: Delivery> Background<D> {
    pub(super) fn new(delivery: D, capacity: usize, byte_budget: usize) -> Self {
        Self {
            delivery: Arc::new(delivery),
            capacity,
            byte_budget,
            sender: OnceLock::new(),
            held_bytes: Arc::new(AtomicUsize::new(0)),
            lost: Arc::new(AtomicU64::new(0)),
        }
    }

    pub(super) fn push(&self, runtime: &Handle, record: Record) {
        let sender = self.sender.get_or_init(|| {
            let (sender, records) = mpsc::channel(self.capacity);
            drop(litellm_call_scope::spawn_detached_on(
                runtime,
                drain(
                    records,
                    Arc::clone(&self.held_bytes),
                    Arc::clone(&self.lost),
                    Arc::clone(&self.delivery),
                ),
            ));
            sender
        });
        let bytes = text_bytes(&record);
        let held = self.held_bytes.fetch_add(bytes, Ordering::Relaxed);
        if held.saturating_add(bytes) > self.byte_budget
            || sender.try_send(Held { record, bytes }).is_err()
        {
            self.held_bytes.fetch_sub(bytes, Ordering::Relaxed);
            self.lost.fetch_add(1, Ordering::Relaxed);
        }
    }
}

async fn drain<D: Delivery>(
    mut records: mpsc::Receiver<Held>,
    held_bytes: Arc<AtomicUsize>,
    lost: Arc<AtomicU64>,
    delivery: Arc<D>,
) {
    while let Some(first) = records.recv().await {
        let batch: Vec<Held> = iter::once(first)
            .chain(iter::from_fn(|| records.try_recv().ok()))
            .take(BATCH)
            .collect();
        let lost = lost.swap(0, Ordering::Relaxed);
        let delivery = Arc::clone(&delivery);
        let held_bytes = Arc::clone(&held_bytes);
        drop(
            tokio::task::spawn_blocking(move || {
                batch.into_iter().for_each(|held| {
                    delivery.record(&held.record);
                    held_bytes.fetch_sub(held.bytes, Ordering::Relaxed);
                });
                if lost > 0 {
                    delivery.lost(lost);
                }
            })
            .await,
        );
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc,
            mpsc::{Receiver, Sender, channel},
        },
        time::Duration,
    };

    use litellm_tracing::{Logger, Metadata, Record, Sink};
    use parking_lot::Mutex;
    use tokio::runtime::Handle;

    use super::{Background, Delivery};

    struct Probe {
        started: Sender<()>,
        gate: Mutex<Receiver<()>>,
        records: Sender<String>,
        lost: Sender<u64>,
    }

    struct Probed {
        started: Receiver<()>,
        release: Sender<()>,
        records: Receiver<String>,
        lost: Receiver<u64>,
    }

    fn probe() -> (Probe, Probed) {
        let (started_sender, started) = channel();
        let (release, gate) = channel();
        let (records_sender, records) = channel();
        let (lost_sender, lost) = channel();
        (
            Probe {
                started: started_sender,
                gate: Mutex::new(gate),
                records: records_sender,
                lost: lost_sender,
            },
            Probed {
                started,
                release,
                records,
                lost,
            },
        )
    }

    impl Delivery for Probe {
        fn record(&self, record: &Record) {
            let _ = self.started.send(());
            let delivered = match self.gate.lock().recv_timeout(Duration::from_secs(2)) {
                Ok(()) => record.message.clone(),
                Err(_) => "delivery held the runtime worker".to_owned(),
            };
            let _ = self.records.send(delivered);
        }

        fn lost(&self, count: u64) {
            let _ = self.lost.send(count);
        }
    }

    struct Queued(Arc<Background<Probe>>, Handle);

    impl Sink for Queued {
        fn enabled(&self, _: &Metadata<'_>) -> bool {
            true
        }

        fn emit(&self, record: &Record) {
            self.0.push(&self.1, record.clone());
        }
    }

    fn single_worker() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .build()
            .unwrap()
    }

    #[rstest::rstest]
    fn a_background_record_is_delivered_without_holding_a_runtime_worker() {
        let runtime = single_worker();
        let (probe, probed) = probe();
        let logger = Logger::new(Queued(
            Arc::new(Background::new(probe, 8, usize::MAX)),
            runtime.handle().clone(),
        ));
        drop(runtime.spawn(async move { logger.scope(|| tracing::warn!("first")) }));
        let release = probed.release.clone();
        drop(runtime.spawn(async move { release.send(()).unwrap() }));

        assert_eq!(
            probed.records.recv_timeout(Duration::from_secs(5)).unwrap(),
            "first"
        );
    }

    #[rstest::rstest]
    fn a_full_background_queue_drops_records_and_reports_how_many() {
        let runtime = single_worker();
        let (probe, probed) = probe();
        let logger = Logger::new(Queued(
            Arc::new(Background::new(probe, 1, usize::MAX)),
            runtime.handle().clone(),
        ));
        let wait = Duration::from_secs(5);
        logger.scope(|| tracing::warn!("delivering"));
        probed.started.recv_timeout(wait).unwrap();
        logger.scope(|| {
            tracing::warn!("queued");
            tracing::warn!("dropped");
            tracing::warn!("dropped too");
        });
        probed.release.send(()).unwrap();
        probed.release.send(()).unwrap();

        assert_eq!(probed.records.recv_timeout(wait).unwrap(), "delivering");
        assert_eq!(probed.records.recv_timeout(wait).unwrap(), "queued");
        assert_eq!(probed.lost.recv_timeout(wait).unwrap(), 2);
        assert!(
            probed
                .records
                .recv_timeout(Duration::from_millis(200))
                .is_err()
        );
    }

    #[rstest::rstest]
    fn records_past_the_byte_budget_are_dropped_and_reported_while_room_for_more_records_remains() {
        let runtime = single_worker();
        let (probe, probed) = probe();
        let logger = Logger::new(Queued(
            Arc::new(Background::new(
                probe,
                8,
                "delivering".len() + "queued".len(),
            )),
            runtime.handle().clone(),
        ));
        let wait = Duration::from_secs(5);
        logger.scope(|| tracing::warn!("delivering"));
        probed.started.recv_timeout(wait).unwrap();
        logger.scope(|| {
            tracing::warn!("queued");
            tracing::warn!("x");
        });
        probed.release.send(()).unwrap();
        probed.release.send(()).unwrap();

        assert_eq!(probed.records.recv_timeout(wait).unwrap(), "delivering");
        assert_eq!(probed.records.recv_timeout(wait).unwrap(), "queued");
        assert_eq!(probed.lost.recv_timeout(wait).unwrap(), 1);

        logger.scope(|| tracing::warn!("next"));
        probed.release.send(()).unwrap();
        assert_eq!(probed.records.recv_timeout(wait).unwrap(), "next");
    }
}
