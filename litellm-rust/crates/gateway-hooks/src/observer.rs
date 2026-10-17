//! Observers see each request after its response ends, off the request path. Each runs on its own
//! task behind a bounded queue; when the queue is full the record is dropped and counted, so a
//! slow observer never slows a request.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use futures_util::future::BoxFuture;
use tokio::{sync::mpsc, task::JoinHandle};

use crate::facts::{Facts, Record};

/// Records an observer may have waiting before new ones are dropped.
pub const DEFAULT_QUEUE: usize = 10_000;

/// An off-path consumer of finished requests.
pub trait Observer: Send + Sync + 'static {
    /// The name the config and metrics use for this observer.
    fn name(&self) -> &'static str;

    /// The facts this observer reads from each record.
    fn reads(&self) -> Facts;

    fn observe<'a>(&'a self, record: &'a Record) -> BoxFuture<'a, ()>;

    /// Writes out anything held back; called once at shutdown after the last record.
    fn flush(&self) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }
}

/// `None` tells the observer's task to flush and stop.
type Message = Option<Arc<Record>>;

struct Lane {
    name: &'static str,
    sender: mpsc::Sender<Message>,
    dropped: Arc<AtomicU64>,
}

/// Hands records to every observer.
pub struct Observers {
    lanes: Vec<Lane>,
    tasks: Mutex<Vec<JoinHandle<()>>>,
    reads: Facts,
}

impl Observers {
    /// Starts a task per observer. Must be called inside a tokio runtime.
    pub fn start(observers: Vec<Arc<dyn Observer>>, queue: usize) -> Self {
        let mut lanes = Vec::with_capacity(observers.len());
        let mut tasks = Vec::with_capacity(observers.len());
        let mut reads = Facts::NONE;
        for observer in observers {
            reads |= observer.reads();
            let (sender, mut receiver) = mpsc::channel::<Message>(queue.max(1));
            lanes.push(Lane {
                name: observer.name(),
                sender,
                dropped: Arc::new(AtomicU64::new(0)),
            });
            tasks.push(tokio::spawn(async move {
                while let Some(Some(record)) = receiver.recv().await {
                    observer.observe(&record).await;
                }
                observer.flush().await;
            }));
        }
        Self {
            lanes,
            tasks: Mutex::new(tasks),
            reads,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.lanes.is_empty()
    }

    /// The facts some observer reads.
    pub fn reads(&self) -> Facts {
        self.reads
    }

    pub fn send(&self, record: Arc<Record>) {
        for lane in &self.lanes {
            if lane.sender.try_send(Some(Arc::clone(&record))).is_err() {
                lane.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Records each observer has dropped because its queue was full, by observer name.
    pub fn dropped(&self) -> impl Iterator<Item = (&'static str, u64)> + '_ {
        self.lanes
            .iter()
            .map(|lane| (lane.name, lane.dropped.load(Ordering::Relaxed)))
    }

    /// Lets every observer finish its queue, then flushes them. Records sent after this are
    /// dropped.
    pub async fn close(&self) {
        for lane in &self.lanes {
            let _ = lane.sender.send(None).await;
        }
        let tasks = std::mem::take(&mut *self.tasks.lock().unwrap_or_else(|e| e.into_inner()));
        for task in tasks {
            if let Err(error) = task.await {
                tracing::error!(%error, "observer task failed");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::Mutex,
        time::{Duration, SystemTime},
    };

    use bytes::Bytes;
    use http::HeaderMap;

    use super::*;
    use crate::facts::{Annotations, Outcome, RequestFacts, Route};

    fn record(status: u16) -> Arc<Record> {
        Arc::new(Record {
            request: Arc::new(RequestFacts::new(
                "r".into(),
                Route::Messages,
                None,
                None,
                "m".into(),
                Arc::new(HeaderMap::new()),
                false,
                None,
                Bytes::new(),
            )),
            outcome: Outcome {
                deployment: None,
                status,
                usage: None,
                cost: None,
                time_to_first_byte: None,
                ended_at: SystemTime::now(),
                interrupted: false,
            },
            annotations: Annotations::default(),
            response_body: None,
        })
    }

    #[derive(Default)]
    struct Collect {
        seen: Mutex<Vec<u16>>,
        flushed: Mutex<bool>,
        delay: Duration,
    }

    impl Observer for Collect {
        fn name(&self) -> &'static str {
            "collect"
        }

        fn reads(&self) -> Facts {
            Facts::USAGE
        }

        fn observe<'a>(&'a self, record: &'a Record) -> BoxFuture<'a, ()> {
            Box::pin(async move {
                tokio::time::sleep(self.delay).await;
                self.seen.lock().unwrap().push(record.outcome.status);
            })
        }

        fn flush(&self) -> BoxFuture<'_, ()> {
            Box::pin(async { *self.flushed.lock().unwrap() = true })
        }
    }

    #[tokio::test]
    async fn close_drains_then_flushes() {
        let collect = Arc::new(Collect::default());
        let observers = Observers::start(vec![collect.clone()], 16);
        assert_eq!(observers.reads(), Facts::USAGE);
        observers.send(record(200));
        observers.send(record(500));
        observers.close().await;
        assert_eq!(*collect.seen.lock().unwrap(), vec![200, 500]);
        assert!(*collect.flushed.lock().unwrap());
    }

    #[tokio::test]
    async fn a_full_queue_drops_and_counts() {
        let collect = Arc::new(Collect {
            delay: Duration::from_millis(200),
            ..Collect::default()
        });
        let observers = Observers::start(vec![collect.clone()], 1);
        for _ in 0..5 {
            observers.send(record(200));
        }
        let dropped: u64 = observers.dropped().map(|(_, count)| count).sum();
        assert!(dropped >= 3, "dropped {dropped}");
        observers.close().await;
    }
}
