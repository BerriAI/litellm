use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::time;

use super::types::{SpendEntry, SpendUpdateBatch, SpendUpdateItem};

/// Trait for flushing batched spend updates to a persistent store.
/// Implementations handle the actual DB/Redis writes.
pub trait SpendFlush: Send + Sync + 'static {
    fn flush(&self, batch: SpendUpdateBatch) -> impl std::future::Future<Output = ()> + Send;
}

/// A no-op flush sink that discards all data. Used for testing or when
/// no persistent store is configured.
pub struct NullFlush;

impl SpendFlush for NullFlush {
    async fn flush(&self, _batch: SpendUpdateBatch) {}
}

/// An in-memory flush sink that collects batches for inspection.
/// Used for testing.
#[derive(Clone, Default)]
pub struct MemoryFlush {
    batches: Arc<tokio::sync::Mutex<Vec<SpendUpdateBatch>>>,
}

impl MemoryFlush {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn get_batches(&self) -> Vec<SpendUpdateBatch> {
        self.batches.lock().await.clone()
    }

    pub async fn total_entries(&self) -> usize {
        self.batches
            .lock()
            .await
            .iter()
            .map(|b| b.total_entries())
            .sum()
    }
}

impl SpendFlush for MemoryFlush {
    async fn flush(&self, batch: SpendUpdateBatch) {
        self.batches.lock().await.push(batch);
    }
}

enum SpendMessage {
    Update(SpendUpdateItem),
    Log(Box<SpendEntry>),
}

/// The background spend worker. Receives spend entries via a channel,
/// batches them, and flushes periodically or when the batch is full.
pub struct SpendWorker {
    tx: mpsc::Sender<SpendMessage>,
}

impl SpendWorker {
    /// Spawn a new background spend worker.
    ///
    /// - `batch_size`: flush when this many entries accumulate (default 100)
    /// - `flush_interval`: flush at least this often (default 100ms)
    /// - `flush`: the sink to write batched data to
    pub fn spawn<F: SpendFlush>(batch_size: usize, flush_interval: Duration, flush: F) -> Self {
        let (tx, rx) = mpsc::channel(10_000);
        tokio::spawn(worker_loop(rx, batch_size, flush_interval, flush));
        Self { tx }
    }

    /// Record a spend update for a specific entity (key, user, team, etc.).
    /// Non-blocking: sends to the background worker via channel.
    pub fn record_update(&self, item: SpendUpdateItem) {
        if self.tx.try_send(SpendMessage::Update(item)).is_err() {
            eprintln!("[warn] spend worker channel full, dropping spend update");
        }
    }

    /// Record a spend log entry (the per-request log row).
    /// Non-blocking: sends to the background worker via channel.
    pub fn record_log(&self, entry: SpendEntry) {
        if self
            .tx
            .try_send(SpendMessage::Log(Box::new(entry)))
            .is_err()
        {
            eprintln!("[warn] spend worker channel full, dropping spend log");
        }
    }
}

async fn worker_loop<F: SpendFlush>(
    mut rx: mpsc::Receiver<SpendMessage>,
    batch_size: usize,
    flush_interval: Duration,
    flush: F,
) {
    let mut batch = SpendUpdateBatch::new();
    let mut interval = time::interval(flush_interval);
    interval.tick().await;

    loop {
        tokio::select! {
            msg = rx.recv() => {
                match msg {
                    Some(SpendMessage::Update(item)) => batch.add_update(item),
                    Some(SpendMessage::Log(entry)) => batch.add_spend_log(*entry),
                    None => {
                        if !batch.is_empty() {
                            flush.flush(batch).await;
                        }
                        return;
                    }
                }
                if batch.total_entries() >= batch_size {
                    flush.flush(std::mem::take(&mut batch)).await;
                    batch = SpendUpdateBatch::new();
                }
            }
            _ = interval.tick() => {
                if !batch.is_empty() {
                    flush.flush(std::mem::take(&mut batch)).await;
                    batch = SpendUpdateBatch::new();
                }
            }
        }
    }
}
