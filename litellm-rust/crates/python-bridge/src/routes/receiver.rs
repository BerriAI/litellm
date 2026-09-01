use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures_util::{Stream, StreamExt, pin_mut};
use litellm_core::Error;
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinHandle;

const BRIDGE_CHANNEL_CAPACITY: usize = 1;

struct ReceiverState<T> {
    receiver: Mutex<mpsc::Receiver<Result<T, Error>>>,
    reading: AtomicBool,
    closed: AtomicBool,
    producer: std::sync::Mutex<Option<JoinHandle<()>>>,
}

impl<T> Drop for ReceiverState<T> {
    fn drop(&mut self) {
        if let Ok(producer) = self.producer.get_mut()
            && let Some(producer) = producer.take()
        {
            producer.abort();
        }
    }
}

#[derive(Clone)]
pub(super) struct BridgeReceiver<T> {
    state: Arc<ReceiverState<T>>,
}

struct ReadGuard<'a>(&'a AtomicBool);

impl Drop for ReadGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl<T: Send + 'static> BridgeReceiver<T> {
    pub(super) fn from_stream<S>(stream: S) -> Self
    where
        S: Stream<Item = Result<T, Error>> + Send + 'static,
    {
        let (sender, receiver) = mpsc::channel(BRIDGE_CHANNEL_CAPACITY);
        let producer = pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
            pin_mut!(stream);
            while let Some(item) = stream.next().await {
                let terminal = item.is_err();
                if sender.send(item).await.is_err() || terminal {
                    return;
                }
            }
        });
        Self {
            state: Arc::new(ReceiverState {
                receiver: Mutex::new(receiver),
                reading: AtomicBool::new(false),
                closed: AtomicBool::new(false),
                producer: std::sync::Mutex::new(Some(producer)),
            }),
        }
    }

    pub(super) async fn next(&self) -> Result<Option<T>, Error> {
        if self.state.closed.load(Ordering::Acquire) {
            return Ok(None);
        }
        if self.state.reading.swap(true, Ordering::AcqRel) {
            return Err(Error::InvalidRequest(
                "native stream does not support concurrent reads".to_string(),
            ));
        }
        let _guard = ReadGuard(&self.state.reading);
        match self.state.receiver.lock().await.recv().await {
            Some(Ok(item)) => Ok(Some(item)),
            Some(Err(error)) => Err(error),
            None => Ok(None),
        }
    }

    pub(super) fn close(&self) {
        if self.state.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        if let Ok(mut producer) = self.state.producer.lock()
            && let Some(producer) = producer.take()
        {
            producer.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use futures_util::stream;

    use super::*;

    struct DropFlag(Arc<AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn receiver_preserves_items_and_terminal_error() {
        let receiver = BridgeReceiver::from_stream(stream::iter([
            Ok(vec![1_u8]),
            Err(Error::Network("broken".to_string())),
            Ok(vec![2_u8]),
        ]));

        assert_eq!(receiver.next().await.expect("first item"), Some(vec![1]));
        assert!(matches!(
            receiver.next().await,
            Err(Error::Network(message)) if message == "broken"
        ));
        assert_eq!(receiver.next().await.expect("closed after error"), None);
    }

    #[tokio::test]
    async fn close_unblocks_a_pending_read() {
        let receiver = BridgeReceiver::<Vec<u8>>::from_stream(stream::pending());
        let pending = {
            let receiver = receiver.clone();
            tokio::spawn(async move { receiver.next().await })
        };
        tokio::time::sleep(Duration::from_millis(10)).await;
        receiver.close();

        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), pending)
                .await
                .expect("read should unblock")
                .expect("task should finish")
                .expect("close is clean"),
            None
        );
    }

    #[tokio::test]
    async fn capacity_one_stops_the_producer_from_draining_the_source() {
        let polls = Arc::new(AtomicUsize::new(0));
        let source_polls = polls.clone();
        let source = stream::poll_fn(move |_| {
            let item = source_polls.fetch_add(1, Ordering::SeqCst);
            std::task::Poll::Ready(Some(Ok(item)))
        });
        let receiver = BridgeReceiver::from_stream(source);

        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(polls.load(Ordering::SeqCst), 2);
        assert_eq!(receiver.next().await.expect("first item"), Some(0));
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(polls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn concurrent_reads_are_rejected() {
        let receiver = BridgeReceiver::<Vec<u8>>::from_stream(stream::pending());
        let pending = {
            let receiver = receiver.clone();
            tokio::spawn(async move { receiver.next().await })
        };
        tokio::time::sleep(Duration::from_millis(10)).await;

        assert!(matches!(
            receiver.next().await,
            Err(Error::InvalidRequest(message))
                if message == "native stream does not support concurrent reads"
        ));
        receiver.close();
        pending
            .await
            .expect("pending read task")
            .expect("clean close");
    }

    #[tokio::test]
    async fn dropping_the_last_receiver_cancels_the_producer() {
        let dropped = Arc::new(AtomicBool::new(false));
        let producer_dropped = dropped.clone();
        let source = stream::once(async move {
            let _flag = DropFlag(producer_dropped);
            std::future::pending::<Result<Vec<u8>, Error>>().await
        });
        let receiver = BridgeReceiver::from_stream(source);
        tokio::time::sleep(Duration::from_millis(10)).await;

        drop(receiver);
        tokio::time::timeout(Duration::from_secs(1), async {
            while !dropped.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("producer future should be dropped");
    }
}
