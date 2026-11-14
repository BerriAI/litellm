use std::{collections::VecDeque, sync::Arc};

use bytes::Bytes;
use futures_util::{FutureExt, Stream, StreamExt, stream::BoxStream};
use std::sync::Mutex;
use tokio::{sync::Notify, task::JoinHandle};

const READ_SIZE: usize = 16 * 1024;
const LOW_WATER: usize = 64 * 1024;
const HIGH_WATER: usize = 2 * LOW_WATER;
const SETTLE_YIELDS: usize = 64;

struct Buffer<E> {
    chunks: VecDeque<Bytes>,
    size: usize,
    quiet: bool,
    failure: Option<E>,
    ended: bool,
}

impl<E> Buffer<E> {
    fn take(&mut self) -> Bytes {
        let mut joined = Vec::new();
        while let Some(mut chunk) = self.chunks.pop_front() {
            let room = READ_SIZE - joined.len();
            if chunk.len() > room {
                let rest = chunk.split_off(room);
                self.chunks.push_front(rest);
            }
            if joined.is_empty() && (chunk.len() == READ_SIZE || self.chunks.is_empty()) {
                self.size -= chunk.len();
                return chunk;
            }
            joined.extend_from_slice(&chunk);
            self.size -= chunk.len();
            if joined.len() == READ_SIZE {
                break;
            }
        }
        Bytes::from(joined)
    }
}

struct Reads<E> {
    buffer: Mutex<Buffer<E>>,
    received: Notify,
    drained: Notify,
}

enum Read<E> {
    Chunk(Bytes),
    Failure(E),
    End,
}

impl<E> Reads<E> {
    fn readable(&self) -> bool {
        let buffer = self.buffer();
        buffer.failure.is_some()
            || buffer.ended
            || (!buffer.chunks.is_empty() && (buffer.quiet || buffer.size >= READ_SIZE))
    }

    fn quiet(&self) {
        self.buffer().quiet = true;
        self.received.notify_one();
    }

    fn read(&self) -> Option<Read<E>> {
        let mut buffer = self.buffer();
        if !buffer.chunks.is_empty() {
            let chunk = buffer.take();
            drop(buffer);
            self.drained.notify_one();
            return Some(Read::Chunk(chunk));
        }
        if let Some(failure) = buffer.failure.take() {
            return Some(Read::Failure(failure));
        }
        buffer.ended.then_some(Read::End)
    }

    fn buffer(&self) -> std::sync::MutexGuard<'_, Buffer<E>> {
        self.buffer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

struct Reader(JoinHandle<()>);

impl Drop for Reader {
    fn drop(&mut self) {
        self.0.abort();
    }
}

pub(crate) fn joined_reads<E: Send + 'static>(
    chunks: impl Stream<Item = Result<Bytes, E>> + Send + 'static,
) -> BoxStream<'static, Result<Bytes, E>> {
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return chunks.boxed();
    };
    let reads = Arc::new(Reads {
        buffer: Mutex::new(Buffer {
            chunks: VecDeque::new(),
            size: 0,
            quiet: false,
            failure: None,
            ended: false,
        }),
        received: Notify::new(),
        drained: Notify::new(),
    });
    let reader = Reader(runtime.spawn(fill(chunks.boxed(), Arc::clone(&reads))));
    futures_util::stream::unfold(Some((reads, reader)), |state| async move {
        let (reads, reader) = state?;
        loop {
            let received = reads.received.notified();
            if reads.readable() {
                break;
            }
            received.await;
        }
        let read = reads.read()?;
        match read {
            Read::Chunk(chunk) => Some((Ok(chunk), Some((reads, reader)))),
            Read::Failure(failure) => Some((Err(failure), None)),
            Read::End => None,
        }
    })
    .boxed()
}

async fn fill<E>(mut chunks: BoxStream<'static, Result<Bytes, E>>, reads: Arc<Reads<E>>) {
    loop {
        let next = match settled_next(&mut chunks).await {
            Some(next) => next,
            None => {
                reads.quiet();
                chunks.next().await
            }
        };
        let paused = {
            let mut buffer = reads.buffer();
            buffer.quiet = false;
            match next {
                Some(Ok(chunk)) => {
                    buffer.size += chunk.len();
                    if !chunk.is_empty() {
                        buffer.chunks.push_back(chunk);
                    }
                }
                Some(Err(failure)) => {
                    buffer.failure = Some(failure);
                    buffer.ended = true;
                }
                None => buffer.ended = true,
            }
            if buffer.ended {
                drop(buffer);
                reads.received.notify_one();
                return;
            }
            buffer.size > HIGH_WATER
        };
        reads.received.notify_one();
        if paused {
            reads.quiet();
            loop {
                let drained = reads.drained.notified();
                if reads.buffer().size < LOW_WATER {
                    break;
                }
                drained.await;
            }
        }
    }
}

async fn settled_next<E>(
    chunks: &mut BoxStream<'static, Result<Bytes, E>>,
) -> Option<Option<Result<Bytes, E>>> {
    for _ in 0..SETTLE_YIELDS {
        if let Some(next) = chunks.next().now_or_never() {
            return Some(next);
        }
        tokio::task::yield_now().await;
    }
    None
}

#[cfg(test)]
mod tests {
    use std::{
        pin::Pin,
        sync::atomic::{AtomicUsize, Ordering},
        task::{Context, Poll},
        time::Duration,
    };

    use futures_util::{FutureExt, StreamExt, TryStreamExt};
    use tokio::sync::mpsc;

    use super::*;

    struct Upstream {
        receiver: mpsc::UnboundedReceiver<Result<Bytes, &'static str>>,
        pulled: Arc<AtomicUsize>,
    }

    impl Stream for Upstream {
        type Item = Result<Bytes, &'static str>;

        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let next = self.receiver.poll_recv(cx);
            if let Poll::Ready(Some(_)) = next {
                self.pulled.fetch_add(1, Ordering::SeqCst);
            }
            next
        }
    }

    type Item = Result<Bytes, &'static str>;

    fn upstream() -> (
        mpsc::UnboundedSender<Item>,
        BoxStream<'static, Item>,
        Arc<AtomicUsize>,
    ) {
        let (sender, receiver) = mpsc::unbounded_channel();
        let pulled = Arc::new(AtomicUsize::new(0));
        let reads = joined_reads(Upstream {
            receiver,
            pulled: Arc::clone(&pulled),
        });
        (sender, reads, pulled)
    }

    async fn settled() {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn chunks_that_arrived_together_are_read_as_one() {
        let (sender, mut reads, _) = upstream();
        sender.send(Ok(Bytes::from("event: a\n\n"))).unwrap();
        sender.send(Ok(Bytes::from("event: b\n\n"))).unwrap();
        sender.send(Ok(Bytes::from("event: c\n\n"))).unwrap();
        settled().await;
        assert_eq!(
            reads.try_next().await.unwrap().unwrap(),
            "event: a\n\nevent: b\n\nevent: c\n\n"
        );
        sender.send(Ok(Bytes::from("event: d\n\n"))).unwrap();
        settled().await;
        assert_eq!(reads.try_next().await.unwrap().unwrap(), "event: d\n\n");
        drop(sender);
        assert!(reads.try_next().await.unwrap().is_none());
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn chunks_still_being_handed_over_join_the_read() {
        let handed_over = futures_util::stream::unfold(0, |sent| async move {
            tokio::task::yield_now().await;
            (sent < 50).then(|| (Ok::<_, &str>(Bytes::from("event: ping\n\n")), sent + 1))
        });
        let read: Vec<usize> = joined_reads(handed_over)
            .map_ok(|chunk| chunk.len())
            .try_collect()
            .await
            .unwrap();
        assert_eq!(read, [50 * "event: ping\n\n".len()]);
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn a_read_returns_the_first_chunk_without_waiting_for_more() {
        let (sender, mut reads, _) = upstream();
        let first = tokio::spawn(async move { reads.try_next().await });
        settled().await;
        sender.send(Ok(Bytes::from("event: a\n\n"))).unwrap();
        assert_eq!(first.await.unwrap().unwrap().unwrap(), "event: a\n\n");
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn reads_are_cut_at_sixteen_kibibytes() {
        let (sender, reads, _) = upstream();
        sender.send(Ok(Bytes::from(vec![b'a'; 10_000]))).unwrap();
        sender.send(Ok(Bytes::from(vec![b'b'; 30_000]))).unwrap();
        drop(sender);
        settled().await;
        let sizes: Vec<usize> = reads
            .map_ok(|chunk| chunk.len())
            .try_collect()
            .await
            .unwrap();
        assert_eq!(sizes, [READ_SIZE, READ_SIZE, 40_000 - 2 * READ_SIZE]);
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn the_bytes_still_buffered_are_read_before_a_failure() {
        let (sender, mut reads, _) = upstream();
        sender.send(Ok(Bytes::from("event: a\n\n"))).unwrap();
        sender.send(Ok(Bytes::from("event: b\n\n"))).unwrap();
        sender.send(Err("reset")).unwrap();
        settled().await;
        assert_eq!(
            reads.try_next().await.unwrap().unwrap(),
            "event: a\n\nevent: b\n\n"
        );
        assert_eq!(reads.try_next().await.unwrap_err(), "reset");
        assert!(reads.next().await.is_none());
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn reading_pauses_above_the_high_water_mark_until_below_the_low_one() {
        let (sender, mut reads, pulled) = upstream();
        std::iter::repeat_with(|| Bytes::from(vec![b'a'; READ_SIZE]))
            .take(12)
            .try_for_each(|chunk| sender.send(Ok(chunk)))
            .unwrap();
        settled().await;
        assert_eq!(pulled.load(Ordering::SeqCst), 9);
        let read: Vec<Bytes> = (&mut reads).take(5).try_collect().await.unwrap();
        assert_eq!(read.len(), 5);
        settled().await;
        assert_eq!(pulled.load(Ordering::SeqCst), 9);
        reads.try_next().await.unwrap();
        settled().await;
        assert_eq!(pulled.load(Ordering::SeqCst), 12);
    }

    async fn chunked_upstream(frames: usize) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let _ = socket.read(&mut request).await.unwrap();
            let frame = "event: ping\ndata: {\"type\": \"ping\"}\n\n";
            let body: String =
                std::iter::repeat_n(format!("{:x}\r\n{frame}\r\n", frame.len()), frames).collect();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n{body}0\r\n\r\n"
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            tokio::time::sleep(Duration::from_secs(1)).await;
        });
        format!("http://{address}")
    }

    #[rstest::rstest]
    #[case::fifty(50)]
    #[case::a_thousand(1000)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn http_chunks_written_together_are_read_together(#[case] frames: usize) {
        let url = chunked_upstream(frames).await;
        let response = litellm_http::Client::for_test()
            .get(url)
            .send()
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        let sizes: Vec<usize> = joined_reads(response.bytes_stream())
            .map_ok(|chunk| chunk.len())
            .try_collect()
            .await
            .unwrap();
        let total = frames * "event: ping\ndata: {\"type\": \"ping\"}\n\n".len();
        let expected: Vec<usize> = std::iter::repeat_n(READ_SIZE, total / READ_SIZE)
            .chain((!total.is_multiple_of(READ_SIZE)).then_some(total % READ_SIZE))
            .collect();
        assert_eq!(sizes, expected);
    }

    #[rstest::rstest]
    fn without_a_runtime_the_chunks_pass_through() {
        let chunks =
            futures_util::stream::iter([Ok::<_, &str>(Bytes::from("a")), Ok(Bytes::from("b"))]);
        let read: Vec<Bytes> = joined_reads(chunks)
            .try_collect()
            .now_or_never()
            .unwrap()
            .unwrap();
        assert_eq!(read, [Bytes::from("a"), Bytes::from("b")]);
    }
}
