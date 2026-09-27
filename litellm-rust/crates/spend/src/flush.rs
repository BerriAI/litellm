use crate::{Buffer, FlushError, InsertError, Key, LogFlushError, LogQueue, LogSink, Store, Tally};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlushOutcome {
    Empty,
    Committed { entries: usize },
}

pub async fn flush<K, V, B, S>(
    buffer: &B,
    store: &S,
    max: usize,
) -> Result<FlushOutcome, FlushError<B::Error, S::Error>>
where
    K: Key,
    V: Tally,
    B: Buffer<K, V>,
    S: Store<K, V>,
{
    let Some(claimed) = buffer.claim(max).await.map_err(FlushError::Claim)? else {
        return Ok(FlushOutcome::Empty);
    };
    if let Err(commit) = store.commit(&claimed).await {
        return Err(match buffer.release(claimed).await {
            Ok(()) => FlushError::Commit(commit),
            Err(release) => FlushError::Unreleased { commit, release },
        });
    }
    let entries = claimed.batch().len();
    buffer.ack(claimed).await.map_err(FlushError::Ack)?;
    Ok(FlushOutcome::Committed { entries })
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LogFlushOutcome {
    pub inserted: usize,
    pub rejected: Vec<String>,
}

pub async fn flush_logs<S: LogSink>(
    queue: &LogQueue,
    sink: &S,
    max_rows: usize,
) -> Result<LogFlushOutcome, LogFlushError<S::Error>> {
    let mut chunks = vec![queue.dequeue(max_rows)];
    let mut outcome = LogFlushOutcome::default();
    while let Some(mut chunk) = chunks.pop() {
        if chunk.is_empty() {
            continue;
        }
        match sink.insert(&chunk).await {
            Ok(()) => outcome.inserted += chunk.len(),
            Err(InsertError::Rejected(_)) if chunk.len() == 1 => {
                outcome
                    .rejected
                    .extend(chunk.into_iter().map(|row| row.request_id));
            }
            Err(InsertError::Rejected(_)) => {
                let later_half = chunk.split_off(chunk.len() / 2);
                chunks.push(later_half);
                chunks.push(chunk);
            }
            Err(InsertError::Transient(insert)) => {
                let unsent: Vec<_> = chunk
                    .into_iter()
                    .chain(chunks.into_iter().rev().flatten())
                    .collect();
                let requeued = unsent.len();
                let evicted = queue.requeue(unsent);
                return Err(LogFlushError {
                    insert,
                    requeued,
                    evicted,
                });
            }
        }
    }
    Ok(outcome)
}
