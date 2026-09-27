#[derive(Debug, thiserror::Error)]
pub enum FlushError<B, S>
where
    B: std::error::Error + 'static,
    S: std::error::Error + 'static,
{
    #[error("claiming a batch from the spend buffer failed")]
    Claim(#[source] B),
    #[error("committing spend failed; the batch was released for the next flush")]
    Commit(#[source] S),
    #[error("committing spend failed; the batch returns when its claim expires")]
    Unreleased {
        #[source]
        commit: S,
        release: B,
    },
    #[error("spend was committed but not acked; the batch is redelivered and skipped")]
    Ack(#[source] B),
}

#[derive(Debug, thiserror::Error)]
#[error("inserting spend logs failed; {requeued} rows were requeued and {evicted} of them evicted")]
pub struct LogFlushError<S: std::error::Error + 'static> {
    #[source]
    pub insert: S,
    pub requeued: usize,
    pub evicted: usize,
}
