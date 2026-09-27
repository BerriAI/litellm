use std::future::Future;

use crate::{Batch, Claimed, Cost, CounterKey, SpendLogRow};

pub trait Buffer<K, V>: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn push(&self, batch: Batch<K, V>) -> impl Future<Output = Result<(), Self::Error>> + Send;

    fn claim(
        &self,
        max: usize,
    ) -> impl Future<Output = Result<Option<Claimed<K, V>>, Self::Error>> + Send;

    fn ack(&self, claimed: Claimed<K, V>) -> impl Future<Output = Result<(), Self::Error>> + Send;

    fn release(
        &self,
        claimed: Claimed<K, V>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
}

pub trait Store<K, V>: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn commit(
        &self,
        claimed: &Claimed<K, V>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
}

pub trait Counters: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn add(
        &self,
        increments: &[(CounterKey, Cost)],
    ) -> impl Future<Output = Result<Vec<Cost>, Self::Error>> + Send;

    fn current(
        &self,
        key: &CounterKey,
    ) -> impl Future<Output = Result<Option<Cost>, Self::Error>> + Send;
}

#[derive(Debug)]
pub enum InsertError<E> {
    Transient(E),
    Rejected(E),
}

pub trait LogSink: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn insert(
        &self,
        rows: &[SpendLogRow],
    ) -> impl Future<Output = Result<(), InsertError<Self::Error>>> + Send;
}
