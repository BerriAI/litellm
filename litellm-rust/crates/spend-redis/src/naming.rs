use litellm_spend::{Batch, CounterKey};

pub trait CounterNaming: Send + Sync + 'static {
    fn counter(&self, key: &CounterKey) -> Option<String>;
}

pub trait BatchCodec<K, V>: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn list(&self) -> &str;

    fn encode(&self, batch: &Batch<K, V>) -> Result<String, Self::Error>;

    fn decode(&self, blob: &str) -> Result<Batch<K, V>, Self::Error>;
}
