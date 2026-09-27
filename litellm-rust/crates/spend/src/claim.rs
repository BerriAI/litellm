use uuid::Uuid;

use crate::Batch;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BatchId(Uuid);

impl BatchId {
    pub fn random() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_uuid(id: Uuid) -> Self {
        Self(id)
    }

    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

#[must_use = "a claimed batch stays hidden until it is acked, released, or its claim expires"]
#[derive(Debug, PartialEq)]
pub struct Claimed<K, V> {
    id: BatchId,
    batch: Batch<K, V>,
}

impl<K, V> Claimed<K, V> {
    pub fn from_buffer(id: BatchId, batch: Batch<K, V>) -> Self {
        Self { id, batch }
    }

    pub fn id(&self) -> BatchId {
        self.id
    }

    pub fn batch(&self) -> &Batch<K, V> {
        &self.batch
    }
}
