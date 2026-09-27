use std::{future::Future, time::Duration};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct JobName(String);

impl JobName {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HolderId(String);

impl HolderId {
    pub fn random() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Acquire {
    Held,
    Busy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Renewal {
    Extended,
    Lost,
}

pub trait LeaseStore: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn try_acquire(
        &self,
        job: &JobName,
        holder: &HolderId,
        ttl: Duration,
    ) -> impl Future<Output = Result<Acquire, Self::Error>> + Send;

    fn renew(
        &self,
        job: &JobName,
        holder: &HolderId,
        ttl: Duration,
    ) -> impl Future<Output = Result<Renewal, Self::Error>> + Send;

    fn release(
        &self,
        job: &JobName,
        holder: &HolderId,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
}
