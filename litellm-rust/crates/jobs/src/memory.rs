use std::{collections::HashMap, convert::Infallible, sync::Mutex, time::Duration};

use tokio::time::Instant;

use crate::{Acquire, HolderId, JobName, LeaseStore, Renewal};

#[derive(Debug, Default)]
pub struct MemoryLeaseStore {
    leases: Mutex<HashMap<JobName, (HolderId, Instant)>>,
}

impl MemoryLeaseStore {
    fn with_leases<T>(&self, f: impl FnOnce(&mut HashMap<JobName, (HolderId, Instant)>) -> T) -> T {
        f(&mut self
            .leases
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()))
    }
}

fn held_by_other(entry: Option<&(HolderId, Instant)>, holder: &HolderId, now: Instant) -> bool {
    entry.is_some_and(|(owner, expires_at)| owner != holder && *expires_at > now)
}

impl LeaseStore for MemoryLeaseStore {
    type Error = Infallible;

    async fn try_acquire(
        &self,
        job: &JobName,
        holder: &HolderId,
        ttl: Duration,
    ) -> Result<Acquire, Self::Error> {
        let now = Instant::now();
        Ok(self.with_leases(|leases| {
            if held_by_other(leases.get(job), holder, now) {
                return Acquire::Busy;
            }
            leases.insert(job.clone(), (holder.clone(), now + ttl));
            Acquire::Held
        }))
    }

    async fn renew(
        &self,
        job: &JobName,
        holder: &HolderId,
        ttl: Duration,
    ) -> Result<Renewal, Self::Error> {
        let now = Instant::now();
        Ok(self.with_leases(|leases| match leases.get_mut(job) {
            Some((owner, expires_at)) if owner == holder && *expires_at > now => {
                *expires_at = now + ttl;
                Renewal::Extended
            }
            _ => Renewal::Lost,
        }))
    }

    async fn release(&self, job: &JobName, holder: &HolderId) -> Result<(), Self::Error> {
        self.with_leases(|leases| {
            if leases.get(job).is_some_and(|(owner, _)| owner == holder) {
                leases.remove(job);
            }
        });
        Ok(())
    }
}
