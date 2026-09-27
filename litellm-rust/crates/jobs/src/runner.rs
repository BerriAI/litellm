use std::{future::Future, pin::pin, time::Duration};

use tokio_util::sync::CancellationToken;

use crate::{Acquire, Exclusivity, HolderId, JobName, JobSpec, LeaseStore, Renewal};

pub async fn run<S, F, Fut>(
    store: &S,
    holder: &HolderId,
    spec: &JobSpec,
    body: F,
    shutdown: CancellationToken,
) where
    S: LeaseStore,
    F: Fn(CancellationToken) -> Fut + Send + Sync,
    Fut: Future<Output = ()> + Send,
{
    loop {
        tokio::select! {
            () = shutdown.cancelled() => return,
            () = tokio::time::sleep(spec.schedule.next_delay()) => {}
        }
        match spec.exclusivity {
            Exclusivity::EveryPod => body(shutdown.child_token()).await,
            Exclusivity::OnePod { lease_ttl } => {
                run_leased(store, holder, &spec.name, lease_ttl, &body, &shutdown).await;
            }
        }
    }
}

async fn run_leased<S, F, Fut>(
    store: &S,
    holder: &HolderId,
    job: &JobName,
    ttl: Duration,
    body: &F,
    shutdown: &CancellationToken,
) where
    S: LeaseStore,
    F: Fn(CancellationToken) -> Fut,
    Fut: Future<Output = ()>,
{
    match store.try_acquire(job, holder, ttl).await {
        Ok(Acquire::Held) => {}
        Ok(Acquire::Busy) => return,
        Err(error) => {
            tracing::warn!(job = job.as_str(), %error, "lease acquire failed; skipping this run");
            return;
        }
    }
    let token = shutdown.child_token();
    let mut work = pin!(body(token.clone()));
    tokio::select! {
        biased;
        () = &mut work => {}
        () = keep_alive(store, holder, job, ttl) => {
            token.cancel();
            work.await;
        }
    }
    if let Err(error) = store.release(job, holder).await {
        tracing::warn!(job = job.as_str(), %error, "lease release failed; it expires on its own");
    }
}

async fn keep_alive<S: LeaseStore>(store: &S, holder: &HolderId, job: &JobName, ttl: Duration) {
    loop {
        tokio::time::sleep(ttl / 3).await;
        match store.renew(job, holder, ttl).await {
            Ok(Renewal::Extended) => {}
            Ok(Renewal::Lost) => {
                tracing::warn!(job = job.as_str(), "lease lost mid-run; cancelling the job");
                return;
            }
            Err(error) => {
                tracing::warn!(job = job.as_str(), %error, "lease renewal failed; cancelling the job");
                return;
            }
        }
    }
}
