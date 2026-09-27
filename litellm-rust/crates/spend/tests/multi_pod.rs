mod support;

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use litellm_jobs::{Exclusivity, HolderId, JobName, JobSpec, MemoryLeaseStore, Schedule, run};
use litellm_spend::{Buffer, Cost, EntityKey, FlushOutcome, Totals, flush};
use support::{CLAIM_TTL, Database, Faults, buffer};
use tokio_util::sync::CancellationToken;

const PODS: usize = 3;
const REQUESTS_PER_POD: usize = 1_000;
const REQUEST_GAP: Duration = Duration::from_millis(100);
const TICK: Duration = Duration::from_secs(10);
const DRAIN_MAX: usize = 10;
const CRASH_EVERY: usize = 5;
const CRASH_AFTER: Duration = Duration::from_secs(1);

fn request(pod: usize, i: usize) -> [(EntityKey, Cost); 2] {
    let cost = Cost(0.25 * ((i % 4) + 1) as f64);
    [
        (EntityKey::Key(format!("sk-{}", (i * 7 + pod) % 20)), cost),
        (EntityKey::Team(format!("team-{}", (i + pod) % 5)), cost),
    ]
}

fn expected_totals() -> HashMap<EntityKey, f64> {
    let mut totals = HashMap::new();
    for pod in 0..PODS {
        for i in 0..REQUESTS_PER_POD {
            for (entity, cost) in request(pod, i) {
                *totals.entry(entity).or_default() += cost.0;
            }
        }
    }
    totals
}

fn job(name: &str, exclusivity: Exclusivity) -> JobSpec {
    JobSpec {
        name: JobName::new(name),
        schedule: Schedule::Every {
            period: TICK,
            jitter: Duration::from_secs(5),
        },
        exclusivity,
    }
}

async fn drain_visible<B: Buffer<EntityKey, Cost>>(buffer: &B, database: &Database) {
    loop {
        match flush(buffer, database, DRAIN_MAX).await {
            Ok(FlushOutcome::Empty) => return,
            Ok(FlushOutcome::Committed { .. }) | Err(_) => {}
        }
    }
}

async fn drain_all<B: Buffer<EntityKey, Cost>>(buffer: &B, database: &Database) {
    drain_visible(buffer, database).await;
    tokio::time::sleep(CLAIM_TTL).await;
    drain_visible(buffer, database).await;
}

async fn commit_or_crash(
    shared: &impl Buffer<EntityKey, Cost>,
    database: &Database,
    runs: &AtomicUsize,
    crashes: &AtomicUsize,
) {
    let commit = flush(shared, database, DRAIN_MAX);
    if !(runs.fetch_add(1, Ordering::SeqCst) + 1).is_multiple_of(CRASH_EVERY) {
        let _ = commit.await;
        return;
    }
    if tokio::time::timeout(CRASH_AFTER, commit).await.is_err() {
        crashes.fetch_add(1, Ordering::SeqCst);
    }
}

#[tokio::test(start_paused = true)]
async fn redis_buffer_topology_commits_every_cent_exactly_once_despite_failures_and_crashes() {
    let shared = Arc::new(buffer());
    let database = Arc::new(Database::new(Faults {
        latency: Duration::from_secs(2),
        fail_without_applying_every: Some(4),
        fail_after_applying_every: Some(7),
        ..Faults::default()
    }));
    let leases = Arc::new(MemoryLeaseStore::default());
    let shutdown = CancellationToken::new();
    let (runs, crashes) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let locals: Vec<_> = (0..PODS).map(|_| Arc::new(buffer())).collect();

    let traffic: Vec<_> = locals
        .iter()
        .enumerate()
        .map(|(pod, local)| {
            let local = local.clone();
            tokio::spawn(async move {
                for i in 0..REQUESTS_PER_POD {
                    local
                        .push(Totals::from_entries(request(pod, i)))
                        .await
                        .unwrap();
                    tokio::time::sleep(REQUEST_GAP).await;
                }
            })
        })
        .collect();

    let runners: Vec<_> = locals
        .iter()
        .flat_map(|local| {
            let holder = HolderId::random();
            let push = {
                let (local, shared, leases, shutdown, holder) = (
                    local.clone(),
                    shared.clone(),
                    leases.clone(),
                    shutdown.clone(),
                    holder.clone(),
                );
                tokio::spawn(async move {
                    let body = |_token| {
                        let (local, shared) = (local.clone(), shared.clone());
                        async move {
                            flush(&*local, &*shared, usize::MAX).await.unwrap();
                        }
                    };
                    let spec = job("spend_buffer_push", Exclusivity::EveryPod);
                    run(&*leases, &holder, &spec, body, shutdown).await;
                })
            };
            let commit = {
                let (shared, database, leases, shutdown, runs, crashes) = (
                    shared.clone(),
                    database.clone(),
                    leases.clone(),
                    shutdown.clone(),
                    runs.clone(),
                    crashes.clone(),
                );
                tokio::spawn(async move {
                    let body = |_token| {
                        let (shared, database, runs, crashes) = (
                            shared.clone(),
                            database.clone(),
                            runs.clone(),
                            crashes.clone(),
                        );
                        async move { commit_or_crash(&*shared, &database, &runs, &crashes).await }
                    };
                    let spec = job(
                        "db_spend_update_job",
                        Exclusivity::OnePod {
                            lease_ttl: Duration::from_secs(60),
                        },
                    );
                    run(&*leases, &holder, &spec, body, shutdown).await;
                })
            };
            [push, commit]
        })
        .collect();

    for pod in traffic {
        pod.await.unwrap();
    }
    shutdown.cancel();
    for runner in runners {
        runner.await.unwrap();
    }
    for local in &locals {
        flush(&**local, &*shared, usize::MAX).await.unwrap();
    }
    drain_all(&*shared, &database).await;

    assert_eq!(database.totals(), expected_totals());
    assert_eq!(database.max_in_flight.load(Ordering::SeqCst), 1);
    assert!(database.failed_without_applying.load(Ordering::SeqCst) > 0);
    assert!(database.failed_after_applying.load(Ordering::SeqCst) > 0);
    assert!(crashes.load(Ordering::SeqCst) > 0);
    assert!(database.repeats_skipped.load(Ordering::SeqCst) > 0);
}
