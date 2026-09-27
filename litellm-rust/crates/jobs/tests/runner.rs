use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use litellm_jobs::{
    Acquire, Exclusivity, HolderId, JobName, JobSpec, LeaseStore, MemoryLeaseStore, Renewal,
    Schedule, run,
};
use rstest::rstest;
use tokio_util::sync::CancellationToken;

const PERIOD: Duration = Duration::from_secs(10);

fn spec(exclusivity: Exclusivity) -> JobSpec {
    JobSpec {
        name: JobName::new("job"),
        schedule: Schedule::Every {
            period: PERIOD,
            jitter: Duration::ZERO,
        },
        exclusivity,
    }
}

#[derive(Default)]
struct Probe {
    runs: AtomicUsize,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
}

impl Probe {
    async fn work(&self, duration: Duration) {
        self.runs.fetch_add(1, Ordering::SeqCst);
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(duration).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

#[rstest]
#[case::every_pod(Exclusivity::EveryPod, 3 * 5, 3)]
#[case::one_pod(Exclusivity::OnePod { lease_ttl: Duration::from_secs(30) }, 5, 1)]
#[tokio::test(start_paused = true)]
async fn exclusivity_decides_how_many_pods_run_each_tick(
    #[case] exclusivity: Exclusivity,
    #[case] expected_runs: usize,
    #[case] expected_max_in_flight: usize,
) {
    let store = Arc::new(MemoryLeaseStore::default());
    let probe = Arc::new(Probe::default());
    let shutdown = CancellationToken::new();
    let pods: Vec<_> = (0..3)
        .map(|_| {
            let (store, probe, shutdown) = (store.clone(), probe.clone(), shutdown.clone());
            tokio::spawn(async move {
                let body = |_token| {
                    let probe = probe.clone();
                    async move { probe.work(Duration::from_secs(1)).await }
                };
                run(
                    &*store,
                    &HolderId::random(),
                    &spec(exclusivity),
                    body,
                    shutdown,
                )
                .await;
            })
        })
        .collect();

    tokio::time::sleep(PERIOD * 5 + PERIOD / 2).await;
    shutdown.cancel();
    for pod in pods {
        pod.await.unwrap();
    }

    assert_eq!(probe.runs.load(Ordering::SeqCst), expected_runs);
    assert_eq!(
        probe.max_in_flight.load(Ordering::SeqCst),
        expected_max_in_flight
    );
}

struct LosesLeaseOnRenew {
    released: AtomicBool,
}

impl LeaseStore for LosesLeaseOnRenew {
    type Error = Infallible;

    async fn try_acquire(
        &self,
        _: &JobName,
        _: &HolderId,
        _: Duration,
    ) -> Result<Acquire, Infallible> {
        Ok(Acquire::Held)
    }

    async fn renew(&self, _: &JobName, _: &HolderId, _: Duration) -> Result<Renewal, Infallible> {
        Ok(Renewal::Lost)
    }

    async fn release(&self, _: &JobName, _: &HolderId) -> Result<(), Infallible> {
        self.released.store(true, Ordering::SeqCst);
        Ok(())
    }
}

#[tokio::test(start_paused = true)]
async fn losing_the_lease_cancels_the_running_body() {
    let store = Arc::new(LosesLeaseOnRenew {
        released: AtomicBool::new(false),
    });
    let cancelled_after = Arc::new(std::sync::Mutex::new(None));
    let shutdown = CancellationToken::new();
    let spec = spec(Exclusivity::OnePod {
        lease_ttl: Duration::from_secs(30),
    });

    let runner = {
        let (store, cancelled_after, shutdown) =
            (store.clone(), cancelled_after.clone(), shutdown.clone());
        tokio::spawn(async move {
            let body = |token: CancellationToken| {
                let cancelled_after = cancelled_after.clone();
                async move {
                    let started = tokio::time::Instant::now();
                    tokio::select! {
                        () = token.cancelled() => {
                            *cancelled_after.lock().unwrap() = Some(started.elapsed());
                        }
                        () = tokio::time::sleep(Duration::from_secs(3600)) => {}
                    }
                }
            };
            run(&*store, &HolderId::random(), &spec, body, shutdown).await;
        })
    };

    tokio::time::sleep(PERIOD + Duration::from_secs(15)).await;
    shutdown.cancel();
    runner.await.unwrap();

    assert_eq!(
        *cancelled_after.lock().unwrap(),
        Some(Duration::from_secs(10))
    );
    assert!(store.released.load(Ordering::SeqCst));
}

#[tokio::test(start_paused = true)]
async fn shutdown_cancels_a_running_body_and_stops_the_loop() {
    let store = Arc::new(MemoryLeaseStore::default());
    let shutdown = CancellationToken::new();
    let saw_shutdown = Arc::new(AtomicBool::new(false));

    let runner = {
        let (store, shutdown, saw_shutdown) =
            (store.clone(), shutdown.clone(), saw_shutdown.clone());
        tokio::spawn(async move {
            let body = |token: CancellationToken| {
                let saw_shutdown = saw_shutdown.clone();
                async move {
                    token.cancelled().await;
                    saw_shutdown.store(true, Ordering::SeqCst);
                }
            };
            let spec = spec(Exclusivity::OnePod {
                lease_ttl: Duration::from_secs(30),
            });
            run(&*store, &HolderId::random(), &spec, body, shutdown).await;
        })
    };

    tokio::time::sleep(PERIOD + Duration::from_secs(1)).await;
    shutdown.cancel();
    runner.await.unwrap();

    assert!(saw_shutdown.load(Ordering::SeqCst));
    assert_eq!(
        store
            .try_acquire(
                &JobName::new("job"),
                &HolderId::random(),
                Duration::from_secs(1)
            )
            .await,
        Ok(Acquire::Held)
    );
}
