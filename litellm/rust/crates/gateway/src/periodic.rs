//! The jobs `proxy_startup_event` schedules every `PROXY_BATCH_WRITE_AT` seconds, such as
//! `update_gateway_requests_job`: a pass that runs on a fixed interval in the background and one
//! last time on shutdown, so what it buffers is written before the process exits.

use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

use litellm_config::Environment;
use parking_lot::Mutex;
use rand::Rng;
use tokio::{sync::watch, task::JoinHandle};

/// `PROXY_BATCH_WRITE_AT`'s default.
const DEFAULT_BATCH_WRITE_SECONDS: u64 = 10;
/// `_BATCH_WRITE_JITTER_MAX_SECONDS`.
const MAX_JITTER_SECONDS: u64 = 5;

type Pass = Arc<dyn Fn() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// A pass run every `interval` on the current runtime until [`PeriodicJob::close`].
pub struct PeriodicJob {
    pass: Pass,
    stop: watch::Sender<bool>,
    job: Mutex<Option<JoinHandle<()>>>,
}

impl PeriodicJob {
    /// Starts running `pass` every `interval`, the first one an interval from now.
    pub fn start<F, Fut>(interval: Duration, pass: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let pass: Pass = Arc::new(move || Box::pin(pass()));
        let (stop, mut stopped) = watch::channel(false);
        let job = tokio::spawn({
            let pass = pass.clone();
            async move {
                loop {
                    tokio::select! {
                        () = tokio::time::sleep(interval) => pass().await,
                        _ = stopped.changed() => return,
                    }
                }
            }
        });
        Self {
            pass,
            stop,
            job: Mutex::new(Some(job)),
        }
    }

    /// `_flush_request_accumulators` on shutdown: stops the schedule, waits out a pass already
    /// running, then runs one last pass.
    pub async fn close(&self) {
        let _ = self.stop.send(true);
        let job = self.job.lock().take();
        if let Some(job) = job {
            let _ = job.await;
        }
        (self.pass)().await;
    }
}

/// `proxy_batch_write_at + random.randint(0, 5)`, the jitter picked once per job, and at least a
/// second so a zero cannot spin. Only `PROXY_BATCH_WRITE_AT` is read:
/// `general_settings.proxy_batch_write_at` refuses native, so Python answers any config that sets
/// it.
pub fn batch_write_interval(environment: &Environment) -> Duration {
    interval(
        environment,
        rand::thread_rng().gen_range(0..=MAX_JITTER_SECONDS),
    )
}

fn interval(environment: &Environment, jitter_seconds: u64) -> Duration {
    let seconds = environment
        .proxy_batch_write_at
        .unwrap_or(DEFAULT_BATCH_WRITE_SECONDS)
        .saturating_add(jitter_seconds);
    Duration::from_secs(seconds.max(1))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::default(None, 0, 10)]
    #[case::configured(Some(30), 0, 30)]
    #[case::jittered(None, 5, 15)]
    #[case::zero(Some(0), 0, 1)]
    fn the_interval_is_the_batch_write_interval_plus_jitter(
        #[case] proxy_batch_write_at: Option<u64>,
        #[case] jitter_seconds: u64,
        #[case] expected_seconds: u64,
    ) {
        let environment = Environment {
            proxy_batch_write_at,
            ..Environment::default()
        };

        assert_eq!(
            interval(&environment, jitter_seconds),
            Duration::from_secs(expected_seconds)
        );
    }

    fn counting(passes: &Arc<AtomicUsize>) -> impl Fn() -> std::future::Ready<()> + use<> {
        let passes = passes.clone();
        move || {
            passes.fetch_add(1, Ordering::SeqCst);
            std::future::ready(())
        }
    }

    #[rstest]
    #[tokio::test(start_paused = true)]
    async fn a_pass_runs_every_interval_and_once_more_on_close() {
        let passes = Arc::new(AtomicUsize::new(0));
        let job = PeriodicJob::start(Duration::from_secs(10), counting(&passes));

        tokio::time::sleep(Duration::from_secs(25)).await;
        let scheduled = passes.load(Ordering::SeqCst);
        job.close().await;

        assert_eq!((scheduled, passes.load(Ordering::SeqCst)), (2, 3));
    }

    #[rstest]
    #[tokio::test(start_paused = true)]
    async fn close_waits_out_a_running_pass_before_the_last_one() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let job = PeriodicJob::start(Duration::from_secs(1), {
            let order = order.clone();
            move || {
                let order = order.clone();
                async move {
                    order.lock().push("started");
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    order.lock().push("finished");
                }
            }
        });

        tokio::time::sleep(Duration::from_millis(1500)).await;
        job.close().await;

        assert_eq!(
            *order.lock(),
            ["started", "finished", "started", "finished"]
        );
    }
}
