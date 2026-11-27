//! Reads the stored config again soon after the proxy announces a change on Redis, as Python's
//! `ConfigSyncSubscriber` (`config_sync_pubsub.py`) does, so a write lands in seconds instead of
//! at the next poll.
//!
//! A change waits a quiet second plus a random jitter of up to five seconds, so a burst of writes,
//! and every pod hearing the same one, cause one read each, and reads are at least ten seconds
//! apart. Changes heard while waiting join the read that is pending, and a change heard during a
//! read starts the next wait. After a lost subscription, one read catches up on whatever was
//! published meanwhile. Every message causes a full read, whose change detection skips the
//! rebuild when nothing the gateway serves changed, rather than Python's extra targeted read of
//! the `model_ids` a message names.

use std::time::Duration;

use futures_util::{Stream, StreamExt};
use litellm_redis_store::config_sync::Heard;
use tokio::time::Instant;

/// How long a heard change waits before the read it causes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Pacing {
    quiet: Duration,
    jitter: Duration,
    apart: Duration,
}

impl Pacing {
    /// `CONFIG_SYNC_DEBOUNCE_SECONDS`, `CONFIG_SYNC_JITTER_MAX_SECONDS` and
    /// `CONFIG_SYNC_MIN_RESYNC_INTERVAL_SECONDS`.
    pub(crate) const PYTHON: Self = Self {
        quiet: Duration::from_secs(1),
        jitter: Duration::from_secs(5),
        apart: Duration::from_secs(10),
    };
}

/// A random wait of up to `max`, as Python's `random.uniform(0, max)`.
pub(crate) fn uniform(max: Duration) -> Duration {
    max.mul_f64(rand::random::<f64>())
}

/// Runs `reload` for what `heard` brings, paced by `pacing` with `jitter` drawing each jitter
/// from its maximum, until `heard` ends.
pub(crate) async fn listen<J, R, F>(
    mut heard: impl Stream<Item = Heard> + Unpin,
    pacing: Pacing,
    mut jitter: J,
    reload: R,
) where
    J: FnMut(Duration) -> Duration,
    R: Fn() -> F,
    F: Future<Output = ()>,
{
    let mut lost = false;
    let mut last: Option<Instant> = None;
    loop {
        let Some(why) = next_trigger(&mut heard, &mut lost).await else {
            return;
        };
        let quiet = Instant::now() + pacing.quiet + jitter(pacing.jitter);
        let due = last.map_or(quiet, |last| quiet.max(last + pacing.apart));
        if !absorb_until(&mut heard, due, &mut lost).await {
            return;
        }
        tracing::info!(
            why,
            "reading the config the proxy stores in its database again"
        );
        reload().await;
        last = Some(Instant::now());
    }
}

/// Waits for something that calls for a read: a change, or subscribing again after the
/// subscription was lost. `None` when `heard` ends.
async fn next_trigger(
    heard: &mut (impl Stream<Item = Heard> + Unpin),
    lost: &mut bool,
) -> Option<&'static str> {
    loop {
        match heard.next().await? {
            Heard::Change(change) => {
                tracing::debug!(
                    object_type = change.object_type,
                    model_ids = change.model_ids.join(","),
                    "the proxy announced a config change"
                );
                return Some("the proxy announced a config change on Redis");
            }
            Heard::Lost => *lost = true,
            Heard::Subscribed if std::mem::take(lost) => {
                return Some(
                    "subscribed to config changes on Redis again, so one may have been missed",
                );
            }
            Heard::Subscribed => (),
        }
    }
}

/// Takes in what `heard` brings until `due`, which the read pending then covers. `false` when
/// `heard` ends first.
async fn absorb_until(
    heard: &mut (impl Stream<Item = Heard> + Unpin),
    due: Instant,
    lost: &mut bool,
) -> bool {
    let wait = tokio::time::sleep_until(due);
    tokio::pin!(wait);
    loop {
        tokio::select! {
            () = &mut wait => return true,
            event = heard.next() => match event {
                None => return false,
                Some(Heard::Lost) => *lost = true,
                Some(Heard::Subscribed) => *lost = false,
                Some(Heard::Change(_)) => (),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use futures_util::stream;
    use litellm_redis_store::config_sync::ConfigChange;
    use rstest::rstest;

    use super::*;

    const PACED: Pacing = Pacing::PYTHON;

    fn change() -> Heard {
        Heard::Change(ConfigChange {
            object_type: "litellm_proxymodeltable".to_owned(),
            model_ids: vec!["m-1".to_owned()],
        })
    }

    fn at(seconds: f64, heard: Heard) -> (Duration, Heard) {
        (Duration::from_secs_f64(seconds), heard)
    }

    /// When, in seconds after the start, each read began, for `script` heard at its offsets
    /// after the start, each jitter `jitter` and each read lasting `lasting`. The subscription
    /// ends a minute in.
    async fn reads(
        script: Vec<(Duration, Heard)>,
        jitter: Duration,
        lasting: Duration,
    ) -> Vec<f64> {
        let start = Instant::now();
        let heard = stream::iter(script)
            .then(move |(after, heard)| async move {
                tokio::time::sleep_until(start + after).await;
                Some(heard)
            })
            .chain(stream::once(async move {
                tokio::time::sleep_until(start + Duration::from_secs(60)).await;
                None
            }))
            .filter_map(std::future::ready)
            .boxed();
        let began = Arc::new(Mutex::new(Vec::new()));
        listen(
            heard,
            PACED,
            |_| jitter,
            || {
                let began = began.clone();
                async move {
                    began
                        .lock()
                        .unwrap()
                        .push((Instant::now() - start).as_secs_f64());
                    tokio::time::sleep(lasting).await;
                }
            },
        )
        .await;
        Arc::try_unwrap(began).unwrap().into_inner().unwrap()
    }

    #[rstest]
    #[case::one_change_after_the_quiet_second_and_its_jitter(
        vec![at(0.0, change())],
        2.0,
        vec![3.0]
    )]
    #[case::a_burst_inside_the_wait_is_one_read(
        vec![at(0.0, change()), at(0.5, change()), at(2.5, change())],
        2.0,
        vec![3.0]
    )]
    #[case::reads_are_ten_seconds_apart(
        vec![at(0.0, change()), at(2.0, change())],
        0.0,
        vec![1.0, 11.0]
    )]
    #[case::changes_after_the_ten_seconds_wait_only_their_quiet_second(
        vec![at(0.0, change()), at(20.0, change())],
        0.0,
        vec![1.0, 21.0]
    )]
    #[case::subscribing_is_no_change(vec![at(0.0, Heard::Subscribed)], 0.0, vec![])]
    #[case::subscribing_again_after_a_lost_subscription_reads_once(
        vec![at(0.0, Heard::Subscribed), at(5.0, Heard::Lost), at(10.0, Heard::Lost), at(15.0, Heard::Subscribed)],
        0.0,
        vec![16.0]
    )]
    #[case::subscribing_late_after_failing_at_first_reads_once(
        vec![at(0.0, Heard::Lost), at(5.0, Heard::Subscribed)],
        0.0,
        vec![6.0]
    )]
    #[case::a_reconnect_inside_the_wait_joins_the_pending_read(
        vec![at(0.0, change()), at(0.5, Heard::Lost), at(1.5, Heard::Subscribed)],
        2.0,
        vec![3.0]
    )]
    #[case::a_lost_subscription_not_yet_back_is_caught_up_once_it_is(
        vec![at(0.0, change()), at(2.0, Heard::Lost), at(30.0, Heard::Subscribed)],
        0.0,
        vec![1.0, 31.0]
    )]
    #[tokio::test(start_paused = true)]
    async fn a_heard_change_reads_once_per_burst_paced_like_python(
        #[case] script: Vec<(Duration, Heard)>,
        #[case] jitter: f64,
        #[case] expected: Vec<f64>,
    ) {
        assert_eq!(
            reads(script, Duration::from_secs_f64(jitter), Duration::ZERO).await,
            expected
        );
    }

    #[rstest]
    #[tokio::test(start_paused = true)]
    async fn a_change_heard_during_a_read_causes_one_more_read() {
        let script = vec![at(0.0, change()), at(12.0, change()), at(13.0, change())];

        let began = reads(script, Duration::ZERO, Duration::from_secs(15)).await;

        assert_eq!(began, vec![1.0, 26.0]);
    }

    #[rstest]
    #[case::none(Duration::ZERO)]
    #[case::five_seconds(Duration::from_secs(5))]
    fn a_jitter_is_at_most_its_maximum(#[case] max: Duration) {
        assert!(uniform(max) <= max);
    }
}
