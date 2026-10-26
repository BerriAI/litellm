use std::collections::HashSet;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};

use litellm_model_catalog::{
    BUNDLED_REGISTRY, Catalog, Installed, Located, Origin, RetryPolicy, Revalidation, SourceId,
    Step,
};
use tokio::runtime::Runtime;
use tokio_util::sync::CancellationToken;

use crate::{Fetch, Stale, process_local::ProcessLocal};

pub trait Install: Send + Sync + 'static {
    fn generation(&self) -> u64;
    /// Compare and install atomically against `seen`, returning `Stale` if another writer won.
    fn install(&self, seen: u64, adopted: Adopted) -> Result<u64, Stale>;
    fn checked(&self, outcome: Checked);
}

#[derive(Debug)]
pub struct Adopted {
    pub installed: Installed,
    pub left_out: Vec<Located>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Checked {
    NotModified,
    Failed { reason: String },
}

pub struct Refresher {
    source: SourceId,
    fetch: Arc<dyn Fetch>,
    installed: Installed,
    install: Arc<dyn Install>,
    remote_adopted: Arc<AtomicBool>,
    process: ProcessLocal<Option<Worker>>,
    policy: RetryPolicy,
}

struct Worker {
    cancel: CancellationToken,
    _thread: Option<JoinHandle<()>>,
}

impl Refresher {
    pub fn start(
        source: SourceId,
        fetch: Arc<dyn Fetch>,
        installed: &Installed,
        install: Arc<dyn Install>,
    ) -> Self {
        Self::with_policy(source, fetch, installed, install, RetryPolicy::default())
    }

    fn with_policy(
        source: SourceId,
        fetch: Arc<dyn Fetch>,
        installed: &Installed,
        install: Arc<dyn Install>,
        policy: RetryPolicy,
    ) -> Self {
        let refresher = Self {
            source,
            fetch,
            installed: installed.clone(),
            install,
            policy,
            remote_adopted: Arc::new(AtomicBool::new(matches!(
                installed.origin,
                Origin::Remote { .. }
            ))),
            process: ProcessLocal::new(),
        };
        refresher.process.get_or_init(|| Some(refresher.spawn()));
        refresher
    }

    pub fn stop(&self) {
        if let Some(worker) = self.process.get_or_init(|| None) {
            worker.cancel.cancel();
        }
    }

    /// The host must reset any of its own inherited locks before calling this in the child.
    pub fn after_fork(&self) {
        self.process
            .get_or_init(|| (!self.remote_adopted.load(Ordering::Acquire)).then(|| self.spawn()));
    }

    fn spawn(&self) -> Worker {
        let cancel = CancellationToken::new();
        let worker_cancel = cancel.clone();
        let source = self.source.clone();
        let fetch = self.fetch.clone();
        let installed = self.installed.clone();
        let install = self.install.clone();
        let seen = install.generation();
        let remote_adopted = self.remote_adopted.clone();
        let policy = self.policy;
        let dispatch = tracing::dispatcher::get_default(Clone::clone);
        let span = tracing::Span::current();
        let worker = thread::Builder::new()
            .name("litellm-catalog-refresh".to_owned())
            .spawn(move || {
                tracing::dispatcher::with_default(&dispatch, || {
                    let _entered = span.enter();
                    let runtime = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(runtime) => runtime,
                        Err(error) => {
                            if !worker_cancel.is_cancelled() {
                                install.checked(Checked::Failed {
                                    reason: error.to_string(),
                                });
                            }
                            return;
                        }
                    };
                    let (state, step) = Revalidation::new(source, &installed, policy, None);
                    drive(
                        &runtime,
                        state,
                        step,
                        fetch.as_ref(),
                        install.as_ref(),
                        seen,
                        &worker_cancel,
                        &remote_adopted,
                        policy,
                    );
                    runtime.shutdown_background();
                });
            });
        let handle = match worker {
            Ok(handle) => Some(handle),
            Err(error) => {
                self.install.checked(Checked::Failed {
                    reason: error.to_string(),
                });
                None
            }
        };
        Worker {
            cancel,
            _thread: handle,
        }
    }
}

impl Drop for Refresher {
    fn drop(&mut self) {
        self.stop();
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "the driver owns one refresh's fixed dependencies"
)]
fn drive(
    runtime: &Runtime,
    mut state: Revalidation,
    mut step: Step,
    fetch: &dyn Fetch,
    install: &dyn Install,
    seen: u64,
    cancel: &CancellationToken,
    remote_adopted: &AtomicBool,
    policy: RetryPolicy,
) {
    loop {
        if cancel.is_cancelled() {
            return;
        }
        match step {
            Step::Fetch(request) => {
                let reply = runtime.block_on(async {
                    tokio::select! {
                        biased;
                        _ = cancel.cancelled() => None,
                        reply = fetch.fetch(request) => Some(reply),
                    }
                });
                let Some(reply) = reply else { return };
                if cancel.is_cancelled() {
                    return;
                }
                step = state.on_reply(reply);
            }
            Step::Sleep(delay) => {
                runtime.block_on(async {
                    tokio::select! {
                        biased;
                        _ = cancel.cancelled() => {},
                        _ = tokio::time::sleep(delay) => {},
                    }
                });
                if cancel.is_cancelled() {
                    return;
                }
                step = state.on_slept();
            }
            Step::Adopt {
                mut installed,
                errors,
            } => {
                let catalog = Arc::unwrap_or_clone(installed.catalog);
                let (catalog, left_out) = catalog.without_rejected(&errors);
                let reference = Catalog::parse(BUNDLED_REGISTRY)
                    .expect("bundled registry must parse")
                    .value
                    .model_count();
                if let Err(error) = catalog.check_size(reference, policy.integrity) {
                    if !cancel.is_cancelled() && install.generation() == seen {
                        install.checked(Checked::Failed {
                            reason: error.to_string(),
                        });
                    }
                    return;
                }
                installed.catalog = Arc::new(catalog);
                let paths: HashSet<_> = left_out
                    .into_iter()
                    .map(|error| error.path.clone())
                    .collect();
                let adopted = Adopted {
                    installed,
                    left_out: errors
                        .into_iter()
                        .filter(|error| paths.contains(&error.path))
                        .collect(),
                };
                if cancel.is_cancelled() {
                    return;
                }
                if install.generation() != seen || install.install(seen, adopted).is_err() {
                    tracing::debug!(outcome = "stale", "catalog refresh superseded");
                } else {
                    remote_adopted.store(true, Ordering::Release);
                }
                return;
            }
            Step::Done { reason } => {
                if !cancel.is_cancelled() && install.generation() == seen {
                    install.checked(match reason {
                        Some(reason) => Checked::Failed { reason },
                        None => Checked::NotModified,
                    });
                }
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use litellm_model_catalog::{FetchRequest, IntegrityLimits, Reply};
    use rstest::{fixture, rstest};
    use std::{
        sync::{Mutex, mpsc},
        time::{Duration, Instant},
    };
    use tokio::sync::oneshot;

    struct ScriptedFetch {
        requests: mpsc::Sender<(FetchRequest, oneshot::Sender<Reply>)>,
    }

    #[async_trait]
    impl Fetch for ScriptedFetch {
        async fn fetch(&self, request: FetchRequest) -> Reply {
            let (send, receive) = oneshot::channel();
            self.requests.send((request, send)).unwrap();
            receive.await.unwrap()
        }
    }

    #[derive(Default)]
    struct HostState {
        generation: u64,
        adopted: Option<Adopted>,
        checked: Vec<Checked>,
        install_calls: usize,
    }

    #[derive(Default)]
    struct Host {
        state: Mutex<HostState>,
        race_on_install: AtomicBool,
    }

    impl Install for Host {
        fn generation(&self) -> u64 {
            self.state.lock().unwrap().generation
        }

        fn install(&self, seen: u64, adopted: Adopted) -> Result<u64, Stale> {
            assert!(tokio::runtime::Handle::try_current().is_err());
            let mut state = self.state.lock().unwrap();
            state.install_calls += 1;
            if self.race_on_install.load(Ordering::Relaxed) {
                state.generation += 1;
            }
            if seen != state.generation {
                return Err(Stale);
            }
            state.adopted = Some(adopted);
            state.generation += 1;
            Ok(state.generation)
        }

        fn checked(&self, outcome: Checked) {
            self.state.lock().unwrap().checked.push(outcome);
        }
    }

    struct Harness {
        fetch: Arc<ScriptedFetch>,
        requests: mpsc::Receiver<(FetchRequest, oneshot::Sender<Reply>)>,
        host: Arc<Host>,
        initial: Installed,
        policy: RetryPolicy,
    }

    impl Harness {
        fn start(&self) -> Refresher {
            Refresher::with_policy(
                SourceId("test:catalog".to_owned()),
                self.fetch.clone(),
                &self.initial,
                self.host.clone(),
                self.policy,
            )
        }

        fn request(&self) -> (FetchRequest, oneshot::Sender<Reply>) {
            self.requests.recv_timeout(Duration::from_secs(5)).unwrap()
        }
    }

    #[fixture]
    fn harness() -> Harness {
        let (requests, received) = mpsc::channel();
        Harness {
            fetch: Arc::new(ScriptedFetch { requests }),
            requests: received,
            host: Arc::new(Host::default()),
            initial: Installed {
                catalog: Arc::new(
                    Catalog::parse(br#"{"base":{"litellm_provider":"test"}}"#)
                        .unwrap()
                        .value,
                ),
                revision: "base".to_owned(),
                validator: Some("\"old\"".to_owned()),
                origin: Origin::Bundled,
            },
            policy: RetryPolicy {
                max_attempts: 2,
                backoff_base_seconds: 0.0,
                jitter_min_seconds: 0.0,
                jitter_max_seconds: 0.0,
                integrity: IntegrityLimits::new(1, 0.0).unwrap(),
                ..RetryPolicy::default()
            },
        }
    }

    fn fetched(body: &'static str) -> Reply {
        Reply::Fetched {
            body: body.into(),
            validator: Some("\"new\"".to_owned()),
        }
    }

    const VALID: &str = r#"{"good":{"litellm_provider":"test","input_cost_per_token":0.5}}"#;
    const MIXED: &str = r#"{"good":{"litellm_provider":"test","input_cost_per_token":0.5},"bad":{"litellm_provider":"test","mode":"chatt"}}"#;

    fn finish(refresher: &Refresher) {
        let thread = refresher
            .process
            .get_or_init(|| None)
            .as_ref()
            .unwrap()
            ._thread
            .as_ref()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while !thread.is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
        assert!(thread.is_finished(), "refresh worker did not exit");
    }

    #[rstest]
    fn held_reply_is_adopted_after_release(harness: Harness) {
        let refresher = harness.start();
        let (request, send) = harness.request();
        assert_eq!(request.if_none_match, harness.initial.validator);
        assert_eq!(request.timeout, harness.policy.timeout);
        assert_eq!(harness.host.generation(), 0);
        send.send(fetched(VALID)).unwrap();
        finish(&refresher);
        let state = harness.host.state.lock().unwrap();
        assert_eq!(state.generation, 1);
        assert_eq!(state.install_calls, 1);
        let adopted = state.adopted.as_ref().unwrap();
        assert_eq!(
            adopted.installed.revision,
            litellm_model_catalog::git_blob_id(VALID.as_bytes())
        );
        assert_eq!(adopted.installed.validator.as_deref(), Some("\"new\""));
        assert_eq!(
            adopted.installed.origin,
            Origin::Remote {
                source: SourceId("test:catalog".to_owned())
            }
        );
        assert!(adopted.installed.catalog.lookup("good").is_some());
        assert!(adopted.left_out.is_empty());
        assert!(state.checked.is_empty());
        assert!(harness.requests.try_recv().is_err());
    }

    #[rstest]
    #[case::not_modified(Reply::NotModified, Checked::NotModified)]
    #[case::failed(Reply::Failed { reason: "unavailable".to_owned() }, Checked::Failed { reason: "unavailable".to_owned() })]
    fn terminal_check_never_installs(
        harness: Harness,
        #[case] reply: Reply,
        #[case] expected: Checked,
    ) {
        let refresher = harness.start();
        harness.request().1.send(reply).unwrap();
        finish(&refresher);
        let state = harness.host.state.lock().unwrap();
        assert_eq!(state.install_calls, 0);
        assert_eq!(state.generation, 0);
        assert_eq!(state.checked, vec![expected]);
    }

    #[rstest]
    #[case::unchanged(false)]
    #[case::changed_between_attempts(true)]
    fn retry_uses_the_same_revalidation_and_generation(harness: Harness, #[case] changed: bool) {
        let refresher = harness.start();
        let (first, send) = harness.request();
        if changed {
            harness.host.state.lock().unwrap().generation += 1;
        }
        send.send(Reply::Retry {
            after: Some(Duration::ZERO),
            reason: "temporary".to_owned(),
        })
        .unwrap();
        let (second, send) = harness.request();
        assert_eq!(first, second);
        send.send(fetched(VALID)).unwrap();
        finish(&refresher);
        assert_eq!(harness.host.generation(), 1);
        assert_eq!(
            harness.host.state.lock().unwrap().adopted.is_some(),
            !changed
        );
        assert!(harness.host.state.lock().unwrap().checked.is_empty());
    }

    #[rstest]
    #[case::before_reply(false, 0)]
    #[case::at_install(true, 1)]
    fn newer_install_fences_the_response(
        harness: Harness,
        #[case] race: bool,
        #[case] calls: usize,
    ) {
        let refresher = harness.start();
        let (_, send) = harness.request();
        if race {
            harness.host.race_on_install.store(true, Ordering::Relaxed);
        } else {
            harness.host.state.lock().unwrap().generation += 1;
        }
        send.send(fetched(VALID)).unwrap();
        finish(&refresher);
        let state = harness.host.state.lock().unwrap();
        assert_eq!(state.generation, 1);
        assert_eq!(state.install_calls, calls);
        assert!(state.adopted.is_none());
        assert!(state.checked.is_empty());
    }

    #[rstest]
    #[case::not_modified(Reply::NotModified)]
    #[case::failed(Reply::Failed { reason: "old check".to_owned() })]
    fn stale_checks_do_not_update_newer_state(harness: Harness, #[case] reply: Reply) {
        let refresher = harness.start();
        let (_, send) = harness.request();
        harness.host.state.lock().unwrap().generation += 1;
        send.send(reply).unwrap();
        finish(&refresher);
        assert!(harness.host.state.lock().unwrap().checked.is_empty());
    }

    #[rstest]
    #[case::valid_rows_remain(1, true)]
    #[case::filtered_below_minimum(2, false)]
    fn rejected_rows_are_removed_before_size_check(
        mut harness: Harness,
        #[case] minimum: usize,
        #[case] succeeds: bool,
    ) {
        harness.policy.integrity = IntegrityLimits::new(minimum, 0.0).unwrap();
        let refresher = harness.start();
        harness.request().1.send(fetched(MIXED)).unwrap();
        finish(&refresher);
        let state = harness.host.state.lock().unwrap();
        if succeeds {
            let adopted = state.adopted.as_ref().unwrap();
            assert!(adopted.installed.catalog.lookup("good").is_some());
            assert!(adopted.installed.catalog.lookup("bad").is_none());
            assert_eq!(adopted.left_out.len(), 1);
            assert_eq!(adopted.left_out[0].path, "/bad/mode");
        } else {
            assert_eq!(state.generation, 0);
            assert_eq!(state.install_calls, 0);
            assert!(matches!(state.checked.as_slice(), [Checked::Failed { .. }]));
        }
    }

    #[rstest]
    fn invalid_json_preserves_the_installed_catalog(harness: Harness) {
        let refresher = harness.start();
        harness.request().1.send(fetched("{invalid")).unwrap();
        finish(&refresher);
        let state = harness.host.state.lock().unwrap();
        assert_eq!(state.generation, 0);
        assert!(state.adopted.is_none());
        assert!(matches!(state.checked.as_slice(), [Checked::Failed { .. }]));
    }

    #[rstest]
    #[case::stop(false)]
    #[case::drop(true)]
    fn held_fetch_is_cancelled_promptly(harness: Harness, #[case] drop_handle: bool) {
        let refresher = harness.start();
        let (_, send) = harness.request();
        let before = Instant::now();
        if drop_handle {
            drop(refresher);
        } else {
            refresher.stop();
            refresher.stop();
            finish(&refresher);
        }
        assert!(before.elapsed() < Duration::from_secs(1));
        let deadline = Instant::now() + Duration::from_secs(1);
        while !send.is_closed() && Instant::now() < deadline {
            thread::yield_now();
        }
        assert!(send.is_closed(), "fetch future was not dropped");
        let state = harness.host.state.lock().unwrap();
        assert_eq!(state.install_calls, 0);
        assert!(state.checked.is_empty());
    }

    #[rstest]
    fn sleeping_refresh_is_cancelled_promptly(harness: Harness) {
        let refresher = harness.start();
        harness
            .request()
            .1
            .send(Reply::Retry {
                after: Some(Duration::from_secs(30)),
                reason: "temporary".to_owned(),
            })
            .unwrap();
        assert!(matches!(
            harness.requests.recv_timeout(Duration::from_millis(50)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        let before = Instant::now();
        refresher.stop();
        finish(&refresher);
        assert!(before.elapsed() < Duration::from_secs(1));
        assert_eq!(harness.host.generation(), 0);
        assert!(harness.host.state.lock().unwrap().checked.is_empty());
        assert!(harness.requests.try_recv().is_err());
    }

    #[rstest]
    fn same_process_after_fork_is_a_noop(harness: Harness) {
        let refresher = harness.start();
        let (_, send) = harness.request();
        refresher.after_fork();
        send.send(fetched(VALID)).unwrap();
        finish(&refresher);
        refresher.after_fork();
        assert_eq!(harness.host.generation(), 1);
        assert!(harness.requests.try_recv().is_err());
    }
}
