use std::sync::LazyLock;
use std::time::Duration;

use litellm_model_catalog::{BUNDLED_REGISTRY, Catalog};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sha1::{Digest, Sha1};

use crate::policy::RetryPolicy;
use crate::source::Source;
use crate::step::{FetchRequest, Installed, Origin, Reply, Step};

/// Main's `RETRYABLE_FETCH_STATUS_CODES`
const RETRYABLE_STATUSES: [u16; 5] = [429, 500, 502, 503, 504];

static BUNDLED_MODEL_COUNT: LazyLock<usize> = LazyLock::new(|| {
    Catalog::parse(BUNDLED_REGISTRY)
        .expect("the bundled model registry parses")
        .value
        .model_count()
});

/// The fetch state machine. It asks its driver for an attempt or a sleep and never does IO itself
#[derive(Debug)]
pub struct Revalidation {
    source: Source,
    if_none_match: Option<String>,
    policy: RetryPolicy,
    attempts: u32,
    rng: StdRng,
    state: State,
}

#[derive(Debug)]
enum State {
    AwaitingReply,
    AwaitingSleep,
    Done(String, Option<Duration>),
}

impl Revalidation {
    pub fn new(
        source: &Source,
        installed: &Installed,
        policy: RetryPolicy,
        seed: Option<u64>,
    ) -> (Self, Step) {
        let if_none_match = match &installed.origin {
            Origin::Remote { source: from } if from == source => installed.etag.clone(),
            Origin::Remote { .. } | Origin::Bundled => None,
        };
        let revalidation = Self {
            source: source.clone(),
            if_none_match,
            policy,
            attempts: 1,
            rng: seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64),
            state: State::AwaitingReply,
        };
        let step = revalidation.fetch();
        (revalidation, step)
    }

    pub fn on_reply(&mut self, reply: Reply) -> Step {
        if !matches!(self.state, State::AwaitingReply) {
            return self.invalid_state("revalidation is not awaiting a reply");
        }
        match reply {
            Reply::Transport { reason } => self.retry_or_done(reason, None),
            Reply::Response {
                status: 200,
                etag,
                body,
                ..
            } => self.adopt(body.unwrap_or_default(), etag),
            Reply::Response { status: 304, .. } => {
                self.finish(format!("{} is not modified", self.source))
            }
            Reply::Response {
                status,
                retry_after,
                ..
            } if RETRYABLE_STATUSES.contains(&status) => {
                self.retry_or_done(format!("HTTP {status} from {}", self.source), retry_after)
            }
            Reply::Response { status, .. } => {
                self.finish(format!("HTTP {status} from {}", self.source))
            }
        }
    }

    pub fn on_slept(&mut self) -> Step {
        if !matches!(self.state, State::AwaitingSleep) {
            return self.invalid_state("revalidation is not awaiting a sleep");
        }
        self.attempts += 1;
        self.state = State::AwaitingReply;
        self.fetch()
    }

    fn fetch(&self) -> Step {
        Step::Fetch(FetchRequest {
            if_none_match: self.if_none_match.clone(),
            timeout: self.policy.timeout,
        })
    }

    fn adopt(&mut self, body: Vec<u8>, etag: Option<String>) -> Step {
        let loaded = match Catalog::parse(&body) {
            Ok(loaded) => loaded,
            Err(error) => {
                return self.finish(format!(
                    "model cost map from {} is unusable: {error}",
                    self.source
                ));
            }
        };
        if let Err(error) = loaded
            .value
            .check_size(*BUNDLED_MODEL_COUNT, self.policy.limits)
        {
            return self.finish(format!(
                "model cost map from {} is unusable: {error}",
                self.source
            ));
        }
        let installed = Installed {
            origin: Origin::Remote {
                source: self.source.clone(),
            },
            etag,
            revision: git_blob_id(&body),
            model_count: loaded.value.model_count(),
        };
        self.state = State::Done("adopted".to_owned(), None);
        Step::Adopt {
            catalog: Box::new(loaded.value),
            installed,
            errors: loaded.errors,
        }
    }

    fn retry_or_done(&mut self, reason: String, retry_after: Option<Duration>) -> Step {
        let delay = retry_after
            .unwrap_or_else(|| self.backoff())
            .min(self.policy.max_wait);
        if self.attempts >= self.policy.max_attempts {
            let reason = if self.policy.max_attempts > 1 {
                format!("{reason} (after {} attempts)", self.policy.max_attempts)
            } else {
                reason
            };
            self.state = State::Done(reason.clone(), Some(delay));
            return Step::Done {
                reason,
                retry_after: Some(delay),
            };
        }
        self.state = State::AwaitingSleep;
        Step::Sleep(delay)
    }

    fn backoff(&mut self) -> Duration {
        let jitter = if self.policy.jitter_max_seconds > self.policy.jitter_min_seconds {
            self.rng
                .gen_range(self.policy.jitter_min_seconds..=self.policy.jitter_max_seconds)
        } else {
            self.policy.jitter_min_seconds
        };
        let seconds = self.policy.backoff_base_seconds.powi(self.attempts as i32) + jitter;
        Duration::try_from_secs_f64(seconds.min(self.policy.max_wait.as_secs_f64()))
            .unwrap_or(self.policy.max_wait)
    }

    fn finish(&mut self, reason: String) -> Step {
        self.state = State::Done(reason.clone(), None);
        Step::Done {
            reason,
            retry_after: None,
        }
    }

    fn invalid_state(&mut self, reason: &str) -> Step {
        if let State::Done(previous, retry_after) = &self.state {
            return Step::Done {
                reason: previous.clone(),
                retry_after: *retry_after,
            };
        }
        self.finish(reason.to_owned())
    }
}

/// The digest `git hash-object` prints, which main records as the source revision
pub fn git_blob_id(body: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(format!("blob {}\0", body.len()).as_bytes());
    hasher.update(body);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};

    use super::*;

    fn source() -> Source {
        Source::parse("https://example.com/map.json").unwrap()
    }

    #[fixture]
    fn bundled() -> Installed {
        Installed {
            origin: Origin::Bundled,
            etag: None,
            revision: git_blob_id(BUNDLED_REGISTRY),
            model_count: *BUNDLED_MODEL_COUNT,
        }
    }

    fn response(status: u16) -> Reply {
        Reply::Response {
            status,
            etag: None,
            retry_after: None,
            body: None,
        }
    }

    fn done(step: Step) -> String {
        match step {
            Step::Done { reason, .. } => reason,
            other => panic!("expected done, got {other:?}"),
        }
    }

    fn slept(step: Step) -> Duration {
        match step {
            Step::Sleep(wait) => wait,
            other => panic!("expected a sleep, got {other:?}"),
        }
    }

    fn started(installed: &Installed) -> Revalidation {
        Revalidation::new(&source(), installed, RetryPolicy::default(), Some(0)).0
    }

    #[rstest]
    #[case::too_many_requests(429)]
    #[case::internal_error(500)]
    #[case::bad_gateway(502)]
    #[case::unavailable(503)]
    #[case::gateway_timeout(504)]
    fn mains_retryable_statuses_sleep_before_the_next_attempt(
        bundled: Installed,
        #[case] status: u16,
    ) {
        let mut revalidation = started(&bundled);
        assert!(matches!(
            revalidation.on_reply(response(status)),
            Step::Sleep(_)
        ));
    }

    #[rstest]
    #[case::not_implemented(501)]
    #[case::version_not_supported(505)]
    #[case::not_found(404)]
    fn other_statuses_are_final(bundled: Installed, #[case] status: u16) {
        let mut revalidation = started(&bundled);
        assert_eq!(
            done(revalidation.on_reply(response(status))),
            format!("HTTP {status} from https://example.com/map.json")
        );
    }

    #[rstest]
    #[case::honored(Duration::from_secs(7), Duration::from_secs(7))]
    #[case::capped(Duration::from_secs(120), Duration::from_secs(30))]
    fn retry_after_sets_the_wait_up_to_the_cap(
        bundled: Installed,
        #[case] header: Duration,
        #[case] wait: Duration,
    ) {
        let mut revalidation = started(&bundled);
        let reply = Reply::Response {
            status: 429,
            etag: None,
            retry_after: Some(header),
            body: None,
        };
        assert_eq!(slept(revalidation.on_reply(reply)), wait);
    }

    #[rstest]
    fn backoff_grows_from_two_seconds_with_jitter(bundled: Installed) {
        let mut revalidation = started(&bundled);
        let first = slept(revalidation.on_reply(response(503)));
        revalidation.on_slept();
        let second = slept(revalidation.on_reply(response(503)));
        assert!((2.0..=3.0).contains(&first.as_secs_f64()), "{first:?}");
        assert!((4.0..=5.0).contains(&second.as_secs_f64()), "{second:?}");
    }

    #[rstest]
    fn the_last_attempt_names_how_many_were_made(bundled: Installed) {
        let mut revalidation = started(&bundled);
        revalidation.on_reply(response(503));
        revalidation.on_slept();
        revalidation.on_reply(response(503));
        revalidation.on_slept();
        assert_eq!(
            done(revalidation.on_reply(Reply::Transport {
                reason: "connection refused".into()
            })),
            "connection refused (after 3 attempts)"
        );
    }

    #[rstest]
    fn a_single_attempt_reports_the_wait_a_retry_should_use(bundled: Installed) {
        let policy = RetryPolicy {
            max_attempts: 1,
            ..RetryPolicy::default()
        };
        let (mut revalidation, _) = Revalidation::new(&source(), &bundled, policy, Some(0));
        let reply = Reply::Response {
            status: 503,
            etag: None,
            retry_after: Some(Duration::from_secs(4)),
            body: None,
        };
        let Step::Done {
            reason,
            retry_after,
        } = revalidation.on_reply(reply)
        else {
            panic!("expected done");
        };
        assert_eq!(reason, "HTTP 503 from https://example.com/map.json");
        assert_eq!(retry_after, Some(Duration::from_secs(4)));
    }

    #[rstest]
    fn a_body_that_is_not_a_catalog_is_final(bundled: Installed) {
        let mut revalidation = started(&bundled);
        let reply = Reply::Response {
            status: 200,
            etag: None,
            retry_after: None,
            body: Some(b"not json".to_vec()),
        };
        let reason = done(revalidation.on_reply(reply));
        assert!(reason.contains("invalid JSON"), "{reason}");
    }

    #[rstest]
    fn the_bundled_body_is_adopted_with_its_revision_and_etag(bundled: Installed) {
        let mut revalidation = started(&bundled);
        let reply = Reply::Response {
            status: 200,
            etag: Some("W/\"abc\"".into()),
            retry_after: None,
            body: Some(BUNDLED_REGISTRY.to_vec()),
        };
        let Step::Adopt { installed, .. } = revalidation.on_reply(reply) else {
            panic!("expected adopt");
        };
        assert_eq!(installed.revision, bundled.revision);
        assert_eq!(installed.etag.as_deref(), Some("W/\"abc\""));
        assert_eq!(installed.origin, Origin::Remote { source: source() });
    }

    #[rstest]
    #[case::same_source(Origin::Remote { source: source() }, Some("v1"))]
    #[case::another_source(
        Origin::Remote { source: Source::parse("https://example.com/other.json").unwrap() },
        None
    )]
    #[case::bundled(Origin::Bundled, None)]
    fn the_etag_is_sent_only_back_to_the_source_that_gave_it(
        bundled: Installed,
        #[case] origin: Origin,
        #[case] expected: Option<&str>,
    ) {
        let installed = Installed {
            origin,
            etag: Some("v1".into()),
            ..bundled
        };
        let (_, step) = Revalidation::new(&source(), &installed, RetryPolicy::default(), Some(0));
        let Step::Fetch(request) = step else {
            panic!("expected a fetch");
        };
        assert_eq!(request.if_none_match.as_deref(), expected);
    }

    #[rstest]
    fn git_blob_id_is_what_git_hash_object_prints() {
        assert_eq!(
            git_blob_id(b"{\"gpt-5.4-mini\": {\"mode\": \"chat\"}}\n"),
            "18b9a8381e13a3b38a2128f184f631f95829e987"
        );
    }
}
