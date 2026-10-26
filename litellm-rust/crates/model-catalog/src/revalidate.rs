use std::sync::{Arc, LazyLock};
use std::time::Duration;

use bytes::Bytes;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sha1::{Digest, Sha1};

use crate::bundled::BUNDLED_REGISTRY;
use crate::catalog::{Catalog, IntegrityLimits};
use crate::error::{Loaded, Located};

#[derive(Clone, Debug)]
pub struct Installed {
    pub catalog: Arc<Catalog>,
    pub revision: String,
    pub validator: Option<String>,
    pub origin: Origin,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SourceId(pub String);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Origin {
    Bundled,
    Remote { source: SourceId },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FetchRequest {
    pub if_none_match: Option<String>,
    pub timeout: Duration,
}

#[derive(Clone, Debug)]
pub enum Reply {
    Fetched {
        body: Bytes,
        validator: Option<String>,
    },
    NotModified,
    Retry {
        after: Option<Duration>,
        reason: String,
    },
    Failed {
        reason: String,
    },
}

#[derive(Debug)]
pub enum Step {
    Fetch(FetchRequest),
    Sleep(Duration),
    Adopt {
        installed: Installed,
        errors: Vec<Located>,
    },
    Done {
        reason: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetryPolicy {
    pub max_attempts: usize,
    pub max_wait: Duration,
    pub timeout: Duration,
    pub backoff_base_seconds: f64,
    pub jitter_min_seconds: f64,
    pub jitter_max_seconds: f64,
    pub integrity: IntegrityLimits,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            max_wait: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            backoff_base_seconds: 2.0,
            jitter_min_seconds: 0.0,
            jitter_max_seconds: 1.0,
            integrity: IntegrityLimits::DEFAULT,
        }
    }
}

impl RetryPolicy {
    fn is_valid(self) -> bool {
        self.max_attempts > 0
            && self.backoff_base_seconds.is_finite()
            && self.backoff_base_seconds >= 0.0
            && self.jitter_min_seconds.is_finite()
            && self.jitter_min_seconds >= 0.0
            && self.jitter_max_seconds.is_finite()
            && self.jitter_max_seconds >= self.jitter_min_seconds
    }
}

#[derive(Debug)]
pub struct Revalidation {
    source: SourceId,
    if_none_match: Option<String>,
    policy: RetryPolicy,
    attempts: usize,
    rng: StdRng,
    state: State,
}

#[derive(Debug)]
enum State {
    AwaitingReply,
    AwaitingSleep,
    Done(Option<String>),
}

impl Revalidation {
    pub fn new(
        source: SourceId,
        installed: &Installed,
        policy: RetryPolicy,
        seed: Option<u64>,
    ) -> (Self, Step) {
        let if_none_match = match &installed.origin {
            Origin::Remote { source: from } if *from != source => None,
            Origin::Bundled | Origin::Remote { .. } => installed.validator.clone(),
        };
        let mut revalidation = Self {
            source,
            if_none_match,
            policy,
            attempts: 1,
            rng: seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64),
            state: State::AwaitingReply,
        };
        let step = if policy.is_valid() {
            revalidation.fetch()
        } else {
            revalidation.finish(Some("invalid retry policy".to_owned()))
        };
        (revalidation, step)
    }

    pub fn on_reply(&mut self, reply: Reply) -> Step {
        if !matches!(&self.state, State::AwaitingReply) {
            return self.invalid_state("revalidation is not awaiting a reply");
        }
        match reply {
            Reply::Fetched { body, validator } => self.adopt(&body, validator),
            Reply::NotModified => self.finish(None),
            Reply::Retry { after, reason } => self.retry_or_done(reason, after),
            Reply::Failed { reason } => self.finish(Some(reason)),
        }
    }

    pub fn on_slept(&mut self) -> Step {
        if !matches!(&self.state, State::AwaitingSleep) {
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

    fn adopt(&mut self, body: &[u8], validator: Option<String>) -> Step {
        let Loaded { value, errors } = match Catalog::parse(body) {
            Ok(loaded) => loaded,
            Err(error) => return self.finish(Some(error.to_string())),
        };
        let served = value.model_count_without_rejected(&errors);
        if let Err(error) = self.policy.integrity.check(served, bundled_model_count()) {
            return self.finish(Some(error.to_string()));
        }
        self.state = State::Done(None);
        Step::Adopt {
            installed: Installed {
                catalog: Arc::new(value),
                revision: git_blob_id(body),
                validator,
                origin: Origin::Remote {
                    source: self.source.clone(),
                },
            },
            errors,
        }
    }

    fn retry_or_done(&mut self, reason: String, after: Option<Duration>) -> Step {
        if self.attempts >= self.policy.max_attempts {
            return self.finish(Some(reason));
        }
        let delay = after.unwrap_or_else(|| self.backoff());
        self.state = State::AwaitingSleep;
        Step::Sleep(delay.min(self.policy.max_wait))
    }

    fn backoff(&mut self) -> Duration {
        let exponent = i32::try_from(self.attempts).unwrap_or(i32::MAX);
        let seconds = self.policy.backoff_base_seconds.powi(exponent)
            + self
                .rng
                .gen_range(self.policy.jitter_min_seconds..=self.policy.jitter_max_seconds);
        Duration::try_from_secs_f64(seconds.min(self.policy.max_wait.as_secs_f64()))
            .unwrap_or(self.policy.max_wait)
    }

    fn finish(&mut self, reason: Option<String>) -> Step {
        self.state = State::Done(reason.clone());
        Step::Done { reason }
    }

    fn invalid_state(&mut self, reason: &str) -> Step {
        if let State::Done(previous) = &self.state {
            return Step::Done {
                reason: previous.clone(),
            };
        }
        self.finish(Some(reason.to_owned()))
    }
}

fn bundled_model_count() -> usize {
    static COUNT: LazyLock<usize> = LazyLock::new(|| {
        Catalog::parse(BUNDLED_REGISTRY)
            .expect("bundled registry must parse")
            .value
            .model_count()
    });
    *COUNT
}

pub fn git_blob_id(body: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(format!("blob {}\0", body.len()).as_bytes());
    hasher.update(body);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serde_json::Value;

    const SOURCE: &str = "https://example.test/catalog.json";
    const TWO_ROWS: &str =
        r#"{"good":{"litellm_provider":"test"},"bad":{"litellm_provider":"test","mode":"chatt"}}"#;

    fn source(id: &str) -> SourceId {
        SourceId(id.to_owned())
    }

    fn installed(validator: Option<&str>, origin: Origin) -> Installed {
        Installed {
            catalog: Arc::new(
                Catalog::parse(br#"{"base":{"litellm_provider":"test"}}"#)
                    .unwrap()
                    .value,
            ),
            revision: "bundled".into(),
            validator: validator.map(str::to_owned),
            origin,
        }
    }

    fn policy(
        max_attempts: usize,
        min_model_count: usize,
        min_reference_ratio: f64,
    ) -> RetryPolicy {
        RetryPolicy {
            max_attempts,
            integrity: IntegrityLimits::new(min_model_count, min_reference_ratio).unwrap(),
            ..RetryPolicy::default()
        }
    }

    fn start(policy: RetryPolicy, seed: u64) -> Revalidation {
        let (revalidation, step) = Revalidation::new(
            source(SOURCE),
            &installed(None, Origin::Bundled),
            policy,
            Some(seed),
        );
        assert!(matches!(step, Step::Fetch(_)), "{step:?}");
        revalidation
    }

    fn fetched(body: &str) -> Reply {
        Reply::Fetched {
            body: Bytes::copy_from_slice(body.as_bytes()),
            validator: None,
        }
    }

    fn retry(after: Option<u64>) -> Reply {
        Reply::Retry {
            after: after.map(Duration::from_secs),
            reason: "busy".into(),
        }
    }

    fn done_reason(step: Step) -> Option<String> {
        match step {
            Step::Done { reason } => reason,
            step => panic!("expected Done, got {step:?}"),
        }
    }

    #[rstest]
    fn fetched_adopts_with_its_row_errors_validator_revision_and_source() {
        let mut revalidation = start(policy(3, 1, 0.0), 1);
        let step = revalidation.on_reply(Reply::Fetched {
            body: Bytes::from_static(TWO_ROWS.as_bytes()),
            validator: Some("tag".into()),
        });
        let Step::Adopt { installed, errors } = step else {
            panic!("expected Adopt, got {step:?}")
        };
        assert_eq!(installed.catalog.model_count(), 2);
        assert_eq!(
            errors
                .iter()
                .map(|error| error.path.as_str())
                .collect::<Vec<_>>(),
            ["/bad/mode"]
        );
        assert_eq!(installed.validator.as_deref(), Some("tag"));
        assert_eq!(installed.revision, git_blob_id(TWO_ROWS.as_bytes()));
        assert_eq!(
            installed.origin,
            Origin::Remote {
                source: source(SOURCE)
            }
        );
    }

    #[rstest]
    #[case::not_modified(Reply::NotModified, None)]
    #[case::failed(Reply::Failed { reason: "gone".into() }, Some("gone"))]
    fn replies_that_end_at_once(#[case] reply: Reply, #[case] reason: Option<&str>) {
        let mut revalidation = start(policy(3, 0, 0.0), 1);
        assert_eq!(done_reason(revalidation.on_reply(reply)).as_deref(), reason);
        assert_eq!(done_reason(revalidation.on_slept()).as_deref(), reason);
    }

    #[rstest]
    #[case::named_delay(Some(2), 2)]
    #[case::named_delay_capped_at_max_wait(Some(100), 30)]
    fn retry_sleeps_the_named_delay_then_fetches_again(
        #[case] after: Option<u64>,
        #[case] slept: u64,
    ) {
        let mut revalidation = start(policy(3, 0, 0.0), 1);
        assert!(matches!(
            revalidation.on_reply(retry(after)),
            Step::Sleep(duration) if duration == Duration::from_secs(slept)
        ));
        assert!(matches!(revalidation.on_slept(), Step::Fetch(_)));
    }

    #[rstest]
    fn retry_without_a_delay_sleeps_the_seeded_exponential_backoff() {
        let fixture: Value =
            serde_json::from_str(include_str!("../tests/git_blob_id.json")).unwrap();
        let first_retry = fixture["retry"]["first_retry_index"].as_i64().unwrap() as i32;
        let sleeps = |seed| {
            let mut revalidation = start(policy(3, 0, 0.0), seed);
            let first = revalidation.on_reply(retry(None));
            revalidation.on_slept();
            let second = revalidation.on_reply(retry(None));
            match (first, second) {
                (Step::Sleep(first), Step::Sleep(second)) => {
                    (first.as_secs_f64(), second.as_secs_f64())
                }
                other => panic!("expected two sleeps, got {other:?}"),
            }
        };
        let (first, second) = sleeps(42);
        assert_eq!((first, second), sleeps(42));
        let window = |index: i32| 2f64.powi(index)..=2f64.powi(index) + 1.0;
        assert!(window(first_retry).contains(&first), "{first}");
        assert!(window(first_retry + 1).contains(&second), "{second}");
    }

    #[rstest]
    fn attempts_run_out_into_done_with_the_last_reason() {
        let mut revalidation = start(policy(2, 0, 0.0), 3);
        assert!(matches!(
            revalidation.on_reply(retry(Some(1))),
            Step::Sleep(_)
        ));
        assert!(matches!(revalidation.on_slept(), Step::Fetch(_)));
        let last = Reply::Retry {
            after: Some(Duration::from_secs(1)),
            reason: "still busy".into(),
        };
        assert_eq!(
            done_reason(revalidation.on_reply(last)).as_deref(),
            Some("still busy")
        );
    }

    #[rstest]
    #[case::below_minimum_after_rejected_rows(policy(3, 2, 0.0), "below minimum 2")]
    #[case::shrunk_against_the_bundled_count(policy(3, 0, 1.0), "reference count")]
    fn a_document_failing_the_integrity_check_is_not_adopted(
        #[case] policy: RetryPolicy,
        #[case] reason: &str,
    ) {
        let mut revalidation = start(policy, 8);
        let reason_text = done_reason(revalidation.on_reply(fetched(TWO_ROWS))).unwrap();
        assert!(reason_text.contains(reason), "{reason_text}");
    }

    #[rstest]
    #[case::invalid_json("{")]
    #[case::list_root("[]")]
    #[case::no_models("{}")]
    fn a_fatal_document_is_done_with_its_message(#[case] body: &str) {
        let mut revalidation = start(policy(3, 0, 0.0), 9);
        let expected = Catalog::parse(body.as_bytes()).unwrap_err().to_string();
        assert_eq!(
            done_reason(revalidation.on_reply(fetched(body))),
            Some(expected)
        );
    }

    #[rstest]
    #[case::bundled(Origin::Bundled, Some("tag"))]
    #[case::same_source(Origin::Remote { source: source(SOURCE) }, Some("tag"))]
    #[case::other_source(Origin::Remote { source: source("file:///other.json") }, None)]
    fn the_validator_is_sent_only_back_to_where_it_came_from(
        #[case] origin: Origin,
        #[case] expected: Option<&str>,
    ) {
        let (_, step) = Revalidation::new(
            source(SOURCE),
            &installed(Some("tag"), origin),
            RetryPolicy::default(),
            Some(1),
        );
        let Step::Fetch(FetchRequest {
            if_none_match,
            timeout,
        }) = step
        else {
            panic!("expected Fetch, got {step:?}")
        };
        assert_eq!(if_none_match.as_deref(), expected);
        assert_eq!(timeout, RetryPolicy::default().timeout);
    }

    #[rstest]
    #[case::zero_attempts(RetryPolicy { max_attempts: 0, ..RetryPolicy::default() })]
    #[case::nan_backoff(RetryPolicy { backoff_base_seconds: f64::NAN, ..RetryPolicy::default() })]
    #[case::negative_backoff(RetryPolicy { backoff_base_seconds: -1.0, ..RetryPolicy::default() })]
    #[case::negative_jitter(RetryPolicy { jitter_min_seconds: -1.0, ..RetryPolicy::default() })]
    #[case::infinite_jitter(RetryPolicy { jitter_max_seconds: f64::INFINITY, ..RetryPolicy::default() })]
    #[case::reversed_jitter(RetryPolicy { jitter_min_seconds: 2.0, ..RetryPolicy::default() })]
    fn an_invalid_policy_is_done_before_fetching(#[case] policy: RetryPolicy) {
        let (_, step) = Revalidation::new(
            source(SOURCE),
            &installed(None, Origin::Bundled),
            policy,
            Some(1),
        );
        assert_eq!(done_reason(step).as_deref(), Some("invalid retry policy"));
    }

    #[rstest]
    fn a_reply_out_of_turn_ends_the_revalidation() {
        let mut revalidation = start(policy(3, 0, 0.0), 1);
        assert!(matches!(
            revalidation.on_reply(retry(Some(1))),
            Step::Sleep(_)
        ));
        assert_eq!(
            done_reason(revalidation.on_reply(Reply::NotModified)).as_deref(),
            Some("revalidation is not awaiting a reply")
        );
    }

    #[rstest]
    fn the_shared_fixture_pins_the_default_policy_and_the_git_blob_formula() {
        let fixture: Value =
            serde_json::from_str(include_str!("../tests/git_blob_id.json")).unwrap();
        let retry = &fixture["retry"];
        let blob = &fixture["git_blob_id"];
        let seconds = |name: &str| retry[name].as_f64().unwrap();
        let default = RetryPolicy::default();

        assert_eq!(
            default,
            RetryPolicy {
                max_attempts: retry["max_attempts"].as_u64().unwrap() as usize,
                max_wait: Duration::from_secs_f64(seconds("max_wait_seconds")),
                timeout: Duration::from_secs_f64(seconds("timeout_seconds")),
                backoff_base_seconds: seconds("backoff_base_seconds"),
                jitter_min_seconds: seconds("jitter_min_seconds"),
                jitter_max_seconds: seconds("jitter_max_seconds"),
                integrity: IntegrityLimits::new(
                    retry["minimum_model_count"].as_u64().unwrap() as usize,
                    retry["minimum_reference_ratio"].as_f64().unwrap(),
                )
                .unwrap(),
            }
        );
        assert_eq!(
            git_blob_id(blob["example_body_utf8"].as_str().unwrap().as_bytes()),
            blob["example_revision"].as_str().unwrap()
        );
    }
}
