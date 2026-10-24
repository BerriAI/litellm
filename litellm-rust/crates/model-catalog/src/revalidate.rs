use std::sync::{Arc, LazyLock};
use std::time::Duration;

use bytes::Bytes;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sha1::{Digest, Sha1};

use crate::bundled::BUNDLED_REGISTRY;
use crate::catalog::{Catalog, IntegrityLimits};
use crate::error::{Loaded, Located};

pub const DEFAULT_REGISTRY_URL: &str =
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";

const FIRST_RETRY_INDEX: usize = 1;

#[derive(Clone, Debug)]
pub struct Installed {
    pub catalog: Arc<Catalog>,
    pub revision: String,
    pub etag: Option<String>,
    pub origin: Origin,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Origin {
    Bundled,
    Remote { url: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FetchRequest {
    pub url: String,
    pub if_none_match: Option<String>,
    pub timeout: Duration,
}

#[derive(Clone, Debug)]
pub enum Reply {
    Response {
        status: u16,
        etag: Option<String>,
        retry_after: Option<String>,
        body: Bytes,
    },
    Transport {
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
    pub min_model_count: usize,
    pub min_reference_ratio: f64,
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
            min_model_count: 50,
            min_reference_ratio: 0.5,
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
            && (0.0..=1.0).contains(&self.min_reference_ratio)
    }
}

#[derive(Debug)]
pub struct Revalidation {
    url: String,
    installed: Installed,
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
        url: String,
        installed: &Installed,
        policy: RetryPolicy,
        seed: Option<u64>,
    ) -> (Self, Step) {
        let mut revalidation = Self {
            url,
            installed: installed.clone(),
            policy,
            attempts: FIRST_RETRY_INDEX,
            rng: seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64),
            state: State::AwaitingReply,
        };
        if revalidation.url.starts_with("s3://") {
            let step = revalidation.finish(Some("s3 is fetched by the host".to_owned()));
            return (revalidation, step);
        }
        if !revalidation.policy.is_valid() {
            let step = revalidation.finish(Some("invalid retry policy".to_owned()));
            return (revalidation, step);
        }
        let step = revalidation.fetch();
        (revalidation, step)
    }

    pub fn on_reply(&mut self, reply: Reply) -> Step {
        if !matches!(&self.state, State::AwaitingReply) {
            return self.invalid_state("revalidation is not awaiting a reply");
        }
        match reply {
            Reply::Transport { reason } => self.retry_or_done(reason, None),
            Reply::Response {
                status,
                etag,
                retry_after,
                body,
            } => match status {
                200 => self.adopt(body, etag),
                304 => self.finish(None),
                429 | 500..=599 => self.retry_or_done(format!("HTTP status {status}"), retry_after),
                _ => self.finish(Some(format!("unexpected HTTP status {status}"))),
            },
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
        let if_none_match = match &self.installed.origin {
            Origin::Bundled => self.installed.etag.clone(),
            Origin::Remote { url } if url == &self.url => self.installed.etag.clone(),
            Origin::Remote { .. } => None,
        };
        Step::Fetch(FetchRequest {
            url: self.url.clone(),
            if_none_match,
            timeout: self.policy.timeout,
        })
    }

    fn adopt(&mut self, body: Bytes, etag: Option<String>) -> Step {
        let Loaded { value, errors } = match Catalog::parse(&body) {
            Ok(loaded) => loaded,
            Err(error) => return self.finish(Some(error.to_string())),
        };
        let Some(limits) =
            IntegrityLimits::new(self.policy.min_model_count, self.policy.min_reference_ratio)
        else {
            return self.finish(Some("invalid integrity limits".to_owned()));
        };
        let (filtered, _) = value.clone().without_rejected(&errors);
        if let Err(error) = filtered.check_size(bundled_model_count(), limits) {
            return self.finish(Some(error.to_string()));
        }
        let revision = git_blob_id(&body);
        let installed = Installed {
            catalog: Arc::new(value),
            revision,
            etag,
            origin: crate::revalidate::Origin::Remote {
                url: self.url.clone(),
            },
        };
        self.state = State::Done(None);
        Step::Adopt { installed, errors }
    }

    fn retry_or_done(&mut self, reason: String, retry_after: Option<String>) -> Step {
        self.retry_or_done_with_delay(
            reason,
            retry_after.and_then(|value| parse_retry_after(&value, self.policy.max_wait)),
        )
    }

    fn retry_or_done_with_delay(&mut self, reason: String, retry_after: Option<Duration>) -> Step {
        if self.attempts >= self.policy.max_attempts {
            return self.finish(Some(reason));
        }
        let delay = retry_after.unwrap_or_else(|| self.backoff());
        self.state = State::AwaitingSleep;
        Step::Sleep(delay.min(self.policy.max_wait))
    }

    fn backoff(&mut self) -> Duration {
        let seconds = self.policy.backoff_base_seconds.powi(self.attempts as i32)
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

fn parse_retry_after(value: &str, max_wait: Duration) -> Option<Duration> {
    let seconds = value.parse::<f64>().ok()?;
    if !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    if seconds >= max_wait.as_secs_f64() {
        return Some(max_wait);
    }
    Duration::try_from_secs_f64(seconds).ok()
}

pub fn git_blob_id(body: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(format!("blob {}\0", body.len()).as_bytes());
    hasher.update(body);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn installed(document: &str, etag: Option<&str>) -> Installed {
        Installed {
            catalog: Arc::new(Catalog::parse(document.as_bytes()).unwrap().value),
            revision: "bundled".into(),
            etag: etag.map(str::to_owned),
            origin: Origin::Bundled,
        }
    }

    fn policy(min_model_count: usize, min_reference_ratio: f64) -> RetryPolicy {
        RetryPolicy {
            min_model_count,
            min_reference_ratio,
            ..RetryPolicy::default()
        }
    }

    fn response(status: u16, body: &str) -> Reply {
        Reply::Response {
            status,
            etag: None,
            retry_after: None,
            body: Bytes::copy_from_slice(body.as_bytes()),
        }
    }

    #[rstest]
    fn initial_request_uses_installed_etag_and_policy_timeout() {
        let installed = installed(r#"{"base":{"litellm_provider":"test"}}"#, Some("tag"));
        let (mut state, step) = Revalidation::new(
            "https://example.test/catalog.json".into(),
            &installed,
            policy(0, 0.0),
            Some(1),
        );
        assert!(matches!(
            step,
            Step::Fetch(FetchRequest {
                ref url,
                if_none_match: Some(ref etag),
                timeout
            }) if url == "https://example.test/catalog.json"
                && etag == "tag"
                && timeout == Duration::from_secs(5)
        ));
        assert!(matches!(
            state.on_reply(response(304, "")),
            Step::Done { reason: None }
        ));
        assert!(matches!(state.on_slept(), Step::Done { reason: None }));
    }

    #[rstest]
    fn remote_etags_are_only_reused_for_the_same_url() {
        let mut installed = installed(r#"{"base":{"litellm_provider":"test"}}"#, Some("tag"));
        installed.origin = Origin::Remote {
            url: "https://same.test/catalog.json".into(),
        };
        let (_, same_url) = Revalidation::new(
            "https://same.test/catalog.json".into(),
            &installed,
            policy(0, 0.0),
            Some(1),
        );
        assert!(matches!(
            same_url,
            Step::Fetch(FetchRequest {
                if_none_match: Some(ref etag),
                ..
            }) if etag == "tag"
        ));

        let (_, changed_url) = Revalidation::new(
            "https://changed.test/catalog.json".into(),
            &installed,
            policy(0, 0.0),
            Some(1),
        );
        assert!(matches!(
            changed_url,
            Step::Fetch(FetchRequest {
                if_none_match: None,
                ..
            })
        ));
    }

    #[rstest]
    #[case::zero_attempts(0, 2.0, 0.0, 1.0, 0.5)]
    #[case::nonfinite_backoff(3, f64::NAN, 0.0, 1.0, 0.5)]
    #[case::negative_backoff(3, -1.0, 0.0, 1.0, 0.5)]
    #[case::negative_jitter(3, 2.0, -1.0, 1.0, 0.5)]
    #[case::nonfinite_jitter(3, 2.0, 0.0, f64::INFINITY, 0.5)]
    #[case::reversed_jitter(3, 2.0, 2.0, 1.0, 0.5)]
    #[case::negative_ratio(3, 2.0, 0.0, 1.0, -0.1)]
    #[case::invalid_ratio(3, 2.0, 0.0, 1.0, 1.1)]
    #[case::nonfinite_ratio(3, 2.0, 0.0, 1.0, f64::NAN)]
    fn invalid_retry_policy_finishes_without_panicking(
        #[case] max_attempts: usize,
        #[case] backoff_base_seconds: f64,
        #[case] jitter_min_seconds: f64,
        #[case] jitter_max_seconds: f64,
        #[case] min_reference_ratio: f64,
    ) {
        let installed = installed(r#"{"base":{"litellm_provider":"test"}}"#, None);
        let retry_policy = RetryPolicy {
            max_attempts,
            backoff_base_seconds,
            jitter_min_seconds,
            jitter_max_seconds,
            min_reference_ratio,
            ..RetryPolicy::default()
        };
        let (_, step) = Revalidation::new(
            "https://example.test/catalog.json".into(),
            &installed,
            retry_policy,
            Some(1),
        );
        assert!(matches!(
            step,
            Step::Done {
                reason: Some(reason)
            } if reason == "invalid retry policy"
        ));
    }

    #[rstest]
    #[case::rate_limited(429, "100", 30)]
    #[case::server_error(503, "2", 2)]
    #[case::upper_server_error(599, "2", 2)]
    fn retryable_statuses_sleep_then_fetch_again(
        #[case] status: u16,
        #[case] retry_after: &str,
        #[case] expected_seconds: u64,
    ) {
        let installed = installed(r#"{"base":{"litellm_provider":"test"}}"#, None);
        let (mut state, _) = Revalidation::new(
            "https://example.test/catalog.json".into(),
            &installed,
            policy(0, 0.0),
            Some(2),
        );
        assert!(matches!(
            state.on_reply(Reply::Response {
                status,
                etag: None,
                retry_after: Some(retry_after.into()),
                body: Bytes::new()
            }),
            Step::Sleep(duration) if duration == Duration::from_secs(expected_seconds)
        ));
        assert!(matches!(
            state.on_slept(),
            Step::Fetch(FetchRequest {
                if_none_match: None,
                ..
            })
        ));
    }

    #[rstest]
    fn transport_failures_stop_at_the_attempt_limit() {
        let installed = installed(r#"{"base":{"litellm_provider":"test"}}"#, None);
        let retry_policy = RetryPolicy {
            max_attempts: 2,
            ..policy(0, 0.0)
        };
        let (mut state, _) = Revalidation::new(
            "https://example.test/catalog.json".into(),
            &installed,
            retry_policy,
            Some(3),
        );
        assert!(matches!(
            state.on_reply(Reply::Transport {
                reason: "offline".into()
            }),
            Step::Sleep(_)
        ));
        assert!(matches!(state.on_slept(), Step::Fetch(_)));
        assert!(matches!(
            state.on_reply(Reply::Transport {
                reason: "offline".into()
            }),
            Step::Done { reason: Some(reason) } if reason == "offline"
        ));
    }

    #[rstest]
    fn invalid_retry_after_uses_seeded_jitter_and_guards_sleep_state() {
        let installed = installed(r#"{"base":{"litellm_provider":"test"}}"#, None);
        let make = || {
            Revalidation::new(
                "https://example.test/catalog.json".into(),
                &installed,
                policy(0, 0.0),
                Some(42),
            )
        };
        let (mut first, _) = make();
        let (mut second, _) = make();
        let first_sleep = first.on_reply(Reply::Response {
            status: 429,
            etag: None,
            retry_after: Some("NaN".into()),
            body: Bytes::new(),
        });
        let second_sleep = second.on_reply(Reply::Response {
            status: 429,
            etag: None,
            retry_after: Some("-1".into()),
            body: Bytes::new(),
        });
        let (Step::Sleep(first_sleep), Step::Sleep(second_sleep)) = (first_sleep, second_sleep)
        else {
            panic!("invalid Retry-After should use the retry delay")
        };
        assert_eq!(first_sleep, second_sleep);
        assert!((2.0..=3.0).contains(&first_sleep.as_secs_f64()));
        assert!(matches!(
            first.on_reply(response(304, "")),
            Step::Done {
                reason: Some(ref reason)
            } if reason == "revalidation is not awaiting a reply"
        ));
    }

    #[rstest]
    fn successful_response_adopts_raw_catalog_and_reports_all_row_errors() {
        let installed = installed(r#"{"base":{"litellm_provider":"test"}}"#, None);
        let (mut state, _) = Revalidation::new(
            "https://example.test/catalog.json".into(),
            &installed,
            policy(1, 0.0),
            Some(7),
        );
        let body = r#"{"good":{"litellm_provider":"test"},"bad":{"litellm_provider":"test","mode":"chatt"}}"#;
        let Step::Adopt {
            installed: adopted,
            errors,
        } = state.on_reply(Reply::Response {
            status: 200,
            etag: Some("new-tag".into()),
            retry_after: None,
            body: Bytes::copy_from_slice(body.as_bytes()),
        })
        else {
            panic!("expected catalog adoption")
        };
        assert_eq!(adopted.catalog.model_count(), 2);
        assert_eq!(errors.len(), 1);
        assert_eq!(adopted.etag.as_deref(), Some("new-tag"));
        assert_eq!(
            adopted.origin,
            Origin::Remote {
                url: "https://example.test/catalog.json".into()
            }
        );
        assert_eq!(adopted.revision, git_blob_id(body.as_bytes()));
        let (filtered, removed) = (*adopted.catalog).clone().without_rejected(&errors);
        assert_eq!(filtered.model_count(), 1);
        assert_eq!(removed.len(), 1);
    }

    #[rstest]
    fn rejected_rows_are_filtered_before_integrity_check() {
        let installed = installed(
            r#"{"base":{"litellm_provider":"test"},"other":{"litellm_provider":"test"}}"#,
            None,
        );
        let (mut state, _) = Revalidation::new(
            "https://example.test/catalog.json".into(),
            &installed,
            policy(4, 0.0),
            Some(8),
        );
        let step = state.on_reply(response(
            200,
            r#"{"valid":{"litellm_provider":"test"},"invalid-1":{"litellm_provider":"test","mode":"chatt"},"invalid-2":{"litellm_provider":"test","mode":"chatt"},"invalid-3":{"litellm_provider":"test","mode":"chatt"}}"#,
        ));
        assert!(matches!(
            step,
            Step::Done {
                reason: Some(ref reason)
            } if reason.contains("below minimum 4")
        ));
    }

    #[rstest]
    fn integrity_uses_the_bundled_count_not_installed_catalog_size() {
        let installed = installed(r#"{"base":{"litellm_provider":"test"}}"#, None);
        let (mut state, _) = Revalidation::new(
            "https://example.test/catalog.json".into(),
            &installed,
            policy(0, 1.0),
            Some(8),
        );
        let step = state.on_reply(response(200, r#"{"valid":{"litellm_provider":"test"}}"#));
        assert!(matches!(
            step,
            Step::Done {
                reason: Some(ref reason)
            } if reason.contains("reference count")
        ));
    }

    #[rstest]
    #[case::invalid_json("{")]
    #[case::non_object_root("[]")]
    fn fatal_catalog_responses_are_not_adopted(#[case] body: &str) {
        let installed = installed(r#"{"base":{"litellm_provider":"test"}}"#, None);
        let (mut state, _) = Revalidation::new(
            "https://example.test/catalog.json".into(),
            &installed,
            policy(0, 0.0),
            Some(9),
        );
        assert!(matches!(
            state.on_reply(response(200, body)),
            Step::Done { reason: Some(_) }
        ));
    }

    #[rstest]
    fn s3_is_left_for_the_host_and_git_revision_uses_blob_framing() {
        let installed = installed(r#"{"base":{"litellm_provider":"test"}}"#, None);
        let (_, step) = Revalidation::new(
            "s3://bucket/catalog.json".into(),
            &installed,
            policy(0, 0.0),
            Some(9),
        );
        assert!(matches!(
            step,
            Step::Done {
                reason: Some(reason)
            } if reason == "s3 is fetched by the host"
        ));
        assert_eq!(
            git_blob_id(b"hello\n"),
            "ce013625030ba8dba906f756967f9e9ca394464a"
        );
    }

    #[rstest]
    fn shared_fixture_pins_the_registry_policy_and_git_blob_formula() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/git_blob_id.json")).unwrap();
        let retry = &fixture["retry"];
        let blob = &fixture["git_blob_id"];
        let default_policy = RetryPolicy::default();
        let body = blob["example_body_utf8"].as_str().unwrap();

        assert_eq!(
            fixture["registry_url"].as_str().unwrap(),
            DEFAULT_REGISTRY_URL
        );
        assert_eq!(
            retry["max_attempts"].as_u64().unwrap() as usize,
            default_policy.max_attempts
        );
        assert_eq!(
            retry["max_wait_seconds"].as_u64().unwrap(),
            default_policy.max_wait.as_secs()
        );
        assert_eq!(
            retry["timeout_seconds"].as_u64().unwrap(),
            default_policy.timeout.as_secs()
        );
        assert_eq!(
            retry["minimum_model_count"].as_u64().unwrap() as usize,
            default_policy.min_model_count
        );
        assert_eq!(
            retry["minimum_reference_ratio"].as_f64().unwrap(),
            default_policy.min_reference_ratio
        );
        assert_eq!(
            retry["backoff_base_seconds"].as_f64().unwrap(),
            default_policy.backoff_base_seconds
        );
        assert_eq!(
            retry["jitter_min_seconds"].as_f64().unwrap(),
            default_policy.jitter_min_seconds
        );
        assert_eq!(
            retry["jitter_max_seconds"].as_f64().unwrap(),
            default_policy.jitter_max_seconds
        );
        assert_eq!(
            retry["first_retry_index"].as_u64(),
            Some(FIRST_RETRY_INDEX as u64)
        );
        assert_eq!(retry["backoff_base_seconds"].as_u64(), Some(2));
        assert_eq!(retry["jitter_min_seconds"].as_u64(), Some(0));
        assert_eq!(retry["jitter_max_seconds"].as_u64(), Some(1));
        assert_eq!(blob["algorithm"].as_str(), Some("sha1"));
        assert_eq!(
            blob["formula"].as_str(),
            Some("SHA1(b\"blob \" + len(body) + b\"\\0\" + body)")
        );
        assert_eq!(
            git_blob_id(body.as_bytes()),
            blob["example_revision"].as_str().unwrap()
        );
    }

    #[rstest]
    fn revalidation_can_move_between_host_threads() {
        fn assert_send<T: Send>() {}
        assert_send::<Revalidation>();
    }
}
