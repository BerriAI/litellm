use std::time::Duration;

use litellm_model_catalog::{BUNDLED_REGISTRY, IntegrityLimits, git_blob_id};
use litellm_model_catalog_refresh::{
    Failed, Fetched, Kept, Refreshed, Remote, RetryPolicy, Source, refresh,
};
use rstest::rstest;

struct Scripted(Result<Fetched, Failed>);

impl Source for Scripted {
    async fn fetch(&self) -> Result<Fetched, Failed> {
        self.0.clone()
    }
}

fn limits() -> IntegrityLimits {
    IntegrityLimits::DEFAULT
}

const PRESIGNED: &str = "https://user:secret@example.com/cost.json?X-Amz-Signature=abc#part";
const SHOWN: &str = "https://example.com/cost.json";

fn document(body: &[u8]) -> Scripted {
    Scripted(Ok(Fetched {
        body: body.to_vec(),
        etag: Some("\"v1\"".into()),
    }))
}

#[rstest]
#[tokio::test]
async fn a_fetched_document_within_the_limits_replaces_the_served_one() {
    let Refreshed::Replaced(cost_map) = refresh(&document(BUNDLED_REGISTRY), limits()).await else {
        panic!("the bundled document passes its own limits");
    };
    assert_eq!(
        cost_map.provenance().revision,
        git_blob_id(BUNDLED_REGISTRY)
    );
    assert_eq!(cost_map.provenance().etag.as_deref(), Some("\"v1\""));
    assert!(cost_map.catalog().model_count() > 50);
}

#[rstest]
#[tokio::test]
async fn a_failed_fetch_is_unavailable_with_the_fetch_failure() {
    let failed = Failed {
        reason: "HTTP 503 from https://example.com/cost.json (after 3 attempts)".into(),
        retryable: true,
        retry_after: Some(Duration::from_secs(1)),
    };
    let Refreshed::Unavailable(unavailable) =
        refresh(&Scripted(Err(failed.clone())), limits()).await
    else {
        panic!("a failed fetch reads no document");
    };
    assert_eq!(unavailable, failed);
}

#[rstest]
#[case::lost_most_of_its_models(br#"{"only": {"litellm_provider": "openai"}}"#, true)]
#[case::not_an_object(b"[1, 2]", false)]
#[tokio::test]
async fn a_document_the_limits_refuse_is_rejected(#[case] body: &[u8], #[case] integrity: bool) {
    let Refreshed::Rejected(error) = refresh(&document(body), limits()).await else {
        panic!("the document must be refused");
    };
    assert_eq!(error.is_integrity(), integrity);
}

#[rstest]
#[tokio::test]
async fn a_replaced_document_is_handed_to_the_host() {
    let cost_map = refresh(&document(BUNDLED_REGISTRY), limits())
        .await
        .or_kept(PRESIGNED)
        .unwrap();
    assert_eq!(
        cost_map.provenance().revision,
        git_blob_id(BUNDLED_REGISTRY)
    );
}

#[rstest]
#[tokio::test]
async fn a_failed_fetch_reports_the_fetch_failure() {
    let failed = Failed {
        reason: format!("HTTP 404 from {SHOWN}"),
        retryable: false,
        retry_after: None,
    };
    let refreshed = refresh(&Scripted(Err(failed)), limits()).await;
    assert_eq!(
        refreshed.or_kept(PRESIGNED).unwrap_err(),
        Kept {
            detail: format!("HTTP 404 from {SHOWN}"),
            reason: format!("HTTP 404 from {SHOWN}"),
            fallback_reason: format!("Remote fetch failed: HTTP 404 from {SHOWN}"),
        }
    );
}

#[rstest]
#[tokio::test]
async fn a_document_that_lost_its_models_reports_integrity_without_the_url_credentials() {
    let body = br#"{"only": {"litellm_provider": "openai"}}"#;
    let kept = refresh(&document(body), limits())
        .await
        .or_kept(PRESIGNED)
        .unwrap_err();
    assert_eq!(
        kept.reason,
        format!("model cost map from {SHOWN} failed integrity validation")
    );
    assert_eq!(
        kept.fallback_reason,
        "Remote data failed integrity validation"
    );
    assert!(
        kept.detail.contains("has 1 models, below minimum"),
        "{}",
        kept.detail
    );
}

#[rstest]
#[tokio::test]
async fn a_document_that_does_not_read_reports_invalid_json_without_the_url_credentials() {
    let kept = refresh(&document(b"[1, 2]"), limits())
        .await
        .or_kept(PRESIGNED)
        .unwrap_err();
    assert_eq!(
        kept.reason,
        format!("invalid JSON from {SHOWN}: {}", kept.detail)
    );
    assert_eq!(
        kept.fallback_reason,
        format!(
            "Remote fetch failed: invalid JSON from {SHOWN}: {}",
            kept.detail
        )
    );
    assert!(!kept.detail.is_empty());
}

#[rstest]
#[case::remote(false, false, Remote::At("https://example.com/cost.json".into()))]
#[case::local_override(true, false, Remote::Off)]
#[case::offline(false, true, Remote::Off)]
#[case::both(true, true, Remote::Off)]
fn the_local_override_and_offline_mode_each_turn_the_remote_document_off(
    #[case] local_only: bool,
    #[case] offline: bool,
    #[case] remote: Remote,
) {
    assert_eq!(
        Remote::new("https://example.com/cost.json".into(), local_only, offline),
        remote
    );
}

#[rstest]
fn the_default_policy_is_three_attempts_of_five_seconds_two_to_thirty_apart() {
    assert_eq!(
        RetryPolicy::DEFAULT,
        RetryPolicy {
            max_attempts: 3,
            timeout: Duration::from_secs(5),
            min_delay: Duration::from_secs(2),
            max_delay: Duration::from_secs(30),
        }
    );
}

#[rstest]
fn the_default_limits_are_fifty_models_and_half_the_bundled_count() {
    assert_eq!(
        (
            IntegrityLimits::DEFAULT.min_model_count(),
            IntegrityLimits::DEFAULT.min_reference_ratio()
        ),
        (50, 0.5)
    );
}
