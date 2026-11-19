use std::time::Duration;

use litellm_model_catalog::{BUNDLED_REGISTRY, IntegrityLimits, git_blob_id};
use litellm_model_catalog_refresh::{Failed, Fetched, Refreshed, Source, refresh};
use rstest::rstest;

struct Scripted(Result<Fetched, Failed>);

impl Source for Scripted {
    async fn fetch(&self) -> Result<Fetched, Failed> {
        self.0.clone()
    }
}

fn limits() -> IntegrityLimits {
    IntegrityLimits::new(50, 0.5).unwrap()
}

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
