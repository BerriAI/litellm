use litellm_error::Report;
use litellm_model_catalog::{CostMap, IntegrityLimits};

use crate::fetch::redacted;
use crate::{Failed, Source};

/// What a refresh found. Only `Replaced` hands the host a new document, and it keeps what it serves otherwise
#[derive(Debug)]
pub enum Refreshed {
    Replaced(CostMap),
    Unavailable(Failed),
    Rejected(litellm_model_catalog::Error),
}

/// Why a refresh installed nothing, in the words every host reports
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Kept {
    /// The cause as the failure itself tells it, for a log line
    pub detail: String,
    /// What a reload answers its caller
    pub reason: String,
    /// What the source info shows while the host serves the document it had
    pub fallback_reason: String,
    /// Another fetch could succeed. Only the attempt limit ended this one
    pub retryable: bool,
}

impl Refreshed {
    /// The new document, or why the host keeps the one it serves. `url` is shown without userinfo, query and
    /// fragment, where presigned URLs carry credentials
    pub fn or_kept(self, url: &str) -> Result<CostMap, Kept> {
        match self {
            Self::Replaced(cost_map) => Ok(cost_map),
            Self::Unavailable(failed) => Err(Kept {
                fallback_reason: format!("Remote fetch failed: {}", failed.reason),
                detail: failed.reason.clone(),
                reason: failed.reason,
                retryable: failed.retryable,
            }),
            Self::Rejected(error) if error.is_integrity() => Err(Kept {
                detail: Report::new(&error).to_string(),
                reason: format!(
                    "model cost map from {} failed integrity validation",
                    redacted(url)
                ),
                fallback_reason: "Remote data failed integrity validation".to_owned(),
                retryable: false,
            }),
            Self::Rejected(error) => {
                let detail = Report::new(&error).to_string();
                let reason = format!("invalid JSON from {}: {detail}", redacted(url));
                Err(Kept {
                    fallback_reason: format!("Remote fetch failed: {reason}"),
                    detail,
                    reason,
                    retryable: false,
                })
            }
        }
    }
}

pub async fn refresh(source: &impl Source, limits: IntegrityLimits) -> Refreshed {
    let fetched = match source.fetch().await {
        Ok(fetched) => fetched,
        Err(failed) => return Refreshed::Unavailable(failed),
    };
    match CostMap::read(&fetched.body, fetched.etag, limits) {
        Ok(cost_map) => Refreshed::Replaced(cost_map),
        Err(error) => Refreshed::Rejected(error),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use litellm_model_catalog::{BUNDLED_REGISTRY, git_blob_id};
    use rstest::{fixture, rstest};

    use super::*;
    use crate::Fetched;

    struct Scripted(Result<Fetched, Failed>);

    impl Source for Scripted {
        async fn fetch(&self) -> Result<Fetched, Failed> {
            self.0.clone()
        }
    }

    #[fixture]
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
    async fn a_fetched_document_within_the_limits_replaces_the_served_one(limits: IntegrityLimits) {
        let Refreshed::Replaced(cost_map) = refresh(&document(BUNDLED_REGISTRY), limits).await
        else {
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
    async fn a_failed_fetch_is_unavailable_with_the_fetch_failure(limits: IntegrityLimits) {
        let failed = Failed {
            reason: "HTTP 503 from https://example.com/cost.json (after 3 attempts)".into(),
            retryable: true,
            retry_after: Some(Duration::from_secs(1)),
        };
        let Refreshed::Unavailable(unavailable) =
            refresh(&Scripted(Err(failed.clone())), limits).await
        else {
            panic!("a failed fetch reads no document");
        };
        assert_eq!(unavailable, failed);
    }

    #[rstest]
    #[case::lost_most_of_its_models(br#"{"only": {"litellm_provider": "openai"}}"#, true)]
    #[case::not_an_object(b"[1, 2]", false)]
    #[tokio::test]
    async fn a_document_the_limits_refuse_is_rejected(
        limits: IntegrityLimits,
        #[case] body: &[u8],
        #[case] integrity: bool,
    ) {
        let Refreshed::Rejected(error) = refresh(&document(body), limits).await else {
            panic!("the document must be refused");
        };
        assert_eq!(error.is_integrity(), integrity);
    }

    #[rstest]
    #[tokio::test]
    async fn a_replaced_document_is_handed_to_the_host(limits: IntegrityLimits) {
        let cost_map = refresh(&document(BUNDLED_REGISTRY), limits)
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
    #[case::final_failure(false)]
    #[case::transient_failure(true)]
    async fn a_failed_fetch_reports_the_fetch_failure(
        limits: IntegrityLimits,
        #[case] retryable: bool,
    ) {
        let failed = Failed {
            reason: format!("HTTP 404 from {SHOWN}"),
            retryable,
            retry_after: None,
        };
        let refreshed = refresh(&Scripted(Err(failed)), limits).await;
        assert_eq!(
            refreshed.or_kept(PRESIGNED).unwrap_err(),
            Kept {
                detail: format!("HTTP 404 from {SHOWN}"),
                reason: format!("HTTP 404 from {SHOWN}"),
                fallback_reason: format!("Remote fetch failed: HTTP 404 from {SHOWN}"),
                retryable,
            }
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_document_that_lost_its_models_reports_integrity_without_the_url_credentials(
        limits: IntegrityLimits,
    ) {
        let body = br#"{"only": {"litellm_provider": "openai"}}"#;
        let kept = refresh(&document(body), limits)
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
    async fn a_document_that_does_not_read_reports_invalid_json_without_the_url_credentials(
        limits: IntegrityLimits,
    ) {
        let kept = refresh(&document(b"[1, 2]"), limits)
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
}
