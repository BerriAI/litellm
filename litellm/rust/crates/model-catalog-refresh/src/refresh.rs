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
            }),
            Self::Rejected(error) if error.is_integrity() => Err(Kept {
                detail: Report::new(&error).to_string(),
                reason: format!(
                    "model cost map from {} failed integrity validation",
                    redacted(url)
                ),
                fallback_reason: "Remote data failed integrity validation".to_owned(),
            }),
            Self::Rejected(error) => {
                let detail = Report::new(&error).to_string();
                let reason = format!("invalid JSON from {}: {detail}", redacted(url));
                Err(Kept {
                    fallback_reason: format!("Remote fetch failed: {reason}"),
                    detail,
                    reason,
                })
            }
        }
    }
}

/// Fetches the document and reads it against `limits`
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
