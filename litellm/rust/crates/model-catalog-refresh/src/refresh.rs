use litellm_model_catalog::{CostMap, IntegrityLimits};

use crate::{Failed, Source};

/// What a refresh found. Only `Replaced` hands the host a new document, and it keeps what it serves otherwise
#[derive(Debug)]
pub enum Refreshed {
    Replaced(CostMap),
    Unavailable(Failed),
    Rejected(litellm_model_catalog::Error),
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
