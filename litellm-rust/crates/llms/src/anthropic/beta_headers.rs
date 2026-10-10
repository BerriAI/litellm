use litellm_http::request::{with_header, without_headers};
use litellm_llms_types::providers::anthropic::{
    BETA_HEADER,
    beta::{BetaProvider, BetaSet},
};

use crate::{anthropic::common_utils::existing_betas, base_llm::auth::Headers};

/// How the `anthropic-beta` header is treated before a request leaves for a host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BetaPolicy {
    /// Send the header untouched and let the host decide.
    Forward,
    /// Keep only the betas the provider accepts, under the names it expects.
    Filter(BetaProvider),
    /// Remove every `anthropic-beta` header.
    Drop,
}

impl BetaPolicy {
    pub fn apply(self, headers: Headers) -> Headers {
        let provider = match self {
            Self::Forward => return headers,
            Self::Drop => return without_headers(headers, &[BETA_HEADER]),
            Self::Filter(provider) => provider,
        };
        let accepted: BetaSet = existing_betas(&headers)
            .iter()
            .filter_map(|beta| beta.on(provider))
            .collect();
        let headers = without_headers(headers, &[BETA_HEADER]);
        if accepted.is_empty() {
            return headers;
        }
        with_header(headers, BETA_HEADER, accepted.to_string())
    }
}
