//! What the proxy's admin has allowed callers to set, apart from any one deployment.

use super::{
    names::{BUCKET_PARAMS, ENDPOINT_URL_PARAM},
    value::CallerValue,
};
use crate::error::body::Allowlist;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Policy {
    /// `general_settings.allow_client_side_credentials`: every override is the caller's to set.
    pub allow_client_side_credentials: bool,
    /// `general_settings.allowed_client_side_s3_buckets`, and the caller team's own buckets once
    /// the team is known.
    pub s3_buckets: Vec<String>,
    /// `general_settings.allowed_client_side_s3_endpoint_urls`.
    pub s3_endpoint_urls: Vec<String>,
    /// `litellm_settings.provider_url_destination_allowed_hosts`: the hosts a URL-valued
    /// fallback target may send to.
    pub url_destination_allowed_hosts: Vec<String>,
}

impl Policy {
    /// Whether a setting lists this one value of `param`.
    pub(crate) fn admits(&self, param: &str, value: &CallerValue) -> bool {
        let allowed = match allowlist(param) {
            Allowlist::None => return false,
            Allowlist::Buckets => &self.s3_buckets,
            Allowlist::EndpointUrls => &self.s3_endpoint_urls,
        };
        value
            .text()
            .is_some_and(|text| allowed.iter().any(|entry| entry == text))
    }
}

pub(crate) fn allowlist(param: &str) -> Allowlist {
    if BUCKET_PARAMS.contains(&param) {
        Allowlist::Buckets
    } else if param == ENDPOINT_URL_PARAM {
        Allowlist::EndpointUrls
    } else {
        Allowlist::None
    }
}

/// The checks a caller of the body check runs itself, later, once it knows more.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Deferred {
    /// A deployment's opt-in depends on the caller's team, which is not authenticated yet.
    pub model_checks: bool,
    /// The handler checks bucket params against the caller team's own allowlist.
    pub s3_buckets: bool,
}
