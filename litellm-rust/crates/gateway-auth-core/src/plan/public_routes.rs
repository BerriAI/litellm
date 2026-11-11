use std::collections::HashSet;

use litellm_config::GeneralSettings;

use crate::routes::PUBLIC_ROUTES;

const METRICS_ROUTE: &str = "/metrics";

/// Routes a request reaches without any credential.
#[derive(Clone, Debug, Default)]
pub struct PublicRoutes {
    /// [`PUBLIC_ROUTES`]
    builtin: HashSet<String>,
    /// [`general_settings.public_routes`](GeneralSettings::public_routes), exact or ending in `*`.
    /// Empty without an enterprise license ([`HostFacts::premium`](crate::HostFacts::premium)).
    configured: Box<[String]>,
    /// `litellm.require_auth_for_metrics_endpoint` ([`HostFacts::metrics_require_auth`](crate::HostFacts::metrics_require_auth))
    /// keeps `/metrics` behind auth even when `public_routes` lists it.
    metrics_require_auth: bool,
}

impl PublicRoutes {
    pub fn new(general: &GeneralSettings, premium: bool, metrics_require_auth: bool) -> Self {
        Self {
            builtin: PUBLIC_ROUTES
                .iter()
                .map(|route| (*route).to_owned())
                .collect(),
            configured: match premium {
                true => general.public_routes.clone().unwrap_or_default(),
                false => Box::default(),
            },
            metrics_require_auth,
        }
    }

    pub(super) fn contains(&self, route: &str) -> bool {
        if self.metrics_require_auth && without_trailing_slash(route) == METRICS_ROUTE {
            return false;
        }
        self.builtin.contains(route)
            || self
                .configured
                .iter()
                .any(|pattern| matches_pattern(route, pattern))
    }
}

fn matches_pattern(route: &str, pattern: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => route.starts_with(prefix),
        None => route == pattern,
    }
}

fn without_trailing_slash(route: &str) -> &str {
    match route {
        "/" => route,
        _ => route.trim_end_matches('/'),
    }
}
