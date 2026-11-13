use litellm_config::GeneralSettings;

use crate::routes::{RouteGroup, RouteGroups, matches_wildcard, table::METRICS};

/// Routes a request reaches without any credential.
#[derive(Clone, Debug, Default)]
pub struct PublicRoutes {
    route_groups: RouteGroups,
    /// [`general_settings.public_routes`](GeneralSettings::public_routes), exact or ending in `*`.
    /// Empty without an enterprise license ([`HostFacts::premium`](crate::HostFacts::premium)).
    configured: Box<[String]>,
    /// `litellm.require_auth_for_metrics_endpoint` ([`HostFacts::metrics_require_auth`](crate::HostFacts::metrics_require_auth))
    /// keeps `/metrics` behind auth even when `public_routes` lists it.
    metrics_require_auth: bool,
}

impl PublicRoutes {
    pub fn new(
        general: &GeneralSettings,
        premium: bool,
        metrics_require_auth: bool,
        route_groups: &RouteGroups,
    ) -> Self {
        Self {
            route_groups: route_groups.clone(),
            configured: if premium {
                general.public_routes.clone().unwrap_or_default()
            } else {
                Box::default()
            },
            metrics_require_auth,
        }
    }

    /// `route in LiteLLMRoutes.public_routes.value or route_in_additonal_public_routes(route)`
    pub(super) fn contains(&self, route: &str) -> bool {
        if self.metrics_require_auth && without_trailing_slash(route) == METRICS {
            return false;
        }
        self.route_groups.lists(route, RouteGroup::Public)
            || self
                .configured
                .iter()
                .any(|pattern| matches_wildcard(route, pattern))
    }
}

fn without_trailing_slash(route: &str) -> &str {
    match route {
        "/" => route,
        _ => route.trim_end_matches('/'),
    }
}
