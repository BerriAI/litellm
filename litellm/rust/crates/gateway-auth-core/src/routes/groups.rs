use std::{collections::HashMap, sync::Arc};

use strum::VariantArray;

use super::{MissingRouteGroup, table::RouteGroup, template::Template};

#[derive(Debug)]
pub(super) struct Entries {
    pub(super) routes: Box<[String]>,
    pub(super) templates: Box<[Template]>,
}

impl Entries {
    pub(super) fn matches(&self, route: &str) -> bool {
        self.templates
            .iter()
            .any(|template| template.matches(route))
    }
}

/// `LiteLLMRoutes` by member name, which the host hands in with its settings.
#[derive(Clone, Debug, Default)]
pub struct RouteGroups(Arc<HashMap<String, Entries>>);

/// An operator's route list paired with the groups its entries may name, so the list is only ever
/// read through `_allowed_routes_check`.
#[derive(Clone, Copy, Debug)]
pub struct RouteAllowlist<'a> {
    entries: &'a [String],
    groups: &'a RouteGroups,
}

impl<'a> RouteAllowlist<'a> {
    pub fn permits(&self, route: &str) -> bool {
        self.groups.is_route_allowed(route, self.entries)
    }

    pub fn entries(&self) -> &'a [String] {
        self.entries
    }

    /// `get_actual_routes`
    pub fn expanded(&self) -> Vec<String> {
        self.groups.actual_routes(self.entries)
    }

    /// `jwt_team_routes_grant_pass_through`: explicit paths and trailing-wildcard prefixes grant,
    /// group names and blanket entries (`*`, `/*`) never do.
    pub fn grants_pass_through(&self, route: &str) -> bool {
        self.entries
            .iter()
            .filter(|allowed| !allowed.trim_end_matches('*').trim_matches('/').is_empty())
            .any(|allowed| matches_wildcard(route, allowed))
    }
}

impl RouteGroups {
    /// Fails on a table without a group the core's rules read, so a group the host renames stops
    /// the settings build instead of reading as empty.
    pub fn new(
        groups: impl IntoIterator<Item = (String, Vec<String>)>,
    ) -> Result<Self, MissingRouteGroup> {
        let groups: HashMap<String, Entries> = groups
            .into_iter()
            .map(|(name, routes)| {
                let templates = routes
                    .iter()
                    .map(|route| Template::compile(route))
                    .collect();
                (
                    name,
                    Entries {
                        routes: routes.into(),
                        templates,
                    },
                )
            })
            .collect();
        let missing = RouteGroup::VARIANTS
            .iter()
            .copied()
            .find(|group| !groups.contains_key(<&str>::from(*group)));
        match missing {
            Some(group) => Err(MissingRouteGroup { group }),
            None => Ok(Self(Arc::new(groups))),
        }
    }

    pub(super) fn entries(&self, group: RouteGroup) -> Option<&Entries> {
        self.0.get(<&str>::from(group))
    }

    /// `LiteLLMRoutes.<group>.value`
    pub fn routes(&self, group: RouteGroup) -> &[String] {
        self.entries(group)
            .map(|entries| &*entries.routes)
            .unwrap_or_default()
    }

    /// `route in LiteLLMRoutes.<group>.value`: the route is one of the group's entries as written.
    pub fn lists(&self, route: &str, group: RouteGroup) -> bool {
        self.routes(group).iter().any(|entry| entry == route)
    }

    /// The route matches one of the group's entries read as starlette templates.
    pub fn matches(&self, route: &str, group: RouteGroup) -> bool {
        self.entries(group)
            .is_some_and(|entries| entries.matches(route))
    }

    /// `_allowed_routes_check` over an operator's list: each entry is a `LiteLLMRoutes` group name,
    /// an exact route, or a prefix ending in `*`.
    pub fn is_route_allowed<S: AsRef<str>>(&self, route: &str, allowed_routes: &[S]) -> bool {
        allowed_routes
            .iter()
            .any(|allowed| match self.0.get(allowed.as_ref()) {
                Some(entries) => entries.matches(route),
                None => matches_wildcard(route, allowed.as_ref()),
            })
    }

    pub fn allowlist<'a>(&'a self, entries: &'a [String]) -> RouteAllowlist<'a> {
        RouteAllowlist {
            entries,
            groups: self,
        }
    }

    /// `get_actual_routes`: group names expanded to their routes, anything else kept as given.
    pub fn actual_routes<S: AsRef<str>>(&self, allowed_routes: &[S]) -> Vec<String> {
        allowed_routes
            .iter()
            .flat_map(|allowed| match self.0.get(allowed.as_ref()) {
                Some(entries) => entries.routes.to_vec(),
                None => vec![allowed.as_ref().to_owned()],
            })
            .collect()
    }
}

/// `route_matches_wildcard_pattern`: a trailing `*` matches by prefix, anything else exactly.
pub fn matches_wildcard(route: &str, pattern: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => route.starts_with(prefix),
        None => route == pattern,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::test_support::{route_groups, route_table};

    #[rstest]
    #[case::exact("/scim/v2", "/scim/v2", true)]
    #[case::wildcard_child("/scim/v2/Users/1", "/scim/v2/*", true)]
    #[case::wildcard_other("/chat/completions", "/scim/v2/*", false)]
    #[case::no_wildcard_child("/scim/v2/Users", "/scim/v2", false)]
    fn wildcard(#[case] route: &str, #[case] pattern: &str, #[case] expected: bool) {
        assert_eq!(matches_wildcard(route, pattern), expected);
    }

    #[rstest]
    #[case::group("info_routes", "/key/info", true)]
    #[case::group_template("openai_routes", "/engines/gpt/chat/completions", true)]
    #[case::exact_route("/v1/messages", "/v1/messages", true)]
    #[case::wildcard_prefix("/scim/v2/*", "/scim/v2/Users/1", true)]
    #[case::wildcard_other("/scim/v2/*", "/scim/v3", false)]
    #[case::unknown_name_is_exact("not_a_group", "/not_a_group", false)]
    fn allowed_routes(#[case] allowed: &str, #[case] route: &str, #[case] expected: bool) {
        assert_eq!(route_groups().is_route_allowed(route, &[allowed]), expected);
    }

    #[test]
    fn a_table_without_a_group_the_core_reads_is_refused() {
        let table = route_table()
            .into_iter()
            .filter(|(name, _)| name != "public_routes");
        assert_eq!(
            RouteGroups::new(table).unwrap_err(),
            MissingRouteGroup {
                group: RouteGroup::Public
            }
        );
    }

    #[test]
    fn actual_routes_expand_groups_in_order() {
        assert_eq!(
            route_groups().actual_routes(&["/first", "master_key_only_routes", "/last"]),
            [
                "/first",
                "/global/spend/reset",
                "/memory-usage-in-mem-cache",
                "/memory-usage-in-mem-cache-items",
                "/last",
            ]
        );
    }
}
