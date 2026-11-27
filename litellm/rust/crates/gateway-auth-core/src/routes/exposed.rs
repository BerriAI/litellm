//! Which routes this proxy serves at all, whoever calls.

use litellm_config::{GeneralSettings, RouteList};

use super::error::RouteNotExposed;

/// The proxy's own route allowlist, read before any credential.
#[derive(Clone, Debug, Default)]
pub struct ExposedRoutes {
    /// [`general_settings.allowed_routes`](GeneralSettings::allowed_routes). A key left out takes
    /// every route, and a value that is not a list of strings takes none, a key written with no
    /// value included.
    allowed: Option<Box<[String]>>,
}

impl ExposedRoutes {
    pub fn new(general: &GeneralSettings) -> Self {
        Self {
            allowed: general
                .allowed_routes
                .as_ref()
                .map(|allowed| RouteList::items(allowed).into()),
        }
    }

    /// `RouteChecks.is_allowed_on_proxy(route)`: the route as written, with no pattern or group.
    pub fn permits(&self, route: &str) -> Result<(), RouteNotExposed> {
        let Some(allowed) = &self.allowed else {
            return Ok(());
        };
        if allowed.iter().any(|entry| entry == route) {
            return Ok(());
        }
        Err(RouteNotExposed {
            route: route.to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const ROUTE: &str = "/v1/messages";

    fn list(entries: &[&str]) -> RouteList {
        RouteList::Valid(entries.iter().map(|entry| (*entry).to_owned()).collect())
    }

    #[rstest]
    #[case::unset_takes_every_route(None, true)]
    #[case::a_listed_route(Some(list(&["/health", ROUTE])), true)]
    #[case::an_unlisted_route(Some(list(&["/health"])), false)]
    #[case::a_prefix_is_not_a_match(Some(list(&["/v1"])), false)]
    #[case::a_pattern_is_not_read(Some(list(&["/v1/*"])), false)]
    #[case::an_empty_list_takes_none(Some(list(&[])), false)]
    #[case::a_malformed_list_takes_none(Some(RouteList::Malformed), false)]
    fn the_route_is_checked_against_allowed_routes(
        #[case] allowed: Option<RouteList>,
        #[case] permitted: bool,
    ) {
        let exposed = ExposedRoutes::new(&GeneralSettings {
            allowed_routes: allowed,
            ..GeneralSettings::default()
        });
        let expected = if permitted {
            Ok(())
        } else {
            Err(RouteNotExposed {
                route: ROUTE.to_owned(),
            })
        };

        assert_eq!(exposed.permits(ROUTE), expected);
    }
}
