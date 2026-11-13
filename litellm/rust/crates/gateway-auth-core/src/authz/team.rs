use super::{RoutePolicy, Target};
use crate::{
    JwtDenial, TeamDenial,
    identity::jwt::Team,
    routes::passthrough::{PassthroughAccess, PassthroughRequest},
};

impl RoutePolicy<'_> {
    /// A managed agent's grant may select another team than the token claimed only on a route
    /// teams may call.
    pub(crate) fn granting_team(
        &self,
        granting_team: Option<&str>,
        claimed_team: Option<&str>,
        target: &Target<'_>,
    ) -> Result<(), JwtDenial> {
        let route_allowed = self.team_eligibility(target).route_allowed();
        match granting_team {
            Some(granting) if Some(granting) != claimed_team && !route_allowed => {
                Err(JwtDenial::GrantingTeamRouteNotAllowed)
            }
            _ => Ok(()),
        }
    }

    /// The pass-through gate on the team a request acts for.
    pub(crate) fn team<R>(
        &self,
        team_id: Option<&str>,
        team: Option<&Team<R>>,
        target: &Target<'_>,
    ) -> Result<(), TeamDenial> {
        passthrough_gate(team_id, team, &self.team_eligibility(target).passthrough)
    }
}

/// A team the core never loaded grants nothing.
fn passthrough_gate<R>(
    team_id: Option<&str>,
    team: Option<&Team<R>>,
    passthrough: &PassthroughRequest<'_>,
) -> Result<(), TeamDenial> {
    if team_id.filter(|text| !text.is_empty()).is_none() {
        return Ok(());
    }
    let grants = team
        .map(|team| &team.passthrough)
        .cloned()
        .unwrap_or_default();
    match passthrough.team_access(&grants) {
        PassthroughAccess::Allowed => Ok(()),
        PassthroughAccess::Denied { denied_route } => Err(TeamDenial::PassthroughRouteDenied {
            route: passthrough.route.to_owned(),
            denied_route,
        }),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::{
        Error,
        identity::jwt::TeamRead,
        routes::passthrough::{ForwardedRoutes, PassthroughGrants},
    };

    const ROUTE: &str = "/chat/completions";

    fn team(team_id: &str, passthrough: PassthroughGrants) -> Team<String> {
        Team {
            team_id: team_id.to_owned(),
            models: Some(vec![]),
            passthrough,
            members: vec![],
            row: team_id.to_owned(),
        }
    }

    fn found(team_id: &str) -> Result<TeamRead<String>, Error> {
        Ok(TeamRead::Found(team(team_id, PassthroughGrants::default())))
    }

    fn granted(team_id: &str) -> Result<TeamRead<String>, Error> {
        Ok(TeamRead::Found(team(
            team_id,
            PassthroughGrants {
                allowed: Some(vec![ROUTE.to_owned()]),
                denied: vec![],
            },
        )))
    }

    fn denying(team_id: &str, denied_route: &str) -> Result<TeamRead<String>, Error> {
        Ok(TeamRead::Found(team(
            team_id,
            PassthroughGrants {
                allowed: None,
                denied: vec![denied_route.to_owned()],
            },
        )))
    }

    fn enforced_route(forwarded: &[String]) -> PassthroughRequest<'_> {
        PassthroughRequest {
            route: ROUTE,
            auth_enforced: true,
            forwarded_routes: ForwardedRoutes(forwarded),
            jwt_team_allowed_routes: None,
        }
    }

    fn forwarded() -> Vec<String> {
        vec![ROUTE.to_owned()]
    }

    fn loaded(read: Result<TeamRead<String>, Error>) -> Option<Team<String>> {
        match read {
            Ok(TeamRead::Found(team)) => Some(team),
            _ => None,
        }
    }

    fn passthrough_denied(denied_route: Option<&str>) -> Result<(), TeamDenial> {
        Err(TeamDenial::PassthroughRouteDenied {
            route: ROUTE.to_owned(),
            denied_route: denied_route.map(str::to_owned),
        })
    }

    #[rstest]
    #[case::no_team(None, None, Ok(()))]
    #[case::empty_team(Some(""), None, Ok(()))]
    #[case::granted(Some("a"), loaded(granted("a")), Ok(()))]
    #[case::not_granted(Some("b"), loaded(found("b")), passthrough_denied(None))]
    #[case::denied(
        Some("c"),
        loaded(denying("c", "/chat/*")),
        passthrough_denied(Some("/chat/*"))
    )]
    #[case::never_loaded(Some("d"), None, passthrough_denied(None))]
    fn gates_passthrough_routes_on_the_selected_team(
        #[case] team_id: Option<&str>,
        #[case] team: Option<Team<String>>,
        #[case] expected: Result<(), TeamDenial>,
    ) {
        let forwarded = forwarded();
        assert_eq!(
            passthrough_gate(team_id, team.as_ref(), &enforced_route(&forwarded)),
            expected
        );
    }
}
