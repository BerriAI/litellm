use super::{RoutePolicy, Target};
use crate::{
    JwtDenial, Rows, TeamDenial,
    identity::{Principal, jwt::HeaderTeam},
};

impl RoutePolicy<'_> {
    /// What the whole identity settles: a provisional header team needs the user's membership and
    /// a team route, and with `enforce_rbac` the request has to name a user or a team.
    pub(crate) fn principal<R: Rows>(
        &self,
        principal: &Principal<R>,
        target: &Target<'_>,
    ) -> Result<(), JwtDenial> {
        if let Some(header_team) = principal
            .header_team
            .as_ref()
            .filter(|header_team| header_team.provisional)
            && principal.team_id.as_deref() == Some(header_team.team_id.as_str())
        {
            let user_teams = principal
                .user
                .as_ref()
                .map(|user| user.teams.as_slice())
                .unwrap_or_default();
            let route_allowed = self.team_eligibility(target).route_allowed();
            check_provisional_header_team(header_team, user_teams, route_allowed, target.route)?;
        }
        if self.jwt.enforce_rbac
            && principal.user_id.as_deref().is_none_or(str::is_empty)
            && principal.team_id.as_deref().is_none_or(str::is_empty)
        {
            return Err(JwtDenial::NoObjectId);
        }
        Ok(())
    }
}

/// `_validate_header_team_in_db_membership` and the route check `authorize_jwt` runs after it.
fn check_provisional_header_team(
    header_team: &HeaderTeam,
    user_team_ids: &[String],
    team_route_allowed: bool,
    route: &str,
) -> Result<(), TeamDenial> {
    if !user_team_ids.contains(&header_team.team_id) {
        return Err(TeamDenial::HeaderTeamNotMember {
            header_value: header_team.header_value.clone(),
        });
    }
    if !team_route_allowed {
        return Err(TeamDenial::HeaderTeamRouteNotAllowed {
            header_value: header_team.header_value.clone(),
            route: route.to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const ROUTE: &str = "/chat/completions";

    fn owned(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|id| (*id).to_owned()).collect()
    }

    fn header(header_value: &str, team_id: &str, provisional: bool) -> HeaderTeam {
        HeaderTeam {
            header_value: header_value.to_owned(),
            team_id: team_id.to_owned(),
            provisional,
        }
    }

    #[rstest]
    #[case::member(&["b", "t"], true, Ok(()))]
    #[case::not_member(&["b"], true, Err(TeamDenial::HeaderTeamNotMember { header_value: "eng".to_owned() }))]
    #[case::not_member_beats_route(&["b"], false, Err(TeamDenial::HeaderTeamNotMember { header_value: "eng".to_owned() }))]
    #[case::route_not_allowed(
        &["t"], false,
        Err(TeamDenial::HeaderTeamRouteNotAllowed { header_value: "eng".to_owned(), route: ROUTE.to_owned() })
    )]
    fn checks_a_provisional_header_team(
        #[case] user_team_ids: &[&str],
        #[case] team_route_allowed: bool,
        #[case] expected: Result<(), TeamDenial>,
    ) {
        assert_eq!(
            check_provisional_header_team(
                &header("eng", "t", true),
                &owned(user_team_ids),
                team_route_allowed,
                ROUTE
            ),
            expected
        );
    }
}
