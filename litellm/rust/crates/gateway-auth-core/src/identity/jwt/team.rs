use std::collections::BTreeSet;

use litellm_config::ClaimPath;

use super::claims;
use crate::{
    AuthHost, Error, Rows, TeamDenial,
    authn::jwt::{Claims, JwtRequest, JwtSettings},
    authz::{TeamEligibility, Verdict as Eligibility},
    host::{MembershipRow, TeamRow},
    routes::passthrough::PassthroughGrants,
};

/// A team the host loaded: the facts the core decides on and the host's row, which the core only
/// hands back.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Team<R> {
    pub team_id: String,
    pub models: Option<Vec<String>>,
    pub passthrough: PassthroughGrants,
    /// The user ids `members_with_roles` lists.
    pub members: Vec<String>,
    pub row: R,
}

type HostTeam<H> = Team<TeamRow<H>>;

/// A team a fallback picked for the user, with the user's membership row in it.
#[derive(Clone, Debug, PartialEq)]
pub struct Joined<R: Rows> {
    pub team: Team<R::Team>,
    pub membership: Option<R::Membership>,
}

/// The answer to a decision the host still owns, `can_team_access_model`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelAccess {
    Allowed,
    Denied,
}

/// `get_team_object`: `Absent` is its `TeamNotFoundError`, `Unreadable` any other `HTTPException`
/// it raises. Anything else it raises is a `Error`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TeamRead<R> {
    Found(Team<R>),
    Absent,
    Unreadable,
}

impl<R> TeamRead<R> {
    pub fn found(self) -> Option<Team<R>> {
        match self {
            Self::Found(team) => Some(team),
            Self::Absent | Self::Unreadable => None,
        }
    }

    pub fn or_not_found(self, team_id: &str) -> Result<Team<R>, TeamDenial> {
        self.found().ok_or_else(|| TeamDenial::TeamNotFound {
            team_id: team_id.to_owned(),
        })
    }
}

/// `get_team_object_by_alias`: `Absent` is its 404, `Ambiguous` its 400, `NoDb` no database
/// connected. Its 500 and anything else it raises is a `Error`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AliasRead<R> {
    Found(Team<R>),
    Absent,
    Ambiguous,
    NoDb,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TeamSettings {
    pub enforce_team_based_model_access: bool,
    pub fallback_to_db_teams: bool,
    pub team_claim_fallback: bool,
}

/// What `find_and_validate_specific_team_id` reads from the token and `litellm_jwtauth`.
#[derive(Clone, Copy, Debug, Default)]
pub struct TeamClaims<'a> {
    /// `get_team_id`, with `team_id_default` already applied.
    pub team_id: Option<&'a str>,
    /// Whether `get_all_jwt_team_ids` found any team id, which `team_id_default` never fills.
    pub has_team_id_claim: bool,
    pub team_alias: Option<&'a str>,
    pub team_id_jwt_field: Option<&'a str>,
    pub team_alias_jwt_field: Option<&'a str>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeaderTeam {
    pub header_value: String,
    pub team_id: String,
    /// Accepted under `fallback_to_db_teams` from outside the token's teams, so it still has to
    /// pass `check_provisional_header_team`.
    pub provisional: bool,
}

/// What one of the token's teams contributes to `find_team_with_model_access`.
enum Verdict<R> {
    /// The team did not load.
    Unresolved,
    /// Loaded, but it neither qualifies nor explains a denial.
    NoAccess,
    /// The route and pass-through grants allow it, the model does not.
    Fallback(Team<R>),
    RouteDenied,
    PassthroughDenied(Option<String>),
    Selected(Team<R>),
}

async fn check_team<H: AuthHost>(
    host: &H,
    team_id: &str,
    model: Option<&str>,
    eligibility: &TeamEligibility<'_>,
) -> Verdict<TeamRow<H>> {
    let read = host.team(team_id, false).await.ok();
    let Some(team) = read.and_then(TeamRead::found) else {
        return Verdict::Unresolved;
    };
    if team.models.is_none() {
        return Verdict::NoAccess;
    }
    let has_model = match model {
        None => true,
        Some(model) => matches!(
            host.team_model_access(&team.row, model).await,
            Ok(ModelAccess::Allowed)
        ),
    };
    match eligibility.verdict(&team.passthrough, has_model) {
        Eligibility::Selected => Verdict::Selected(team),
        Eligibility::Fallback => Verdict::Fallback(team),
        Eligibility::RouteDenied => Verdict::RouteDenied,
        Eligibility::PassthroughDenied(denied_route) => Verdict::PassthroughDenied(denied_route),
        Eligibility::NoAccess => Verdict::NoAccess,
    }
}

/// `find_team_with_model_access`
pub(crate) async fn select_team<H: AuthHost>(
    host: &H,
    team_ids: &BTreeSet<String>,
    model: Option<&str>,
    eligibility: &TeamEligibility<'_>,
    settings: TeamSettings,
) -> Result<Option<HostTeam<H>>, TeamDenial> {
    let model = model.filter(|text| !text.is_empty());
    let route = eligibility.route();
    if team_ids.is_empty() {
        if settings.enforce_team_based_model_access && !settings.fallback_to_db_teams {
            return Err(TeamDenial::NoTeamsInToken);
        }
        return Ok(None);
    }
    let mut any_resolved = false;
    let mut fallback = None;
    let mut route_denied = None;
    let mut passthrough_denied = None;
    for team_id in team_ids {
        let verdict = check_team(host, team_id, model, eligibility).await;
        any_resolved |= !matches!(verdict, Verdict::Unresolved);
        match verdict {
            Verdict::Selected(team) => return Ok(Some(team)),
            Verdict::Fallback(team) => fallback = fallback.or(Some(team)),
            Verdict::RouteDenied => route_denied = Some(team_id),
            Verdict::PassthroughDenied(denied_route) => passthrough_denied = Some(denied_route),
            Verdict::Unresolved | Verdict::NoAccess => {}
        }
    }
    if let Some(team) = fallback {
        return Ok(Some(team));
    }
    if let Some(denied_route) = passthrough_denied {
        return Err(TeamDenial::PassthroughRouteDenied {
            route: route.to_owned(),
            denied_route,
        });
    }
    match model {
        Some(model) if any_resolved || !settings.team_claim_fallback => Err(match route_denied {
            Some(team_id) => TeamDenial::RouteNotAllowed {
                team_id: team_id.clone(),
                model: model.to_owned(),
                route: route.to_owned(),
            },
            None => TeamDenial::NoModelAccess {
                model: model.to_owned(),
                team_ids: team_ids.clone(),
            },
        }),
        _ => Ok(None),
    }
}

/// `find_and_validate_specific_team_id`
pub async fn specific_team<H: AuthHost>(
    host: &H,
    claims: TeamClaims<'_>,
    upsert: bool,
    settings: TeamSettings,
) -> Result<Option<HostTeam<H>>, Error> {
    let team_alias = claims.team_alias.filter(|text| !text.is_empty());
    let team_id = claims
        .team_id
        .filter(|text| !text.is_empty())
        .filter(|_| team_alias.is_none() || claims.has_team_id_claim);
    if let Some(team_id) = team_id {
        return match host.team(team_id, upsert).await?.found() {
            Some(team) => Ok(Some(team)),
            None if settings.team_claim_fallback => Ok(None),
            None => Err(TeamDenial::TeamNotFound {
                team_id: team_id.to_owned(),
            }
            .into()),
        };
    }
    if let Some(alias) = team_alias {
        return match host.team_by_alias(alias).await? {
            AliasRead::Found(team) => Ok(Some(team)),
            AliasRead::Absent => Err(TeamDenial::TeamAliasNotFound {
                alias: alias.to_owned(),
            }
            .into()),
            AliasRead::Ambiguous => Err(TeamDenial::TeamAliasAmbiguous {
                alias: alias.to_owned(),
            }
            .into()),
            AliasRead::NoDb => Err(TeamDenial::NoDb.into()),
        };
    }
    if claims.team_id_jwt_field.is_some() || claims.team_alias_jwt_field.is_some() {
        return Err(TeamDenial::NoTeamInToken {
            team_id_jwt_field: claims.team_id_jwt_field.map(str::to_owned),
            team_alias_jwt_field: claims.team_alias_jwt_field.map(str::to_owned),
        }
        .into());
    }
    Ok(None)
}

async fn team_id_by_alias(host: &impl AuthHost, alias: &str) -> Result<Option<String>, Error> {
    Ok(match host.team_by_alias(alias).await? {
        AliasRead::Found(team) => Some(team.team_id),
        AliasRead::Absent | AliasRead::Ambiguous | AliasRead::NoDb => None,
    })
}

/// `resolve_team_from_header`
pub async fn header_team(
    host: &impl AuthHost,
    header_value: Option<&str>,
    allowed_team_ids: &BTreeSet<String>,
    fallback_to_db_teams: bool,
) -> Result<Option<HeaderTeam>, Error> {
    let Some(header_value) = header_value.filter(|text| !text.is_empty()) else {
        return Ok(None);
    };
    let pick = |team_id: &str| HeaderTeam {
        header_value: header_value.to_owned(),
        team_id: team_id.to_owned(),
        provisional: fallback_to_db_teams && !allowed_team_ids.contains(team_id),
    };
    if allowed_team_ids.contains(header_value) {
        return Ok(Some(pick(header_value)));
    }
    let not_member = || TeamDenial::HeaderTeamNotMember {
        header_value: header_value.to_owned(),
    };
    if fallback_to_db_teams {
        return match host.team(header_value, false).await? {
            TeamRead::Found(_) => Ok(Some(pick(header_value))),
            TeamRead::Unreadable => Err(not_member().into()),
            TeamRead::Absent => match team_id_by_alias(host, header_value).await? {
                Some(team_id) => Ok(Some(pick(&team_id))),
                None => Err(not_member().into()),
            },
        };
    }
    match team_id_by_alias(host, header_value).await? {
        Some(team_id) if allowed_team_ids.contains(&team_id) => Ok(Some(pick(&team_id))),
        _ => Err(TeamDenial::HeaderTeamNotAllowed {
            header_value: header_value.to_owned(),
            allowed_team_ids: allowed_team_ids.clone(),
        }
        .into()),
    }
}

/// The team load `authorize_jwt` runs for a header team. A provisional team is never upserted.
pub async fn load_header_team<H: AuthHost>(
    host: &H,
    header_team: &HeaderTeam,
    upsert: bool,
) -> Result<HostTeam<H>, Error> {
    let read = host
        .team(&header_team.team_id, upsert && !header_team.provisional)
        .await?;
    match read.found() {
        Some(team) => Ok(team),
        None if header_team.provisional => Err(TeamDenial::HeaderTeamNotMember {
            header_value: header_team.header_value.clone(),
        }
        .into()),
        None => Err(TeamDenial::TeamNotFound {
            team_id: header_team.team_id.clone(),
        }
        .into()),
    }
}

/// `_attach_team_from_header_for_admin`: the header team to attach, if it loads.
pub async fn admin_header_team<H: AuthHost>(
    host: &H,
    header_value: Option<&str>,
    is_llm_api_route: bool,
    upsert: bool,
) -> Option<HostTeam<H>> {
    let header_value = header_value.filter(|text| !text.is_empty())?;
    if !is_llm_api_route {
        return None;
    }
    host.team(header_value, upsert).await.ok()?.found()
}

/// The team with the user's membership in it, read when there is a user.
async fn join<H: AuthHost>(
    host: &H,
    user_id: Option<&str>,
    team: HostTeam<H>,
) -> Result<Joined<H::Rows>, Error> {
    let membership: Option<MembershipRow<H>> = match user_id.filter(|text| !text.is_empty()) {
        Some(user_id) => host.load_team_membership(user_id, &team.team_id).await?,
        None => None,
    };
    Ok(Joined { team, membership })
}

/// `_resolve_single_team_fallback`. `user_team_ids` is empty when there is no user row.
pub async fn single_team_fallback<H: AuthHost>(
    host: &H,
    user_team_ids: &[String],
    user_id: Option<&str>,
    upsert: bool,
) -> Result<Option<Joined<H::Rows>>, Error> {
    let [team_id] = user_team_ids else {
        return Ok(None);
    };
    let Some(team) = host.team(team_id, upsert).await?.found() else {
        return Ok(None);
    };
    join(host, user_id, team).await.map(Some)
}

/// `_resolve_db_team_fallback`
pub(crate) async fn db_team_fallback<H: AuthHost>(
    host: &H,
    user_team_ids: &[String],
    user_id: Option<&str>,
    model: Option<&str>,
    eligibility: &TeamEligibility<'_>,
    upsert: bool,
    settings: TeamSettings,
) -> Result<Option<Joined<H::Rows>>, Error> {
    let model = model.filter(|text| !text.is_empty());
    let mut any_resolved = false;
    let mut fallback = None;
    for team_id in user_team_ids {
        let Some(team) = host.team(team_id, upsert).await?.found() else {
            continue;
        };
        any_resolved = true;
        if fallback.is_none() && eligibility.member_fallback(&team.passthrough) {
            fallback = Some(team.clone());
        }
        if let Some(model) = model
            && host.team_model_access(&team.row, model).await? == ModelAccess::Denied
        {
            continue;
        }
        if !eligibility.route_allowed() {
            continue;
        }
        return join(host, user_id, team).await.map(Some);
    }
    if let Some(team) = fallback {
        return join(host, user_id, team).await.map(Some);
    }
    if !settings.enforce_team_based_model_access {
        return Ok(None);
    }
    Err(match model {
        Some(model) if any_resolved => TeamDenial::NoMemberTeamHasModel {
            model: model.to_owned(),
        },
        _ => TeamDenial::NotTeamMember,
    }
    .into())
}

/// The team `authorize_jwt` takes from the team header, the token's claims or an RBAC team token,
/// before a managed agent's delegation or a fallback to the user's own teams.
pub(crate) struct ClaimedTeam<H: AuthHost> {
    pub(crate) team_id: Option<String>,
    /// Its row, when resolving it read one.
    pub(crate) team: Option<HostTeam<H>>,
    pub(crate) header_team: Option<HeaderTeam>,
    /// The token names no team, so one of the user's DB teams is picked later.
    pub(crate) db_fallback: bool,
    /// The teams a managed agent's delegation may act for.
    pub(crate) scope: Option<BTreeSet<String>>,
}

pub(crate) async fn claimed_team<H: AuthHost>(
    host: &H,
    claims: &Claims,
    request: &JwtRequest<'_>,
    settings: &JwtSettings,
    rbac_team_id: Option<String>,
    team_id_upsert: bool,
    eligibility: &TeamEligibility<'_>,
) -> Result<ClaimedTeam<H>, Error> {
    let team_settings = TeamSettings::from(&settings.config);
    let claim_team_ids = claims::all_team_ids(claims, settings);
    let team_alias = claims::team_alias(claims, settings).filter(|text| !text.is_empty());
    let specific_team_id = claims::team_id(claims, settings);
    let db_fallback = settings.config.fallback_to_db_teams
        && claim_team_ids.is_empty()
        && team_alias.is_none()
        && rbac_team_id.is_none();
    let team_ids: BTreeSet<String> = claims::team_ids(claims, settings)
        .into_iter()
        .chain(
            specific_team_id
                .clone()
                .filter(|text| !text.is_empty())
                .filter(|_| !db_fallback),
        )
        .collect();
    let header_db_fallback = settings.config.fallback_to_db_teams && rbac_team_id.is_none();
    let header_team = header_team(host, request.team_header, &team_ids, header_db_fallback).await?;

    let (team_id, team) = match &header_team {
        Some(header_team) => {
            let loaded = load_header_team(host, header_team, team_id_upsert).await?;
            (Some(header_team.team_id.clone()), Some(loaded))
        }
        None if rbac_team_id.is_none() && !db_fallback => {
            let team_claims = TeamClaims {
                team_id: specific_team_id.as_deref(),
                has_team_id_claim: !claim_team_ids.is_empty(),
                team_alias: team_alias.as_deref(),
                team_id_jwt_field: settings
                    .config
                    .team_id_jwt_field
                    .as_ref()
                    .map(ClaimPath::as_str),
                team_alias_jwt_field: settings
                    .config
                    .team_alias_jwt_field
                    .as_ref()
                    .map(ClaimPath::as_str),
            };
            let found = specific_team(host, team_claims, team_id_upsert, team_settings).await?;
            (found.as_ref().map(|team| team.team_id.clone()), found)
        }
        None => (rbac_team_id, None),
    };
    let (team_id, team) = match team_id {
        None => {
            let found =
                select_team(host, &team_ids, request.model, eligibility, team_settings).await?;
            (found.as_ref().map(|team| team.team_id.clone()), found)
        }
        Some(team_id) => (Some(team_id), team),
    };
    let team = match team_id.as_deref() {
        Some(id) if team.is_none() && request.auth_enforced_pass_through => {
            Some(host.team(id, team_id_upsert).await?.or_not_found(id)?)
        }
        _ => team,
    };
    let scope = match (&team_id, &team_alias) {
        _ if !claim_team_ids.is_empty() => Some(claim_team_ids.iter().cloned().collect()),
        (Some(team_id), Some(_)) => Some(BTreeSet::from([team_id.clone()])),
        _ => None,
    };
    Ok(ClaimedTeam {
        team_id,
        team,
        header_team,
        db_fallback,
        scope,
    })
}

/// The team a request acts for: the one a managed agent's grant selected, read from the
/// database, else the one the token claimed.
pub(crate) async fn selected<H: AuthHost>(
    host: &H,
    granting_team: Option<String>,
    claimed: &ClaimedTeam<H>,
) -> Result<(Option<String>, Option<HostTeam<H>>), Error> {
    let team_id = granting_team.or_else(|| claimed.team_id.clone());
    let team = match team_id.as_deref() {
        Some(id) if team_id != claimed.team_id => {
            Some(host.granting_team(id).await?.or_not_found(id)?)
        }
        _ => claimed.team.clone(),
    };
    Ok((team_id, team))
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use rstest::{fixture, rstest};

    use super::*;
    use crate::{
        routes::passthrough::{ForwardedRoutes, PassthroughRequest},
        testing::{Labels, ScriptedHost},
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

    /// A team whose `allowed_passthrough_routes` grants `ROUTE`.
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

    fn alias_of(team_id: &str) -> Result<AliasRead<String>, Error> {
        Ok(AliasRead::Found(Team {
            models: None,
            ..team(team_id, PassthroughGrants::default())
        }))
    }

    fn id<R>(team: Option<Team<R>>) -> Option<String> {
        team.map(|team| team.team_id)
    }

    fn joined_id(joined: Option<Joined<Labels>>) -> Option<String> {
        joined.map(|joined| joined.team.team_id)
    }

    /// `ROUTE` as an `auth: true` pass-through endpoint that forwards it unchanged.
    fn enforced_route(forwarded: &[String]) -> PassthroughRequest<'_> {
        PassthroughRequest {
            route: ROUTE,
            auth_enforced: true,
            forwarded_routes: ForwardedRoutes(forwarded),
            jwt_team_allowed_routes: None,
        }
    }

    /// Team eligibility on `ROUTE`, which `team_allowed_routes` lists or not.
    fn eligibility(
        forwarded: &[String],
        route_listed: bool,
        route_allowed: bool,
    ) -> TeamEligibility<'_> {
        TeamEligibility {
            route_listed,
            route_allowed,
            passthrough: enforced_route(forwarded),
        }
    }

    fn forwarded() -> Vec<String> {
        vec![ROUTE.to_owned()]
    }

    fn ids(ids: &[&str]) -> BTreeSet<String> {
        ids.iter().map(|id| (*id).to_owned()).collect()
    }

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

    #[fixture]
    fn enforced() -> TeamSettings {
        TeamSettings {
            enforce_team_based_model_access: true,
            ..TeamSettings::default()
        }
    }

    async fn select(
        host: &ScriptedHost,
        team_ids: &[&str],
        model: Option<&str>,
        route_allowed: bool,
        settings: TeamSettings,
    ) -> Result<Option<String>, TeamDenial> {
        let forwarded = forwarded();
        select_team(
            host,
            &ids(team_ids),
            model,
            &eligibility(&forwarded, route_allowed, false),
            settings,
        )
        .await
        .map(id)
    }

    #[rstest]
    #[case::enforced(true, false, Err(TeamDenial::NoTeamsInToken))]
    #[case::db_fallback(true, true, Ok(None))]
    #[case::not_enforced(false, false, Ok(None))]
    #[tokio::test]
    async fn token_without_teams(
        #[case] enforce_team_based_model_access: bool,
        #[case] fallback_to_db_teams: bool,
        #[case] expected: Result<Option<String>, TeamDenial>,
    ) {
        let settings = TeamSettings {
            enforce_team_based_model_access,
            fallback_to_db_teams,
            ..TeamSettings::default()
        };
        let host = ScriptedHost::default();
        assert_eq!(
            select(&host, &[], Some("gpt"), true, settings).await,
            expected
        );
        assert!(host.calls().is_empty());
    }

    #[rstest]
    #[tokio::test]
    async fn picks_first_team_in_sorted_order_with_access(enforced: TeamSettings) {
        let host = ScriptedHost {
            teams: HashMap::from([("b", granted("b")), ("c", granted("c"))]),
            model_access: HashSet::from(["c"]),
            ..ScriptedHost::default()
        };
        assert_eq!(
            select(&host, &["c", "a", "b"], Some("gpt"), true, enforced).await,
            Ok(Some("c".to_owned()))
        );
        assert_eq!(
            host.calls(),
            ["team a", "team b", "model b gpt", "team c", "model c gpt"]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn route_allowed_team_without_model_access_is_the_fallback(enforced: TeamSettings) {
        let host = ScriptedHost {
            teams: HashMap::from([("a", granted("a"))]),
            ..ScriptedHost::default()
        };
        assert_eq!(
            select(&host, &["a"], Some("gpt"), true, enforced).await,
            Ok(Some("a".to_owned()))
        );
    }

    #[rstest]
    #[case::not_granted(None)]
    #[case::denied_by_team(Some("/chat/*"))]
    #[tokio::test]
    async fn team_with_model_but_no_passthrough_grant_is_denied(
        enforced: TeamSettings,
        #[case] denied_route: Option<&'static str>,
    ) {
        let team = denied_route.map_or_else(|| found("a"), |route| denying("a", route));
        let host = ScriptedHost {
            teams: HashMap::from([("a", team)]),
            model_access: HashSet::from(["a"]),
            ..ScriptedHost::default()
        };
        assert_eq!(
            select(&host, &["a"], Some("gpt"), true, enforced).await,
            Err(TeamDenial::PassthroughRouteDenied {
                route: ROUTE.to_owned(),
                denied_route: denied_route.map(str::to_owned),
            })
        );
    }

    #[rstest]
    #[tokio::test]
    async fn team_with_model_on_disallowed_route_names_the_team(enforced: TeamSettings) {
        let host = ScriptedHost {
            teams: HashMap::from([("a", found("a"))]),
            model_access: HashSet::from(["a"]),
            ..ScriptedHost::default()
        };
        assert_eq!(
            select(&host, &["a"], Some("gpt"), false, enforced).await,
            Err(TeamDenial::RouteNotAllowed {
                team_id: "a".to_owned(),
                model: "gpt".to_owned(),
                route: ROUTE.to_owned(),
            })
        );
    }

    #[rstest]
    #[case::resolved_team(
        found("a"), true,
        Err(TeamDenial::NoModelAccess { model: "gpt".to_owned(), team_ids: ids(&["a"]) })
    )]
    #[case::unresolved_without_claim_fallback(
        Ok(TeamRead::Absent), false,
        Err(TeamDenial::NoModelAccess { model: "gpt".to_owned(), team_ids: ids(&["a"]) })
    )]
    #[case::unresolved_with_claim_fallback(Ok(TeamRead::Absent), true, Ok(None))]
    #[case::unreadable_with_claim_fallback(Ok(TeamRead::Unreadable), true, Ok(None))]
    #[case::lookup_failure_with_claim_fallback(Err(Error::HostFailed), true, Ok(None))]
    #[tokio::test]
    async fn no_team_has_the_model(
        #[case] team: Result<TeamRead<String>, Error>,
        #[case] team_claim_fallback: bool,
        #[case] expected: Result<Option<String>, TeamDenial>,
    ) {
        let host = ScriptedHost {
            teams: HashMap::from([("a", team)]),
            ..ScriptedHost::default()
        };
        let settings = TeamSettings {
            enforce_team_based_model_access: true,
            team_claim_fallback,
            ..TeamSettings::default()
        };
        assert_eq!(
            select(&host, &["a"], Some("gpt"), false, settings).await,
            expected
        );
    }

    #[rstest]
    #[case::no_model(None)]
    #[case::empty_model(Some(""))]
    #[tokio::test]
    async fn without_a_model_any_team_with_models_qualifies(
        enforced: TeamSettings,
        #[case] model: Option<&str>,
    ) {
        let without_models = Ok(TeamRead::Found(Team {
            models: None,
            ..team("a", PassthroughGrants::default())
        }));
        let host = ScriptedHost {
            teams: HashMap::from([("a", without_models), ("b", granted("b"))]),
            ..ScriptedHost::default()
        };
        assert_eq!(
            select(&host, &["a", "b"], model, true, enforced).await,
            Ok(Some("b".to_owned()))
        );
        assert_eq!(host.calls(), ["team a", "team b"]);
    }

    fn claims<'a>(
        team_id: Option<&'a str>,
        has_team_id_claim: bool,
        team_alias: Option<&'a str>,
    ) -> TeamClaims<'a> {
        TeamClaims {
            team_id,
            has_team_id_claim,
            team_alias,
            team_id_jwt_field: Some("team"),
            team_alias_jwt_field: Some("team_name"),
        }
    }

    #[rstest]
    #[case::claimed_id(claims(Some("a"), true, None), Ok(Some("a".to_owned())), &["team a upsert"])]
    #[case::claimed_id_beats_alias(claims(Some("a"), true, Some("eng")), Ok(Some("a".to_owned())), &["team a upsert"])]
    #[case::alias_beats_default_id(claims(Some("default"), false, Some("eng")), Ok(Some("t-eng".to_owned())), &["alias eng"])]
    #[case::default_id_without_alias(claims(Some("a"), false, None), Ok(Some("a".to_owned())), &["team a upsert"])]
    #[case::empty_alias_keeps_default_id(claims(Some("a"), false, Some("")), Ok(Some("a".to_owned())), &["team a upsert"])]
    #[case::empty_id_falls_to_alias(claims(Some(""), true, Some("eng")), Ok(Some("t-eng".to_owned())), &["alias eng"])]
    #[case::unknown_alias(
        claims(None, false, Some("ops")),
        Err(TeamDenial::TeamAliasNotFound { alias: "ops".to_owned() }),
        &["alias ops"]
    )]
    #[case::shared_alias(
        claims(None, false, Some("dup")),
        Err(TeamDenial::TeamAliasAmbiguous { alias: "dup".to_owned() }),
        &["alias dup"]
    )]
    #[case::alias_without_db(claims(None, false, Some("nodb")), Err(TeamDenial::NoDb), &["alias nodb"])]
    #[case::nothing_in_token(
        claims(None, false, None),
        Err(TeamDenial::NoTeamInToken {
            team_id_jwt_field: Some("team".into()),
            team_alias_jwt_field: Some("team_name".into()),
        }),
        &[]
    )]
    #[case::nothing_in_token_with_only_the_alias_field(
        TeamClaims { team_alias_jwt_field: Some("team_name"), ..TeamClaims::default() },
        Err(TeamDenial::NoTeamInToken {
            team_id_jwt_field: None,
            team_alias_jwt_field: Some("team_name".into()),
        }),
        &[]
    )]
    #[case::nothing_in_token_with_only_the_id_field(
        TeamClaims { team_id_jwt_field: Some("team"), ..TeamClaims::default() },
        Err(TeamDenial::NoTeamInToken {
            team_id_jwt_field: Some("team".into()),
            team_alias_jwt_field: None,
        }),
        &[]
    )]
    #[case::nothing_configured(TeamClaims::default(), Ok(None), &[])]
    #[tokio::test]
    async fn specific_team_resolves_id_then_alias(
        #[case] claims: TeamClaims<'static>,
        #[case] expected: Result<Option<String>, TeamDenial>,
        #[case] calls: &[&str],
    ) {
        let host = ScriptedHost {
            teams: HashMap::from([("a", found("a")), ("default", found("default"))]),
            aliases: HashMap::from([
                ("eng", alias_of("t-eng")),
                ("dup", Ok(AliasRead::Ambiguous)),
                ("nodb", Ok(AliasRead::NoDb)),
            ]),
            ..ScriptedHost::default()
        };
        assert_eq!(
            specific_team(&host, claims, true, TeamSettings::default())
                .await
                .map(id),
            expected.map_err(Error::from)
        );
        assert_eq!(host.calls(), calls);
    }

    #[rstest]
    #[case::absent_with_fallback(Ok(TeamRead::Absent), true, Ok(None))]
    #[case::unreadable_with_fallback(Ok(TeamRead::Unreadable), true, Ok(None))]
    #[case::absent_without_fallback(
        Ok(TeamRead::Absent), false,
        Err(Error::from(TeamDenial::TeamNotFound { team_id: "x".to_owned() }))
    )]
    #[case::unreadable_without_fallback(
        Ok(TeamRead::Unreadable), false,
        Err(Error::from(TeamDenial::TeamNotFound { team_id: "x".to_owned() }))
    )]
    #[case::host_error_with_fallback(Err(Error::HostFailed), true, Err(Error::HostFailed))]
    #[tokio::test]
    async fn specific_team_claim_that_does_not_load(
        #[case] team: Result<TeamRead<String>, Error>,
        #[case] team_claim_fallback: bool,
        #[case] expected: Result<Option<String>, Error>,
    ) {
        let host = ScriptedHost {
            teams: HashMap::from([("x", team)]),
            ..ScriptedHost::default()
        };
        let settings = TeamSettings {
            team_claim_fallback,
            ..TeamSettings::default()
        };
        assert_eq!(
            specific_team(&host, claims(Some("x"), true, Some("eng")), false, settings)
                .await
                .map(id),
            expected
        );
        assert_eq!(host.calls(), ["team x"]);
    }

    fn header_host() -> ScriptedHost {
        ScriptedHost {
            teams: HashMap::from([
                ("db", found("db")),
                ("broken", Ok(TeamRead::Unreadable)),
                ("down", Err(Error::HostFailed)),
            ]),
            aliases: HashMap::from([
                ("eng", alias_of("allowed")),
                ("ops", alias_of("other")),
                ("dup", Ok(AliasRead::Ambiguous)),
                ("nodb", Ok(AliasRead::NoDb)),
                ("fails", Err(Error::HostFailed)),
            ]),
            ..ScriptedHost::default()
        }
    }

    type HeaderResult = Result<Option<HeaderTeam>, Error>;

    fn not_allowed(header_value: &str) -> HeaderResult {
        Err(Error::from(TeamDenial::HeaderTeamNotAllowed {
            header_value: header_value.to_owned(),
            allowed_team_ids: ids(&["allowed", "zeta"]),
        }))
    }

    fn not_member(header_value: &str) -> HeaderResult {
        Err(Error::from(TeamDenial::HeaderTeamNotMember {
            header_value: header_value.to_owned(),
        }))
    }

    fn picked(header_value: &str, team_id: &str, provisional: bool) -> HeaderResult {
        Ok(Some(header(header_value, team_id, provisional)))
    }

    #[rstest]
    #[case::no_header(None, false, Ok(None), &[])]
    #[case::empty_header(Some(""), true, Ok(None), &[])]
    #[case::allowed_id(Some("allowed"), false, picked("allowed", "allowed", false), &[])]
    #[case::allowed_id_under_db_fallback(Some("allowed"), true, picked("allowed", "allowed", false), &[])]
    #[case::alias_of_allowed(Some("eng"), false, picked("eng", "allowed", false), &["alias eng"])]
    #[case::alias_of_other(Some("ops"), false, not_allowed("ops"), &["alias ops"])]
    #[case::unknown(Some("nope"), false, not_allowed("nope"), &["alias nope"])]
    #[case::shared_alias(Some("dup"), false, not_allowed("dup"), &["alias dup"])]
    #[case::alias_without_db(Some("nodb"), false, not_allowed("nodb"), &["alias nodb"])]
    #[case::alias_lookup_fails(Some("fails"), false, Err(Error::HostFailed), &["alias fails"])]
    #[case::db_team(Some("db"), true, picked("db", "db", true), &["team db"])]
    #[case::db_alias_of_other(Some("ops"), true, picked("ops", "other", true), &["team ops", "alias ops"])]
    #[case::db_alias_of_allowed(Some("eng"), true, picked("eng", "allowed", false), &["team eng", "alias eng"])]
    #[case::db_unknown(Some("nope"), true, not_member("nope"), &["team nope", "alias nope"])]
    #[case::db_unreadable_skips_alias(Some("broken"), true, not_member("broken"), &["team broken"])]
    #[case::db_lookup_fails(Some("down"), true, Err(Error::HostFailed), &["team down"])]
    #[tokio::test]
    async fn resolves_the_header_team(
        #[case] header_value: Option<&str>,
        #[case] fallback_to_db_teams: bool,
        #[case] expected: HeaderResult,
        #[case] calls: &[&str],
    ) {
        let host = header_host();
        assert_eq!(
            header_team(
                &host,
                header_value,
                &ids(&["zeta", "allowed"]),
                fallback_to_db_teams
            )
            .await,
            expected
        );
        assert_eq!(host.calls(), calls);
    }

    #[rstest]
    #[case::claimed_team_upserts(header("a", "a", false), Ok(()), "team a upsert")]
    #[case::provisional_never_upserts(header("a", "a", true), Ok(()), "team a")]
    #[case::claimed_missing(
        header("x", "x", false),
        Err(Error::from(TeamDenial::TeamNotFound { team_id: "x".to_owned() })),
        "team x upsert"
    )]
    #[case::provisional_missing(
        header("alias-x", "x", true),
        Err(Error::from(TeamDenial::HeaderTeamNotMember { header_value: "alias-x".to_owned() })),
        "team x"
    )]
    #[case::provisional_unreadable(
        header("u", "u", true),
        Err(Error::from(TeamDenial::HeaderTeamNotMember { header_value: "u".to_owned() })),
        "team u"
    )]
    #[case::lookup_fails(header("down", "down", true), Err(Error::HostFailed), "team down")]
    #[tokio::test]
    async fn loads_the_header_team(
        #[case] header_team: HeaderTeam,
        #[case] expected: Result<(), Error>,
        #[case] call: &str,
    ) {
        let host = ScriptedHost {
            teams: HashMap::from([
                ("a", found("a")),
                ("u", Ok(TeamRead::Unreadable)),
                ("down", Err(Error::HostFailed)),
            ]),
            ..ScriptedHost::default()
        };
        assert_eq!(
            load_header_team(&host, &header_team, true)
                .await
                .map(|_| ()),
            expected
        );
        assert_eq!(host.calls(), [call]);
    }

    #[rstest]
    #[case::llm_route(Some("a"), true, Some("a".to_owned()), &["team a upsert"])]
    #[case::management_route(Some("a"), false, None, &[])]
    #[case::no_header(None, true, None, &[])]
    #[case::absent(Some("x"), true, None, &["team x upsert"])]
    #[case::unreadable(Some("u"), true, None, &["team u upsert"])]
    #[case::lookup_fails(Some("down"), true, None, &["team down upsert"])]
    #[tokio::test]
    async fn attaches_an_admin_header_team(
        #[case] header_value: Option<&str>,
        #[case] is_llm_api_route: bool,
        #[case] expected: Option<String>,
        #[case] calls: &[&str],
    ) {
        let host = ScriptedHost {
            teams: HashMap::from([
                ("a", found("a")),
                ("u", Ok(TeamRead::Unreadable)),
                ("down", Err(Error::HostFailed)),
            ]),
            ..ScriptedHost::default()
        };
        assert_eq!(
            id(admin_header_team(&host, header_value, is_llm_api_route, true).await),
            expected
        );
        assert_eq!(host.calls(), calls);
    }

    #[rstest]
    #[case::one_team_with_user(
        &["a"], Some("u"), false,
        Ok(Some(("a", Some("membership u a")))),
        &["team a upsert", "membership u a"]
    )]
    #[case::one_team_without_user(&["a"], None, false, Ok(Some(("a", None))), &["team a upsert"])]
    #[case::one_team_empty_user(&["a"], Some(""), false, Ok(Some(("a", None))), &["team a upsert"])]
    #[case::no_teams(&[], Some("u"), false, Ok(None), &[])]
    #[case::two_teams(&["a", "b"], Some("u"), false, Ok(None), &[])]
    #[case::absent(&["x"], Some("u"), false, Ok(None), &["team x upsert"])]
    #[case::unreadable(&["un"], Some("u"), false, Ok(None), &["team un upsert"])]
    #[case::lookup_fails(&["down"], Some("u"), false, Err(Error::HostFailed), &["team down upsert"])]
    #[case::membership_fails(&["a"], Some("u"), true, Err(Error::HostFailed), &["team a upsert", "membership u a"])]
    #[tokio::test]
    async fn falls_back_to_the_only_team(
        #[case] user_team_ids: &[&str],
        #[case] user_id: Option<&str>,
        #[case] membership_fails: bool,
        #[case] expected: Result<Option<(&str, Option<&str>)>, Error>,
        #[case] calls: &[&str],
    ) {
        let host = ScriptedHost {
            teams: HashMap::from([
                ("a", found("a")),
                ("un", Ok(TeamRead::Unreadable)),
                ("down", Err(Error::HostFailed)),
            ]),
            membership_fails,
            ..ScriptedHost::default()
        };
        let joined = single_team_fallback(&host, &owned(user_team_ids), user_id, true).await;
        assert_eq!(
            joined.as_ref().map(|joined| {
                joined
                    .as_ref()
                    .map(|joined| (joined.team.team_id.as_str(), joined.membership.as_deref()))
            }),
            expected.as_ref().map(|joined| *joined)
        );
        assert_eq!(host.calls(), calls);
    }

    fn db_host() -> ScriptedHost {
        ScriptedHost {
            teams: HashMap::from([
                ("a", granted("a")),
                ("b", granted("b")),
                ("un", Ok(TeamRead::Unreadable)),
                ("down", Err(Error::HostFailed)),
            ]),
            model_access: HashSet::from(["b"]),
            ..ScriptedHost::default()
        }
    }

    #[rstest]
    #[case::first_team_with_model(
        &["x", "un", "a", "b"], Some("gpt"), true,
        Ok(Some("b".to_owned())),
        &["team x", "team un", "team a", "model a gpt", "team b", "model b gpt", "membership u b"]
    )]
    #[case::no_model_takes_first_loaded(
        &["x", "a", "b"], None, true,
        Ok(Some("a".to_owned())),
        &["team x", "team a", "membership u a"]
    )]
    #[case::route_allowed_team_without_model_is_the_fallback(
        &["a"], Some("gpt"), true,
        Ok(Some("a".to_owned())),
        &["team a", "model a gpt", "membership u a"]
    )]
    #[case::route_not_allowed_with_model(
        &["b"], Some("gpt"), false,
        Err(Error::from(TeamDenial::NoMemberTeamHasModel { model: "gpt".to_owned() })),
        &["team b", "model b gpt"]
    )]
    #[case::route_not_allowed_without_model(&["b"], None, false, Err(Error::from(TeamDenial::NotTeamMember)), &["team b"])]
    #[case::nothing_loads(&["x", "un"], Some("gpt"), true, Err(Error::from(TeamDenial::NotTeamMember)), &["team x", "team un"])]
    #[case::no_memberships(&[], Some("gpt"), true, Err(Error::from(TeamDenial::NotTeamMember)), &[])]
    #[case::lookup_fails(&["down", "b"], Some("gpt"), true, Err(Error::HostFailed), &["team down"])]
    #[tokio::test]
    async fn falls_back_to_a_db_membership(
        enforced: TeamSettings,
        #[case] user_team_ids: &[&str],
        #[case] model: Option<&str>,
        #[case] team_route_allowed: bool,
        #[case] expected: Result<Option<String>, Error>,
        #[case] calls: &[&str],
    ) {
        let (host, forwarded) = (db_host(), forwarded());
        assert_eq!(
            db_team_fallback(
                &host,
                &owned(user_team_ids),
                Some("u"),
                model,
                &eligibility(&forwarded, false, team_route_allowed),
                false,
                enforced
            )
            .await
            .map(joined_id),
            expected
        );
        assert_eq!(host.calls(), calls);
    }

    #[rstest]
    #[tokio::test]
    async fn db_fallback_without_enforcement_selects_nothing() {
        let (host, forwarded) = (db_host(), forwarded());
        assert_eq!(
            db_team_fallback(
                &host,
                &owned(&["b"]),
                None,
                Some("gpt"),
                &eligibility(&forwarded, false, false),
                true,
                TeamSettings::default()
            )
            .await
            .map(joined_id),
            Ok(None)
        );
        assert_eq!(host.calls(), ["team b upsert", "model b gpt"]);
    }
}
