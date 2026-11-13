use litellm_config::RbacRole;

use super::{
    ClaimIdentity, Provisioning, User, UserLookup, caller,
    claims::{self},
};
use crate::{
    AuthHost, Error, JwtDenial,
    authn::jwt::{Claims, JwtSettings},
    host::{MembershipRow, UserRow},
};

type HostUser<H> = User<UserRow<H>>;

/// What `GrantResolver.resolve_identity` resolved a claim's user to.
pub(crate) struct ResolvedUser<H: AuthHost> {
    pub(crate) user: Option<HostUser<H>>,
    /// The id the user rebinds the claim to.
    pub(crate) user_id: Option<String>,
    pub(crate) membership: Option<MembershipRow<H>>,
}

/// `GrantResolver.resolve_identity`
pub(crate) async fn resolve_user<H: AuthHost>(
    host: &H,
    user_id: Option<&str>,
    user_email: Option<&str>,
    upsert: bool,
    team_id: Option<&str>,
) -> Result<ResolvedUser<H>, Error> {
    let Some(user_id) = user_id.filter(|text| !text.is_empty()) else {
        return Ok(ResolvedUser {
            user: None,
            user_id: user_id.map(str::to_owned),
            membership: None,
        });
    };
    let lookup = UserLookup {
        user_id,
        user_email,
        sso_user_id: Some(user_id),
        upsert,
    };
    let read = host.resolve_user_identity(&lookup, team_id).await?;
    let Some(user) = read.user.found() else {
        return Err(JwtDenial::UserNotFound {
            user_id: user_id.to_owned(),
        }
        .into());
    };
    let effective = if user.user_id.is_empty() {
        user_id.to_owned()
    } else {
        user.user_id.clone()
    };
    Ok(ResolvedUser {
        user: Some(user),
        user_id: Some(effective),
        membership: read.membership,
    })
}

/// `JWTAuthManager._resolve_claim_identity`, with the user it loaded.
pub(crate) async fn claim_identity<H: AuthHost>(
    host: &H,
    claims: &Claims,
    settings: &JwtSettings,
) -> Result<(ClaimIdentity<UserRow<H>>, Option<HostUser<H>>), Error> {
    let info = claims::user_info(claims, settings);
    let user_id = match claims::rbac_role(claims, settings)? {
        Some(RbacRole::InternalUser) => claims::object_id(claims, settings)
            .filter(|text| !text.is_empty())
            .or(info.user_id),
        _ => info.user_id,
    };
    let agent_id = caller::resolve_agent_id(host, claims, settings).await?;
    let is_admin = claims::is_admin(&claims::scopes(claims)?, settings);
    match resolve_user(
        host,
        user_id.as_deref(),
        info.user_email.as_deref(),
        false,
        None,
    )
    .await
    {
        Ok(resolved) => Ok((
            ClaimIdentity {
                user_id: if is_admin { user_id } else { resolved.user_id },
                agent_id,
                user: resolved.user.as_ref().map(|user| user.row.clone()),
            },
            resolved.user,
        )),
        Err(Error::Jwt(JwtDenial::UserNotFound { .. })) if is_admin => Ok((
            ClaimIdentity {
                user_id,
                agent_id,
                user: None,
            },
            None,
        )),
        Err(error) => Err(error),
    }
}

/// The membership changes `sync_user_role_and_teams` makes from the token's teams.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TeamSync {
    pub add: Vec<String>,
    pub remove: Vec<String>,
    /// The user's teams afterwards.
    pub teams: Vec<String>,
}

fn unique(ids: impl IntoIterator<Item = String>) -> Vec<String> {
    ids.into_iter().fold(Vec::new(), |mut seen, id| {
        if !seen.contains(&id) {
            seen.push(id);
        }
        seen
    })
}

/// The team half of `sync_user_role_and_teams`. Under `fallback_to_db_teams` a token without
/// team claims keeps the user's DB teams. `remove` follows the user's team order.
pub fn team_sync(claims: &Claims, user_teams: &[String], settings: &JwtSettings) -> TeamSync {
    let fallback = settings.config.fallback_to_db_teams;
    let claimed = unique(match fallback {
        true => claims::all_team_ids(claims, settings),
        false => claims::team_ids(claims, settings),
    });
    let existing = unique(user_teams.iter().cloned());
    let add: Vec<String> = claimed
        .iter()
        .filter(|id| !existing.contains(id))
        .cloned()
        .collect();
    let remove: Vec<String> = match fallback && claimed.is_empty() {
        true => Vec::new(),
        false => existing
            .iter()
            .filter(|id| !claimed.contains(id))
            .cloned()
            .collect(),
    };
    let teams = existing
        .iter()
        .filter(|id| !remove.contains(id))
        .chain(&add)
        .cloned()
        .collect();
    TeamSync { add, remove, teams }
}

/// `JWTAuthManager.sync_user_role_and_teams`
pub(crate) async fn sync_user<H: AuthHost>(
    host: &H,
    claims: &Claims,
    user: HostUser<H>,
    settings: &JwtSettings,
) -> Result<HostUser<H>, Error> {
    let new_role = claims::litellm_role(claims, settings)
        .filter(|role| user.user_role.as_deref() != Some(<&str>::from(*role)));
    let user = match new_role {
        Some(role) => User {
            row: host.set_user_role(&user.row, role).await?,
            user_role: Some(<&str>::from(role).to_owned()),
            ..user
        },
        None => user,
    };
    let sync = team_sync(claims, &user.teams, settings);
    if sync.add.is_empty() && sync.remove.is_empty() {
        return Ok(user);
    }
    Ok(User {
        row: host
            .patch_team_membership(&user.row, &sync.add, &sync.remove)
            .await?,
        teams: sync.teams,
        ..user
    })
}

/// The user row a proxy admin's identity carries.
pub(crate) async fn admin_user<H: AuthHost>(
    host: &H,
    claims: &Claims,
    settings: &JwtSettings,
    provisioning: Option<Provisioning>,
    user_id: Option<&str>,
    user_email: Option<&str>,
) -> Result<Option<HostUser<H>>, Error> {
    match (provisioning, user_id) {
        (None, _) => Ok(claim_identity(host, claims, settings).await?.1),
        (Some(_), _) if !settings.has_db => Ok(None),
        (Some(_), None) => Ok(None),
        (Some(_), Some(id)) => {
            let lookup = UserLookup {
                user_id: id,
                user_email,
                sso_user_id: Some(id),
                upsert: false,
            };
            Ok(host.user(&lookup).await?.found())
        }
    }
}
