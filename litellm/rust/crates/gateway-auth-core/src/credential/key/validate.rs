use jiff::Timestamp;

use crate::{
    AuthHost, Error, KeyDenial,
    host::{EndUserRow, KeyRow, UserRow},
};

use super::{
    KeyContext, KeyIdentity, KeyOutcome, KeyRecord, KeyRequest, KeySettings, Keyed, TeamGrant,
    UI_TEAM_ID, refused, token::Shown,
};

pub(super) fn expired(
    key: &KeyRecord,
    now: Timestamp,
    shown: Shown<'_>,
    settings: &KeySettings,
) -> Option<KeyDenial> {
    key.expires
        .filter(|expires| *expires < now)
        .map(|expires| KeyDenial::Expired {
            expires,
            now,
            key: shown.abbreviated(settings.min_custom_key_length),
        })
}

/// The key owner, refusing a key whose owner SCIM deactivated. An owner that cannot be read is absent.
async fn key_owner<H: AuthHost>(
    host: &H,
    record: &KeyRecord,
) -> Result<Option<UserRow<H>>, KeyDenial> {
    let Some(user_id) = record.user_id.as_deref() else {
        return Ok(None);
    };
    match host.key_owner(user_id).await.ok().flatten() {
        Some(owner) if owner.deactivated => Err(KeyDenial::UserDeactivated {
            user_id: user_id.to_owned(),
        }),
        owner => Ok(owner.map(|owner| owner.row)),
    }
}

/// The budgets main runs inside authentication, around the expiry check, in main's order. A free
/// request skips them.
async fn admitted<H: AuthHost>(
    host: &H,
    record: &KeyRecord,
    key: &KeyRow<H>,
    owner: Option<&UserRow<H>>,
    request: &KeyRequest<'_>,
    settings: &KeySettings,
    shown: Shown<'_>,
) -> Result<(), Error> {
    let budgets_apply = !host.key_skips_budget_checks(key).await?;
    if budgets_apply {
        host.team_member_budget(key, owner).await?;
    }
    if let Some(denial) = expired(record, request.now, shown, settings) {
        return Err(denial.into());
    }
    if budgets_apply {
        host.key_budgets(key, owner, request.is_llm_api_route)
            .await?;
    }
    Ok(())
}

pub(super) async fn key_team<H: AuthHost>(
    host: &H,
    record: &KeyRecord,
) -> Result<TeamGrant<H::Rows>, Error> {
    let loaded = match record.team_id.as_deref() {
        None => return Ok(TeamGrant::NoTeam),
        Some(UI_TEAM_ID) => None,
        Some(team_id) => host.key_team(team_id).await?,
    };
    if let Some(team) = loaded {
        return Ok(TeamGrant::Loaded(team));
    }
    let permission = match record.team_object_permission_id.as_deref() {
        None => None,
        Some(id) => host.object_permission(id).await?,
    };
    Ok(TeamGrant::KeyCopy(permission))
}

pub(super) async fn validate<H: AuthHost>(
    host: &H,
    key: Keyed<KeyRow<H>>,
    end_user: EndUserRow<H>,
    request: &KeyRequest<'_>,
    settings: &KeySettings,
    shown: Shown<'_>,
) -> KeyOutcome<H::Rows> {
    if key.record.blocked {
        return Err(refused(KeyDenial::Blocked, Some(&key.row)));
    }
    let Keyed { record, row } = host.with_temp_budget(key).await?;
    host.normalize_model(&row).await?;
    host.prefetch_auth_objects(&row, &end_user).await?;
    let owner = key_owner(host, &record)
        .await
        .map_err(|denial| refused(denial, Some(&row)))?;
    admitted(
        host,
        &record,
        &row,
        owner.as_ref(),
        request,
        settings,
        shown,
    )
    .await
    .map_err(|error| refused(error, Some(&row)))?;
    let team = key_team(host, &record).await?;
    let project = match record.project_id.as_deref() {
        None => None,
        Some(project_id) => host.project(project_id).await?,
    };
    if settings.proxy_budget && settings.has_db {
        host.proxy_budget_alert(&row).await?;
    }
    if record.token.is_none() {
        return Err(refused(KeyDenial::NoToken, Some(&row)));
    }
    Ok(KeyIdentity::VirtualKey(Box::new(KeyContext {
        key: row,
        end_user,
        owner,
        team,
        project,
    })))
}
