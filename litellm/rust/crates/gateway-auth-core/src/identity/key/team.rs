use crate::{
    AuthHost, Error,
    authn::key::{KeyRecord, UI_TEAM_ID},
};

use super::TeamGrant;

pub(super) async fn read<H: AuthHost>(
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
