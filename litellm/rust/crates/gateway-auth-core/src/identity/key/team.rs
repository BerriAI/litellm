use crate::{
    AuthHost, Error, Team, TeamCopy,
    authn::key::{KeyRecord, UI_TEAM_ID},
};

pub(super) async fn read<H: AuthHost>(
    host: &H,
    record: &KeyRecord,
) -> Result<Option<Team<H::Rows>>, Error> {
    let Some(team_id) = record.team_id.as_deref() else {
        return Ok(None);
    };
    let loaded = match team_id {
        UI_TEAM_ID => None,
        team_id => host.key_team(team_id).await?,
    };
    if let Some(team) = loaded {
        return Ok(Some(Team::Row(team)));
    }
    let object_permission = match record.team_object_permission_id.as_deref() {
        None => None,
        Some(id) => host.object_permission(id).await?,
    };
    Ok(Some(Team::KeyCopy(TeamCopy {
        team_id: team_id.to_owned(),
        object_permission,
    })))
}
