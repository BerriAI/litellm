use crate::{
    AuthHost, Error,
    host::{EndUserRow, KeyRow},
};

use super::{KeyHash, KeySettings, Keyed, Refused, Token, UI_TEAM_ID, refused};

/// The end user, and the key the credential names, with its team values current.
pub(super) struct Lookup<H: AuthHost> {
    pub(super) hash: KeyHash,
    pub(super) end_user: EndUserRow<H>,
    pub(super) key: Option<Keyed<KeyRow<H>>>,
}

pub(super) async fn presented<H: AuthHost>(
    host: &H,
    token: &Token,
    requested_end_user: Option<&str>,
    settings: &KeySettings,
) -> Result<Lookup<H>, Refused<H::Rows>> {
    let end_user = request_end_user(host, Some(token), requested_end_user, settings).await?;
    let hash = token.hash();
    let key = found_key(host, token, settings, &hash).await?;
    Ok(Lookup {
        hash,
        end_user,
        key,
    })
}

/// A key the caller already holds names the end user before the request does.
pub(super) async fn resolved<H: AuthHost>(
    host: &H,
    hash: KeyHash,
    key: Keyed<KeyRow<H>>,
    requested_end_user: Option<&str>,
    settings: &KeySettings,
) -> Result<Lookup<H>, Refused<H::Rows>> {
    let named = key
        .record
        .end_user_id
        .as_deref()
        .filter(|end_user_id| !end_user_id.is_empty())
        .or(requested_end_user);
    let end_user = request_end_user(host, None, named, settings)
        .await
        .map_err(|error| refused(error, Some(&key.row)))?;
    Ok(Lookup {
        hash,
        end_user,
        key: Some(key),
    })
}

/// The end user `named`, which the host may drop when the proxy does not know the id. The cache
/// is warmed for the id that is left, and for the key `unread` names.
async fn request_end_user<H: AuthHost>(
    host: &H,
    unread: Option<&Token>,
    named: Option<&str>,
    settings: &KeySettings,
) -> Result<EndUserRow<H>, Error> {
    let end_user_id = match named {
        None => None,
        Some(end_user_id) => host.validated_end_user_id(end_user_id).await?,
    };
    if settings.has_db {
        host.prefetch_identity(unread, end_user_id.as_deref())
            .await?;
    }
    host.end_user(end_user_id.as_deref()).await
}

async fn found_key<H: AuthHost>(
    host: &H,
    token: &Token,
    settings: &KeySettings,
    hash: &KeyHash,
) -> Result<Option<Keyed<KeyRow<H>>>, Error> {
    match host.cached_key(hash).await.ok().flatten() {
        None if !token.is_virtual_key() && settings.ui_login => host.ui_key(token).await,
        cached => Ok(cached),
    }
}

/// A key whose cached team was refreshed after it takes that team's values.
pub(super) async fn with_cached_team<H: AuthHost>(
    host: &H,
    key: Option<Keyed<KeyRow<H>>>,
) -> Result<Option<Keyed<KeyRow<H>>>, Error> {
    match key {
        None => Ok(None),
        Some(key) => cached_team_values(host, key).await.map(Some),
    }
}

async fn cached_team_values<H: AuthHost>(
    host: &H,
    key: Keyed<KeyRow<H>>,
) -> Result<Keyed<KeyRow<H>>, Error> {
    let Some(team_id) = key
        .record
        .team_id
        .as_deref()
        .filter(|team_id| *team_id != UI_TEAM_ID)
    else {
        return Ok(key);
    };
    let Some(team) = host.cached_team(team_id).await.ok().flatten() else {
        return Ok(key);
    };
    match (team.refreshed_at, key.record.last_refreshed_at) {
        (Some(team_refreshed), Some(key_refreshed)) if team_refreshed > key_refreshed => {
            Ok(Keyed {
                row: host.with_team(&key.row, &team.row).await?,
                record: key.record,
            })
        }
        _ => Ok(key),
    }
}
