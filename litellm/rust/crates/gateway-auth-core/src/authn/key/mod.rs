use std::collections::HashSet;

use jiff::Timestamp;

use crate::{AuthHost, KeyHash, MasterKey, Token, host::Rows, outcome::Refused, token::Shown};

mod expiry;
mod lookup;
mod precheck;

pub(crate) use precheck::{Named, Prechecked, precheck};

pub(crate) const UI_TEAM_ID: &str = "litellm-dashboard";

#[derive(Clone, Debug, Default)]
pub struct KeySettings {
    /// [`HostFacts::master_key`](crate::HostFacts::master_key)
    pub master_key: Option<MasterKey>,
    /// [`HostFacts::has_db`](crate::HostFacts::has_db)
    pub has_db: bool,
    /// [`general_settings.allow_user_auth`](litellm_config::GeneralSettings::allow_user_auth)
    pub allow_user_auth: bool,
    /// [`HostFacts::ui_login`](crate::HostFacts::ui_login)
    pub ui_login: bool,
    /// [`general_settings.enable_jwt_auth`](litellm_config::GeneralSettings::enable_jwt_auth), so a JWT
    /// sent while it is off gets a hint in the denial.
    pub jwt_auth: bool,
    /// [`HostFacts::proxy_budget`](crate::HostFacts::proxy_budget)
    pub proxy_budget: bool,
    /// The `master_key_only_routes` group of [`RouteGroups`](crate::RouteGroups)
    pub master_key_only_routes: HashSet<String>,
    /// `MINIMUM_CUSTOM_KEY_LENGTH`: shorter keys are never echoed back in an error, not even in part.
    pub min_custom_key_length: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct KeyRequest<'a> {
    pub route: &'a str,
    /// [`Request::end_user_id`](crate::Request::end_user_id)
    pub end_user_id: Option<&'a str>,
    pub now: Timestamp,
    /// `RouteChecks.is_llm_api_route(route)`, where the key's max budget applies.
    pub is_llm_api_route: bool,
}

/// What the key path authenticates.
#[derive(Debug)]
pub enum Credential<'a, K> {
    /// The token the request presented, if it presented one.
    Presented(Option<&'a Token>),
    /// A key the caller already read, which a JWT mapped to that key authenticates as.
    Resolved(Keyed<K>),
}

/// The key fields the core decides on.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KeyRecord {
    pub token: Option<KeyHash>,
    pub blocked: bool,
    pub expires: Option<Timestamp>,
    pub is_proxy_admin: bool,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub team_object_permission_id: Option<String>,
    pub project_id: Option<String>,
    /// The end user the key itself names, which wins over the request's.
    pub end_user_id: Option<String>,
    pub last_refreshed_at: Option<Timestamp>,
}

/// A host's key row with the fields the core decides on.
#[derive(Clone, Debug, PartialEq)]
pub struct Keyed<K> {
    pub record: KeyRecord,
    pub row: K,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CachedTeam<T> {
    pub refreshed_at: Option<Timestamp>,
    pub row: T,
}

/// What the credential proved, before anything is read about who holds it.
#[derive(Clone, Debug, PartialEq)]
pub enum Verified<R: Rows> {
    /// No master key is configured, so any token, the one a pass-through endpoint read included, is let through.
    NoMasterKey {
        token: Option<Token>,
    },
    UserAuthRoute,
    /// `hash` is the master key's hash, which the host caches the admin identity under.
    MasterKey {
        hash: KeyHash,
        end_user: R::EndUser,
    },
    CachedAdmin {
        key: R::Key,
        end_user: R::EndUser,
    },
    /// A virtual key that is not blocked, expired or missing its token.
    VirtualKey {
        key: Keyed<R::Key>,
        end_user: R::EndUser,
    },
}

/// Reads the key a named credential stands for and checks it is still usable: not blocked,
/// expired or missing its token. Who holds it is read after.
pub(crate) async fn verify<H: AuthHost>(
    host: &H,
    named: Named<'_, H::Rows>,
    request: &KeyRequest<'_>,
    settings: &KeySettings,
) -> Result<Verified<H::Rows>, Refused<H::Rows>> {
    let (presented, master_key, found) = match named {
        Named::Presented { token, master_key } => {
            let found = lookup::presented(host, token, request.end_user_id, settings).await?;
            (Some(token), master_key, found)
        }
        Named::Resolved {
            hash,
            key,
            master_key,
        } => {
            let found = lookup::resolved(host, hash, *key, request.end_user_id, settings).await?;
            (None, master_key, found)
        }
    };
    let lookup::Lookup {
        hash,
        end_user,
        key,
    } = found;
    let shown = presented.map_or(Shown::Stored(&hash), Shown::Presented);
    let key = match key {
        Some(admin) if admin.record.is_proxy_admin => {
            return match expiry::expired(&admin.record, request.now, shown, settings) {
                Some(denial) => {
                    host.evict_key(&hash).await?;
                    Err(Refused::new(denial, Some(&admin.row)))
                }
                None => Ok(Verified::CachedAdmin {
                    key: admin.row,
                    end_user,
                }),
            };
        }
        key => lookup::with_cached_team(host, key).await?,
    };
    let row = key.as_ref().map(|key| key.row.clone());
    let (key, shown) = match lookup::gate(key, presented, master_key, settings) {
        lookup::Gate::MasterKey => return Ok(Verified::MasterKey { hash, end_user }),
        lookup::Gate::Denied(denial) => return Err(Refused::new(denial, row.as_ref())),
        lookup::Gate::Held(key) => (key, shown),
        lookup::Gate::Unread => match host.db_key(&hash, &end_user).await? {
            Some(key) => (key, Shown::Stored(&hash)),
            None => return Err(Refused::new(crate::KeyDenial::NotFound, None)),
        },
    };
    expiry::usable(&key.record, request.now, shown, settings)
        .map_err(|denial| Refused::new(denial, Some(&key.row)))?;
    Ok(Verified::VirtualKey { key, end_user })
}
