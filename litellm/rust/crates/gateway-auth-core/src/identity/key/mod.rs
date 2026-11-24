use crate::{
    AuthHost, Belongs, KeyHash, Token,
    authn::key::Keyed,
    host::{KeyRow, Rows},
    outcome::Refused,
};

mod owner;
mod project;
mod team;

/// An object permission row, every column of it, which a key, a team and an end user can each
/// point at. `mcp_tool_permissions` keeps the order its row was written in.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ObjectPermissionRecord {
    pub object_permission_id: String,
    pub mcp_servers: Option<Vec<String>>,
    pub mcp_access_groups: Option<Vec<String>>,
    pub mcp_tool_permissions: Option<Vec<(String, Vec<String>)>>,
    pub vector_stores: Option<Vec<String>>,
    pub agents: Option<Vec<String>>,
    pub agent_access_groups: Option<Vec<String>>,
    pub models: Option<Vec<String>>,
    pub mcp_toolsets: Option<Vec<String>>,
    pub blocked_tools: Option<Vec<String>>,
    pub search_tools: Option<Vec<String>>,
    pub mcp_tool_search_enabled: Option<bool>,
    pub skills: Option<Vec<String>>,
    pub apps: Option<Vec<String>>,
}

/// The request's end user, as far as any host reads one. `found` is whether it has a row, which
/// decides whether its values replace the key's own.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EndUserRecord {
    pub end_user_id: Option<String>,
    pub found: bool,
    pub object_permission: Option<Box<ObjectPermissionRecord>>,
    pub params: Box<EndUserParams>,
}

/// The values an end user sets on a key, each under the key's own field name. One that is `None`
/// leaves the key's value as it is.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EndUserParams {
    pub end_user_id: Option<String>,
    /// `Some(None)` is a row that names no region, which still replaces the key's own.
    pub allowed_model_region: Option<Option<String>>,
    pub end_user_tpm_limit: Option<i64>,
    pub end_user_rpm_limit: Option<i64>,
    pub end_user_tpd_limit: Option<i64>,
    pub end_user_max_budget: Option<f64>,
    pub end_user_model_max_budget: Option<serde_json::Map<String, serde_json::Value>>,
    pub end_user_model_tpm_limit: Option<Vec<(String, i64)>>,
    pub end_user_model_rpm_limit: Option<Vec<(String, i64)>>,
}

/// A key's project, as far as any host reads it: the two values a key in it carries.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProjectRecord {
    pub project_alias: Option<String>,
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Owner<U> {
    /// Deactivated through SCIM.
    pub deactivated: bool,
    pub row: U,
}

/// A virtual key that passed its checks, with the rows read about who holds it.
#[derive(Clone, Debug, PartialEq)]
pub struct KeyContext<R: Rows> {
    pub key: R::Key,
    pub belongs: Belongs<R>,
}

/// What the key path authenticated.
#[derive(Clone, Debug, PartialEq)]
pub enum KeyOutcome<R: Rows> {
    /// No master key is configured, so any token, the one a pass-through endpoint read included, is let through.
    NoMasterKey {
        token: Option<Token>,
    },
    UserAuthRoute,
    CachedAdmin {
        key: R::Key,
        end_user: EndUserRecord,
    },
    /// `hash` is the master key's hash, which the host caches the admin identity under.
    MasterKey {
        hash: KeyHash,
        end_user: EndUserRecord,
    },
    VirtualKey(Box<KeyContext<R>>),
}

/// Who holds a verified virtual key: its owner, team and project, read before any budget.
pub(crate) async fn principal<H: AuthHost>(
    host: &H,
    key: Keyed<KeyRow<H>>,
    end_user: EndUserRecord,
) -> Result<KeyPrincipal<H::Rows>, Refused<H::Rows>> {
    host.prefetch_auth_objects(&key.row, &end_user).await?;
    let owner = owner::read(host, &key.record)
        .await
        .map_err(|denial| Refused::new(denial, Some(&key.row)))?;
    let team = team::read(host, &key.record).await?;
    let project = project::read(host, &key.record).await?;
    Ok(KeyPrincipal {
        key,
        belongs: Belongs {
            user: owner,
            team,
            project,
            end_user: Some(end_user),
            ..Belongs::default()
        },
    })
}

/// A virtual key with everything read about who holds it, which admission checks next.
pub(crate) struct KeyPrincipal<R: Rows> {
    pub(crate) key: Keyed<R::Key>,
    pub(crate) belongs: Belongs<R>,
}

impl<R: Rows> From<KeyPrincipal<R>> for KeyContext<R> {
    fn from(principal: KeyPrincipal<R>) -> Self {
        Self {
            key: principal.key.row,
            belongs: principal.belongs,
        }
    }
}
