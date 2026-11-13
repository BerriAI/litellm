use crate::{
    AuthHost, KeyHash, Token,
    authn::key::Keyed,
    host::{EndUserRow, KeyRow, Rows},
    outcome::Refused,
};

mod owner;
mod project;
mod team;

#[derive(Clone, Debug, PartialEq)]
pub struct Owner<U> {
    /// Deactivated through SCIM.
    pub deactivated: bool,
    pub row: U,
}

/// The team values a virtual key carries into the request.
#[derive(Clone, Debug, PartialEq)]
pub enum TeamGrant<R: Rows> {
    NoTeam,
    Loaded(R::Team),
    /// The UI team, or a team `get_team_object` did not report: the key's own copy, with the
    /// object permission it names.
    KeyCopy(Option<R::ObjectPermission>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct KeyContext<R: Rows> {
    pub key: R::Key,
    pub end_user: R::EndUser,
    pub owner: Option<R::User>,
    pub team: TeamGrant<R>,
    pub project: Option<R::Project>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum KeyIdentity<R: Rows> {
    /// No master key is configured, so any token, the one a pass-through endpoint read included, is let through.
    NoMasterKey {
        token: Option<Token>,
    },
    UserAuthRoute,
    CachedAdmin {
        key: R::Key,
        end_user: R::EndUser,
    },
    /// `hash` is the master key's hash, which the host caches the admin identity under.
    MasterKey {
        hash: KeyHash,
        end_user: R::EndUser,
    },
    VirtualKey(Box<KeyContext<R>>),
}

/// Who holds a verified virtual key: its owner, team and project, read before any budget.
pub(crate) async fn principal<H: AuthHost>(
    host: &H,
    key: Keyed<KeyRow<H>>,
    end_user: EndUserRow<H>,
) -> Result<KeyPrincipal<H::Rows>, Refused<H::Rows>> {
    host.prefetch_auth_objects(&key.row, &end_user).await?;
    let owner = owner::read(host, &key.record)
        .await
        .map_err(|denial| Refused::new(denial, Some(&key.row)))?;
    let team = team::read(host, &key.record).await?;
    let project = project::read(host, &key.record).await?;
    Ok(KeyPrincipal {
        key,
        end_user,
        owner,
        team,
        project,
    })
}

/// A virtual key with everything read about who holds it, which admission checks next.
pub(crate) struct KeyPrincipal<R: Rows> {
    pub(crate) key: Keyed<R::Key>,
    pub(crate) end_user: R::EndUser,
    pub(crate) owner: Option<R::User>,
    pub(crate) team: TeamGrant<R>,
    pub(crate) project: Option<R::Project>,
}

impl<R: Rows> From<KeyPrincipal<R>> for KeyContext<R> {
    fn from(principal: KeyPrincipal<R>) -> Self {
        Self {
            key: principal.key.row,
            end_user: principal.end_user,
            owner: principal.owner,
            team: principal.team,
            project: principal.project,
        }
    }
}
