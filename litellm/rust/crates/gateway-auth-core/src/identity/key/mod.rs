use crate::{
    AuthHost, Belongs, Holder, KeyHash, Token,
    authn::key::Keyed,
    host::{EndUserRow, KeyRow, Rows},
    outcome::Refused,
};

mod owner;
pub(crate) mod project;
mod team;

#[derive(Clone, Debug, PartialEq)]
pub struct Owner<U> {
    /// Deactivated through SCIM.
    pub deactivated: bool,
    pub row: U,
}

/// A virtual key that passed its checks, with what it belongs to and the request's end user.
#[derive(Clone, Debug, PartialEq)]
pub struct KeyContext<R: Rows> {
    pub key: R::Key,
    pub belongs: Belongs<R>,
    pub end_user: R::EndUser,
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
        belongs: Belongs {
            holder: Holder::new(owner, team.map(|team| (team, None))),
            org: None,
            project,
        },
        end_user,
    })
}

/// A virtual key with everything read about who holds it, which admission checks next.
pub(crate) struct KeyPrincipal<R: Rows> {
    pub(crate) key: Keyed<R::Key>,
    pub(crate) belongs: Belongs<R>,
    pub(crate) end_user: R::EndUser,
}

impl<R: Rows> From<KeyPrincipal<R>> for KeyContext<R> {
    fn from(principal: KeyPrincipal<R>) -> Self {
        Self {
            key: principal.key.row,
            belongs: principal.belongs,
            end_user: principal.end_user,
        }
    }
}
