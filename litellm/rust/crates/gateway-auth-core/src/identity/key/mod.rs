use crate::{
    Affiliation, AuthHost, Holder, KeyHash, Named, Team, Token,
    authn::key::Keyed,
    host::{AffiliationRead, EndUserRow, KeyRow, Rows},
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

#[derive(Clone, Debug, PartialEq)]
pub struct KeyTeam<T> {
    pub row: T,
    pub organization_id: Option<String>,
}

/// A virtual key that passed its checks, with what it belongs to and the request's end user.
#[derive(Clone, Debug, PartialEq)]
pub struct KeyContext<R: Rows> {
    pub key: R::Key,
    pub affiliation: Affiliation<R>,
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

/// Who holds a verified virtual key: its owner, team, organization and project, read before any budget.
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
    let org = if H::READ_KEY_AFFILIATIONS {
        let org_id = key.record.org_id.clone().or_else(|| match &team {
            Some(team::Read::Stored(team)) => team.organization_id.clone(),
            _ => None,
        });
        match org_id {
            Some(id) => {
                let org = Box::pin(host.org(&id, AffiliationRead::Key)).await?;
                Named::of(Some(id), org.map(|org| org.row))
            }
            None => None,
        }
    } else {
        None
    };
    let membership = match (&owner, &team, &key.record.user_id, &key.record.team_id) {
        (Some(_), Some(team::Read::Stored(_)), Some(user_id), Some(team_id))
            if H::READ_KEY_AFFILIATIONS =>
        {
            Box::pin(host.load_team_membership(user_id, team_id, AffiliationRead::Key)).await?
        }
        _ => None,
    };
    let team = team.map(|team| match team {
        team::Read::Stored(team) => Team::Row(team.row),
        team::Read::KeyCopy(copy) => Team::KeyCopy(copy),
    });
    let project = project::read(host, &key.record).await?;
    Ok(KeyPrincipal {
        key,
        affiliation: Affiliation {
            holder: Holder::new(owner, team.map(|team| (team, membership))),
            org,
            project,
        },
        end_user,
    })
}

/// A virtual key with everything read about who holds it, which admission checks next.
pub(crate) struct KeyPrincipal<R: Rows> {
    pub(crate) key: Keyed<R::Key>,
    pub(crate) affiliation: Affiliation<R>,
    pub(crate) end_user: R::EndUser,
}

impl<R: Rows> From<KeyPrincipal<R>> for KeyContext<R> {
    fn from(principal: KeyPrincipal<R>) -> Self {
        Self {
            key: principal.key.row,
            affiliation: principal.affiliation,
            end_user: principal.end_user,
        }
    }
}
