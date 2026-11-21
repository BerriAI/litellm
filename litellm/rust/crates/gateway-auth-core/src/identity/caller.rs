//! The authenticated caller a host serves a request for, built in one place from the identity the
//! pipeline returns.

use litellm_config::LitellmRole;

use crate::{
    KeyHash, Rows,
    authn::key::{KeyRecord, Keyed},
    identity::key::{KeyContext, KeyIdentity, TeamGrant},
    outcome::Identity,
};

/// One variant per way a credential is proven.
#[derive(Clone, Debug, PartialEq)]
pub enum Caller<R: Rows> {
    MasterKey,
    Key(Box<KeyCaller<R>>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct KeyCaller<R: Rows> {
    /// The stored row's hash, which is the active key's when a rotated hash resolved to it.
    pub hash: KeyHash,
    pub role: LitellmRole,
    /// The facts the core decided on.
    pub record: KeyRecord,
    pub key: R::Key,
    pub scope: Scope<R>,
}

/// What the core read about who holds the key.
#[derive(Clone, Debug, PartialEq)]
pub enum Scope<R: Rows> {
    /// A cached proxy admin's key: the core reads no owner, team or project for it.
    Unread,
    Read {
        owner: Option<R::User>,
        team: TeamGrant<R>,
        project: Option<R::Project>,
    },
}

/// An identity that authenticated but is not a caller a host can serve from `Caller`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum NotACaller {
    #[error("no master key is configured, so the request proved no credential")]
    NoMasterKey,
    #[error("the user auth route authenticates no caller")]
    UserAuthRoute,
    #[error("the key's record has no stored token")]
    KeyWithoutToken,
    #[error("a JWT mapped to a virtual key is not a caller yet")]
    MappedJwtKey,
    #[error("a proxy admin's JWT is not a caller yet")]
    AdminJwt,
    #[error("a JWT the host finished is not a caller yet")]
    FinishedJwt,
}

impl<R: Rows> Caller<R> {
    pub fn role(&self) -> LitellmRole {
        match self {
            Self::MasterKey => LitellmRole::ProxyAdmin,
            Self::Key(key) => key.role,
        }
    }
}

impl<R: Rows> KeyCaller<R> {
    // TODO: the role is the record's until owners are served natively and `Owner` carries the
    // owner's `user_role`, which the proxy falls back to before `internal_user`
    fn new(key: Keyed<R::Key>, scope: Scope<R>) -> Result<Self, NotACaller> {
        let Keyed { record, row } = key;
        let hash = record.token.clone().ok_or(NotACaller::KeyWithoutToken)?;
        let role = match record.is_proxy_admin {
            true => LitellmRole::ProxyAdmin,
            false => LitellmRole::InternalUser,
        };
        Ok(Self {
            hash,
            role,
            record,
            key: row,
            scope,
        })
    }
}

impl<R: Rows> TryFrom<Identity<R>> for Caller<R> {
    type Error = NotACaller;

    fn try_from(identity: Identity<R>) -> Result<Self, NotACaller> {
        match identity {
            Identity::Key(KeyIdentity::MasterKey { .. }) => Ok(Self::MasterKey),
            Identity::Key(KeyIdentity::VirtualKey(context)) => {
                let KeyContext {
                    key,
                    owner,
                    team,
                    project,
                    end_user: _,
                } = *context;
                let scope = Scope::Read {
                    owner,
                    team,
                    project,
                };
                KeyCaller::new(key, scope).map(|key| Self::Key(Box::new(key)))
            }
            Identity::Key(KeyIdentity::CachedAdmin { key, .. }) => {
                KeyCaller::new(key, Scope::Unread).map(|key| Self::Key(Box::new(key)))
            }
            Identity::Key(KeyIdentity::NoMasterKey { .. }) => Err(NotACaller::NoMasterKey),
            Identity::Key(KeyIdentity::UserAuthRoute) => Err(NotACaller::UserAuthRoute),
            Identity::MappedJwtKey(..) => Err(NotACaller::MappedJwtKey),
            Identity::AdminJwt(_) => Err(NotACaller::AdminJwt),
            Identity::FinishedJwt(_) => Err(NotACaller::FinishedJwt),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::Value;

    use super::*;
    use crate::{
        Token,
        authn::jwt::{KeyOrigin, MappedJwt},
        identity::jwt::{JwtIdentity, JwtRows},
        testing::Labels,
    };

    fn record(is_proxy_admin: bool) -> KeyRecord {
        KeyRecord {
            token: Some(KeyHash::new("stored")),
            is_proxy_admin,
            user_id: Some("u".to_owned()),
            ..KeyRecord::default()
        }
    }

    fn keyed(record: KeyRecord) -> Keyed<String> {
        Keyed {
            record,
            row: "key row".to_owned(),
        }
    }

    fn context(record: KeyRecord) -> Box<KeyContext<Labels>> {
        Box::new(KeyContext {
            key: keyed(record),
            end_user: "end user".to_owned(),
            owner: Some("owner".to_owned()),
            team: TeamGrant::Loaded("team".to_owned()),
            project: Some("project".to_owned()),
        })
    }

    fn virtual_key(record: KeyRecord) -> Identity<Labels> {
        Identity::Key(KeyIdentity::VirtualKey(context(record)))
    }

    fn cached_admin(record: KeyRecord) -> Identity<Labels> {
        Identity::Key(KeyIdentity::CachedAdmin {
            key: keyed(record),
            end_user: "end user".to_owned(),
        })
    }

    fn mapped_jwt() -> MappedJwt<String> {
        MappedJwt {
            claims: Value::Null,
            end_user_id: None,
            user_email: None,
            origin: KeyOrigin::Mapping,
            user: None,
        }
    }

    fn admin_jwt() -> JwtIdentity<Labels> {
        JwtIdentity {
            claims: Value::Null,
            is_proxy_admin: true,
            user_id: None,
            user_email: None,
            team_id: None,
            org_id: None,
            end_user_id: None,
            agent_id: None,
            managed_agent: None,
            rows: JwtRows::default(),
        }
    }

    #[rstest]
    fn the_master_key_is_a_proxy_admin() {
        let identity: Identity<Labels> = Identity::Key(KeyIdentity::MasterKey {
            hash: Token::new("sk-master").hash(),
            end_user: "end user".to_owned(),
        });

        let caller = Caller::try_from(identity);

        assert_eq!(caller, Ok(Caller::MasterKey));
        assert_eq!(caller.unwrap().role(), LitellmRole::ProxyAdmin);
    }

    #[rstest]
    #[case::a_key_with_no_role(false, LitellmRole::InternalUser)]
    #[case::a_proxy_admins_key(true, LitellmRole::ProxyAdmin)]
    fn a_virtual_key_keeps_its_stored_hash_its_record_and_what_was_read(
        #[case] is_proxy_admin: bool,
        #[case] role: LitellmRole,
    ) {
        let caller = Caller::try_from(virtual_key(record(is_proxy_admin)));

        assert_eq!(
            caller,
            Ok(Caller::Key(Box::new(KeyCaller {
                hash: KeyHash::new("stored"),
                role,
                record: record(is_proxy_admin),
                key: "key row".to_owned(),
                scope: Scope::Read {
                    owner: Some("owner".to_owned()),
                    team: TeamGrant::Loaded("team".to_owned()),
                    project: Some("project".to_owned()),
                },
            })))
        );
        assert_eq!(caller.unwrap().role(), role);
    }

    #[rstest]
    fn a_cached_admin_key_has_nothing_read_for_it() {
        assert_eq!(
            Caller::try_from(cached_admin(record(true))),
            Ok(Caller::Key(Box::new(KeyCaller {
                hash: KeyHash::new("stored"),
                role: LitellmRole::ProxyAdmin,
                record: record(true),
                key: "key row".to_owned(),
                scope: Scope::Unread,
            })))
        );
    }

    #[rstest]
    #[case::no_master_key(
        Identity::Key(KeyIdentity::NoMasterKey { token: None }),
        NotACaller::NoMasterKey
    )]
    #[case::the_user_auth_route(
        Identity::Key(KeyIdentity::UserAuthRoute),
        NotACaller::UserAuthRoute
    )]
    #[case::a_virtual_key_without_a_token(
        virtual_key(KeyRecord::default()),
        NotACaller::KeyWithoutToken
    )]
    #[case::a_cached_admin_key_without_a_token(
        cached_admin(KeyRecord { is_proxy_admin: true, ..KeyRecord::default() }),
        NotACaller::KeyWithoutToken
    )]
    #[case::a_jwt_mapped_to_a_key(
        Identity::MappedJwtKey(mapped_jwt(), context(record(false))),
        NotACaller::MappedJwtKey
    )]
    #[case::an_admin_jwt(Identity::AdminJwt(Box::new(admin_jwt())), NotACaller::AdminJwt)]
    #[case::a_finished_jwt(Identity::FinishedJwt("token".to_owned()), NotACaller::FinishedJwt)]
    fn an_identity_that_is_not_a_caller_says_what_it_was(
        #[case] identity: Identity<Labels>,
        #[case] expected: NotACaller,
    ) {
        assert_eq!(Caller::try_from(identity), Err(expected));
    }
}
