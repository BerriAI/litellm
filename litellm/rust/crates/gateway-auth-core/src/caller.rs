//! The caller a host serves once authentication is over. It is the only thing a host carries past
//! authentication, so everything a host later asks about the caller is answered here, from what
//! the key path read.

use std::convert::Infallible;

use litellm_config::LitellmRole;

use crate::{
    Affiliation, KeyHash, Rows,
    authn::key::KeyRecord,
    identity::key::{KeyContext, KeyIdentity},
    outcome::Identity,
};

/// A host's key row that keeps the record the key path read it with, which gives a caller its id
/// and role.
pub trait StoredKey {
    fn record(&self) -> &KeyRecord;
}

impl StoredKey for KeyRecord {
    fn record(&self) -> &KeyRecord {
        self
    }
}

impl StoredKey for Infallible {
    fn record(&self) -> &KeyRecord {
        match *self {}
    }
}

/// Who a request is served for: what it presented, what that belongs to, and the end user the
/// request names, which any credential can carry.
#[derive(Clone, Debug, PartialEq)]
pub struct Caller<R: Rows> {
    pub credential: Credential<R>,
    pub affiliation: Affiliation<R>,
    pub end_user: R::EndUser,
}

/// What a caller presented and the core verified.
#[derive(Clone, Debug, PartialEq)]
pub enum Credential<R: Rows> {
    MasterKey,
    Key { key: R::Key },
}

/// An identity the core authenticated that no host serves a request for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum NotACaller {
    #[error("no master key is configured, so no credential was checked")]
    NoMasterKey,
    #[error("the user auth route authenticates no caller")]
    UserAuthRoute,
    #[error("a cached admin key is served by the host that cached it")]
    CachedAdmin,
    #[error("a JWT is not served as a key's caller")]
    Jwt,
}

impl<R: Rows> TryFrom<Identity<R>> for Caller<R> {
    type Error = NotACaller;

    fn try_from(identity: Identity<R>) -> Result<Self, NotACaller> {
        match identity {
            Identity::Key(KeyIdentity::MasterKey { end_user, .. }) => {
                Ok(Self::master_key(end_user))
            }
            Identity::Key(KeyIdentity::VirtualKey(context)) => {
                let KeyContext {
                    key,
                    affiliation,
                    end_user,
                } = *context;
                Ok(Self {
                    credential: Credential::Key { key },
                    affiliation,
                    end_user,
                })
            }
            Identity::Key(KeyIdentity::NoMasterKey { .. }) => Err(NotACaller::NoMasterKey),
            Identity::Key(KeyIdentity::UserAuthRoute) => Err(NotACaller::UserAuthRoute),
            Identity::Key(KeyIdentity::CachedAdmin { .. }) => Err(NotACaller::CachedAdmin),
            Identity::MappedJwtKey(..) | Identity::AdminJwt(_) | Identity::FinishedJwt(_) => {
                Err(NotACaller::Jwt)
            }
        }
    }
}

impl<R: Rows> Caller<R> {
    /// The master key belongs to nothing, and can still name an end user.
    pub fn master_key(end_user: R::EndUser) -> Self {
        Self {
            credential: Credential::MasterKey,
            affiliation: Affiliation::nobody(),
            end_user,
        }
    }
}

impl<R: Rows<Key: StoredKey>> Caller<R> {
    /// The hash a key's row is stored under, which is the key's id wherever a host counts, caches
    /// or logs by it. Never the hash of the token the request presented, since a key a JWT maps
    /// to presents none. The key path refuses a key whose row has none, so only the master key,
    /// which has no row, has none.
    pub fn hash(&self) -> Option<&KeyHash> {
        match &self.credential {
            Credential::MasterKey => None,
            Credential::Key { key } => key.record().token.as_ref(),
        }
    }

    /// Admin is a role, never a kind of credential: the master key has it, and a key only when
    /// its row says so.
    // TODO: read the key owner's `user_role` once the key path returns it beside the owner's row,
    // as `UserAPIKeyAuth.user_role` is the owner's
    pub fn role(&self) -> LitellmRole {
        match &self.credential {
            Credential::MasterKey => LitellmRole::ProxyAdmin,
            Credential::Key { key } if key.record().is_proxy_admin => LitellmRole::ProxyAdmin,
            Credential::Key { .. } => LitellmRole::InternalUser,
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
        identity::jwt::JwtIdentity,
    };

    #[derive(Clone, Debug, PartialEq)]
    struct Stored;

    impl Rows for Stored {
        type Key = KeyRecord;
        type User = String;
        type Team = String;
        type EndUser = String;
        type ObjectPermission = String;
        type Project = String;
        type Membership = String;
        type Org = String;
        type Agent = String;
    }

    fn stored(hash: &str) -> KeyRecord {
        KeyRecord {
            token: Some(KeyHash::new(hash)),
            ..KeyRecord::default()
        }
    }

    fn key(record: KeyRecord) -> Box<KeyContext<Stored>> {
        Box::new(KeyContext {
            key: record,
            affiliation: Affiliation::nobody(),
            end_user: "end-user".to_owned(),
        })
    }

    fn cached_admin() -> KeyIdentity<Stored> {
        KeyIdentity::CachedAdmin {
            key: KeyRecord {
                is_proxy_admin: true,
                ..stored("stored")
            },
            end_user: "end-user".to_owned(),
        }
    }

    fn mapped_jwt() -> Identity<Stored> {
        let jwt = MappedJwt {
            claims: Value::Null,
            user_email: None,
            end_user_id: None,
            origin: KeyOrigin::Mapping,
            user: None,
        };
        Identity::MappedJwtKey(jwt, key(stored("stored")))
    }

    fn admin_jwt() -> Identity<Stored> {
        Identity::AdminJwt(Box::new(JwtIdentity {
            claims: Value::Null,
            is_proxy_admin: true,
            user_id: None,
            user_email: None,
            team_id: None,
            org_id: None,
            end_user_id: None,
            agent_id: None,
            managed_agent: None,
            affiliation: Affiliation::nobody(),
            end_user: None,
        }))
    }

    #[rstest]
    #[case::no_master_key(
        Identity::Key(KeyIdentity::NoMasterKey { token: Some(Token::new("anything")) }),
        NotACaller::NoMasterKey
    )]
    #[case::user_auth_route(Identity::Key(KeyIdentity::UserAuthRoute), NotACaller::UserAuthRoute)]
    #[case::cached_admin(Identity::Key(cached_admin()), NotACaller::CachedAdmin)]
    #[case::jwt_mapped_to_a_key(mapped_jwt(), NotACaller::Jwt)]
    #[case::admin_jwt(admin_jwt(), NotACaller::Jwt)]
    #[case::finished_jwt(Identity::FinishedJwt(stored("stored")), NotACaller::Jwt)]
    fn an_identity_no_host_serves_never_becomes_a_caller(
        #[case] identity: Identity<Stored>,
        #[case] expected: NotACaller,
    ) {
        assert_eq!(Caller::try_from(identity), Err(expected));
    }

    #[rstest]
    fn the_master_key_is_an_admin_with_no_row_that_keeps_the_requests_end_user() {
        let master = Identity::<Stored>::Key(KeyIdentity::MasterKey {
            hash: Token::new("sk-master").hash(),
            end_user: "end-user".to_owned(),
        });

        let caller = Caller::try_from(master).unwrap();

        assert_eq!(
            (
                caller.hash(),
                caller.role(),
                &caller.affiliation,
                caller.end_user.as_str()
            ),
            (
                None,
                LitellmRole::ProxyAdmin,
                &Affiliation::nobody(),
                "end-user"
            )
        );
    }

    #[rstest]
    #[case::a_plain_key(false, LitellmRole::InternalUser)]
    #[case::a_key_its_row_calls_an_admin(true, LitellmRole::ProxyAdmin)]
    fn a_key_is_named_by_its_stored_hash_and_holds_the_role_its_row_says(
        #[case] is_proxy_admin: bool,
        #[case] role: LitellmRole,
    ) {
        let record = KeyRecord {
            is_proxy_admin,
            ..stored("stored")
        };
        let identity = Identity::Key(KeyIdentity::VirtualKey(key(record)));

        let caller = Caller::try_from(identity).unwrap();

        assert_eq!(
            (caller.hash(), caller.role()),
            (Some(&KeyHash::new("stored")), role)
        );
    }
}
