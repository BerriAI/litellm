use crate::{KeyDenial, host::Rows, routes::table::USER_AUTH};

use super::{Credential, KeyHash, KeyIdentity, KeySettings, Keyed, MasterKey, Token};

/// What the request alone decides, before any host call.
pub(super) enum Precheck<'a, R: Rows> {
    Decided(KeyIdentity<R>),
    Presented {
        token: &'a Token,
        master_key: &'a MasterKey,
    },
    /// `hash` is the token the key stores.
    Resolved {
        hash: KeyHash,
        key: Box<Keyed<R::Key>>,
        master_key: &'a MasterKey,
    },
}

pub(super) fn precheck<'a, R: Rows>(
    route: &str,
    credential: Credential<'a, R::Key>,
    settings: &'a KeySettings,
) -> Result<Precheck<'a, R>, KeyDenial> {
    let Some(master_key) = settings.master_key.as_ref() else {
        let token = match credential {
            Credential::Presented(token) => token.cloned(),
            Credential::Resolved(key) => key.record.token.as_ref().map(Token::from_stored),
        };
        return Ok(Precheck::Decided(KeyIdentity::NoMasterKey { token }));
    };
    let named = match credential {
        Credential::Presented(None) => return Err(KeyDenial::MissingKey),
        Credential::Presented(Some(token)) if token.is_empty() => {
            return Err(KeyDenial::MalformedKey);
        }
        Credential::Presented(Some(token)) => Precheck::Presented { token, master_key },
        Credential::Resolved(key) => match key.record.token.clone() {
            None => return Err(KeyDenial::MissingKey),
            Some(hash) if hash.as_str().is_empty() => return Err(KeyDenial::MalformedKey),
            Some(hash) => Precheck::Resolved {
                hash,
                key: Box::new(key),
                master_key,
            },
        },
    };
    if route != USER_AUTH {
        return Ok(named);
    }
    match settings.allow_user_auth {
        true => Ok(Precheck::Decided(KeyIdentity::UserAuthRoute)),
        false => Err(KeyDenial::UserAuthDisabled),
    }
}

/// Where a looked-up key goes once it is not a cached admin key.
pub(super) enum Gate<K> {
    MasterKey,
    Denied(KeyDenial),
    /// Already read, from the cache or by the caller.
    Held(Keyed<K>),
    /// Not read yet, so read from the database and named by its hash.
    Unread,
}

/// `presented` is absent for a key the caller resolved, which is never the master key.
pub(super) fn gate<K>(
    key: Option<Keyed<K>>,
    presented: Option<&Token>,
    master_key: &MasterKey,
    route: &str,
    settings: &KeySettings,
) -> Gate<K> {
    if presented.is_some_and(|token| master_key.matches(token)) {
        return Gate::MasterKey;
    }
    if settings.master_key_only_routes.contains(route) {
        return Gate::Denied(KeyDenial::MasterKeyOnlyRoute {
            route: route.to_owned(),
        });
    }
    if !settings.has_db {
        return Gate::Denied(KeyDenial::NoDb);
    }
    match (key, presented) {
        (Some(key), _) => Gate::Held(key),
        (None, Some(token)) if !token.is_virtual_key() => Gate::Denied(KeyDenial::NotVirtualKey {
            received: token.masked(),
            jwt_hint: !settings.jwt_auth && token.is_jwt(),
        }),
        (None, _) => Gate::Unread,
    }
}
