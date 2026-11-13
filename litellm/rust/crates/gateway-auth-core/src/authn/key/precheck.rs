use crate::{KeyDenial, KeyHash, MasterKey, Token, host::Rows, routes::table::USER_AUTH};

use super::{Credential, KeySettings, Keyed, Verified};

/// What the request alone decides, before any host call.
pub(crate) enum Prechecked<'a, R: Rows> {
    Decided(Verified<R>),
    Named(Named<'a, R>),
}

/// A credential that names a key the host still has to read.
pub(crate) enum Named<'a, R: Rows> {
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

impl<R: Rows> Named<'_, R> {
    /// A key the caller resolved is never the master key.
    pub(crate) fn is_master_key(&self) -> bool {
        match self {
            Self::Presented { token, master_key } => master_key.matches(token),
            Self::Resolved { .. } => false,
        }
    }
}

pub(crate) fn precheck<'a, R: Rows>(
    route: &str,
    credential: Credential<'a, R::Key>,
    settings: &'a KeySettings,
) -> Result<Prechecked<'a, R>, KeyDenial> {
    let Some(master_key) = settings.master_key.as_ref() else {
        let token = match credential {
            Credential::Presented(token) => token.cloned(),
            Credential::Resolved(key) => key.record.token.as_ref().map(Token::from_stored),
        };
        return Ok(Prechecked::Decided(Verified::NoMasterKey { token }));
    };
    let named = match credential {
        Credential::Presented(None) => return Err(KeyDenial::MissingKey),
        Credential::Presented(Some(token)) if token.is_empty() => {
            return Err(KeyDenial::MalformedKey);
        }
        Credential::Presented(Some(token)) => Named::Presented { token, master_key },
        Credential::Resolved(key) => match key.record.token.clone() {
            None => return Err(KeyDenial::MissingKey),
            Some(hash) if hash.as_str().is_empty() => return Err(KeyDenial::MalformedKey),
            Some(hash) => Named::Resolved {
                hash,
                key: Box::new(key),
                master_key,
            },
        },
    };
    if route != USER_AUTH {
        return Ok(Prechecked::Named(named));
    }
    match settings.allow_user_auth {
        true => Ok(Prechecked::Decided(Verified::UserAuthRoute)),
        false => Err(KeyDenial::UserAuthDisabled),
    }
}
