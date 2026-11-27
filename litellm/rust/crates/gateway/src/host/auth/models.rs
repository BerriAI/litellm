//! The model lists the native host read for a caller, as the auth core's model rule and the
//! router read them.

use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::{
    Credential,
    authz::{Layer, ModelList},
};

use super::NativeRows;

/// The model lists the native host read for `identity`: a key's own, and none for the master
/// key. A key that names a team, an owner or anything else that carries a list was declined when
/// it was authenticated.
pub(crate) fn layers(identity: &Authenticated<NativeRows>) -> impl Iterator<Item = Layer<'_>> {
    let key = match &identity.credential {
        Credential::MasterKey => None,
        Credential::Key { key } => Some(Layer::Key(ModelList::read(key.models()))),
    };
    key.into_iter()
}

/// Every entry of the model lists the native host read for `identity`, as their rows hold them.
/// The router narrows a caller that reaches a model only through an access group by these, as
/// Python's does by a key's `models`, special values included.
pub(crate) fn allowed_models(identity: &Authenticated<NativeRows>) -> Vec<String> {
    match &identity.credential {
        Credential::MasterKey => Vec::new(),
        Credential::Key { key } => key.models().to_vec(),
    }
}
