/// The `litellm_params` keys whose `os.environ/` value is a secret pointer the identity source
/// dereferences at use time, so a loader leaves it as written. `_WIF_POINTER_FIELDS` in
/// litellm/types/router.py.
const SECRET_POINTER_PARAMS: &[&str] = &[
    "anthropic_issuer_signing_key_ref",
    "anthropic_keycloak_client_secret_ref",
];

/// `holds_secret_pointer` in litellm/types/router.py.
pub fn holds_secret_pointer(param: &str) -> bool {
    SECRET_POINTER_PARAMS.contains(&param)
}
