use litellm_gateway_auth::hash_token;
use litellm_repositories::KeyView;

/// The row of a key generated with no parameters, as `/key/generate` stores it, under the hash of
/// `token`.
pub fn plain(token: &str) -> KeyView {
    let row: KeyView = serde_json::from_str(include_str!("plain.json")).unwrap();
    KeyView {
        token: hash_token(token),
        ..row
    }
}
