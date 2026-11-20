use crate::{AuthHost, Error, host::KeyRow};

/// Rewrites the request's model to the one the key's aliases resolve to, which the budgets read.
pub(super) async fn normalize<H: AuthHost>(host: &H, key: &KeyRow<H>) -> Result<(), Error> {
    host.normalize_model(key).await
}
