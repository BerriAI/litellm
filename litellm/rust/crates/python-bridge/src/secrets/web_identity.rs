use litellm_auth_aws::{WebIdentityTokenFuture, WebIdentityTokenSource};
use litellm_error::Source;
use pyo3::prelude::*;

use super::error::external_error;

#[derive(Debug)]
pub(crate) struct HostWebIdentityTokens;

impl WebIdentityTokenSource for HostWebIdentityTokens {
    fn web_identity_token<'a>(&'a self, reference: &'a str) -> WebIdentityTokenFuture<'a> {
        Box::pin(async move {
            let source = tokio::task::spawn_blocking(|| {
                Python::attach(|py| super::source(py).map_err(|error| external_error(py, error)))
            })
            .await
            .map_err(Source::new)?
            .map_err(Source::new)?;
            source.get_secret_str(reference).await.map_err(Source::new)
        })
    }
}
