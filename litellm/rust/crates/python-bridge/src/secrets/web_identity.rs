use litellm_auth_aws::{WebIdentityTokenFuture, WebIdentityTokenSource};
use litellm_error::Source;
use litellm_host_python::{PythonContext, attach_blocking};
use pyo3::prelude::*;

use super::error::external_error;

#[derive(Debug)]
pub(crate) struct HostWebIdentityTokens;

impl WebIdentityTokenSource for HostWebIdentityTokens {
    fn web_identity_token<'a>(&'a self, reference: &'a str) -> WebIdentityTokenFuture<'a> {
        Box::pin(async move {
            let context = Python::attach(|py| {
                PythonContext::empty(py).map_err(|error| external_error(py, error))
            })
            .map_err(Source::new)?;
            let source = match attach_blocking(context, |py| {
                super::source(py).map_err(|error| external_error(py, error))
            })
            .await
            {
                Ok(source) => source,
                Err(error) => Err(Python::attach(|py| external_error(py, error))),
            }
            .map_err(Source::new)?;
            source.get_secret_str(reference).await.map_err(Source::new)
        })
    }
}
