use std::future::Future;

use crate::Deployment;

pub trait RouterHooks: Send + Sync {
    fn filter_deployments<'a>(
        &self,
        _model: &str,
        deployments: Vec<&'a Deployment>,
    ) -> impl Future<Output = Vec<&'a Deployment>> + Send {
        async { deployments }
    }
}

impl RouterHooks for () {}
