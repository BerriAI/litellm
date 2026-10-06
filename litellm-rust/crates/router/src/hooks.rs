use futures_util::future::BoxFuture;

use crate::Deployment;

pub trait RouterHooks: Send + Sync {
    fn filter_deployments<'h, 'd: 'h>(
        &'h self,
        _model: &'h str,
        deployments: Vec<&'d Deployment>,
    ) -> BoxFuture<'h, Vec<&'d Deployment>> {
        Box::pin(async move { deployments })
    }
}

impl RouterHooks for () {}
