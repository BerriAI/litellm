use std::future::Future;
use std::path::Path;

use crate::release::Release;
use crate::{Error, Fetch, Gateway, LaunchSpec, Target};

pub trait Agent: Sync {
    fn binary(&self) -> &'static str;

    fn release(
        &self,
        fetch: &impl Fetch,
        version: &str,
        target: Target,
    ) -> impl Future<Output = Result<Release, Error>> + Send;

    fn launch_spec(&self, gateway: &Gateway, home: &Path) -> LaunchSpec;
}
