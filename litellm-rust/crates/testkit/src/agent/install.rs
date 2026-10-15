use std::future::Future;

use semver::Version;

use crate::install::Release;
use crate::{Error, Fetch, Target};

pub trait Install: Sync {
    fn binary(&self) -> &'static str;

    fn release(
        &self,
        fetch: &impl Fetch,
        version: &Version,
        target: Target,
    ) -> impl Future<Output = Result<Release, Error>> + Send;
}
