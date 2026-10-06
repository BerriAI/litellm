use serde::Serialize;

use super::CacheScope;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Deployment {
    model: String,
    provider: Option<String>,
    api_base: Option<String>,
}

impl Deployment {
    pub fn new(model: &str, provider: Option<&str>, api_base: Option<&str>) -> Self {
        Self {
            model: model.to_owned(),
            provider: provider.map(str::to_owned),
            api_base: api_base.map(str::to_owned),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum CacheTarget<'a> {
    ModelGroup(&'a str),
    Deployment(&'a Deployment),
}

impl<'a> CacheTarget<'a> {
    pub(super) fn resolve(scope: &'a CacheScope, deployment: &'a Deployment) -> Self {
        match &scope.model_group {
            Some(group) => Self::ModelGroup(group),
            None => Self::Deployment(deployment),
        }
    }
}
