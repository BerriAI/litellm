mod deployment;

use std::collections::HashMap;

pub use deployment::Deployment;

#[derive(Clone, Debug, Default)]
pub struct Router(HashMap<String, Deployment>);

impl Router {
    pub fn get(&self, model_name: &str) -> Option<&Deployment> {
        self.0.get(model_name)
    }
}

impl FromIterator<(String, Deployment)> for Router {
    fn from_iter<I: IntoIterator<Item = (String, Deployment)>>(entries: I) -> Self {
        Self(entries.into_iter().collect())
    }
}
