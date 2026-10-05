pub mod call;
pub mod config;
mod deployment;
mod error;
pub mod retry;
pub mod selection;
pub mod state;

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use call::{Call, RoutingRequest};
use config::{DeploymentConfig, RouterConfig};
use deployment::CatalogEntry;
pub use deployment::Deployment;
pub use error::Error;
use state::Snapshot;

#[derive(Clone, Debug, Default)]
pub struct Router {
    config: Arc<RouterConfig>,
    catalog: Arc<[CatalogEntry]>,
    lookup: Arc<HashMap<String, usize>>,
    generation: u64,
    closed: bool,
}

fn catalog(deployments: &[DeploymentConfig]) -> Arc<[CatalogEntry]> {
    deployments
        .iter()
        .enumerate()
        .map(|(index, config)| CatalogEntry {
            id: config
                .deployment_id
                .clone()
                .unwrap_or_else(|| format!("deployment-{index}")),
            model_name: config.model_name.clone(),
            deployment: config.deployment.clone(),
        })
        .collect()
}

impl Router {
    fn from_catalog(catalog: Arc<[CatalogEntry]>) -> Self {
        Self {
            lookup: Arc::new(
                catalog
                    .iter()
                    .enumerate()
                    .map(|(index, entry)| (entry.model_name.clone(), index))
                    .collect(),
            ),
            catalog,
            ..Self::default()
        }
    }

    pub fn new(config: RouterConfig) -> Result<Self, Error> {
        let catalog = catalog(&config.deployments);
        let mut ids = HashSet::new();
        for entry in catalog.iter() {
            if !ids.insert(&entry.id) {
                return Err(Error::DuplicateDeployment {
                    id: entry.id.clone(),
                });
            }
        }
        Ok(Self {
            config: Arc::new(config),
            ..Self::from_catalog(catalog)
        })
    }

    pub fn get(&self, model_name: &str) -> Option<&Deployment> {
        if self.closed {
            return None;
        }
        self.lookup
            .get(model_name)
            .map(|index| &self.catalog[*index].deployment)
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            generation: self.generation,
            closed: self.closed,
            deployments: self.catalog.iter().map(CatalogEntry::candidate).collect(),
        }
    }

    pub fn reconfigure(&mut self, config: RouterConfig) -> Result<(), Error> {
        if self.closed {
            return Err(Error::Closed);
        }
        let replacement = Self::new(config)?;
        *self = Self {
            generation: self.generation + 1,
            ..replacement
        };
        Ok(())
    }

    pub fn start(&self, request: RoutingRequest) -> Result<Call, Error> {
        if self.closed {
            return Err(Error::Closed);
        }
        if self.get(&request.model).is_none() {
            return Err(Error::UnknownModel {
                model: request.model,
            });
        }
        Ok(Call {
            config: Arc::clone(&self.config),
            catalog: Arc::clone(&self.catalog),
            request,
            generation: self.generation,
        })
    }

    pub fn close(&mut self) {
        self.closed = true;
    }
}

impl FromIterator<(String, Deployment)> for Router {
    fn from_iter<I: IntoIterator<Item = (String, Deployment)>>(entries: I) -> Self {
        Self::from_catalog(
            entries
                .into_iter()
                .enumerate()
                .map(|(index, (model_name, deployment))| CatalogEntry {
                    id: format!("deployment-{index}"),
                    model_name,
                    deployment,
                })
                .collect(),
        )
    }
}
