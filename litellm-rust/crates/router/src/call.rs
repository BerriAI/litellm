use std::{sync::Arc, time::Duration};

use crate::{
    Deployment, Error,
    config::RouterConfig,
    deployment::CatalogEntry,
    retry::{AttemptOutcome, RetryDecision},
    selection::{SelectionContext, Selector},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endpoint {
    ChatCompletions,
    Messages,
    Responses,
    Embeddings,
}

#[derive(Clone, Debug, Default)]
pub struct RoutingOptions {
    pub session_id: Option<String>,
    pub retries: Option<u32>,
    pub timeout: Option<Duration>,
}

#[derive(Clone, Debug)]
pub struct RoutingRequest {
    pub model: String,
    pub endpoint: Endpoint,
    pub options: RoutingOptions,
}

pub struct Call {
    pub(crate) config: Arc<RouterConfig>,
    pub(crate) catalog: Arc<[CatalogEntry]>,
    pub(crate) request: RoutingRequest,
    pub(crate) generation: u64,
}

impl Call {
    pub fn request(&self) -> &RoutingRequest {
        &self.request
    }

    pub fn config(&self) -> &RouterConfig {
        &self.config
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn selection_context(&self) -> SelectionContext {
        SelectionContext {
            model: self.request.model.clone(),
            candidates: self
                .catalog
                .iter()
                .filter(|entry| entry.model_name == self.request.model)
                .map(CatalogEntry::candidate)
                .collect(),
            attempt: 0,
            session_id: self.request.options.session_id.clone(),
        }
    }

    pub fn select(&self, selector: &impl Selector) -> Result<Attempt, Error> {
        let id = selector.select(&self.selection_context())?;
        let entry = self
            .catalog
            .iter()
            .find(|entry| entry.id == id && entry.model_name == self.request.model)
            .ok_or(Error::InvalidSelection { id })?;
        Ok(Attempt {
            deployment_id: entry.id.clone(),
            deployment: entry.deployment.clone(),
        })
    }

    pub fn next_attempt(&mut self) -> Result<Attempt, Error> {
        Err(Error::NotImplemented)
    }
}

// TODO: Own the deployment permit here so streams retain capacity until termination
#[must_use]
pub struct Attempt {
    deployment_id: String,
    deployment: Deployment,
}

impl Attempt {
    pub fn deployment_id(&self) -> &str {
        &self.deployment_id
    }

    pub fn deployment(&self) -> &Deployment {
        &self.deployment
    }

    pub fn finish(self, _outcome: AttemptOutcome) -> Result<RetryDecision, Error> {
        Err(Error::NotImplemented)
    }
}

pub trait AttemptAuthorizer {
    fn authorize(&self, request: &RoutingRequest, attempt: &Attempt) -> Result<(), Error>;
}
