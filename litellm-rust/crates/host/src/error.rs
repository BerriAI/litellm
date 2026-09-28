use std::sync::Arc;

#[derive(Clone, Debug, thiserror::Error)]
#[error("call interceptor failed: {0}")]
pub struct HookError(#[source] Arc<dyn std::error::Error + Send + Sync>);

impl HookError {
    pub fn new(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self(Arc::new(error))
    }
}

impl PartialEq for HookError {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
