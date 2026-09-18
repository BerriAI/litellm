use litellm_callbacks::layer::Layer;

use crate::attempt::{Attempt, AttemptContext, AttemptFactory};
use crate::clock::Clock;
use crate::router::Router;
use crate::routing::{Picker, PlanSource};

/// Wraps an attempt factory in a [`Router`]. Applied once per logical call, so the layer
/// owns the plan source, the picker, the seed and the trace id the host supplied.
pub struct RouterLayer<C: Clock> {
    plan: PlanSource,
    picker: Picker,
    clock: C,
    seed: u64,
    trace_id: Option<String>,
}

impl<C: Clock> RouterLayer<C> {
    pub fn new(plan: impl Into<PlanSource>, picker: Picker, clock: C, seed: u64) -> Self {
        Self {
            plan: plan.into(),
            picker,
            clock,
            seed,
            trace_id: None,
        }
    }

    /// The trace id the host already has for this call; minted from the seed otherwise.
    pub fn with_trace_id(self, trace_id: Option<String>) -> Self {
        Self { trace_id, ..self }
    }
}

impl<F: AttemptFactory, C: Clock + Clone> Layer<F> for RouterLayer<C> {
    type Output = Router<F, C>;

    fn layer(&self, factory: F) -> Router<F, C> {
        Router::new(
            self.plan.clone(),
            self.picker.clone(),
            factory,
            self.clock.clone(),
            self.seed,
            self.trace_id.clone(),
        )
    }
}

/// Applies a machine layer to every attempt the factory produces, so per-attempt behavior
/// (a cache lookup, a timeout) composes inside the router without the layer knowing about
/// factories.
pub struct PerAttempt<L>(pub L);

pub struct PerAttemptFactory<F, L> {
    inner: F,
    layer: L,
}

impl<F, L> Layer<F> for PerAttempt<L>
where
    F: AttemptFactory,
    L: Layer<F::Attempt> + Clone,
    L::Output: Attempt + 'static,
{
    type Output = PerAttemptFactory<F, L>;

    fn layer(&self, inner: F) -> Self::Output {
        PerAttemptFactory {
            inner,
            layer: self.0.clone(),
        }
    }
}

impl<F, L> AttemptFactory for PerAttemptFactory<F, L>
where
    F: AttemptFactory,
    L: Layer<F::Attempt> + Send + Sync,
    L::Output: Attempt + 'static,
{
    type Attempt = L::Output;

    fn start(&self, context: &AttemptContext) -> Self::Attempt {
        self.layer.layer(self.inner.start(context))
    }
}
