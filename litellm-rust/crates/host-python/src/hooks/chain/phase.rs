use litellm_host::{
    interceptors::{RequestContext, WireRequest},
    lifecycle::Timing,
};
use pyo3::{prelude::*, types::PyDict};

use crate::HookStep;

use super::adapter::{ChainHooks, ChainStep};
use super::dispatch::{HookChain, resume_unless_cancelled};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Order {
    Inbound,
    Outbound,
}

impl Order {
    pub(super) fn first(self, len: usize) -> Option<usize> {
        match self {
            Self::Inbound => (len > 0).then_some(0),
            Self::Outbound => len.checked_sub(1),
        }
    }

    pub(super) fn next(self, index: usize, len: usize) -> Option<usize> {
        match self {
            Self::Inbound => (index + 1 < len).then_some(index + 1),
            Self::Outbound => index.checked_sub(1),
        }
    }

    /// Every index in walking order, for the boundaries that only notify.
    pub(super) fn walk(self, len: usize) -> impl Iterator<Item = usize> {
        std::iter::successors(self.first(len), move |&index| self.next(index, len))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Stage {
    Wrapper { started_at: f64 },
    Body,
}

pub(super) trait Transformed: Sized + 'static {
    type Context: Clone + Send + Sync + 'static;

    fn invoke(
        hooks: &mut dyn ChainHooks,
        py: Python<'_>,
        value: Self,
        context: &Self::Context,
    ) -> PyResult<ChainStep<Self>>;

    fn resume(
        hooks: &mut dyn ChainHooks,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<Self>>;
}

impl Transformed for Py<PyDict> {
    type Context = Stage;

    fn invoke(
        hooks: &mut dyn ChainHooks,
        py: Python<'_>,
        value: Self,
        context: &Stage,
    ) -> PyResult<ChainStep<Self>> {
        match *context {
            Stage::Wrapper { started_at } => hooks.prepare_arguments(py, value, started_at),
            Stage::Body => hooks.prepare_request(py, value),
        }
    }

    fn resume(
        hooks: &mut dyn ChainHooks,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<Self>> {
        hooks.resume_arguments(py, result)
    }
}

impl Transformed for Box<WireRequest> {
    type Context = RequestContext;

    fn invoke(
        hooks: &mut dyn ChainHooks,
        py: Python<'_>,
        value: Self,
        context: &RequestContext,
    ) -> PyResult<ChainStep<Self>> {
        hooks.before_provider_request(py, value, context)
    }

    fn resume(
        hooks: &mut dyn ChainHooks,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<Self>> {
        hooks.resume_wire(py, result)
    }
}

impl Transformed for Py<PyAny> {
    type Context = Timing;

    fn invoke(
        hooks: &mut dyn ChainHooks,
        py: Python<'_>,
        value: Self,
        context: &Timing,
    ) -> PyResult<ChainStep<Self>> {
        hooks.transform_response(py, value, *context)
    }

    fn resume(
        hooks: &mut dyn ChainHooks,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<ChainStep<Self>> {
        hooks.resume_response(py, result)
    }
}

pub(super) fn transform<P: Transformed>(
    py: Python<'_>,
    hooks: &mut [Box<dyn ChainHooks>],
    order: Order,
    value: P,
    context: P::Context,
) -> PyResult<HookStep<HookChain, P>> {
    advance(py, hooks, order, order.first(hooks.len()), value, context)
}

fn advance<P: Transformed>(
    py: Python<'_>,
    hooks: &mut [Box<dyn ChainHooks>],
    order: Order,
    index: Option<usize>,
    value: P,
    context: P::Context,
) -> PyResult<HookStep<HookChain, P>> {
    let Some(index) = index else {
        return Ok(HookStep::Ready(value));
    };
    let step = P::invoke(hooks[index].as_mut(), py, value, &context)?;
    continue_from(py, hooks, order, index, step, context)
}

fn continue_from<P: Transformed>(
    py: Python<'_>,
    hooks: &mut [Box<dyn ChainHooks>],
    order: Order,
    index: usize,
    step: ChainStep<P>,
    context: P::Context,
) -> PyResult<HookStep<HookChain, P>> {
    match step {
        ChainStep::Ready(value) => {
            let next = order.next(index, hooks.len());
            advance(py, hooks, order, next, value, context)
        }
        ChainStep::Await(awaitable) => Ok(HookStep::Await(
            awaitable,
            Box::new(move |chain, py, result| {
                let result = resume_unless_cancelled(py, result)?;
                let step = P::resume(chain.hooks[index].as_mut(), py, result)?;
                continue_from(py, &mut chain.hooks, order, index, step, context)
            }),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::Order;
    use rstest::rstest;

    #[rstest]
    #[case::inbound_empty(Order::Inbound, 0, &[])]
    #[case::outbound_empty(Order::Outbound, 0, &[])]
    #[case::inbound(Order::Inbound, 3, &[0, 1, 2])]
    #[case::outbound(Order::Outbound, 3, &[2, 1, 0])]
    fn orders_walk_the_chain_from_either_end(
        #[case] order: Order,
        #[case] len: usize,
        #[case] expected: &[usize],
    ) {
        let mut walked = Vec::new();
        let mut index = order.first(len);
        while let Some(current) = index {
            walked.push(current);
            index = order.next(current, len);
        }
        assert_eq!(walked, expected);
        assert_eq!(order.walk(len).collect::<Vec<_>>(), expected);
    }
}
