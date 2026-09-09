use super::{ActionResult, Delivery, FailurePolicy};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallbackRuntime {
    Native,
    Python,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResultPolicy {
    Observe,
    Transform,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperationContract {
    pub delivery: Delivery,
    pub result: ResultPolicy,
    pub failure: FailurePolicy,
}

impl OperationContract {
    pub fn is_awaited(self) -> bool {
        self.delivery == Delivery::InlineAwaited
    }

    pub(crate) fn apply<T>(self, result: ActionResult<T, crate::Error>) -> Result<T, crate::Error> {
        match result {
            ActionResult::Continue(value) => Ok(value),
            ActionResult::Replace(value) if self.result == ResultPolicy::Transform => Ok(value),
            ActionResult::Replace(_) => Err(crate::Error::InvalidRequest(
                "observational operation cannot replace its input".into(),
            )),
            ActionResult::Reject(error) => Err(error),
        }
    }
}
