use serde::Serialize;

use super::{ActionKind, Delivery, FailurePolicy};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ResultPolicy {
    Continue,
    Replace,
    Reject,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Owner {
    Core,
    Route,
    Host,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ActionBinding {
    pub kind: ActionKind,
    pub delivery: Delivery,
    pub on_result: ResultPolicy,
    pub on_error: FailurePolicy,
    pub owner: Owner,
}
