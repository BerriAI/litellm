use std::ffi::CString;

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Callback, CallbackError, Field, FieldSet, Hook, HookInput, HookOutput, Manifest, Patch,
    Selection, Subscription, Verdict, typed,
};
use litellm_callbacks_python::RustCallback;
use pyo3::{prelude::*, types::PyDict};
use rstest::rstest;

struct Echo(Manifest);

impl Callback for Echo {
    type State = ();

    fn manifest(&self) -> &Manifest {
        &self.0
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        _state: Option<&'a ()>,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            let mut reservation = input
                .fields
                .get::<typed::BudgetReservation>()?
                .unwrap_or_default();
            reservation.input_cost = reservation.reserved_cost.map(|cost| -cost);
            Ok(Some(Verdict::Patch(
                Patch::default().set(
                    Field::BudgetReservation,
                    serde_json::to_value(reservation)
                        .map_err(|error| CallbackError::Failed(error.to_string()))?,
                ),
            )))
        })
    }
}

const SCRIPT: &str = r#"
import asyncio
import math


async def handle():
    return await rust.handle(
        "pre_call",
        {"budget_reservation": {"reserved_cost": float("inf"), "entries": [{"applied_adjustment": float("nan")}]}},
        None,
        None,
    )


written = asyncio.run(handle())["patch"]["ops"][0]["value"]
checks = (
    written["reserved_cost"] == float("inf"),
    written["input_cost"] == float("-inf"),
    math.isnan(written["entries"][0]["applied_adjustment"]),
)
"#;

#[rstest]
fn a_non_finite_reservation_cost_reads_and_writes_back_as_a_python_float() {
    let echo = Echo(Manifest {
        selection: Selection::named("non_finite_echo"),
        subscriptions: vec![Subscription {
            reads: FieldSet::of(&[Field::BudgetReservation]),
            writes: FieldSet::of(&[Field::BudgetReservation]),
            ..Subscription::on(Hook::PreCall)
        }],
        has_request_state: false,
    });
    Python::initialize();
    let checks = Python::attach(|py| {
        let locals = PyDict::new(py);
        locals
            .set_item(
                "rust",
                Bound::new(py, RustCallback::new(std::sync::Arc::new(echo)).unwrap()).unwrap(),
            )
            .unwrap();
        if let Err(error) = py.run(&CString::new(SCRIPT).unwrap(), Some(&locals), Some(&locals)) {
            error.display(py);
            panic!("the non-finite script failed");
        }
        locals
            .get_item("checks")
            .unwrap()
            .unwrap()
            .extract::<(bool, bool, bool)>()
            .unwrap()
    });
    assert_eq!(checks, (true, true, true));
}
