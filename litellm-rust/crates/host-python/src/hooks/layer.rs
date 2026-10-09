use pyo3::Python;

use super::Hooks;

pub trait HookLayer<Call> {
    fn layer(self, py: Python<'_>, call: Call) -> impl IntoIterator<Item = Hooks>;
}

impl<Call, F, I> HookLayer<Call> for F
where
    F: FnOnce(Python<'_>, Call) -> I,
    I: IntoIterator<Item = Hooks>,
{
    fn layer(self, py: Python<'_>, call: Call) -> impl IntoIterator<Item = Hooks> {
        self(py, call)
    }
}
