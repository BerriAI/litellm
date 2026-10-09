use pyo3::Python;

use super::Hooks;

pub trait HookLayer<Call> {
    fn layer(&self, py: Python<'_>, call: Call) -> impl IntoIterator<Item = Hooks>;
}
