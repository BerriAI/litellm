use pyo3::Python;

use super::Hooks;

/// One scope's contribution to a call's hook chain (gateway, router, SDK call, a callback
/// family). It builds that call's hooks from whatever the scope knows about the call, and
/// may mix Python hooks and Rust built-ins.
pub trait HookLayer<Call> {
    fn layer(&self, py: Python<'_>, call: Call) -> impl IntoIterator<Item = Hooks>;
}
