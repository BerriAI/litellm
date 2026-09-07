use std::ffi::CStr;

/// Python source of the coroutine adapter that awaits a retained callback
/// inline in the caller's task. It is compiled once per interpreter.
pub(crate) const AWAIT_ADAPTER_SOURCE: &CStr =
    c"async def invoke_awaited(callable, positional, keywords):
    if keywords is None:
        return await callable(*positional)
    return await callable(*positional, **keywords)
";

/// Filename recorded on the adapter's code object. Visible to Python
/// `compile` audit hooks and tracebacks.
pub const AWAIT_ADAPTER_FILENAME: &CStr = c"retained_callback.py";

pub(crate) const AWAIT_ADAPTER_MODULE: &CStr = c"_retained_callback";

pub(crate) const AWAIT_ADAPTER_FUNCTION: &str = "invoke_awaited";
