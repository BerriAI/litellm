//! Host-side logging integrations. The callback traits and payload types live
//! in [`litellm_core::callbacks`]; this module holds the I/O implementations:
//!   - [`litellm_python_proxy_api::LiteLLMPythonProxyAPILogger`] — ships events
//!     to the Python proxy's `/v1/rust_control_plane/logs` endpoint

pub mod litellm_python_proxy_api;
