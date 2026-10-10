use prost::Message;
use pyo3::{prelude::*, types::PyBytes};

#[derive(Message)]
struct OtlpErrorStatus {
    #[prost(int32, tag = "1")]
    code: i32,
    #[prost(string, tag = "2")]
    message: String,
}

#[pyfunction]
pub fn trace_encode_error<'py>(py: Python<'py>, message: &str) -> Bound<'py, PyBytes> {
    let status = OtlpErrorStatus {
        code: 0,
        message: message.to_owned(),
    };
    PyBytes::new(py, &status.encode_to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    fn error_status_preserves_otlp_wire_contract() {
        Python::initialize();
        Python::attach(|py| {
            let encoded = trace_encode_error(py, "invalid trace");
            assert_eq!(encoded.as_bytes(), b"\x12\x0dinvalid trace");
        });
    }
}
