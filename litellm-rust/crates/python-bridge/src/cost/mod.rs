use pyo3::prelude::*;

use crate::errors::RustBridgeDeclined;

#[pyfunction]
pub(crate) fn cost_api(name: &str) -> PyResult<()> {
    Err(RustBridgeDeclined::new_err(format!(
        "cost API {name} is not implemented in Rust"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffold_declines_before_execution() {
        Python::initialize();
        Python::attach(|py| {
            let error = cost_api("cost_per_token").unwrap_err();
            assert!(error.is_instance_of::<RustBridgeDeclined>(py));
            assert_eq!(
                error.to_string(),
                "RustBridgeDeclined: cost API cost_per_token is not implemented in Rust"
            );
        });
    }
}
