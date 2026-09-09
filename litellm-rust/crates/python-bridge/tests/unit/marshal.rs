use super::*;

#[test]
fn timeout_rejects_values_that_cannot_form_a_positive_duration() {
    Python::initialize();
    Python::attach(|py| {
        for timeout in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MAX] {
            let error = optional_timeout(Some(timeout)).expect_err("timeout must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
        }
    });
}

#[test]
fn timeout_accepts_none_and_positive_finite_values() {
    assert_eq!(optional_timeout(None).unwrap(), None);
    assert_eq!(
        optional_timeout(Some(1.5)).unwrap(),
        Some(Duration::from_millis(1500))
    );
}
