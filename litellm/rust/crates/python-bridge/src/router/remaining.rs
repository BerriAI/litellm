use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_callbacks::{RemainingUsage, RouterReadError, RouterUsage, UsageAmount};
use litellm_callbacks_python::router::{PyRouterRef, headers as usage_headers};
use litellm_router_core::remaining::{
    Amount, GroupLimits, Limit, RemainingError, RemainingReads, Totals, remaining_usage,
    usage_totals,
};
use num_bigint::BigInt;
use pyo3::{
    exceptions::{PyOverflowError, PyTypeError},
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt},
};

fn limit(value: &Bound<'_, PyAny>) -> PyResult<Option<Limit>> {
    if value.is_none() {
        return Ok(None);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(Some(Limit::Amount(Amount::Int(BigInt::from(u8::from(
            value.extract::<bool>()?,
        ))))));
    }
    if value.is_instance_of::<PyInt>() {
        return Ok(Some(Limit::Amount(Amount::Int(value.extract()?))));
    }
    if value.is_instance_of::<PyFloat>() {
        return Ok(Some(Limit::Amount(Amount::Float(value.extract()?))));
    }
    Ok(Some(Limit::Other(value.get_type().name()?.to_string())))
}

fn counted(value: &Bound<'_, PyAny>) -> PyResult<Option<BigInt>> {
    if value.is_instance_of::<PyBool>() {
        return Ok(Some(BigInt::from(u8::from(value.extract::<bool>()?))));
    }
    if value.is_instance_of::<PyInt>() {
        return value.extract().map(Some);
    }
    Ok(None)
}

fn totals(value: &Bound<'_, PyAny>) -> PyResult<Option<Totals>> {
    if value.is_none() {
        return Ok(None);
    }
    let (first, second): (Bound<'_, PyAny>, Bound<'_, PyAny>) = value.extract()?;
    let total = |value: &Bound<'_, PyAny>| -> PyResult<Option<BigInt>> {
        if value.is_none() {
            Ok(None)
        } else {
            counted(value)
        }
    };
    Ok(Some((total(&first)?, total(&second)?)))
}

fn reads(value: &Bound<'_, PyAny>) -> PyResult<RemainingReads> {
    let (limits, io, usage, reset): (Bound<'_, PyAny>, Bound<'_, PyAny>, Bound<'_, PyAny>, i64) =
        value.extract()?;
    let limits = if limits.is_none() {
        None
    } else {
        let (tpm, rpm, itpm, otpm): (
            Bound<'_, PyAny>,
            Bound<'_, PyAny>,
            Bound<'_, PyAny>,
            Bound<'_, PyAny>,
        ) = limits.extract()?;
        Some(GroupLimits {
            tpm: limit(&tpm)?,
            rpm: limit(&rpm)?,
            itpm: limit(&itpm)?,
            otpm: limit(&otpm)?,
        })
    };
    Ok(RemainingReads {
        limits,
        io: totals(&io)?,
        usage: totals(&usage)?,
        reset,
    })
}

fn raised(error: &RemainingError) -> PyErr {
    match error {
        RemainingError::Unsubtractable(_) => PyTypeError::new_err(error.to_string()),
        RemainingError::IntTooLarge => PyOverflowError::new_err(error.to_string()),
    }
}

fn remaining(reads: &RemainingReads) -> PyResult<RemainingUsage> {
    let headers = remaining_usage(reads).map_err(|error| raised(&error))?;
    Ok(RemainingUsage {
        headers: headers
            .into_iter()
            .map(|(name, amount)| {
                (
                    name.to_owned(),
                    match amount {
                        Amount::Int(value) => UsageAmount::Int(value),
                        Amount::Float(value) => UsageAmount::Float(value),
                    },
                )
            })
            .collect(),
    })
}

struct HostedRouterUsage {
    reads: Py<PyAny>,
}

impl HostedRouterUsage {
    fn read(
        &self,
        model_group: String,
    ) -> PyResult<impl Future<Output = PyResult<Py<PyAny>>> + Send + use<>> {
        Python::attach(|py| {
            let awaitable = self.reads.bind(py).call1((model_group,))?;
            pyo3_async_runtimes::tokio::into_future(awaitable)
        })
    }
}

impl RouterUsage for HostedRouterUsage {
    fn remaining(
        &self,
        model_group: &str,
    ) -> BoxFuture<'static, Result<RemainingUsage, RouterReadError>> {
        let read = self.read(model_group.to_owned());
        Box::pin(async move {
            let answered = match read {
                Ok(read) => read.await,
                Err(error) => Err(error),
            };
            Python::attach(|py| answered.and_then(|answered| remaining(&reads(answered.bind(py))?)))
                .map_err(|error| Box::new(error) as RouterReadError)
        })
    }
}

#[pyfunction]
pub(crate) fn router_usage_ref(reads: Py<PyAny>) -> PyRouterRef {
    PyRouterRef::new(Arc::new(HostedRouterUsage { reads }))
}

#[pyfunction]
pub(crate) fn router_remaining_usage<'py>(
    py: Python<'py>,
    reads: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    usage_headers(py, &remaining(&self::reads(reads)?)?)
}

#[pyfunction]
pub(crate) fn router_usage_totals(
    values: &Bound<'_, PyAny>,
    tpm_count: usize,
) -> PyResult<(Option<BigInt>, Option<BigInt>)> {
    let counted = values
        .try_iter()?
        .map(|value| counted(&value?))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(usage_totals(&counted, tpm_count))
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use rstest::rstest;

    use super::*;

    fn evaluated<'py>(py: Python<'py>, source: &str) -> Bound<'py, PyAny> {
        py.eval(&CString::new(source).unwrap(), None, None)
            .unwrap_or_else(|error| panic!("{source}: {error}"))
    }

    fn remaining(source: &str) -> PyResult<Vec<(String, String)>> {
        Python::initialize();
        Python::attach(|py| {
            let answer = router_remaining_usage(py, &evaluated(py, source))?;
            Ok(answer
                .iter()
                .map(|(name, value)| (name.to_string(), value.repr().unwrap().to_string()))
                .collect())
        })
    }

    fn pairs(expected: &[(&str, &str)]) -> Vec<(String, String)> {
        expected
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    fn python_error(source: &str) -> (String, String) {
        let error = remaining(source).unwrap_err();
        Python::attach(|py| {
            (
                error.get_type(py).name().unwrap().to_string(),
                error.value(py).str().unwrap().to_string(),
            )
        })
    }

    #[rstest]
    #[case::tpm_and_rpm(
        "((1100, 15, None, None), None, (30, 3), 0)",
        &[
            ("x-ratelimit-remaining-tokens", "1070"),
            ("x-ratelimit-limit-tokens", "1100"),
            ("x-ratelimit-remaining-requests", "12"),
            ("x-ratelimit-limit-requests", "15"),
        ],
    )]
    #[case::io(
        "((None, None, 300, 40), (7, None), None, 17)",
        &[
            ("x-ratelimit-limit-input-tokens", "300"),
            ("x-ratelimit-remaining-input-tokens", "293"),
            ("x-ratelimit-reset-input-tokens", "17"),
            ("x-ratelimit-limit-output-tokens", "40"),
            ("x-ratelimit-remaining-output-tokens", "40"),
            ("x-ratelimit-reset-output-tokens", "17"),
        ],
    )]
    #[case::float_limit(
        "((10.5, None, None, None), None, (None, None), 0)",
        &[
            ("x-ratelimit-remaining-tokens", "10.5"),
            ("x-ratelimit-limit-tokens", "10.5"),
        ],
    )]
    #[case::bool_limit(
        "((True, None, None, None), None, (None, None), 0)",
        &[
            ("x-ratelimit-remaining-tokens", "1"),
            ("x-ratelimit-limit-tokens", "1"),
        ],
    )]
    #[case::past_any_machine_int(
        "((10**40, None, None, None), None, (10**39, None), 0)",
        &[
            ("x-ratelimit-remaining-tokens", "9000000000000000000000000000000000000000"),
            ("x-ratelimit-limit-tokens", "10000000000000000000000000000000000000000"),
        ],
    )]
    #[case::no_group("(None, None, None, 0)", &[])]
    fn the_headers_are_the_python_dict_in_its_order(
        #[case] source: &str,
        #[case] expected: &[(&str, &str)],
    ) {
        assert_eq!(remaining(source).unwrap(), pairs(expected));
    }

    #[rstest]
    #[case::unsubtractable(
        "(('100', None, None, None), None, (None, None), 0)",
        "TypeError",
        "unsupported operand type(s) for -: 'str' and 'int'"
    )]
    #[case::int_too_large_for_a_float(
        "((10.5, None, None, None), None, (10**400, None), 0)",
        "OverflowError",
        "int too large to convert to float"
    )]
    fn what_python_cannot_compute_raises_its_error(
        #[case] source: &str,
        #[case] kind: &str,
        #[case] message: &str,
    ) {
        assert_eq!(python_error(source), (kind.to_owned(), message.to_owned()));
    }

    #[rstest]
    #[case::ints("[30, 4, 1, None]", 2, (Some(34), Some(1)))]
    #[case::floats_are_not_counted("[1.5, 2, None, 3.0]", 2, (Some(2), None))]
    #[case::bools_count_as_ints("[True, 2, False]", 2, (Some(3), Some(0)))]
    #[case::nothing("[]", 0, (None, None))]
    fn totals_count_only_the_ints_python_sums(
        #[case] source: &str,
        #[case] tpm_count: usize,
        #[case] expected: (Option<i64>, Option<i64>),
    ) {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(
                router_usage_totals(&evaluated(py, source), tpm_count).unwrap(),
                (expected.0.map(BigInt::from), expected.1.map(BigInt::from))
            );
        });
    }

    #[rstest]
    fn a_total_past_any_machine_int_stays_exact() {
        Python::initialize();
        Python::attach(|py| {
            let (tpm, _) = router_usage_totals(&evaluated(py, "[10**40, 1]"), 2).unwrap();
            assert_eq!(
                tpm.unwrap().to_string(),
                "10000000000000000000000000000000000000001"
            );
        });
    }
}
