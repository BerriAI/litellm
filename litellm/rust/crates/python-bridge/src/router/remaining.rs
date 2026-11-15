use std::{sync::Arc, time::SystemTime};

use futures_util::future::BoxFuture;
use litellm_callbacks::{
    RemainingUsage, RouterGroupLimits, RouterLimit, RouterLimitKind, RouterLimits, RouterReadError,
    RouterUsage, UsageAmount, UsageMember,
};
use litellm_callbacks_python::router::{PyRouterRef, headers as usage_headers};
use litellm_redis_store::{
    Number,
    router_usage::{Counter, Kind, RouterUsage as UsageStore, Scope},
};
use litellm_router_core::{
    health::current_minute,
    remaining::{
        Amount, GroupLimits, Limit, RemainingError, RemainingReads, Totals, remaining_usage,
        usage_totals,
    },
};
use num_bigint::BigInt;
use pyo3::{
    exceptions::{PyBaseException, PyOverflowError, PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt, PyType},
};

use crate::redis::{
    events::RedisCall,
    store::{kind_error, router_usage::NativeRouterUsage, targeted_call},
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

/// What `ROUTER_USAGE_TARGET` names: the Python store reads under `with_service_target` of it.
const ROUTER_USAGE_TARGET: (&str, &str) = (
    "litellm.redis.stores.router_usage.types",
    "ROUTER_USAGE_TARGET",
);

/// A limit as the record carries it, back to the value Python holds.
fn carried_limit(limit: &RouterLimit) -> PyResult<Limit> {
    let malformed = || PyValueError::new_err(format!("a router limit cannot read {limit:?}"));
    Ok(match limit.kind {
        RouterLimitKind::Int => Limit::Amount(Amount::Int(
            limit.text.parse::<BigInt>().map_err(|_| malformed())?,
        )),
        RouterLimitKind::Float => Limit::Amount(Amount::Float(
            limit.text.parse::<f64>().map_err(|_| malformed())?,
        )),
        RouterLimitKind::Other => Limit::Other(limit.text.clone()),
    })
}

fn carried_limits(limits: &RouterGroupLimits) -> PyResult<GroupLimits> {
    let each = |limit: &Option<RouterLimit>| limit.as_ref().map(carried_limit).transpose();
    Ok(GroupLimits {
        tpm: each(&limits.tpm)?,
        rpm: each(&limits.rpm)?,
        itpm: each(&limits.itpm)?,
        otpm: each(&limits.otpm)?,
    })
}

/// This minute's `global_router` counters of `members`, every member's first kind before any of
/// its second, refused as the Python store refused a member that is not a `str`.
fn counters(members: &[UsageMember], minute: &str, kinds: [Kind; 2]) -> PyResult<Vec<Counter>> {
    let text = |value: &Option<String>, name: &str| {
        value.clone().ok_or_else(|| {
            PyTypeError::new_err(format!("failed to extract field PyCounter.{name}"))
        })
    };
    let keyed = members
        .iter()
        .map(|member| {
            Ok((
                text(&member.deployment_id, "deployment_id")?,
                text(&member.model, "model")?,
            ))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(kinds
        .into_iter()
        .flat_map(|kind| {
            keyed.iter().map(move |(deployment_id, model)| Counter {
                scope: Scope::GlobalRouter,
                deployment_id: deployment_id.clone(),
                model: Some(model.clone()),
                window: minute.to_owned(),
                kind,
            })
        })
        .collect())
}

/// What `router_usage_totals` sums: the ints read, split at `tpm_count`.
fn read_totals(read: Vec<Option<Number>>, tpm_count: usize) -> Totals {
    let counted: Vec<Option<BigInt>> = read
        .into_iter()
        .map(|count| match count {
            Some(Number::Int(value)) => Some(BigInt::from(value)),
            _ => None,
        })
        .collect();
    usage_totals(&counted, tpm_count)
}

/// The Python caller's error, raised again from its type name and text.
fn resolution_error(py: Python<'_>, error_type: &str, error_text: &str) -> PyErr {
    let builtin = py
        .import("builtins")
        .and_then(|builtins| builtins.getattr(error_type))
        .ok()
        .filter(|class| {
            class
                .cast::<PyType>()
                .is_ok_and(|class| class.is_subclass_of::<PyBaseException>().unwrap_or(false))
        });
    match builtin {
        Some(class) => PyErr::from_type(
            class
                .cast_into::<PyType>()
                .unwrap_or_else(|_| unreachable!("checked above")),
            error_text.to_owned(),
        ),
        None => PyRuntimeError::new_err(format!("{error_type}: {error_text}")),
    }
}

/// The two store reads one remaining read makes, reported under the router usage target.
struct ReadCalls {
    io: RedisCall,
    usage: RedisCall,
}

#[derive(Clone, Copy)]
enum Caller {
    Awaiting,
    Blocking,
}

impl ReadCalls {
    fn of_caller(py: Python<'_>, store: &UsageStore, caller: Caller) -> PyResult<Self> {
        let target: String = py
            .import(ROUTER_USAGE_TARGET.0)?
            .getattr(ROUTER_USAGE_TARGET.1)?
            .extract()?;
        // The Python store awaited the remote read, or read memory through its sync method.
        let awaited = if store.has_redis() {
            "async_batch_get_cache"
        } else {
            "batch_get_cache"
        };
        let usage = match caller {
            Caller::Awaiting => awaited,
            Caller::Blocking => "batch_get_cache",
        };
        Ok(Self {
            io: targeted_call(py, awaited, &target)?,
            usage: targeted_call(py, usage, &target)?,
        })
    }
}

async fn group_read(
    store: &UsageStore,
    call: RedisCall,
    members: Option<&[UsageMember]>,
    kinds: [Kind; 2],
) -> PyResult<Totals> {
    let Some(members) = members else {
        return Ok((None, None));
    };
    let counters = counters(members, &current_minute(SystemTime::now()), kinds)?;
    let read = call
        .scope(store.get_many(&counters))
        .await
        .map_err(kind_error)?;
    Ok(read_totals(read, members.len()))
}

async fn read_remaining(
    store: Arc<UsageStore>,
    calls: ReadCalls,
    limits: RouterLimits,
    group_limits: GroupLimits,
) -> PyResult<RemainingUsage> {
    let io = if group_limits.itpm.is_some() || group_limits.otpm.is_some() {
        Some(
            group_read(
                &store,
                calls.io,
                limits.io_members.as_deref(),
                [Kind::Itpm, Kind::Otpm],
            )
            .await?,
        )
    } else {
        None
    };
    let usage = if group_limits.tpm.is_some() || group_limits.rpm.is_some() {
        Some(
            group_read(
                &store,
                calls.usage,
                limits.members.as_deref(),
                [Kind::Tpm, Kind::Rpm],
            )
            .await?,
        )
    } else {
        None
    };
    let reset = if io.is_some() {
        limits.reset.unwrap_or_default()
    } else {
        0
    };
    remaining(&RemainingReads {
        limits: Some(group_limits),
        io,
        usage,
        reset,
    })
}

/// The router's remaining usage, read from the Rust usage store the router writes.
struct HostedRouterUsage {
    store: Arc<UsageStore>,
}

impl HostedRouterUsage {
    /// What the read needs from the Python stack it is called on: the limits as values, and the
    /// caller's context for the store's service events. `None` reads nothing.
    fn prepared(
        &self,
        limits: &RouterLimits,
        caller: Caller,
    ) -> PyResult<Option<(ReadCalls, GroupLimits)>> {
        if let Some(error_type) = &limits.error_type {
            let error_text = limits.error_text.clone().unwrap_or_default();
            return Err(Python::attach(|py| {
                resolution_error(py, error_type, &error_text)
            }));
        }
        let Some(group) = &limits.limits else {
            return Ok(None);
        };
        let group_limits = carried_limits(group)?;
        let reads = group_limits.tpm.is_some()
            || group_limits.rpm.is_some()
            || group_limits.itpm.is_some()
            || group_limits.otpm.is_some();
        if !reads {
            // A group with no limits answers no headers, as `remaining_usage` would.
            return Ok(None);
        }
        let calls = Python::attach(|py| ReadCalls::of_caller(py, &self.store, caller))?;
        Ok(Some((calls, group_limits)))
    }
}

impl HostedRouterUsage {
    fn read(
        &self,
        limits: &RouterLimits,
        caller: Caller,
    ) -> BoxFuture<'static, Result<RemainingUsage, RouterReadError>> {
        let prepared = self.prepared(limits, caller);
        let store = Arc::clone(&self.store);
        let limits = limits.clone();
        Box::pin(async move {
            let answered = match prepared {
                Err(error) => Err(error),
                Ok(None) => Ok(RemainingUsage::default()),
                Ok(Some((calls, group_limits))) => {
                    read_remaining(store, calls, limits, group_limits).await
                }
            };
            answered.map_err(|error| Box::new(error) as RouterReadError)
        })
    }
}

impl RouterUsage for HostedRouterUsage {
    fn remaining(
        &self,
        _model_group: &str,
        limits: &RouterLimits,
    ) -> BoxFuture<'static, Result<RemainingUsage, RouterReadError>> {
        self.read(limits, Caller::Awaiting)
    }

    fn remaining_blocking(
        &self,
        _model_group: &str,
        limits: &RouterLimits,
    ) -> BoxFuture<'static, Result<RemainingUsage, RouterReadError>> {
        self.read(limits, Caller::Blocking)
    }
}

/// A reference that reads `store`, the `NativeRouterUsage` the router's usage store wraps.
#[pyfunction]
pub(crate) fn router_usage_ref(store: PyRef<'_, NativeRouterUsage>) -> PyRouterRef {
    PyRouterRef::new(Arc::new(HostedRouterUsage {
        store: store.store(),
    }))
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

    fn carried(kind: RouterLimitKind, text: &str) -> RouterLimit {
        RouterLimit {
            kind,
            text: text.to_owned(),
        }
    }

    #[rstest]
    #[case::int(RouterLimitKind::Int, "10000000000000000000000000000000000000000", Limit::Amount(Amount::Int(BigInt::from(10_u8).pow(40))))]
    #[case::negative_int(RouterLimitKind::Int, "-3", Limit::Amount(Amount::Int(BigInt::from(-3))))]
    #[case::float(RouterLimitKind::Float, "10.5", Limit::Amount(Amount::Float(10.5)))]
    #[case::infinity(
        RouterLimitKind::Float,
        "-inf",
        Limit::Amount(Amount::Float(f64::NEG_INFINITY))
    )]
    #[case::other(RouterLimitKind::Other, "str", Limit::Other("str".to_owned()))]
    fn a_carried_limit_reads_back_as_the_python_value(
        #[case] kind: RouterLimitKind,
        #[case] text: &str,
        #[case] expected: Limit,
    ) {
        assert_eq!(carried_limit(&carried(kind, text)).unwrap(), expected);
    }

    #[rstest]
    fn a_carried_nan_reads_back_as_nan() {
        let limit = carried_limit(&carried(RouterLimitKind::Float, "nan")).unwrap();
        assert!(matches!(limit, Limit::Amount(Amount::Float(value)) if value.is_nan()));
    }

    #[rstest]
    #[case::int(RouterLimitKind::Int, "1.5")]
    #[case::float(RouterLimitKind::Float, "ten")]
    fn a_malformed_carried_limit_is_refused(#[case] kind: RouterLimitKind, #[case] text: &str) {
        assert!(carried_limit(&carried(kind, text)).is_err());
    }

    fn member(deployment_id: Option<&str>, model: Option<&str>) -> UsageMember {
        UsageMember {
            deployment_id: deployment_id.map(str::to_owned),
            model: model.map(str::to_owned),
        }
    }

    #[rstest]
    fn counters_key_every_member_by_the_first_kind_then_the_second() {
        let members = [member(Some("a"), Some("m1")), member(Some("b"), Some("m2"))];
        let keyed: Vec<_> = counters(&members, "12-34", [Kind::Tpm, Kind::Rpm])
            .unwrap()
            .into_iter()
            .map(|counter| {
                (
                    counter.scope,
                    counter.deployment_id,
                    counter.model,
                    counter.window,
                    counter.kind,
                )
            })
            .collect();
        let expected = |id: &str, model: &str, kind| {
            (
                Scope::GlobalRouter,
                id.to_owned(),
                Some(model.to_owned()),
                "12-34".to_owned(),
                kind,
            )
        };
        assert_eq!(
            keyed,
            vec![
                expected("a", "m1", Kind::Tpm),
                expected("b", "m2", Kind::Tpm),
                expected("a", "m1", Kind::Rpm),
                expected("b", "m2", Kind::Rpm),
            ]
        );
    }

    #[rstest]
    #[case::id(
        member(None, Some("m")),
        "failed to extract field PyCounter.deployment_id"
    )]
    #[case::model(member(Some("a"), None), "failed to extract field PyCounter.model")]
    fn a_member_that_is_not_a_str_raises_the_type_error_the_store_raised(
        #[case] refused: UsageMember,
        #[case] message: &str,
    ) {
        Python::initialize();
        let error = counters(&[refused], "12-34", [Kind::Tpm, Kind::Rpm]).unwrap_err();
        Python::attach(|py| {
            assert!(error.is_instance_of::<PyTypeError>(py));
            assert_eq!(error.value(py).str().unwrap().to_string(), message);
        });
    }

    #[rstest]
    fn totals_count_only_the_ints_the_store_holds() {
        let read = vec![
            Some(Number::Int(30)),
            Some(Number::Float(1.5)),
            None,
            Some(Number::Int(2)),
        ];
        assert_eq!(
            read_totals(read, 2),
            (Some(BigInt::from(30)), Some(BigInt::from(2)))
        );
    }

    #[rstest]
    #[case::builtin("KeyError", "'gpt'", "KeyError")]
    #[case::not_an_exception("print", "text", "RuntimeError")]
    #[case::unknown("NoSuchError", "text", "RuntimeError")]
    fn a_carried_resolution_error_raises_its_builtin_type_or_a_runtime_error(
        #[case] error_type: &str,
        #[case] error_text: &str,
        #[case] raised: &str,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let error = resolution_error(py, error_type, error_text);
            assert_eq!(error.get_type(py).name().unwrap().to_string(), raised);
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
