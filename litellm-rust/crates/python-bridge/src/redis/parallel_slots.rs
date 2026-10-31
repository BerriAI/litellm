use std::time::Duration;

use litellm_redis_client::RedisClient;
use litellm_redis_store::parallel_slots::{self, Acquire, Descriptor, Gauge};
use pyo3::{IntoPyObjectExt, prelude::*};

use super::{NativeRedisClient, store_call};

#[pyclass(frozen, name = "NativeParallelSlots")]
pub(crate) struct NativeParallelSlots(RedisClient);

fn descriptor((key, value): (String, String)) -> Descriptor {
    Descriptor { key, value }
}

#[pymethods]
impl NativeParallelSlots {
    #[new]
    fn new(client: &NativeRedisClient) -> Self {
        Self(client.0.clone())
    }

    #[pyo3(signature = (gauges, slot_id, *, slot_ttl_seconds))]
    fn acquire<'py>(
        &self,
        py: Python<'py>,
        gauges: Vec<(String, String, u64)>,
        slot_id: String,
        slot_ttl_seconds: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.0.clone();
        let gauges: Vec<Gauge> = gauges
            .into_iter()
            .map(|(key, value, limit)| Gauge {
                descriptor: descriptor((key, value)),
                limit,
            })
            .collect();
        let slot_ttl = Duration::from_secs(slot_ttl_seconds);
        store_call(
            py,
            async move { parallel_slots::acquire(&client, &gauges, &slot_id, slot_ttl).await },
            |py, acquired| match acquired {
                Acquire::Admitted { in_flight } => ("admitted", in_flight).into_py_any(py),
                Acquire::Rejected {
                    index,
                    in_flight,
                    limit,
                } => ("rejected", index, in_flight, limit).into_py_any(py),
            },
        )
    }

    fn release<'py>(
        &self,
        py: Python<'py>,
        descriptors: Vec<(String, String)>,
        slot_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.0.clone();
        let descriptors: Vec<Descriptor> = descriptors.into_iter().map(descriptor).collect();
        store_call(
            py,
            async move { parallel_slots::release(&client, &descriptors, &slot_id).await },
            |py, in_flight| in_flight.into_py_any(py),
        )
    }

    #[pyo3(signature = (descriptors, *, slot_ttl_seconds))]
    fn count<'py>(
        &self,
        py: Python<'py>,
        descriptors: Vec<(String, String)>,
        slot_ttl_seconds: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.0.clone();
        let descriptors: Vec<Descriptor> = descriptors.into_iter().map(descriptor).collect();
        let slot_ttl = Duration::from_secs(slot_ttl_seconds);
        store_call(
            py,
            async move { parallel_slots::count_in_flight(&client, &descriptors, slot_ttl).await },
            |py, in_flight| in_flight.into_py_any(py),
        )
    }
}
