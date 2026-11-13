use std::sync::Arc;

use litellm_python_compat::Value;
use litellm_redis_store::auth_cache::{AuthCache, AuthObject, Settings};
use pyo3::prelude::*;

use super::{
    NativePending, PyNumber, kind_error, pending, redis_async, redis_sync, remote,
    request::{NativeRequestRedis, NativeSend},
    ttl,
    value::PyValue,
};
use crate::redis::client::NativeRedisClient;

fn values(values: Vec<Option<Value>>) -> Vec<Option<PyValue>> {
    values.into_iter().map(|value| value.map(PyValue)).collect()
}

/// One object the auth cache holds; `key` is the wire key `main` writes for it.
#[pyclass(frozen, eq, hash, from_py_object, name = "AuthObject")]
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyAuthObject(AuthObject);

#[pymethods]
impl PyAuthObject {
    #[staticmethod]
    fn virtual_key(hashed_token: String) -> Self {
        Self(AuthObject::Key(hashed_token))
    }

    #[staticmethod]
    fn user(user_id: String) -> Self {
        Self(AuthObject::User(user_id))
    }

    #[staticmethod]
    fn team(team_id: String) -> Self {
        Self(AuthObject::Team(team_id))
    }

    #[staticmethod]
    fn team_alias(team_alias: String) -> Self {
        Self(AuthObject::TeamAlias(team_alias))
    }

    #[staticmethod]
    fn team_membership(team_id: String, user_id: String) -> Self {
        Self(AuthObject::TeamMembership { team_id, user_id })
    }

    #[staticmethod]
    fn team_membership_reservation(user_id: String, team_id: String) -> Self {
        Self(AuthObject::TeamMembershipReservation { user_id, team_id })
    }

    #[staticmethod]
    fn org(org_id: String) -> Self {
        Self(AuthObject::Org(org_id))
    }

    #[staticmethod]
    fn org_with_budget(org_id: String) -> Self {
        Self(AuthObject::OrgWithBudget(org_id))
    }

    #[staticmethod]
    fn org_with_budget_last_known(org_id: String) -> Self {
        Self(AuthObject::OrgWithBudgetLastKnown(org_id))
    }

    #[staticmethod]
    fn org_alias(org_alias: String) -> Self {
        Self(AuthObject::OrgAlias(org_alias))
    }

    #[staticmethod]
    fn project(project_id: String) -> Self {
        Self(AuthObject::Project(project_id))
    }

    #[staticmethod]
    fn end_user(end_user_id: String) -> Self {
        Self(AuthObject::EndUser(end_user_id))
    }

    #[staticmethod]
    fn end_user_restricted_registry() -> Self {
        Self(AuthObject::EndUserRestrictedRegistry)
    }

    #[staticmethod]
    fn end_user_validation(end_user_id: String) -> Self {
        Self(AuthObject::EndUserValidation(end_user_id))
    }

    #[staticmethod]
    fn customer_member(customer_member_id: String) -> Self {
        Self(AuthObject::CustomerMember(customer_member_id))
    }

    #[staticmethod]
    fn customer_org_alias(customer_org_alias: String) -> Self {
        Self(AuthObject::CustomerOrgAlias(customer_org_alias))
    }

    #[staticmethod]
    fn default_end_user_budget(budget_id: String) -> Self {
        Self(AuthObject::DefaultEndUserBudget(budget_id))
    }

    #[staticmethod]
    fn team_member_default_budget(budget_id: String) -> Self {
        Self(AuthObject::TeamMemberDefaultBudget(budget_id))
    }

    #[staticmethod]
    fn tag(tag_name: String) -> Self {
        Self(AuthObject::Tag(tag_name))
    }

    #[staticmethod]
    fn tag_registry() -> Self {
        Self(AuthObject::TagRegistry)
    }

    #[staticmethod]
    fn model_access_group(access_group_name: String) -> Self {
        Self(AuthObject::ModelAccessGroup(access_group_name))
    }

    #[staticmethod]
    fn model_access_group_registry() -> Self {
        Self(AuthObject::ModelAccessGroupRegistry)
    }

    #[staticmethod]
    fn access_group(access_group_id: String) -> Self {
        Self(AuthObject::AccessGroup(access_group_id))
    }

    #[staticmethod]
    fn access_group_rule_name(name: String) -> Self {
        Self(AuthObject::AccessGroupRuleName(name))
    }

    #[staticmethod]
    fn object_permission(object_permission_id: String) -> Self {
        Self(AuthObject::ObjectPermission(object_permission_id))
    }

    #[staticmethod]
    fn user_object_permission(user_id: String) -> Self {
        Self(AuthObject::UserObjectPermission(user_id))
    }

    #[staticmethod]
    fn agent_object_permission(agent_id: String) -> Self {
        Self(AuthObject::AgentObjectPermission(agent_id))
    }

    #[staticmethod]
    fn managed_vector_store(vector_store_id: String) -> Self {
        Self(AuthObject::ManagedVectorStore(vector_store_id))
    }

    /// `jwt_key_mapping_cache_key`: an empty issuer is the issuer-agnostic scope.
    #[staticmethod]
    #[pyo3(signature = (claim_name, claim_value, issuer = None))]
    fn jwt_key_mapping(claim_name: String, claim_value: String, issuer: Option<String>) -> Self {
        Self(AuthObject::JwtKeyMapping {
            issuer: issuer.filter(|issuer| !issuer.is_empty()),
            claim_name,
            claim_value,
        })
    }

    #[staticmethod]
    fn agent_identity(tenant_id: String, client_id: String) -> Self {
        Self(AuthObject::AgentIdentity {
            tenant_id,
            client_id,
        })
    }

    #[staticmethod]
    fn global_proxy_spend() -> Self {
        Self(AuthObject::GlobalProxySpend)
    }

    #[getter]
    fn key(&self) -> String {
        self.0.key()
    }

    fn __repr__(&self) -> String {
        format!("AuthObject({:?})", self.0)
    }
}

/// `UserApiKeyCache`, always native: memory only until `attach` adds Redis. The sync methods back `DualCache`'s sync
/// API and block with the GIL released, like redis-py's sync client.
#[pyclass(frozen, name = "NativeAuthCache")]
pub(crate) struct NativeAuthCache(pub(super) Arc<AuthCache>);

#[pymethods]
impl NativeAuthCache {
    #[new]
    #[pyo3(signature = (client, *, memory_entries, memory_ttl_seconds = None))]
    fn new(
        client: Option<PyRef<'_, NativeRedisClient>>,
        memory_entries: usize,
        memory_ttl_seconds: Option<f64>,
    ) -> PyResult<Self> {
        Ok(Self(Arc::new(AuthCache::new(
            client.map(|client| remote(&client)),
            Settings {
                memory_entries,
                memory_ttl: ttl(memory_ttl_seconds)?,
            },
        ))))
    }

    /// Adds the Redis tier and keeps memory; `false` when one was already attached.
    fn attach(&self, client: PyRef<'_, NativeRedisClient>) -> bool {
        self.0.attach(remote(&client))
    }

    #[getter]
    fn has_redis(&self) -> bool {
        self.0.has_redis()
    }

    /// `request` answers a key its pre-call reads already found absent without another `GET`.
    #[pyo3(signature = (key, request))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        key: PyAuthObject,
        request: Option<&NativeRequestRedis>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let request = request.map(|request| request.0.clone());
        redis_async(py, "async_get_cache", kind_error, async move {
            store
                .get(request.as_ref(), &key.0)
                .await
                .map(|value| value.map(PyValue))
        })
    }

    fn get_sync(&self, py: Python<'_>, key: PyAuthObject) -> PyResult<Option<PyValue>> {
        let store = self.0.clone();
        redis_sync(py, "get_cache", kind_error, async move {
            store
                .get(None, &key.0)
                .await
                .map(|value| value.map(PyValue))
        })
    }

    /// `update_cache_ttl` and `update_in_memory_max_size`: both partitions keep what they hold.
    #[pyo3(signature = (*, memory_entries, memory_ttl_seconds = None))]
    fn reconfigure(&self, memory_entries: usize, memory_ttl_seconds: Option<f64>) -> PyResult<()> {
        self.0.reconfigure(Settings {
            memory_entries,
            memory_ttl: ttl(memory_ttl_seconds)?,
        });
        Ok(())
    }

    /// `request` reads the key again from now on, though one of its reads found it absent.
    #[pyo3(signature = (key, value, ttl_seconds, request))]
    fn set<'py>(
        &self,
        py: Python<'py>,
        key: PyAuthObject,
        value: PyValue,
        ttl_seconds: Option<f64>,
        request: Option<&NativeRequestRedis>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let ttl = ttl(ttl_seconds)?;
        let request = request.map(|request| request.0.clone());
        redis_async(py, "async_set_cache", kind_error, async move {
            store.set(request.as_ref(), &key.0, value.0, ttl).await
        })
    }

    #[pyo3(signature = (key, value, ttl_seconds, request))]
    fn set_sync(
        &self,
        py: Python<'_>,
        key: PyAuthObject,
        value: PyValue,
        ttl_seconds: Option<f64>,
        request: Option<&NativeRequestRedis>,
    ) -> PyResult<()> {
        let store = self.0.clone();
        let ttl = ttl(ttl_seconds)?;
        let request = request.map(|request| request.0.clone());
        redis_sync(py, "set_cache", kind_error, async move {
            store.set_str(request.as_ref(), &key.0, value.0, ttl).await
        })
    }

    /// Memory now and the `SET` on the batch `send` picks.
    #[pyo3(signature = (key, value, ttl_seconds, send))]
    fn declare_set(
        &self,
        key: PyAuthObject,
        value: PyValue,
        ttl_seconds: Option<f64>,
        send: &NativeSend,
    ) -> PyResult<NativePending> {
        let written = self
            .0
            .declare_set(send.send(), &key.0, value.0, ttl(ttl_seconds)?)
            .map_err(kind_error)?;
        Ok(pending(
            "async_set_cache",
            kind_error,
            written,
            std::convert::identity,
        ))
    }

    fn declare_delete(&self, key: String, send: &NativeSend) -> PyResult<NativePending> {
        let deleted = self
            .0
            .declare_delete(send.send(), &key)
            .map_err(kind_error)?;
        Ok(pending(
            "async_delete_cache",
            kind_error,
            deleted,
            std::convert::identity,
        ))
    }

    #[pyo3(signature = (entries, ttl_seconds, request))]
    fn set_many<'py>(
        &self,
        py: Python<'py>,
        entries: Vec<(PyAuthObject, PyValue)>,
        ttl_seconds: Option<f64>,
        request: Option<&NativeRequestRedis>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        let entries = entries
            .into_iter()
            .map(|(key, value)| (key.0, value.0))
            .collect();
        let ttl = ttl(ttl_seconds)?;
        let request = request.map(|request| request.0.clone());
        redis_async(py, "async_set_cache_pipeline", kind_error, async move {
            store.set_many(request.as_ref(), entries, ttl).await
        })
    }

    /// On a request batch it returns before anything is sent, and `Now` is a pipeline of its own.
    fn write_back<'py>(
        &self,
        py: Python<'py>,
        entries: Vec<(PyAuthObject, PyValue, Option<f64>)>,
        send: &NativeSend,
    ) -> PyResult<Bound<'py, PyAny>> {
        let entries = entries
            .into_iter()
            .map(|(key, value, seconds)| Ok((key.0, value.0, ttl(seconds)?)))
            .collect::<PyResult<_>>()?;
        let written = self
            .0
            .write_back(send.send(), entries)
            .map_err(kind_error)?;
        redis_async(
            py,
            "async_set_cache_pipeline_with_ttls",
            kind_error,
            written,
        )
    }

    fn delete<'py>(&self, py: Python<'py>, keys: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_delete_cache", kind_error, async move {
            store.delete(&keys).await
        })
    }

    fn delete_sync(&self, py: Python<'_>, keys: Vec<String>) -> PyResult<()> {
        let store = self.0.clone();
        redis_sync(py, "delete_cache", kind_error, async move {
            store.delete(&keys).await
        })
    }

    /// One `MGET` of keys the caller found missing in memory, on the batch `send` picks; `None`
    /// without Redis.
    fn get_many(&self, keys: Vec<PyAuthObject>, send: &NativeSend) -> Option<NativePending> {
        let keys: Vec<AuthObject> = keys.into_iter().map(|key| key.0).collect();
        let read = self.0.get_many(send.send(), &keys)?;
        Some(pending("async_batch_get_cache", kind_error, read, values))
    }

    fn increment_if_present<'py>(
        &self,
        py: Python<'py>,
        key: PyAuthObject,
        amount: PyNumber,
    ) -> PyResult<Bound<'py, PyAny>> {
        let store = self.0.clone();
        redis_async(py, "async_increment_if_present", kind_error, async move {
            store.increment_if_present(&key.0, amount.0).await
        })
    }

    fn get_local(&self, key: PyRef<'_, PyAuthObject>) -> PyResult<Option<PyValue>> {
        Ok(self.0.get_local(&key.0).map_err(kind_error)?.map(PyValue))
    }

    #[pyo3(signature = (key, value, ttl_seconds = None))]
    fn set_local(
        &self,
        key: PyRef<'_, PyAuthObject>,
        value: PyValue,
        ttl_seconds: Option<f64>,
    ) -> PyResult<()> {
        self.0
            .set_local(&key.0, value.0, ttl(ttl_seconds)?)
            .map_err(kind_error)
    }

    fn delete_local(&self, key: &str) -> PyResult<()> {
        self.0.delete_local(key).map_err(kind_error)
    }

    fn flush_local(&self) -> PyResult<()> {
        self.0.flush_local().map_err(kind_error)
    }

    fn len_local(&self) -> PyResult<PyLocalSizes> {
        let (objects, key_objects) = self.0.len_local().map_err(kind_error)?;
        Ok(PyLocalSizes {
            objects,
            key_objects,
        })
    }
}

/// Entries the management-object and the user-key partitions hold in memory.
#[pyclass(frozen, eq, get_all, name = "LocalSizes")]
#[derive(PartialEq)]
pub(crate) struct PyLocalSizes {
    objects: usize,
    key_objects: usize,
}

#[pymethods]
impl PyLocalSizes {
    #[new]
    fn new(objects: usize, key_objects: usize) -> Self {
        Self {
            objects,
            key_objects,
        }
    }
}
