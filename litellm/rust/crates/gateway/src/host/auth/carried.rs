//! Whether a field of a cached row holds something. The native host takes each row apart by name,
//! and a field it neither honors nor leaves to a host step declines the row when it is carried.

use std::collections::BTreeMap;

use litellm_config::LitellmRole;
use litellm_gateway_auth_core::record::ModelRegion as AuthModelRegion;
use litellm_python_compat::{Value, datetime::IsoDateTime};
use litellm_redis_store::auth_cache::{
    Bag, BudgetEntry, Lists, ModelRegion, ModelTableEntry, ObjectPermissionEntry,
};

/// The name of the first of `fields` that holds something.
macro_rules! carried {
    ($($field:ident),+ $(,)?) => {
        [$((stringify!($field), Carried::is_carried($field)),)+]
            .into_iter()
            .find(|(_, carried)| *carried)
            .map(|(field, _)| field)
    };
}

pub(super) use carried;

pub(super) trait Carried {
    fn is_carried(&self) -> bool;
}

impl Carried for bool {
    fn is_carried(&self) -> bool {
        *self
    }
}

/// A value a row holds, whatever it is: a limit of zero and an empty owner are still set.
macro_rules! carried_when_held {
    ($($held:ty),+) => {
        $(impl Carried for $held {
            fn is_carried(&self) -> bool {
                true
            }
        })+
    };
}

carried_when_held!(
    String,
    i64,
    f64,
    IsoDateTime,
    LitellmRole,
    ModelRegion,
    AuthModelRegion,
    ObjectPermissionEntry,
    ModelTableEntry,
    BudgetEntry
);

impl<T> Carried for Vec<T> {
    fn is_carried(&self) -> bool {
        !self.is_empty()
    }
}

impl<K, V> Carried for BTreeMap<K, V> {
    fn is_carried(&self) -> bool {
        !self.is_empty()
    }
}

impl Carried for Bag {
    fn is_carried(&self) -> bool {
        !self.is_empty()
    }
}

impl Carried for Lists {
    fn is_carried(&self) -> bool {
        !self.0.is_empty()
    }
}

impl Carried for Value {
    fn is_carried(&self) -> bool {
        !matches!(self, Self::None)
    }
}

impl<T: Carried> Carried for Option<T> {
    fn is_carried(&self) -> bool {
        self.as_ref().is_some_and(Carried::is_carried)
    }
}

/// Whether a field this build does not name, which a newer pod wrote, holds something.
pub(super) fn newer(unread: &[(String, Value)]) -> bool {
    unread.iter().any(|(_, value)| value.is_carried())
}
