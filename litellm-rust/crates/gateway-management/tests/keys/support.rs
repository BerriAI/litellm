use std::{collections::BTreeMap, sync::Mutex};

use litellm_gateway_auth::keys::KeyStatus;
use litellm_gateway_auth::{
    KeyError,
    keys::{KeyHash, KeyLookup, LookupFuture},
};
use litellm_gateway_management::{
    Error,
    keys::{KeyRecord, KeyStore, NewKey, Revocation, StoreFuture},
};

#[derive(Default)]
pub struct MemoryStore(Mutex<BTreeMap<KeyHash, KeyRecord>>);

impl KeyStore for MemoryStore {
    fn create(&self, key: NewKey) -> StoreFuture<'_, ()> {
        Box::pin(async move {
            let mut records = self.0.lock().unwrap();
            match records.entry(key.hash.clone()) {
                std::collections::btree_map::Entry::Occupied(_) => Err(Error::AlreadyExists),
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(KeyRecord {
                        hash: key.hash,
                        status: KeyStatus::Active {
                            expires_at: key.expires_at,
                        },
                    });
                    Ok(())
                }
            }
        })
    }

    fn get<'a>(&'a self, hash: &'a KeyHash) -> StoreFuture<'a, Option<KeyRecord>> {
        Box::pin(async move { Ok(self.0.lock().unwrap().get(hash).cloned()) })
    }

    fn revoke<'a>(&'a self, hash: &'a KeyHash) -> StoreFuture<'a, Revocation> {
        Box::pin(async move {
            let mut records = self.0.lock().unwrap();
            match records.get_mut(hash) {
                Some(record) => {
                    record.status = KeyStatus::Revoked;
                    Ok(Revocation::Revoked)
                }
                None => Ok(Revocation::NotFound),
            }
        })
    }
}

impl KeyLookup for MemoryStore {
    fn lookup<'a>(&'a self, hash: &'a KeyHash) -> LookupFuture<'a> {
        Box::pin(async move {
            self.get(hash)
                .await
                .map(|record| record.map(|record| record.status))
                .map_err(|error| KeyError::Lookup(Box::new(error)))
        })
    }
}

pub struct Unavailable;

fn unavailable<T>() -> Result<T, Error> {
    Err(Error::Storage(Box::new(std::io::Error::other(
        "backend detail",
    ))))
}

impl KeyStore for Unavailable {
    fn create(&self, _: NewKey) -> StoreFuture<'_, ()> {
        Box::pin(async { unavailable() })
    }

    fn get<'a>(&'a self, _: &'a KeyHash) -> StoreFuture<'a, Option<KeyRecord>> {
        Box::pin(async { unavailable() })
    }

    fn revoke<'a>(&'a self, _: &'a KeyHash) -> StoreFuture<'a, Revocation> {
        Box::pin(async { unavailable() })
    }
}
