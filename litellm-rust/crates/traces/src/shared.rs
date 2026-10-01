use std::ops::Deref;

use serde::Serialize;

type Storage<T> = std::sync::Arc<T>;

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Shared<T>(Storage<T>);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SharedIdentity(usize);

impl<T> Shared<T> {
    pub fn new(value: T) -> Self {
        Self(Storage::new(value))
    }

    pub fn identity(&self) -> SharedIdentity {
        SharedIdentity(std::ptr::from_ref(self.as_ref()) as usize)
    }

    pub fn shares_storage_with(&self, other: &Self) -> bool {
        self.identity() == other.identity()
    }
}

impl<T> From<T> for Shared<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T> AsRef<T> for Shared<T> {
    fn as_ref(&self) -> &T {
        self.0.as_ref()
    }
}

impl<T> Deref for Shared<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::Shared;
    use rstest::rstest;

    #[rstest]
    fn clones_preserve_values_and_serialize_transparently() {
        let original = Shared::new(vec!["value".to_owned()]);
        let cloned = original.clone();
        assert_eq!(cloned.as_ref(), original.as_ref());
        assert_eq!(
            serde_json::to_value(&cloned).unwrap(),
            serde_json::json!(["value"])
        );
    }

    #[rstest]
    fn clones_share_storage_without_merging_equal_values() {
        let original = Shared::new("value".to_owned());
        let cloned = original.clone();
        let equal = Shared::new("value".to_owned());
        assert!(original.shares_storage_with(&cloned));
        assert!(!original.shares_storage_with(&equal));
        assert_eq!(*original, *equal);
    }
}
