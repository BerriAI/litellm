//! Whether the proxy accepts a request body, decided from data the caller hands in.
//!
//! A host reads the body into [`Params`] and says what the route reads it as with [`Surface`].
//! [`needs`] names the model groups and stored credentials the body can reach. The host resolves
//! the names to a [`Reachable`] through its router and reads the credentials with
//! [`credentials`], and [`check`] refuses with a [`Rejection`] the caller sees.

mod check;
mod destination;
mod form;
mod grant;
mod names;
mod policy;
mod surface;
mod value;

use std::collections::BTreeSet;

pub use check::{Credentials, Needs, Request, check, needs};
pub use grant::{Deployment, OptIn, Reachable};
pub use names::{HostTables, Names, Reading, VERTEX_TARGET_PARAMS};
pub use policy::{Deferred, Policy};
pub use surface::Surface;
pub use value::{CallerValue, Params};

pub use crate::error::body::{Allowlist, Rejection, ServerOwnedKind};
use crate::{AuthHost, Error};

/// The stored credentials [`Needs::credentials`] names, each read through the host's store. A
/// name the store does not hold is left out, which [`check`] reads as a credential holding
/// nothing.
pub async fn credentials<H: AuthHost>(
    host: &H,
    names: &BTreeSet<String>,
) -> Result<Credentials, Error> {
    let mut read = Vec::with_capacity(names.len());
    for name in names {
        if let Some(values) = host.credential(name).await? {
            read.push((name.clone(), values));
        }
    }
    Ok(read.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rstest::rstest;

    use super::*;
    use crate::testing::ScriptedHost;

    #[rstest]
    #[tokio::test]
    async fn each_named_credential_is_read_and_a_missing_one_left_out() {
        let stored =
            Params::from_iter([("api_key".to_owned(), CallerValue::Text("sk".to_owned()))]);
        let host = ScriptedHost {
            credentials: HashMap::from([("stored", stored.clone())]),
            ..ScriptedHost::default()
        };
        let names = BTreeSet::from(["missing".to_owned(), "stored".to_owned()]);

        let read = credentials(&host, &names).await.unwrap();

        assert_eq!(
            (read, host.calls.into_inner().unwrap()),
            (
                Credentials::from_iter([("stored".to_owned(), stored)]),
                vec![
                    "credential missing".to_owned(),
                    "credential stored".to_owned()
                ]
            )
        );
    }
}
