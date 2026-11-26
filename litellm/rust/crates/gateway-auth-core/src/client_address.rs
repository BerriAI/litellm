//! Which addresses may reach the proxy, decided before any credential is read. No host is asked
//! anything, so every host gets the same answer from the same settings.

use litellm_config::{AddressList, GeneralSettings};

use crate::error::AddressNotAllowed;

/// The proxy's address allowlist and which of a request's addresses it reads.
#[derive(Clone, Debug, Default)]
pub struct ClientAddresses {
    /// [`general_settings.allowed_ips`](GeneralSettings::allowed_ips). Unset takes every address,
    /// and a value that is not a list of strings takes none.
    allowed: Option<Box<[String]>>,
    /// [`general_settings.use_x_forwarded_for`](GeneralSettings::use_x_forwarded_for)
    use_x_forwarded_for: bool,
}

impl ClientAddresses {
    pub fn new(general: &GeneralSettings) -> Self {
        Self {
            allowed: general
                .allowed_ips
                .as_ref()
                .map(|allowed| AddressList::items(allowed).into()),
            use_x_forwarded_for: general.use_x_forwarded_for,
        }
    }

    /// Whether a request's address decides anything, so a host reads it only then.
    pub fn reads_address(&self) -> bool {
        self.allowed.is_some()
    }

    /// `_check_valid_ip`: the address as written, with no range and no normalizing.
    pub fn permits(&self, origin: Origin<'_>) -> Result<(), AddressNotAllowed> {
        let Some(allowed) = &self.allowed else {
            return Ok(());
        };
        let address = origin.address(self.use_x_forwarded_for);
        if allowed.iter().any(|entry| entry == address) {
            return Ok(());
        }
        Err(AddressNotAllowed {
            address: address.to_owned(),
        })
    }
}

/// Where a request came from, as the host's server reports it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Origin<'a> {
    /// `request.client.host`
    pub peer: Option<&'a str>,
    /// The `x-forwarded-for` header as the request sent it.
    pub forwarded_for: Option<&'a str>,
}

impl<'a> Origin<'a> {
    /// `get_request_ip_address`: the forwarded header whole when the operator reads it and the
    /// request sent one, else the peer.
    fn address(&self, use_x_forwarded_for: bool) -> &'a str {
        match self.forwarded_for {
            Some(forwarded) if use_x_forwarded_for => forwarded,
            _ => self.peer.unwrap_or_default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn list(entries: &[&str]) -> AddressList {
        AddressList::Valid(entries.iter().map(|entry| (*entry).to_owned()).collect())
    }

    fn addresses(allowed: Option<AddressList>, use_x_forwarded_for: bool) -> ClientAddresses {
        ClientAddresses::new(&GeneralSettings {
            allowed_ips: allowed,
            use_x_forwarded_for,
            ..GeneralSettings::default()
        })
    }

    fn refused(address: &str) -> Result<(), AddressNotAllowed> {
        Err(AddressNotAllowed {
            address: address.to_owned(),
        })
    }

    #[rstest]
    #[case::unset_takes_every_address(None, false, Some("10.0.0.9"), None, Ok(()))]
    #[case::a_listed_peer(Some(list(&["10.0.0.1", "10.0.0.2"])), false, Some("10.0.0.2"), None, Ok(()))]
    #[case::an_unlisted_peer(Some(list(&["10.0.0.1"])), false, Some("10.0.0.9"), None, refused("10.0.0.9"))]
    #[case::an_empty_list_takes_none(Some(list(&[])), false, Some("10.0.0.1"), None, refused("10.0.0.1"))]
    #[case::a_malformed_list_takes_none(
        Some(AddressList::Malformed),
        false,
        Some("10.0.0.1"),
        None,
        refused("10.0.0.1")
    )]
    #[case::no_peer_is_the_empty_address(Some(list(&["10.0.0.1"])), false, None, None, refused(""))]
    #[case::an_unread_forwarded_header_changes_nothing(Some(list(&["10.0.0.1"])), false, Some("10.0.0.1"), Some("10.0.0.9"), Ok(()))]
    #[case::a_read_forwarded_header_replaces_the_peer(Some(list(&["10.0.0.1"])), true, Some("10.0.0.1"), Some("10.0.0.9"), refused("10.0.0.9"))]
    #[case::a_read_forwarded_header_that_is_listed(Some(list(&["10.0.0.9"])), true, Some("10.0.0.1"), Some("10.0.0.9"), Ok(()))]
    #[case::the_forwarded_header_is_compared_whole(Some(list(&["10.0.0.9"])), true, Some("10.0.0.1"), Some("10.0.0.9, 10.0.0.1"), refused("10.0.0.9, 10.0.0.1"))]
    #[case::no_forwarded_header_falls_back_to_the_peer(Some(list(&["10.0.0.1"])), true, Some("10.0.0.1"), None, Ok(()))]
    fn the_address_is_checked_against_allowed_ips(
        #[case] allowed: Option<AddressList>,
        #[case] use_x_forwarded_for: bool,
        #[case] peer: Option<&str>,
        #[case] forwarded_for: Option<&str>,
        #[case] expected: Result<(), AddressNotAllowed>,
    ) {
        let origin = Origin {
            peer,
            forwarded_for,
        };

        assert_eq!(
            addresses(allowed, use_x_forwarded_for).permits(origin),
            expected
        );
    }

    #[rstest]
    #[case::unset(None, false)]
    #[case::a_list(Some(list(&["10.0.0.1"])), true)]
    #[case::an_empty_list(Some(list(&[])), true)]
    #[case::a_malformed_list(Some(AddressList::Malformed), true)]
    fn the_address_is_read_only_under_an_allowlist(
        #[case] allowed: Option<AddressList>,
        #[case] reads: bool,
    ) {
        assert_eq!(addresses(allowed, true).reads_address(), reads);
    }
}
