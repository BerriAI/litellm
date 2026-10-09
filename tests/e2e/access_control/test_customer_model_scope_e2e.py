"""Live e2e: a customer/end-user model allow-list narrows the discovery endpoints.

The customer analogue of PR #42159: when a request carries an end-user id whose
customer row has a models allow-list, GET /v1/models returns only the models that
customer may call, the same gate inference enforces. An empty allow-list is
unrestricted, not deny-all. This walks the recorded state (the customer's stored
models) and the enforced behavior (the narrowed discovery listing) against a real
booted proxy.
"""

from __future__ import annotations

import pytest

from access_control_client import AccessControlClient
from e2e_config import unique_marker
from lifecycle import ResourceManager
from e2e_metadata import Domain, Subject, meta

pytestmark = pytest.mark.e2e

ALLOWED_MODEL = "kimi"
OTHER_MODEL = "gpt-4o"


class TestCustomerModelScope:
    @meta(Subject(domain=Domain.PROXY_AUTH))
    def test_restricted_customer_sees_only_allowed_models(
        self, client: AccessControlClient, resources: ResourceManager
    ) -> None:
        """A customer with models=[ALLOWED_MODEL] lists only that model, and its
        recorded allow-list matches what discovery enforces."""
        customer_id = f"e2e-cust-scope-{unique_marker()}"
        client.create_customer(customer_id, models=[ALLOWED_MODEL])
        resources.defer(lambda: client.proxy.delete_customers([customer_id]))

        recorded = client.customer_models(customer_id)
        assert recorded == [ALLOWED_MODEL], f"customer must record its allow-list {[ALLOWED_MODEL]!r}, got {recorded!r}"

        listed = client.models_for_customer(customer_id)
        assert ALLOWED_MODEL in listed, (
            f"restricted customer must see its allowed model {ALLOWED_MODEL!r} on /v1/models, got {listed!r}"
        )
        assert OTHER_MODEL not in listed, (
            f"restricted customer must not see {OTHER_MODEL!r} it cannot call, got {listed!r}"
        )

    @meta(Subject(domain=Domain.PROXY_AUTH))
    def test_restricted_customer_list_is_narrower_than_unrestricted(
        self, client: AccessControlClient, resources: ResourceManager
    ) -> None:
        """The narrowing is observable: the restricted customer's list is strictly
        shorter than an unrestricted customer's list against the same proxy."""
        marker = unique_marker()
        restricted = f"e2e-cust-restricted-{marker}"
        unrestricted = f"e2e-cust-unrestricted-{marker}"
        client.create_customer(restricted, models=[ALLOWED_MODEL])
        resources.defer(lambda: client.proxy.delete_customers([restricted]))
        client.create_customer(unrestricted, models=None)
        resources.defer(lambda: client.proxy.delete_customers([unrestricted]))

        restricted_models = client.models_for_customer(restricted)
        unrestricted_models = client.models_for_customer(unrestricted)
        assert len(restricted_models) < len(unrestricted_models), (
            f"restricted list {restricted_models!r} must be shorter than unrestricted {unrestricted_models!r}"
        )

    @meta(Subject(domain=Domain.PROXY_AUTH))
    def test_empty_allowlist_customer_is_not_deny_all(
        self, client: AccessControlClient, resources: ResourceManager
    ) -> None:
        """An empty allow-list means unrestricted: discovery returns the full set,
        a regression guard that an empty list is not read as deny-all."""
        customer_id = f"e2e-cust-empty-{unique_marker()}"
        client.create_customer(customer_id, models=None)
        resources.defer(lambda: client.proxy.delete_customers([customer_id]))

        listed = client.models_for_customer(customer_id)
        assert ALLOWED_MODEL in listed and OTHER_MODEL in listed, (
            f"empty-allowlist customer must see the full set including {ALLOWED_MODEL!r} and {OTHER_MODEL!r}, "
            f"got {listed!r}"
        )
