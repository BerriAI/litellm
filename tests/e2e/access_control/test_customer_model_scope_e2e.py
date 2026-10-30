"""Live e2e: a customer/end-user model allow-list narrows the discovery endpoints.

The customer analogue of PR #42159: when a request carries an end-user id whose
customer row has a models allow-list, GET /v1/models returns only the models that
customer may call, the same gate inference enforces. An empty allow-list is
unrestricted, not deny-all. This walks the recorded state (the customer's stored
models) and the enforced behavior (the narrowed discovery listing) against a real
booted proxy.

The proxy serves no fixed `kimi`/`gpt-4o`, so each test registers the deployments
it names via /model/new (unique marker per run so a leftover row cannot satisfy a
stale assertion) and tears them down, matching the access_control suite's
create-and-clean pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pytest

from access_control_client import AccessControlClient
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody, ModelInfoBody, ModelNewBody
from e2e_metadata import Domain, Subject, meta

pytestmark = pytest.mark.e2e

BACKEND: Final = "openai/gpt-4o-mini"


@dataclass(frozen=True, slots=True)
class ScopedDeployments:
    """Two live deployments: one a restricted customer is granted, one it is not."""

    allowed_model: str
    other_model: str


def _deployment(model_name: str) -> ModelNewBody:
    return ModelNewBody(
        model_name=model_name,
        litellm_params=LiteLLMParamsBody(model=BACKEND),
        model_info=ModelInfoBody(),
    )


def _register(client: AccessControlClient, resources: ResourceManager, name: str) -> str:
    model_id: Final = client.proxy.register_model(_deployment(name))
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return name


@pytest.fixture
def deployments(client: AccessControlClient, resources: ResourceManager) -> ScopedDeployments:
    marker: Final = unique_marker()
    return ScopedDeployments(
        allowed_model=_register(client, resources, f"e2e-custmodel-allowed-{marker}"),
        other_model=_register(client, resources, f"e2e-custmodel-other-{marker}"),
    )


class TestCustomerModelScope:
    @meta(Subject(domain=Domain.PROXY_AUTH))
    def test_restricted_customer_sees_only_allowed_models(
        self, client: AccessControlClient, resources: ResourceManager, deployments: ScopedDeployments
    ) -> None:
        """A customer with models=[allowed_model] lists only that model, and its
        recorded allow-list matches what discovery enforces."""
        customer_id: Final = f"e2e-cust-scope-{unique_marker()}"
        client.create_customer(customer_id, models=[deployments.allowed_model])
        resources.defer(lambda: client.proxy.delete_customers([customer_id]))

        recorded: Final = client.customer_models(customer_id)
        assert recorded == [deployments.allowed_model], (
            f"customer must record its allow-list {[deployments.allowed_model]!r}, got {recorded!r}"
        )

        listed: Final = client.models_for_customer(customer_id)
        assert deployments.allowed_model in listed, (
            f"restricted customer must see its allowed model {deployments.allowed_model!r} on "
            f"/v1/models, got {listed!r}"
        )
        assert deployments.other_model not in listed, (
            f"restricted customer must not see {deployments.other_model!r} it cannot call, got {listed!r}"
        )

    @meta(Subject(domain=Domain.PROXY_AUTH))
    def test_restricted_customer_list_is_narrower_than_unrestricted(
        self, client: AccessControlClient, resources: ResourceManager, deployments: ScopedDeployments
    ) -> None:
        """The narrowing is observable: the restricted customer's list is strictly
        shorter than an unrestricted customer's list against the same proxy."""
        marker: Final = unique_marker()
        restricted: Final = f"e2e-cust-restricted-{marker}"
        unrestricted: Final = f"e2e-cust-unrestricted-{marker}"
        client.create_customer(restricted, models=[deployments.allowed_model])
        resources.defer(lambda: client.proxy.delete_customers([restricted]))
        client.create_customer(unrestricted, models=None)
        resources.defer(lambda: client.proxy.delete_customers([unrestricted]))

        restricted_models: Final = client.models_for_customer(restricted)
        unrestricted_models: Final = client.models_for_customer(unrestricted)
        assert len(restricted_models) < len(unrestricted_models), (
            f"restricted list {restricted_models!r} must be shorter than unrestricted {unrestricted_models!r}"
        )
        assert deployments.other_model in unrestricted_models, (
            f"unrestricted customer must see {deployments.other_model!r} the restricted one cannot, "
            f"got {unrestricted_models!r}"
        )

    @meta(Subject(domain=Domain.PROXY_AUTH))
    def test_empty_allowlist_customer_is_not_deny_all(
        self, client: AccessControlClient, resources: ResourceManager, deployments: ScopedDeployments
    ) -> None:
        """An empty allow-list means unrestricted: discovery returns the full set,
        a regression guard that an empty list is not read as deny-all."""
        customer_id: Final = f"e2e-cust-empty-{unique_marker()}"
        client.create_customer(customer_id, models=None)
        resources.defer(lambda: client.proxy.delete_customers([customer_id]))

        listed: Final = client.models_for_customer(customer_id)
        assert deployments.allowed_model in listed and deployments.other_model in listed, (
            f"empty-allowlist customer must see the full set including {deployments.allowed_model!r} "
            f"and {deployments.other_model!r}, got {listed!r}"
        )
