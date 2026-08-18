"""Resource cleanup for stateful e2e tests.

Shared by every e2e suite under tests/e2e/. The proxy under test is
long-lived and never reset between tests, so anything a test creates (keys,
customers, teams, orgs, users, guardrails, budgets, ...) persists unless
explicitly deleted. The `resources` fixture (see conftest.py) hands each test a
ResourceManager; the test registers a cleanup for every resource it creates, and
the fixture's teardown releases them all even when the test body raises.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Protocol, runtime_checkable

from proxy_client import ProxyClient
from models import KeyGenerateBody


@runtime_checkable
class ResourceClient(Protocol):
    """Proxy operations the convenience creators use. Resource types without a
    creator here are handled generically via ResourceManager.defer(). The ProxyClient
    satisfies this."""

    def generate_key(self, body: KeyGenerateBody) -> str: ...

    def delete_key(self, key: str) -> None: ...

    def delete_customers(self, user_ids: List[str]) -> None: ...


@runtime_checkable
class ProxyClientProvider(Protocol):
    """Every suite's client exposes the shared ProxyClient, which the resources fixture
    uses for cleanup. The client adds its own route methods on top."""

    @property
    def proxy(self) -> ProxyClient: ...


@dataclass
class ResourceManager:
    """Registry of teardown actions for resources a test creates on the stateful
    proxy.

    Not limited to any resource type: register a cleanup with ``defer()`` for a
    key, customer, team, org, user, guardrail, budget, MCP server - anything with
    a delete. The two most common resources have sugar (``key``, ``customer``);
    everything else is ``resources.defer(lambda: client.delete_team(team_id))``.

    Cleanups run LIFO (so a resource is removed before whatever it depends on) and
    best-effort (one failing cleanup never blocks the rest).
    """

    client: ResourceClient
    _cleanups: List[Callable[[], None]] = field(
        default_factory=list
    )  # mutable-ok: append-only teardown registry

    def init(self) -> None:
        """No global setup needed today; present for lifecycle symmetry."""
        return None

    def defer(self, cleanup: Callable[[], None]) -> None:
        """Register a teardown action for any resource the test just created."""
        self._cleanups.append(cleanup)

    def key(self, models: list[str] | None = None, user_id: str | None = "e2e-test-user") -> str:
        """Create a virtual key; delete it on teardown. `models` restricts which
        models the key may call (None/[] means all). `user_id` is required for
        managed-batch ACL: the proxy stores created_by=user_id and checks it on
        retrieve/cancel; None here means the 403 guard fires."""
        key = self.client.generate_key(KeyGenerateBody(models=models or [], user_id=user_id))
        self.defer(lambda: self.client.delete_key(key))
        return key

    def customer(self, customer_id: str) -> str:
        """Track an end-user id (from the `user` param); delete it on teardown."""
        self.defer(lambda: self.client.delete_customers([customer_id]))
        return customer_id

    def teardown(self) -> None:
        for cleanup in reversed(self._cleanups):
            try:
                cleanup()
            except Exception:
                pass  # best-effort: a failed cleanup must not block the rest
