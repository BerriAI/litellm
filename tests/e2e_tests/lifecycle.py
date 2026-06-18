"""Lifecycle contract and resource cleanup for stateful e2e tests.

Shared by every e2e suite under tests/e2e_tests/. The proxy under test is
long-lived and never reset between tests, so anything a test creates (keys,
customers, teams, orgs, users, guardrails, budgets, ...) persists unless
explicitly deleted. Every check follows an init -> run -> teardown lifecycle;
teardown releases each resource init() created, even when run() raises.

In pytest terms (see conftest.py): the `resources` fixture's setup is init(),
the test body is run(), and the fixture's teardown is teardown().
"""

from dataclasses import dataclass, field
from typing import Callable, List, Protocol, runtime_checkable


@runtime_checkable
class E2ECase(Protocol):
    """A stateful e2e check run against a long-lived proxy.

    init() acquires resources, run() exercises behaviour and asserts, teardown()
    releases everything init() created. teardown() must run even if run() raises.
    """

    def init(self) -> None: ...

    def run(self) -> None: ...

    def teardown(self) -> None: ...


@runtime_checkable
class ResourceClient(Protocol):
    """Proxy operations the convenience creators use. Resource types without a
    creator here are handled generically via ResourceManager.defer()."""

    def generate_key(self, *, models: List[str]) -> str: ...

    def delete_key(self, key: str) -> None: ...

    def delete_customers(self, user_ids: List[str]) -> None: ...


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

    def key(self) -> str:
        """Create an all-models virtual key; delete it on teardown."""
        key = self.client.generate_key(models=[])
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
