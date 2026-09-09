"""The load test suite. `make test-load-suite` runs this directory as one job.

New tests here hold to one contract: hermetic (mock deployments, the fault-injecting Redis in
``tests/_fault_injecting_redis.py``, nothing past localhost), measured against a baseline the test
takes itself, and self-aborting when a request exceeds its time or memory budget so a regression
fails the run instead of taking the runner down. The older files predate the contract: the
memory baseline tests take a fresh baseline each but are most sensitive run one per process,
``test_memory_usage.py`` is skipped, the Vertex tests need credentials, the Granian benchmark is
env-gated, and the logging load tests post to an external example endpoint.
"""

from tests.load_tests.memory_leak_utils import (  # noqa: F401  # re-exported so pytest resolves these fixtures by name
    limit_memory,
    mock_server,
    test_router,
)
