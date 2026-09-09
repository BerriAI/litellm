"""The load test suite: tests here run together, in one process, with nothing external.

Every test in this directory holds to the same contract, which is what lets `pytest
tests/load_tests/suite` be one run instead of a list of files to invoke one at a time:

- Hermetic. No provider credentials, no database, no Redis, no network past localhost. Upstreams
  are mock deployments and Redis is ``tests/_fault_injecting_redis.py``
- Relative measurements. A test takes its own RSS and latency baseline after it has set up, and
  asserts growth from there, so the tests before it in the process cannot make it fail
- Bounded. A test aborts on its own when a request exceeds its time or memory budget, so a
  regression fails the run instead of taking the runner down
- Fast enough for a weekly job. Tens of seconds each, a few minutes for the suite

The rest of ``tests/load_tests`` predates this contract: the memory baseline tests must run one
per process, the Vertex tests need live credentials, and ``test_memory_usage.py`` is skipped.
New load tests go here.

    make test-load
"""
