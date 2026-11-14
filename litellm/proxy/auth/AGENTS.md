# Customer hierarchy performance

Customer organization, team and member parameters must preserve warm-request parity with baseline Redis round trips, awaited Postgres calls and gateway CPU per request. Preserve budget rejection ordering, missing-counter seeding, failure fallbacks, token counters and spend attribution

After changing hierarchy resolution, policy caching or customer spend settlement, run the mapped unit tests and `tests/e2e/perf/cases/small_bodies_customer_5rps.py` through the opt-in perf harness. Its CPU and awaited datastore ceilings must remain equal to `small_bodies_5rps`
