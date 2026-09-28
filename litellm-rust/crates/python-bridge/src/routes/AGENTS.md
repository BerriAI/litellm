# Route boundary

These are target invariants; current entrypoints may still perform setup too early

Route modules own public argument projection, route-specific host operations, response and error construction, and composition of the core route with the Python host. Provider dispatch, transport execution and normalization belong to core and provider crates. Runtime waiting, cancellation mechanics and execution state validation belong to `litellm-host-python`

Before execution starts, perform only admission checks needed to select native execution or legacy fallback. Do not fully project a request just to decide admission. Keep effectful settings reads, HTTP client acquisition, secret-source construction and caller context capture inside execution, after argument preparation and SDK preflight. Configure resources from that prepared view, never the original entrypoint kwargs

The host driver owns sequencing and terminal events; the bridge supplies fallible resource composition without exposing route types to the driver. An unstarted async call performs no resource setup. Setup errors after start follow the terminal failure contract and never authorize fallback or provider replay

Pass the public call to `run_legacy_call`; `callbacks-legacy-python` owns legacy argument sharing and `Logging` dispatch. Route bindings may retain the request needed for projection, but must not duplicate the legacy callback contract

Regression tests must observe that an unstarted call does no setup, hook and preflight rewrites affect resource configuration, setup failures reach the selected failure handler once, and provider work is not replayed. Retain existing read-point and object-identity guarantees while changing setup timing
