# Model cost bridge

- `NativeModelCostMap` holds inputs only, the installed `CostMap`, kept `Registrations` and the name index. Python's `litellm.model_cost` is the only copy of the live rows
- Every call that reads or writes rows takes the dict as an argument. Never store rows or build a second table here
- `LiveDict` holds a reference to that same dict and reads it under the GIL on every query, so a Python write is seen on the next read
  - It needs the GIL, so Rust code running without the GIL cannot use it
  - Its stamp is the dict's address, length and last key. [model-catalog/src/store/AGENTS.md](../../../model-catalog/src/store/AGENTS.md) says what that misses
- `host.rs` is the only code that calls Python, by name through `MODULE`. Calls happen with the GIL held and no `NativeModelCostMap` lock held
  - `resolve` reaches `find_key` on this same object, so holding a lock across a host call deadlocks
  - Test `a_host_lookup_that_reaches_find_key_on_the_same_map_finishes` guards this
- Rows and registrations cross as Python objects, never JSON, because router `model_info` holds `datetime`
- `table::build` serves the document as main's `_expand_model_aliases` did, so all names of a model share one row object made by `host.entry`
- `install` touches no state until `CostMap::read` succeeds. `NativeRejected` means nothing changed
- `register` keeps a registration only after its pass finished, so an exception in a host call keeps nothing
- `rows=None` writes each entry into whichever table is published at that moment and never reports a swap
- `fetch_blocking` runs on a throwaway runtime with an unpooled client so an importing process can still fork. `fetch` is for workers after the fork
