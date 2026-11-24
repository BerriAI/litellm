# fnmatch

The glob dialect of Python's `fnmatch.fnmatchcase`, for config patterns Python LiteLLM already matches with it, such as JWT role maps, JWT routing selectors and cost discount model patterns. A crate that reads one of those patterns depends on this crate and never on another glob crate, whose dialect would change what an existing config matches

- No dependencies and no regex. A pattern parses into typed segments, and matching stays linear in the name for any number of `*`
- Every case outside `bang_after_a_reversed_range_is_a_member` takes its expected value from running `fnmatch.fnmatchcase(name, pattern)` on CPython, never from reasoning. Run it before adding or changing a case. `matches_like_cpython` holds the dialect's syntax, and the tests after it hold the patterns each config surface really sees
- There is one divergence, pinned by its own test. CPython looks for the negating `!` after it has dropped reversed ranges, so `[z-a!b]` means "not b" there. Here `!` negates only as the first character of a class
- `fnmatch.fnmatch` also runs `os.path.normcase` on both sides, which does nothing on POSIX. A caller that needs it folds case itself
