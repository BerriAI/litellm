## Readiness Lanes

- [ ] Core lane hermetic (no semantic asserts)
- [ ] ND lane preserved (no accidental dummy/echo in tests/ndsmoke)
- [ ] Live lane opt-in only (no unintended cloud calls)

## ND Guardrails

- [ ] I did not add MINI_AGENT_ALLOW_DUMMY or echo shortcuts to tests/ndsmoke
- [ ] If I touched scripts/mvp_check.py, I preserved the ND_REAL gate logic
- [ ] If I added ND-real tests, they assert robust invariants (non-empty content, model prefix), not exact wording

## Changes Summary

<briefly summarize>

