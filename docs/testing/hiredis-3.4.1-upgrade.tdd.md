# hiredis 3.4.1 upgrade TDD evidence

## User journey

As an operator, I want the proxy to install hiredis 3.4.1 or newer so released native parser security fixes are present

## Test specification

| Guarantee | Test | Result |
|---|---|---|
| The proxy dependency constraint excludes hiredis 3.4.0 | `test_declared_hiredis_floor_excludes_3_4_0` | PASS |
| The frozen lock resolves hiredis 3.4.1 or newer | `test_locked_hiredis_version_includes_3_4_1_fixes` | PASS |
| redis-py still selects the hiredis response parser | `test_redis_uses_the_hiredis_response_parser` | PASS |

## Evidence

RED: both dependency tests failed against the `>=3.0.0,<4.0` declaration and hiredis 3.4.0 lock

GREEN: frozen proxy dependency sync installed hiredis 3.4.1 and all three focused tests passed

The lock was generated with the repository-pinned uv 0.10.9 and retains the existing package cutoff

## Coverage and gaps

This dependency-only change is covered by declaration, lock, runtime import, and parser-selection checks. It does not prove that hiredis caused the previously observed process crash
