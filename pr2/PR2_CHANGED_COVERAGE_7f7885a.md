| File | Changed executable lines covered | Touched functions | Branches covered in touched functions | Uncovered changed lines |
|---|---|---|---|---|
| litellm/integrations/prompt_cache_alerts/__init__.py | 0/0 | 0 | 0/0 | none |
| litellm/integrations/prompt_cache_alerts/detector.py | 159/166 (95.8%) | 14 | 21/22 (95.5%) | 173,174,217,218,314,345,346 |
| litellm/integrations/prompt_cache_alerts/evidence.py | 109/109 (100.0%) | 10 | 34/34 (100.0%) | none |
| litellm/proxy/proxy_server.py | 12/13 (92.3%) | 2 | 32/86 (37.2%) | 1657 |
| **Total** | 280/288 (97.2%) | | 87/142 (61.3%) | |

Missing branches in touched functions (from_line -> to_line):
litellm/integrations/prompt_cache_alerts/detector.py [(313, 314)]
litellm/proxy/proxy_server.py [(1656, 1657), (1692, 1693), (1701, 1702), (1702, 1703), (1702, 1726), (1704, 1705), (1704, 1706), (1707, 1708), (1707, 1711), (1714, 1715), (1714, 1716), (1716, 1717), (1716, 1719), (1727, 1731), (1737, 1738), (1738, 1739), (1738, 1767), (1744, 1745), (1745, 1754), (1745, 1759), (1759, 1760), (1759, 1763), (1764, 1765), (1764, 1767), (1789, 1799), (1809, 1810), (1827, -1821), (1827, 1828), (1832, 1833), (1832, 1836), (1836, 1837), (1836, 1838), (1847, 1827), (1847, 1848), (1867, 1868), (1873, 1878), (1898, 1899), (1949, 1950), (1980, 1981), (1991, 1992), (1999, 2000), (2000, 2001), (2000, 2004), (2001, 2000), (2001, 2002), (2033, 2036), (2049, 2050), (2067, 2069), (2079, 2080), (2083, 2088), (2092, 2099), (2100, 2103), (2106, 2107), (2120, 2121)]

## How this was produced
Head 7f7885ab86ebe27e1343c870970c220d76776722, base a2e1e62893de4ddcf3a26155d8cda15abb955b88. Local run, not Codecov.
```
python -m pytest tests/unit/integrations/prompt_cache_alerts tests/unit/proxy/proxy_server/test_lifecycle.py -q -p no:cacheprovider -n 4 --cov=litellm --cov-branch --cov-report=
python -m coverage json -o coverage.json
python cov_diff.py a2e1e62893de4ddcf3a26155d8cda15abb955b88 coverage.json     # verification helper, not in the repository
```
Result of that run: 131 passed, exit 0. Only these two unit targets feed the numbers. The real-Redis integration suite and the live flow are not instrumented.

## Branch denominator
A touched function is any function containing a changed line, and all of its branches count, changed or not.
- proxy_server.py has two touched functions. `register_prompt_cache_alerts` is new: 4 branches, 3 covered. `_proxy_lifespan` is existing startup and shutdown code that gains one added line (1965): 74 branches, 29 covered. The file aggregate is 86 branches, comprising these 4 new branches, 74 existing lifecycle branches and 8 additional reported branches that were not attributed individually. Treat that aggregate as a reporting limitation, not precise new-code coverage.
- Without proxy_server.py the touched-function branches are 55/56: detector.py 21/22 and evidence.py 34/34.

## Uncovered changed lines
- detector.py 173-174 and 217-218: `except ValidationError` / `except (ValidationError, ValueError)` returning None.
- detector.py 314: early `return` (missing branch 313 -> 314).
- detector.py 345-346: reading the Redis client and registering the Lua script. The live proxy exercises this registration path but is not in the coverage run.
- proxy_server.py 1657: early `return` in `register_prompt_cache_alerts`.
