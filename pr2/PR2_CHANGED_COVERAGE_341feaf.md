# Changed-line and touched-function branch coverage, PR #51973

Head 341feafb5fc2d8a3d72adf5873df6982d4db77da, base 113766d4b90931d7f696576ef56c6721fb424f74. Local, from the 131-unit `-n 4 --cov=litellm --cov-branch` run in the glibc 2.39 container. Real-Redis cases are not in this coverage data.

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

The branch denominator counts every branch of each function the diff touches. `proxy_server.py` contributes 86, of which `_proxy_lifespan` (existing startup and shutdown code that gains one line) holds 74 inherited branches with 29 covered at 7f7885ab86e, and the new `register_prompt_cache_alerts` is 3/4. Excluding `proxy_server.py` the figure is 55/56. The per-function split was not recomputed at this tip, the file totals are identical to 7f7885ab86e.

Command: `python3 cov_diff.py 113766d4b90931d7f696576ef56c6721fb424f74 coverage.json` from the head worktree.
