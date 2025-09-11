# Compile extension with Cmake

```bash
sh clean.sh
cmake -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3.11 . && \
 make -j _native && \
 mv lib_native.so _native.so
```

## Verify compilation was correctly

```bash
python3.11
```
```python
from _native import Source, TaintRange
source = Source(name="aaa", value="bbbb", origin="ccc")
source = Source("aaa", "bbbb", "ccc")
```

## Clean Cmake folders

```bash
./clean.sh
```


## Debug with Valgrind

wget http://svn.python.org/projects/python/trunk/Misc/valgrind-python.supp

valgrind --tool=memcheck --suppressions=ddtrace/appsec/_iast/_taint_tracking/valgrind-python.supp \
python ddtrace/appsec/_iast/_taint_tracking/bench_overload.py --log-file="valgrind_bench_overload.out"

# Debug with gdb

gdb --args python -m pytest tests/appsec/iast/test_command_injection.py

## Generate coverage reports

From the `ddtrace/appsec/_iast/_taint_tracking` dir run:
```
./generate_coverage.sh
```

This will generate a text mode coverage report. If you add `--html` it will generate nice HTML
coverage pages at `ddtrace/appsec/_iast/_taint_tracking/coverage_html` dir. 
Switch to it and open the `index.html` with your browser.


If you use `--summary` it will generate only (at the end of the normal output) the % of line 
coverage, useful for CI and automated checks.
