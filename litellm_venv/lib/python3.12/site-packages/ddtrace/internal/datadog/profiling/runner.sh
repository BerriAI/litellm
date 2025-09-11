#!/bin/bash
set -euox pipefail

./build_standalone.sh -C || { echo "Failed cppcheck"; exit 1; }
./build_standalone.sh -s || { echo "Failed safety tests"; exit 1; }
./build_standalone.sh -f || { echo "Failed -fanalyzer"; exit 1; }
./build_standalone.sh -t || { echo "Failed threading sanitizer"; exit 1; }
./build_standalone.sh -n || { echo "Failed numeric sanitizer"; exit 1; }
./build_standalone.sh -d || { echo "Failed dataflow sanitizer"; exit 1; }
#./build_standalone.sh -m || { echo "Failed memory leak sanitizer"; exit 1; } # Need to propagate msan configuration, currently failing in googletest internals
