#!/bin/bash
set -exu
#cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit

rm -rf CMakeFiles/ CMakeCache.txt Makefile cmake_install.cmake __pycache__/ .cmake *.cbp Testing
rm -rf tests/CMakeFiles/ tests/CMakeCache.txt tests/Makefile tests/cmake_install.cmake tests/__pycache__/ tests/*.cmake *.cbp 
rm -rf cmake-build-debug cmake-build-default cmake-build-tests
rm -rf tests/cmake-build-debug tests/cmake-build-default tests/cmake-build-tests
rm -rf tests/CMakeFiles/native_tests.dir
rm -rf coverage
rm -rf coverage_html
yes|rm -f *.so
yes|rm -f tests/*.so
yes|rm -f tests/native_tests
yes|rm -f gcov_reports/*.gcov
