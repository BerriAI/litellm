#!/bin/bash


# Parse command-line arguments
HTML_REPORT=false
SUMMARY=false

for arg in "$@"
do
    case $arg in
        --html)
        HTML_REPORT=true
        shift
        ;;
        --summary)
        SUMMARY=true
        shift
        ;;
        *)
        shift
        ;;
    esac
done

echo "Cleaning, running CMake with coverage on and generating the files"
./clean.sh
cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE:FILEPATH=python -DNATIVE_TEST_COVERAGE=ON
make -j native_tests && ./tests/native_tests

GCDA_DIR="./tests/CMakeFiles/native_tests.dir"

find_source_file() {
    local gcda_file=$1
    local source_file=$(basename "$gcda_file" .cpp.gcda)

    source_file=${source_file#__}

    find . -type f -name "${source_file}.cpp" \
        -not -path '*/_vendor/*' -not -path '*/_deps/*' -not -path '*/include/*' | head -n 1
}

rename_files() {
    local dir=$1
    local old_ext=$2
    local new_ext=$3
    find "$dir" -name "*$old_ext" | while read -r file; do
        mv "$file" "${file%$old_ext}$new_ext"
    done
}

rename_files "$GCDA_DIR" ".cpp.gcda" ".gcda"
rename_files "$GCDA_DIR" ".cpp.gcno" ".gcno"

echo "Capturing coverage data with lcov..."
lcov --capture --directory "$GCDA_DIR" --output-file coverage.info

echo "Excluding _vendor, _deps, /usr/include, and other /include/ directories from coverage..."
lcov --remove coverage.info \
    '*/_vendor/*' \
    '*/_deps/*' \
    '/usr/include/*' \
    '*/include/*' \
    --output-file coverage_filtered.info

if [ "$SUMMARY" = true ]; then
    echo "Generating coverage summary..."
    # Output only the percentage of coverage by lines
    coverage_percent=$(lcov --summary coverage_filtered.info 2>&1 | grep "lines......:" | awk '{print substr($2, 1, length($2)-1)}')
    echo "$coverage_percent"
else
    if [ "$HTML_REPORT" = true ]; then
        echo "Generating HTML report..."
        genhtml coverage_filtered.info --output-directory coverage_html
        echo "Coverage reports generated in coverage_html/"
    else
        echo "Generating coverage report in text mode..."
        lcov --list coverage_filtered.info
    fi
fi

rm -f coverage.info coverage_filtered.info

rename_files "$GCDA_DIR" ".gcda" ".cpp.gcda"
rename_files "$GCDA_DIR" ".gcno" ".cpp.gcno"

