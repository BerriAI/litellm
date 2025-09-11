#!/usr/bin/env bash
set -euo pipefail

### Useful globals
MY_DIR=$(dirname $(realpath $0))
MY_NAME="$0"
BUILD_DIR="build"

### Compiler discovery
# Initialize variables to store the highest versions
highest_cc=""
highest_cxx=""
highest_gcc=""
highest_gxx=""
highest_clang=""
highest_clangxx=""

if [[ $OSTYPE == 'darwin'* ]]; then
  # Needed for some of ForkDeathTests to pass on Mac
  export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
fi

# Function to find the highest version of compilers
# Note that the product of this check is ignored if the user passes CC/CXX
find_highest_compiler_version() {
  local base_name=$1
  local highest_var_name=$2
  local highest_version=0
  local current_version
  local found_version=""

  # Function to extract the numeric version from the compiler output
  get_version() {
    $1 --version 2>&1 | grep -oE '[0-9]+(\.[0-9]+)?' | head -n 1
  }

  # Try to find the latest versions of both GCC and Clang (numbered versions)
  # The range 5-20 is arbitrary (GCC 5 was released in 2015, and 20 is a high number since Clang is on version 17)
  for version in {20..5}; do
    if command -v "${base_name}-${version}" &> /dev/null; then
      current_version=$(get_version "${base_name}-${version}")
      if (( $(echo "$current_version > $highest_version" | bc -l) )); then
        highest_version=$current_version
        found_version="${base_name}-${version}"
      fi
    fi
  done

  # Check the base version if it exists
  if command -v "$base_name" &> /dev/null; then
    current_version=$(get_version "$base_name")
    if (( $(echo "$current_version > $highest_version" | bc -l) )); then
      found_version="$base_name"
    fi
  fi

  # Assign the result to the variable name passed
  if [[ -n $found_version ]]; then
    eval "$highest_var_name=$found_version"
  fi
}

# Find highest versions for each compiler
find_highest_compiler_version cc highest_cc
find_highest_compiler_version c++ highest_cxx
find_highest_compiler_version gcc highest_gcc
find_highest_compiler_version g++ highest_gxx
find_highest_compiler_version clang highest_clang
find_highest_compiler_version clang++ highest_clangxx

# Get the highest clang_tidy from the $highest_clangxx variable
CLANGTIDY_CMD=${highest_clangxx/clang++/clang-tidy}

### Build setup
# Targets to target dirs
declare -A target_dirs
target_dirs["ddup"]="ddup"
target_dirs["crashtracker"]="crashtracker"
target_dirs["stack_v2"]="stack_v2"
target_dirs["dd_wrapper"]="dd_wrapper"

# Compiler options
declare -A compiler_args
compiler_args["address"]="-DSANITIZE_OPTIONS=address"
compiler_args["leak"]="-DSANITIZE_OPTIONS=leak"
compiler_args["undefined"]="-DSANITIZE_OPTIONS=undefined"
compiler_args["safety"]="-DSANITIZE_OPTIONS=address,leak,undefined"
compiler_args["thread"]="-DSANITIZE_OPTIONS=thread"
compiler_args["numerical"]="-DSANITIZE_OPTIONS=integer,nullability,signed-integer-overflow,bounds,float-divide-by-zero"
compiler_args["dataflow"]="-DSANITIZE_OPTIONS=dataflow"
compiler_args["memory"]="-DSANITIZE_OPTIONS=memory"
compiler_args["fanalyzer"]="-DDO_FANALYZE=ON"
compiler_args["cppcheck"]="-DDO_CPPCHECK=ON"
compiler_args["infer"]="-DDO_INFER=ON"
compiler_args["clangtidy"]="-DDO_CLANGTIDY=ON"
compiler_args["clangtidy_cmd"]="-DCLANGTIDY_CMD=${CLANGTIDY_CMD}"

# Initial cmake args
cmake_args=(
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  -DCMAKE_VERBOSE_MAKEFILE=ON
  -DLIB_INSTALL_DIR=$(realpath $MY_DIR)/lib
  -DPython3_ROOT_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('prefix'))")
)

# Initial build targets; start out empty
targets=()

set_cc() {
  if [ -z "${CC:-}" ]; then
    export CC=$highest_cc
  fi
  if [ -z "${CXX:-}" ]; then
    export CXX=$highest_cxx
  fi
  cmake_args+=(
    -DCMAKE_C_COMPILER=$CC
    -DCMAKE_CXX_COMPILER=$CXX
  )
}

# Helper functions for finding the compiler(s)
set_clang() {
  if [ -z "${CC:-}" ]; then
    export CC=$highest_clang
  fi
  if [ -z "${CXX:-}" ]; then
    export CXX=$highest_clangxx
  fi
  cmake_args+=(
    -DCMAKE_C_COMPILER=$CC
    -DCMAKE_CXX_COMPILER=$CXX
  )
}

set_gcc() {
  # Only set CC or CXX if they're not set
  if [ -z "${CC:-}" ]; then
    export CC=$highest_gcc
  fi
  if [ -z "${CXX:-}" ]; then
    export CXX=$highest_gxx
  fi
  cmake_args+=(
    -DCMAKE_C_COMPILER=$CC
    -DCMAKE_CXX_COMPILER=$CXX
  )
}

### Build runners
run_cmake() {
  target=$1
  dir=${target_dirs[$target]}
  build=${BUILD_DIR}/${dir}
  if [ -z "$dir" ]; then
    echo "No directory specified for cmake"
    exit 1
  fi

  # Make sure we have the build directory
  mkdir -p ${build} && pushd ${build} || { echo "Failed to create build directory for $dir"; exit 1; }

  # Run cmake
  cmake "${cmake_args[@]}" -S=$MY_DIR/$dir || { echo "cmake failed"; exit 1; }
  cmake --build . || { echo "build failed"; exit 1; }
  if [[ " ${cmake_args[*]} " =~ " -DDO_CPPCHECK=ON " ]]; then
    echo "--------------------------------------------------------------------- Running CPPCHECK"
    make cppcheck || { echo "cppcheck failed"; exit 1; }
  fi
  if [[ " ${cmake_args[*]} " =~ " -DBUILD_TESTING=ON " ]]; then
    echo "--------------------------------------------------------------------- Running Tests"
    ctest --output-on-failure || { echo "tests failed!"; exit 1; }
  fi

  # OK, the build or whatever went fine I guess.
  popd
}

### Print help
print_help() {
  echo "Usage: ${MY_NAME} [options] [build_mode] [target]"
  echo "Options (one of)"
  echo "  -h, --help        Show this help message and exit"
  echo "  -a, --address     Clang + " ${compiler_args["address"]}
  echo "  -l, --leak        Clang + " ${compiler_args["leak"]}
  echo "  -u, --undefined   Clang + " ${compiler_args["undefined"]}
  echo "  -s, --safety      Clang + " ${compiler_args["safety"]}
  echo "  -t, --thread      Clang + " ${compiler_args["thread"]}
  echo "  -n, --numerical   Clang + " ${compiler_args["numerical"]}
  echo "  -d, --dataflow    Clang + " ${compiler_args["dataflow"]}  # Requires custom libstdc++ to work
  echo "  -m  --memory      Clang + " ${compiler_args["memory"]}
  echo "  -C  --cppcheck    Clang + " ${compiler_args["cppcheck"]}
  echo "  -I  --infer       Clang + " ${compiler_args["infer"]}
  echo "  -T  --clangtidy   Clang + " ${compiler_args["clangtidy"]}
  echo "  -f, --fanalyze    GCC + " ${compiler_args["fanalyzer"]}
  echo "  -c, --clang       Clang (alone)"
  echo "  -g, --gcc         GCC (alone)"
  echo "  --                Don't do anything special"
  echo ""
  echo "Build Modes:"
  echo "  Debug (default)"
  echo "  Release"
  echo "  RelWithDebInfo"
  echo ""
  echo "(any possible others, depending on what cmake supports for"
  echo "BUILD_TYPE out of the box)"
  echo ""
  echo "Targets:"
  echo "  all"
  echo "  all_test (default)"
  echo "  dd_wrapper"
  echo "  dd_wrapper_test"
  echo "  stack_v2 (also builds dd_wrapper)"
  echo "  stack_v2_test (also builds dd_wrapper_test)"
  echo "  ddup (also builds dd_wrapper)"
  echo "  ddup_test (also builds dd_wrapper_test)"
  echo "  crashtracker (also builds dd_wrapper)"
  echo "  crashtracker_test (also builds dd_wrapper_test)"
}

print_cmake_args() {
  echo "CMake Args: ${cmake_args[*]}"
  echo "Targets: ${targets[*]}"
}

### Check input
# Check the first slot, options
add_compiler_args() {
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    -a|--address)
      cmake_args+=(${compiler_args["address"]})
      set_clang
      ;;
    -l|--leak)
      cmake_args+=(${compiler_args["leak"]})
      set_clang
      ;;
    -u|--undefined)
      cmake_args+=(${compiler_args["undefined"]})
      set_clang
      ;;
    -s|--safety)
      cmake_args+=(${compiler_args["safety"]})
      set_clang
      ;;
    -t|--thread)
      cmake_args+=(${compiler_args["thread"]})
      set_clang
      ;;
    -n|--numerical)
      cmake_args+=(${compiler_args["numerical"]})
      set_clang
      ;;
    -d|--dataflow)
      cmake_args+=(${compiler_args["dataflow"]})
      set_clang
      ;;
    -m|--memory)
      cmake_args+=(${compiler_args["memory"]})
      set_clang
      ;;
    -C|--cppcheck)
      cmake_args+=(${compiler_args["cppcheck"]})
      set_clang
      if command -v cppcheck &> /dev/null; then
        cmake_args+=(-DCPPCHECK_EXECUTABLE=$(which cppcheck))
      fi
      ;;
    -I|--infer)
      cmake_args+=(${compiler_args["infer"]})
      set_clang
      if command -v infer &> /dev/null; then
        cmake_args+=(-DInfer_EXECUTABLE=$(which infer))
      fi
      ;;
    -T|--clangtidy)
      cmake_args+=(${compiler_args["clangtidy"]})
      cmake_args+=(${compiler_args["clangtidy_cmd"]})
      set_clang
      ;;
    -f|--fanalyze)
      cmake_args+=(${compiler_args["fanalyzer"]})
      set_gcc
      ;;
    -c|--clang)
      set_clang
      ;;
    -g|--gcc)
      set_gcc
      ;;
    --)
      set_cc # Use system default compiler
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
}

# Check the second slot, build mode
add_build_mode() {
  case "$1" in
    Debug|Release|RelWithDebInfo)
      cmake_args+=(-DCMAKE_BUILD_TYPE=$1)
      ;;
    ""|--)
      cmake_args+=(-DCMAKE_BUILD_TYPE=Debug)
      ;;
    *)
      echo "Unknown build mode: $1"
      exit 1
      ;;
  esac
}

# Check the third slot, target
add_target() {
  arg=${1:-"all_test"}
  if [[ "${arg}" =~ _test$ ]]; then
    cmake_args+=(-DBUILD_TESTING=ON)
  fi
  target=${arg%_test}

  case "${target}" in
    all|--)
      targets+=("stack_v2")
      targets+=("ddup")
      targets+=("crashtracker")
      ;;
    dd_wrapper)
      # `dd_wrapper` is a dependency of other targets, but the overall structure is weird when it's given explicitly
      # so we only include it when it's called explicitly
      targets+=("dd_wrapper")
      ;;
    stack_v2)
      targets+=("stack_v2")
      ;;
    ddup)
      targets+=("ddup")
      ;;
    crashtracker)
      targets+=("crashtracker")
      ;;
    *)
      echo "Unknown target: $1"
      exit 1
      ;;
  esac
}


### ENTRYPOINT
# Check for basic input validity
if [ $# -eq 0 ]; then
  echo "No arguments given.  At least one is needed, otherwise I'd (a m b i g u o u s l y) do a lot of work!"
  print_help
  exit 1
fi

add_compiler_args "$1"
add_build_mode "$2"
add_target "$3"

# Print cmake args
print_cmake_args

# Run cmake
for target in "${targets[@]}"; do
  run_cmake $target
done
