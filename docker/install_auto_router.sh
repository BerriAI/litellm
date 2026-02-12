#!/bin/bash
# semantic_router does not support Python 3.14+ (requires_python: >=3.9,<3.14)
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PYTHON_MINOR" -lt 14 ]; then
  pip install semantic_router==0.1.11 --no-deps
  pip install aurelio-sdk==0.0.19
else
  echo "Skipping semantic_router and aurelio-sdk: not supported on Python 3.14+"
fi