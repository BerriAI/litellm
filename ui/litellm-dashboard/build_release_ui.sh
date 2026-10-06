#!/bin/bash
set -e

bash ./build_ui.sh
echo "UI artifacts staged for legacy and proxy-extras packages; generated files are not committed."
