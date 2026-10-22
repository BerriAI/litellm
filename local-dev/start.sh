#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
exec ./localization/zh-CN/start.sh "$PWD"
