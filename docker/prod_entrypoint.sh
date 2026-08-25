#!/bin/sh

# Allow the kernel to write a core dump if a worker crashes (SIGSEGV/SIGABRT).
# Defaults to 0 on GKE COS; without this no core is ever written. Inherited by the
# exec'd process and all uvicorn workers it spawns. No effect unless
# kernel.core_pattern is also set to a host path (see coredump-config DaemonSet).
# Capped at 6 GiB (in 1-KiB blocks): captures the full resident set of a pod even
# under high-load RSS growth (current RSS 1.7-2.8 Gi, crashes happen under load),
# while bounding disk use so a SIGSEGV crashloop can't fill the ~44 Gi node disk.
# Pair with a DiskPressure / boot-disk >80% alert as a crashloop backstop.
#
# TWO STEPS, BOTH REQUIRED: GKE COS containerd sets the RLIMIT_CORE hard limit
# to 0 at process birth, so a soft-only `ulimit -c` silently fails (can't exceed
# hard=0) and no core is ever written. Root can raise the hard limit first, then
# the soft. Without the -H raise, every prior SIGSEGV produced no core.
ulimit -H -c 6291456
ulimit -c 6291456

if [ "$USE_DDTRACE" = "true" ]; then
    export DD_TRACE_OPENAI_ENABLED="False"
    exec ddtrace-run litellm "$@"
else
    exec litellm "$@"
fi
