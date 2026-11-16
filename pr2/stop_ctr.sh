#!/bin/bash
# stops every process in the task container except its PID 1 (sleep infinity); the container only ever runs task-owned processes
docker exec cachezero-pr2-glibc239 bash -c 'kill -TERM -1; sleep 3; kill -KILL -1' 2>/dev/null
sleep 1; docker top cachezero-pr2-glibc239 -o pid,cmd | tail -n +2 | grep -vc "sleep infinity"
curl -s -m 2 -o /dev/null -w "%{http_code}\n" http://127.0.0.1:4396/health/liveliness
