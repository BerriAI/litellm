#!/bin/bash
# redis.sh up|down   dedicated Redis :6392; "down" saves state so "up" restores it
if [ "$1" = up ]; then redis-server --port 6392 --appendonly no --dbfilename pr2n.rdb --dir $HOME/cachezero/redis --daemonize yes >/dev/null; sleep 1; redis-cli -p 6392 ping
else redis-cli -p 6392 shutdown save >/dev/null 2>&1; echo down; fi
