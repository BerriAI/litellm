#!/bin/bash
p=$(cat $HOME/cachezero/pr2n/proxy.pid 2>/dev/null); [ -n "$p" ] && kill -TERM -- -$p 2>/dev/null; sleep 3; [ -n "$p" ] && kill -KILL -- -$p 2>/dev/null; pkill -f "proxy_cli.py --config .* --port 4396" 2>/dev/null; true
