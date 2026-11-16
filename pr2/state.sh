#!/bin/bash
# usage: state.sh <out.jsonl> <label>  -- dumps every detector key on the dedicated Redis (6392) as a completion signal
r="redis-cli -p 6392"
for k in $($r --scan --pattern 'prompt_cache_alert:*' | sort); do
  t=$($r type $k)
  case $t in
    hash) v=$($r hgetall $k | tr '\n' ' ');;
    zset) v="zcard $($r zcard $k)";;
    string) v=$($r get $k);;
    *) v="?";;
  esac
  printf '{"state_after":"%s","key":"%s","type":"%s","value":"%s","pttl":%s}\n' "$2" "${k:0:44}..${k: -8}" "$t" "$v" "$($r pttl $k)" | tee -a "$1"
done
