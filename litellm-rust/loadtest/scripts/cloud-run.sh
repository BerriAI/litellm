#!/usr/bin/env bash
# Run one workload on the cloud rig: every load generator takes a shard, results and gateway
# samples come back to results/<run>/, and the summary is printed.
#
#   scripts/cloud-run.sh realistic-ramp            # workloads/realistic-ramp.json
#   KEY=sk-loadtest-master scripts/cloud-run.sh smoke   # one key for all requests (until the
#                                                         # gateway reads virtual keys)
#   MESSAGES_AUTH=x-api-key scripts/cloud-run.sh smoke  # Anthropic SDK style auth
#
# Reads addresses from `terraform output` in terraform/aws. Needs ssh access as ec2-user.
set -euo pipefail

workload="${1:?usage: cloud-run.sh <workload name>}"
here="$(cd "$(dirname "$0")/.." && pwd)"
tf="$here/terraform/aws"
run="${RUN:-$workload-$(date -u +%Y%m%dT%H%M%SZ)}"
out="$here/results/$run"
mkdir -p "$out"

output() { terraform -chdir="$tf" output -json "$1"; }
target="$(output target | jq -r .)"
tools="$(output tools_image | jq -r .)"
mapfile -t loadgens < <(output loadgen_public_ips | jq -r '.[]')
mapfile -t gateways < <(output gateway_public_ips | jq -r '.[]')
mapfile -t mocks < <(output mock_public_ips | jq -r '.[]')
ssh_opts=(-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o BatchMode=yes)

for host in "${loadgens[@]}" "${gateways[@]}" "${mocks[@]}"; do
  ssh "${ssh_opts[@]}" "ec2-user@$host" 'test -f /opt/lt/ready' \
    || { echo "$host is not ready yet (cloud-init still running?)" >&2; exit 1; }
done

extra=()
[[ -n "${KEY:-}" ]] && extra+=(-key "$KEY")
[[ -n "${MESSAGES_AUTH:-}" ]] && extra+=(-messages-auth "$MESSAGES_AUTH")

echo "run $run: ${#loadgens[@]} generators -> $target"
start_ms="$(date +%s%3N)"
pids=()
for i in "${!loadgens[@]}"; do
  host="${loadgens[$i]}"
  ssh "${ssh_opts[@]}" "ec2-user@$host" \
    "sudo docker run --rm --network host --ulimit nofile=1048576:1048576 -v /opt/lt:/work \
      $tools loadgen -workload /workloads/$workload.json -target $target \
      -shard $i -shards ${#loadgens[@]} -seed $RANDOM -out /work/$run-$i.jsonl ${extra[*]}" \
    > "$out/loadgen-$i.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done

for i in "${!loadgens[@]}"; do
  scp "${ssh_opts[@]}" -q "ec2-user@${loadgens[$i]}:/opt/lt/$run-$i.jsonl" "$out/"
done
for i in "${!gateways[@]}"; do
  ssh "${ssh_opts[@]}" "ec2-user@${gateways[$i]}" \
    "sudo docker logs procmon 2>/dev/null | jq -c 'select(.t >= $start_ms)'" > "$out/procmon-gateway-$i.jsonl" || true
done
for i in "${!mocks[@]}"; do
  ssh "${ssh_opts[@]}" "ec2-user@${mocks[$i]}" "curl -s localhost:9000/stats" > "$out/mock-$i-stats.json" || true
done

files=()
for i in "${!loadgens[@]}"; do files+=("/r/$run-$i.jsonl"); done
docker run --rm -v "$out:/r:ro" "$tools" summarize "${files[@]}" > "$out/summary.md" 2>/dev/null \
  || (cd "$here" && go run ./cmd/summarize "$out"/"$run"-*.jsonl > "$out/summary.md")
cat "$out/summary.md"
jq -s 'if length == 0 then "no gateway samples" else
  {peak_cpu_cores: (map(.cpu_cores) | max), peak_rss_mb: (map(.rss_mb) | max),
   peak_fds: (map(.fds) | max), peak_tcp_estab: (map(.tcp_estab) | max)} end' \
  "$out"/procmon-gateway-*.jsonl
echo "results in $out"
exit $status
