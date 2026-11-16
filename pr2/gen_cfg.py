import sys
name, alerts, redis, alt = sys.argv[1], sys.argv[2] == "1", sys.argv[3] == "1", sys.argv[4] == "1"
out = ["model_list:"]
for i in "abcdefghijklmnopqrstuv":
    base = "alt" if (alt and i == "m") else "main"
    key = "fixture-account-2" if (alt and i == "n") else "fixture-account-1" if i == "n" else "os.environ/ANTHROPIC_API_KEY"
    out += [f"  - model_name: pc-{i}", f"    litellm_params: {{model: anthropic/claude-sonnet-5-5, api_key: {key}, api_base: http://127.0.0.1:4395/{base}}}", f"    model_info: {{id: pc-{i}}}"]
out += ["  - model_name: pc-oai", "    litellm_params: {model: openai/fixture, api_key: unused, api_base: http://127.0.0.1:4395/oai/v1}", "    model_info: {id: pc-oai}"]
out += ["general_settings:", "  master_key: os.environ/LITELLM_MASTER_KEY"]
if redis: out += ["  coordination_redis: {host: 127.0.0.1, port: 6392}"]
if alerts: out += ["  prompt_cache_alerts: true"]
out += ["router_settings:", "  num_retries: 0"]
if redis: out += ["  redis_host: 127.0.0.1", "  redis_port: 6392"]
open(f"{name}.yaml", "w").write("\n".join(out) + "\n")
