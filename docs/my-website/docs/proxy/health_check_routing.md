# Health Check Driven Routing

Route traffic away from unhealthy deployments before users hit errors. Background health checks run on a configurable interval, and any deployment that fails gets removed from the routing pool proactively, not after a user request already failed.


## Architecture

<svg viewBox="0 0 860 600" xmlns="http://www.w3.org/2000/svg" style={{maxWidth: '100%', fontFamily: 'system-ui, sans-serif'}}>
  {/* Background */}
  <rect width="860" height="600" fill="#f8fafc" rx="12"/>

  {/* LEFT PANEL: Background health check loop */}
  <rect x="20" y="20" width="240" height="560" fill="#eff6ff" rx="10" stroke="#bfdbfe" strokeWidth="1.5"/>
  <text x="140" y="48" textAnchor="middle" fill="#1d4ed8" fontSize="13" fontWeight="600">Background Loop</text>
  <text x="140" y="64" textAnchor="middle" fill="#3b82f6" fontSize="11">every health_check_interval seconds</text>

  {/* Deployment A */}
  <rect x="40" y="82" width="200" height="50" fill="white" rx="8" stroke="#93c5fd" strokeWidth="1.5"/>
  <text x="140" y="102" textAnchor="middle" fill="#1e40af" fontSize="12" fontWeight="500">Deployment A</text>
  <text x="140" y="120" textAnchor="middle" fill="#64748b" fontSize="11">ahealth_check() → 200 ✓</text>

  {/* Deployment B */}
  <rect x="40" y="148" width="200" height="50" fill="white" rx="8" stroke="#fca5a5" strokeWidth="1.5"/>
  <text x="140" y="168" textAnchor="middle" fill="#991b1b" fontSize="12" fontWeight="500">Deployment B</text>
  <text x="140" y="186" textAnchor="middle" fill="#64748b" fontSize="11">ahealth_check() → 401 ✗</text>

  {/* Deployment C */}
  <rect x="40" y="214" width="200" height="50" fill="white" rx="8" stroke="#fde68a" strokeWidth="1.5"/>
  <text x="140" y="234" textAnchor="middle" fill="#92400e" fontSize="12" fontWeight="500">Deployment C</text>
  <text x="140" y="252" textAnchor="middle" fill="#64748b" fontSize="11">ahealth_check() → 429 ⚡</text>

  {/* ignore_transient box */}
  <rect x="40" y="282" width="200" height="68" fill="#fefce8" rx="8" stroke="#fde047" strokeWidth="1.5"/>
  <text x="140" y="302" textAnchor="middle" fill="#713f12" fontSize="11" fontWeight="600">ignore_transient_errors: true</text>
  <text x="140" y="320" textAnchor="middle" fill="#92400e" fontSize="11">429 / 408 → ignored</text>
  <text x="140" y="338" textAnchor="middle" fill="#92400e" fontSize="11">not written to cache</text>

  {/* allowed_fails_policy box */}
  <rect x="40" y="368" width="200" height="84" fill="#f0fdf4" rx="8" stroke="#86efac" strokeWidth="1.5"/>
  <text x="140" y="388" textAnchor="middle" fill="#166534" fontSize="11" fontWeight="600">allowed_fails_policy</text>
  <text x="140" y="406" textAnchor="middle" fill="#15803d" fontSize="11">401 → increment counter</text>
  <text x="140" y="424" textAnchor="middle" fill="#15803d" fontSize="11">counter &gt; threshold</text>
  <text x="140" y="442" textAnchor="middle" fill="#15803d" fontSize="11">→ cooldown triggered</text>

  {/* CENTER PANEL: Shared State */}
  <rect x="300" y="20" width="220" height="560" fill="#f5f3ff" rx="10" stroke="#c4b5fd" strokeWidth="1.5"/>
  <text x="410" y="48" textAnchor="middle" fill="#6d28d9" fontSize="13" fontWeight="600">Shared State</text>

  {/* Health State Cache */}
  <rect x="320" y="62" width="180" height="116" fill="white" rx="8" stroke="#a78bfa" strokeWidth="1.5"/>
  <text x="410" y="84" textAnchor="middle" fill="#5b21b6" fontSize="12" fontWeight="600">DeploymentHealthCache</text>
  <text x="410" y="104" textAnchor="middle" fill="#64748b" fontSize="11">A → healthy ✓</text>
  <text x="410" y="122" textAnchor="middle" fill="#64748b" fontSize="11">B → unhealthy ✗</text>
  <text x="410" y="140" textAnchor="middle" fill="#64748b" fontSize="11">C → not written (ignored)</text>
  <text x="410" y="164" textAnchor="middle" fill="#94a3b8" fontSize="10">TTL: staleness_threshold × 1.5</text>

  {/* Cooldown Cache */}
  <rect x="320" y="196" width="180" height="104" fill="white" rx="8" stroke="#a78bfa" strokeWidth="1.5"/>
  <text x="410" y="218" textAnchor="middle" fill="#5b21b6" fontSize="12" fontWeight="600">Cooldown Cache</text>
  <text x="410" y="238" textAnchor="middle" fill="#64748b" fontSize="11">B → cooling down</text>
  <text x="410" y="256" textAnchor="middle" fill="#64748b" fontSize="11">(after policy threshold)</text>
  <text x="410" y="278" textAnchor="middle" fill="#94a3b8" fontSize="10">TTL: cooldown_time</text>

  {/* failed_calls counter */}
  <rect x="320" y="318" width="180" height="90" fill="white" rx="8" stroke="#a78bfa" strokeWidth="1.5"/>
  <text x="410" y="340" textAnchor="middle" fill="#5b21b6" fontSize="12" fontWeight="600">failed_calls counter</text>
  <text x="410" y="360" textAnchor="middle" fill="#64748b" fontSize="11">B: 2 / AuthAllowedFails: 1</text>
  <text x="410" y="378" textAnchor="middle" fill="#64748b" fontSize="11">→ threshold exceeded</text>
  <text x="410" y="398" textAnchor="middle" fill="#94a3b8" fontSize="10">TTL: cooldown_time (must &gt; interval)</text>

  {/* RIGHT PANEL: Request path */}
  <rect x="560" y="20" width="280" height="560" fill="#fff7ed" rx="10" stroke="#fed7aa" strokeWidth="1.5"/>
  <text x="700" y="48" textAnchor="middle" fill="#c2410c" fontSize="13" fontWeight="600">Request Path</text>

  {/* Incoming request */}
  <rect x="580" y="62" width="240" height="38" fill="#fff" rx="7" stroke="#fb923c" strokeWidth="1.5"/>
  <text x="700" y="85" textAnchor="middle" fill="#9a3412" fontSize="12" fontWeight="500">Incoming request</text>

  {/* All deployments */}
  <rect x="580" y="120" width="240" height="38" fill="#fff" rx="7" stroke="#fb923c" strokeWidth="1.5"/>
  <text x="700" y="143" textAnchor="middle" fill="#9a3412" fontSize="12">All deployments [A, B, C]</text>

  <line x1="700" y1="100" x2="700" y2="120" stroke="#fb923c" strokeWidth="1.5" markerEnd="url(#arrow-orange)"/>

  {/* Health check filter */}
  <rect x="580" y="178" width="240" height="62" fill="#fff" rx="7" stroke="#fb923c" strokeWidth="1.5"/>
  <text x="700" y="200" textAnchor="middle" fill="#9a3412" fontSize="12" fontWeight="600">① Health Check Filter</text>
  <text x="700" y="218" textAnchor="middle" fill="#64748b" fontSize="11">if policy set → bypass</text>
  <text x="700" y="234" textAnchor="middle" fill="#64748b" fontSize="11">else → remove unhealthy</text>

  <line x1="700" y1="158" x2="700" y2="178" stroke="#fb923c" strokeWidth="1.5" markerEnd="url(#arrow-orange)"/>

  {/* Cooldown filter */}
  <rect x="580" y="262" width="240" height="50" fill="#fff" rx="7" stroke="#fb923c" strokeWidth="1.5"/>
  <text x="700" y="284" textAnchor="middle" fill="#9a3412" fontSize="12" fontWeight="600">② Cooldown Filter</text>
  <text x="700" y="302" textAnchor="middle" fill="#64748b" fontSize="11">remove deployments in cooldown</text>

  <line x1="700" y1="240" x2="700" y2="262" stroke="#fb923c" strokeWidth="1.5" markerEnd="url(#arrow-orange)"/>

  {/* Safety net */}
  <rect x="580" y="334" width="240" height="52" fill="#fef9c3" rx="7" stroke="#fbbf24" strokeWidth="1.5"/>
  <text x="700" y="356" textAnchor="middle" fill="#713f12" fontSize="12" fontWeight="600">Safety Net</text>
  <text x="700" y="376" textAnchor="middle" fill="#713f12" fontSize="11">if all removed → return all</text>

  <line x1="700" y1="312" x2="700" y2="334" stroke="#fb923c" strokeWidth="1.5" markerEnd="url(#arrow-orange)"/>

  {/* Load balancer */}
  <rect x="580" y="408" width="240" height="38" fill="#fff" rx="7" stroke="#fb923c" strokeWidth="1.5"/>
  <text x="700" y="431" textAnchor="middle" fill="#9a3412" fontSize="12" fontWeight="600">③ Load Balancer</text>

  <line x1="700" y1="386" x2="700" y2="408" stroke="#fb923c" strokeWidth="1.5" markerEnd="url(#arrow-orange)"/>

  {/* Selected deployment */}
  <rect x="580" y="468" width="240" height="38" fill="#dcfce7" rx="7" stroke="#4ade80" strokeWidth="1.5"/>
  <text x="700" y="491" textAnchor="middle" fill="#14532d" fontSize="12" fontWeight="600">Selected: Deployment A ✓</text>

  <line x1="700" y1="446" x2="700" y2="468" stroke="#4ade80" strokeWidth="1.5" markerEnd="url(#arrow-green)"/>

  {/* ARROWS: left → center */}
  <line x1="240" y1="107" x2="320" y2="110" stroke="#3b82f6" strokeWidth="1.5" strokeDasharray="4,3" markerEnd="url(#arrow-blue)"/>
  <line x1="240" y1="173" x2="320" y2="240" stroke="#ef4444" strokeWidth="1.5" strokeDasharray="4,3" markerEnd="url(#arrow-red)"/>
  <line x1="240" y1="173" x2="320" y2="348" stroke="#ef4444" strokeWidth="1.5" strokeDasharray="4,3" markerEnd="url(#arrow-red)"/>
  <line x1="240" y1="316" x2="320" y2="130" stroke="#eab308" strokeWidth="1.5" strokeDasharray="4,3" markerEnd="url(#arrow-yellow)"/>

  {/* ARROWS: center → right */}
  <line x1="500" y1="120" x2="580" y2="190" stroke="#8b5cf6" strokeWidth="1.5" strokeDasharray="4,3" markerEnd="url(#arrow-purple)"/>
  <line x1="500" y1="248" x2="580" y2="274" stroke="#8b5cf6" strokeWidth="1.5" strokeDasharray="4,3" markerEnd="url(#arrow-purple)"/>

  {/* Arrow markers */}
  <defs>
    <marker id="arrow-orange" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#fb923c"/>
    </marker>
    <marker id="arrow-blue" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#3b82f6"/>
    </marker>
    <marker id="arrow-red" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#ef4444"/>
    </marker>
    <marker id="arrow-yellow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#eab308"/>
    </marker>
    <marker id="arrow-purple" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#8b5cf6"/>
    </marker>
    <marker id="arrow-green" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#4ade80"/>
    </marker>
  </defs>
</svg>


## What problem does this solve?

By default, LiteLLM routes traffic to all deployments and only stops sending to a broken one after it has already failed a user request. The cooldown system is reactive.

Health check driven routing makes this **proactive**: a background loop pings every deployment on a configurable interval. If a deployment fails its health check, it gets removed from the routing pool immediately, before a user request lands on it.

When you also set `allowed_fails_policy`, you control exactly how many health check failures of each error type (auth errors, rate limits, timeouts) are needed before a deployment enters cooldown. This avoids false positives from transient noise.


## Setup

### Step 1: Enable background health checks

Background health checks are off by default. Turn them on in `general_settings`:

```yaml
general_settings:
  background_health_checks: true
  health_check_interval: 60    # seconds between each full check cycle
```

### Step 2: Enable health check routing

```yaml
general_settings:
  background_health_checks: true
  health_check_interval: 60
  enable_health_check_routing: true  # ← route away from unhealthy deployments
```

At this point, any deployment that fails its health check is immediately excluded from routing until the next check cycle clears it.

### Step 3: Add a policy to control how many failures trigger cooldown

Without a policy, the first health check failure marks a deployment as unhealthy. If you want more tolerance (e.g., only act after 2 consecutive auth failures), use `allowed_fails_policy`:

```yaml
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-5
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-5
      api_key: os.environ/ANTHROPIC_API_KEY_SECONDARY

general_settings:
  background_health_checks: true
  health_check_interval: 30
  enable_health_check_routing: true

router_settings:
  cooldown_time: 60              # how long a deployment stays in cooldown
  allowed_fails_policy:
    AuthenticationErrorAllowedFails: 1   # cooldown after 2nd auth failure
    TimeoutErrorAllowedFails: 3          # cooldown after 4th timeout
```

When `allowed_fails_policy` is set, the binary health check filter is bypassed. Only the cooldown system controls routing exclusion, and it only fires after your configured threshold is crossed.

### Step 4 (optional): Ignore transient errors

429 (rate limit) and 408 (timeout) from a health check usually mean the deployment is temporarily overloaded, not broken. To prevent these from affecting routing at all:

```yaml
general_settings:
  background_health_checks: true
  health_check_interval: 30
  enable_health_check_routing: true
  health_check_ignore_transient_errors: true  # 429 and 408 never affect routing
```

With this on, only hard failures (401, 404, 5xx) from health checks contribute to cooldown.


## Full example

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY_SECONDARY

  - model_name: gpt-4o
    litellm_params:
      model: azure/gpt-4o
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY

general_settings:
  background_health_checks: true
  health_check_interval: 30
  enable_health_check_routing: true
  health_check_ignore_transient_errors: true

router_settings:
  cooldown_time: 60
  allowed_fails_policy:
    AuthenticationErrorAllowedFails: 0   # cooldown immediately on auth failure
    TimeoutErrorAllowedFails: 2          # cooldown after 3 timeouts
    RateLimitErrorAllowedFails: 5        # cooldown after 6 rate limits (if not ignoring transients)
```


## Configuration reference

| Setting | Where | Default | Description |
|---|---|---|---|
| `enable_health_check_routing` | `general_settings` | `false` | Route away from deployments that fail health checks |
| `background_health_checks` | `general_settings` | `false` | Must be `true` for health check routing to work |
| `health_check_interval` | `general_settings` | `300` | Seconds between full health check cycles |
| `health_check_staleness_threshold` | `general_settings` | `interval x 2` | Seconds before cached health state is ignored |
| `health_check_ignore_transient_errors` | `general_settings` | `false` | Ignore 429 and 408 from health checks; these never affect routing |
| `cooldown_time` | `router_settings` | `5` | Seconds a deployment stays in cooldown after threshold is crossed |
| `allowed_fails_policy` | `router_settings` | `null` | Per-error-type failure thresholds before cooldown (see below) |

### `allowed_fails_policy` fields

| Field | Error type | HTTP status |
|---|---|---|
| `AuthenticationErrorAllowedFails` | Bad API key | 401 |
| `TimeoutErrorAllowedFails` | Request timeout | 408 |
| `RateLimitErrorAllowedFails` | Rate limit exceeded | 429 |
| `BadRequestErrorAllowedFails` | Malformed request | 400 |
| `ContentPolicyViolationErrorAllowedFails` | Content filtered | 400 |

The value is the number of failures **tolerated** before cooldown. `0` means cooldown on the first failure. `2` means cooldown on the third.


## Things to keep in mind

- **Counter TTL must be longer than the health check interval.** `allowed_fails_policy` works by incrementing a `failed_calls` counter per deployment. That counter expires after `cooldown_time` seconds. If `cooldown_time` is shorter than `health_check_interval`, the counter resets between every check cycle and failures never accumulate. Set `cooldown_time` greater than `health_check_interval` when using `allowed_fails_policy`.

  ```yaml
  router_settings:
    cooldown_time: 60       # must be > health_check_interval (30s here)

  general_settings:
    health_check_interval: 30
  ```

- **`AllowedFails: N` means cooldown on the (N+1)th failure.** The counter check is `updated_fails > allowed_fails`, so `0` triggers on the 1st failure, `1` on the 2nd, `2` on the 3rd.

  | `AllowedFails` | Cooldown triggers after |
  |---|---|
  | `0` | 1st failure |
  | `1` | 2nd failure |
  | `2` | 3rd failure |

- **Without `allowed_fails_policy`, the first failure is enough.** The first failed health check immediately excludes the deployment from routing. Use `allowed_fails_policy` when you want tolerance for flaky checks.

- **If all deployments are unhealthy, the filter is bypassed.** Traffic keeps flowing rather than returning no deployment at all. Requests will fail, but the router keeps trying.

- **Health check failures and request failures share the same counters.** When `allowed_fails_policy` is set, both sources increment the same `failed_calls` counter. A deployment at 1 health check failure that then receives 1 failing request will hit the threshold for `AllowedFails: 1` and enter cooldown.


## Debugging

Run the proxy with `--detailed_debug` and look for these log lines:

After each health check cycle (written at DEBUG level):
```
health_check_routing_state_updated healthy=2 unhealthy=1
```

When a health check failure increments the counter and triggers cooldown (DEBUG level):
```
checks 'should_run_cooldown_logic'
Attempting to add <deployment_id> to cooldown list
```

When safety net fires because all deployments are in cooldown:
```
All deployments in cooldown via health-check routing, bypassing cooldown filter
```

When safety net fires because all deployments are unhealthy (binary filter, no `allowed_fails_policy`):
```
All deployments marked unhealthy by health checks, bypassing health filter
```
