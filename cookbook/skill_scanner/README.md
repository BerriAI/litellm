# Manual skill security scans

This standalone adapter runs NVIDIA SkillSpector on a separate machine. It is not installed in the LiteLLM proxy and does not require a shared Helm or Compose deployment

Start it with Python 3.12 or newer and uv. The script pins SkillSpector and its HTTP dependencies

```sh
export SKILL_SCANNER_API_KEY="$(openssl rand -hex 32)"
uv run server.py
```

For another host, place the loopback service behind an HTTPS reverse proxy that forwards the Authorization header and enforces a 65 MiB request limit. Run it under an unprivileged account with a writable temporary directory and no production files or credentials. The scanner host needs Git and network access to requested sources and optional model providers. Apply network egress restrictions suitable for those sources

Configure the same scanner secret in the gateway's environment, then add this to LiteLLM's configuration

```yaml
general_settings:
  skill_scanner:
    endpoint: https://scanner.example.com/scan
    api_key_env: SKILL_SCANNER_API_KEY
    use_llm: false
    timeout_seconds: 180
```

The endpoint must be reachable from the gateway. A customer's laptop works while it is reachable through their network, VPN or authenticated HTTPS tunnel. `localhost` refers to the gateway machine, not the admin's browser. HTTP is accepted only for loopback development

Open Skills, Submissions, and Review as a proxy admin. Click **Scan for threats**, inspect the report, then independently approve or reject the proposal. Scanning is optional and never changes approval permissions or blocks publication. The report shows findings, locations, remediation, risk score, scanner version and whether semantic analysis actually ran. Incomplete reports do not establish safety

Uploaded ZIPs are sent privately to the scanner with their exact stored bytes. Metadata-only edits reuse the matching stored archive. If that archive is missing or changed, submit a new ZIP before scanning. The displayed SHA-256 identifies that archive, and the scan examines its files rather than proposed catalog metadata. Remote URL reports describe the source at scan time and do not freeze its contents. GitHub branch/tag refs and explicit-ref subdirectories are supported. Other ref/subdirectory combinations require a ZIP so the adapter does not silently scan the wrong revision

The latest scan result is retained with the submission. Running scans have a deadline. If a gateway restarts or the scanner becomes unavailable, the attempt expires and an admin can retry. Retries replace the latest report. Approving or rejecting during a scan does not cancel it or change the decision

Set `use_llm: true` to request semantic analysis and configure the provider on the scanner machine using [SkillSpector's provider instructions](https://github.com/NVIDIA/skillspector#llm-analysis). Provider credentials stay on that machine. Static analysis can also call OSV for dependency checks

## Gen Agent Trust Hub and Socket

The configured endpoint must implement the [HTTP contract below](#http-contract). Gen and Socket vendor URLs are not interchangeable with this adapter. Their public contracts were checked on November 5, 2026

| Service | Published capability | Private uploaded skills |
| --- | --- | --- |
| [Gen Agent Trust Hub](https://ai.gendigital.com/skill-scanner) | JSON lookup of a public ClawHub skill URL, returning a severity and reasons | No supported arbitrary skill ZIP upload contract is exposed in the public documentation |
| [Socket](https://docs.socket.dev/docs/language-support) | Experimental skills.sh scanning, with access available by contacting Socket | Custom skills are listed as planned, so private skill scanning needs a confirmed vendor contract and access |
| [skills.sh audits](https://skills.sh/docs/api) | Previously generated reports from partners including Gen and Socket | Catalog lookup only, with no uploaded bytes sent for analysis |

Gen documents this request without an API key

```sh
curl --request POST 'https://ai.gendigital.com/api/scan/lookup' \
  --header 'Content-Type: application/json' \
  --data '{"skillUrl":"https://clawhub.ai/steipete/weather"}'
```

The checked response included `severity`, `reasons`, `author` and `displayName`. It provided no content digest or revision. GitHub and skills.sh skill URLs returned HTTP 400 with `Invalid skill URL`. This cannot verify the stored bytes of a Skills Hub submission

Socket's [archive scan API](https://docs.socket.dev/reference/createorgfullscanarchive) accepts ZIPs but extracts supported dependency manifests for a full scan. Its [skills.sh integration](https://www.socket.dev/blog/socket-brings-supply-chain-security-to-skills) describes a separate source-analysis flow. A general Socket API token and successful archive upload do not establish that skill instructions and scripts were analyzed

The skills.sh audit endpoint, `GET /api/v1/skills/audit/{source}/{skill}`, returns partner verdicts, summaries and audit timestamps. Its documented authentication uses a Vercel OIDC token. The audit response does not bind those verdicts to a content digest or commit. Treat these results as historical catalog information, even when the skill detail endpoint separately provides a hash

For private uploads, use the adapter above or another service implementing the complete HTTP contract. A Gen or Socket adapter requires vendor-confirmed private source submission, authentication, result and retention contracts before it can claim to scan submitted bytes. Keep the skill private, send content only to the configured scanner, and preserve unknown or incomplete results instead of translating a lookup into an exact-archive safety verdict

## HTTP contract

The gateway sends an authenticated `POST` to the configured endpoint. The body contains form fields `source` (JSON source descriptor), `use_llm` (boolean), and `expires_at` (timezone-aware ISO timestamp), plus `file` (ZIP) when bytes are available. The service returns HTTP 200 with this JSON shape

```json
{
  "scanner_version": "2.12.0",
  "complete": true,
  "scan_mode": "static-only",
  "risk_score": 70,
  "findings": [{
    "rule_id": "example-rule",
    "severity": "HIGH",
    "message": "Potential data exfiltration",
    "file": "SKILL.md",
    "start_line": 8,
    "explanation": "Instructions request sending local files to an external host",
    "remediation": "Remove the external upload instruction"
  }],
  "limitations": []
}
```

Reports are limited to 1 MiB and 500 findings. Scanner uploads allow 64 MiB because the gateway can expand compressed skill uploads into stored compatibility ZIPs. The gateway rejects malformed reports, refuses redirects and verifies TLS. Only the configured scanner credential is sent. A timeout, HTTP error or incomplete report remains visible to the reviewer

The bundled adapter runs at most two child scans concurrently. Each child and its descendants are terminated at the gateway deadline, capped at 570 seconds. Keep the gateway and scanner clocks synchronized. Direct adapter clients may omit `expires_at` to use the cap. Child processes never execute submitted skill scripts. Raise the gateway deadline up to 600 seconds when needed for semantic analysis and configure the reverse proxy timeout to match
