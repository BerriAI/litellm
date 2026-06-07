/**
 * Regenerates src/lib/http/schema.d.ts from the proxy's OpenAPI spec.
 *
 * Two hops, because the backend is the source of truth: the FastAPI app emits
 * the spec from its route decorators (app.openapi()), then openapi-typescript
 * turns that spec into TypeScript types. There is no live server in the loop —
 * the spec is read straight off the app object, so this runs in CI without a
 * database or proxy boot.
 *
 * The Python interpreter must have litellm installed. Override which one via
 * LITELLM_PYTHON (CI passes "uv run --no-sync python"); defaults to python3.
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const dashboardDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = resolve(dashboardDir, "..", "..");
const outPath = join(dashboardDir, "src", "lib", "http", "schema.d.ts");
const specDir = mkdtempSync(join(tmpdir(), "litellm-openapi-"));
const specPath = join(specDir, "openapi.json");

const python = (process.env.LITELLM_PYTHON ?? "python3").split(" ");
// Two transforms before emitting the spec, both on a throwaway interpreter so the
// spec the proxy actually serves is unchanged:
//   1. Force in internal UI routes the public /openapi.json hides via
//      include_in_schema=False, so the dashboard can type against them.
//   2. Strip every `description`. The dashboard consumes shapes, not docstrings,
//      and FastAPI renders docstring descriptions with environment-dependent
//      indentation, which made schema.d.ts impossible to reproduce across machines
//      and flaked this check. Dropping them makes the types byte-identical
//      regardless of who runs gen:api, and shrinks the file.
const dumpSpec = [
  "import json, sys",
  "from litellm.proxy.proxy_server import app",
  "from fastapi.routing import APIRoute",
  "for route in app.routes:",
  "    if isinstance(route, APIRoute):",
  "        route.include_in_schema = True",
  "app.openapi_schema = None",
  "",
  "def _strip_descriptions(node):",
  "    if isinstance(node, dict):",
  "        node.pop('description', None)",
  "        for value in node.values():",
  "            _strip_descriptions(value)",
  "    elif isinstance(node, list):",
  "        for item in node:",
  "            _strip_descriptions(item)",
  "",
  "spec = app.openapi()",
  "_strip_descriptions(spec)",
  "with open(sys.argv[1], 'w') as f:",
  "    json.dump(spec, f, sort_keys=True)",
].join("\n");

try {
  execFileSync(python[0], [...python.slice(1), "-c", dumpSpec, specPath], {
    cwd: repoRoot,
    stdio: "inherit",
  });

  execFileSync(join(dashboardDir, "node_modules", ".bin", "openapi-typescript"), [specPath, "-o", outPath], {
    cwd: dashboardDir,
    stdio: "inherit",
  });
} finally {
  rmSync(specDir, { recursive: true, force: true });
}
