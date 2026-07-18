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
// The dashboard calls internal UI routes that the public /openapi.json hides via
// include_in_schema=False. Force them in so they get typed here; this mutates a
// throwaway interpreter, so the spec the proxy actually serves is unchanged.
// Python 3.13 strips a docstring's common leading indentation at compile time
// while 3.12 keeps it, so the same model yields differently-indented descriptions
// depending on the interpreter — enough to make this output non-reproducible
// across CI and contributors. inspect.cleandoc normalizes every description to one
// canonical form regardless of interpreter, so the generated file is stable.
const dumpSpec = [
  "import inspect, json, sys",
  "from litellm.proxy.proxy_server import app",
  "from fastapi.routing import APIRoute",
  "for route in app.routes:",
  "    if isinstance(route, APIRoute):",
  "        route.include_in_schema = True",
  "app.openapi_schema = None",
  "def normalize(node):",
  "    if isinstance(node, dict):",
  "        return {k: inspect.cleandoc(v) if k == 'description' and isinstance(v, str) else normalize(v) for k, v in node.items()}",
  "    if isinstance(node, list):",
  "        return [normalize(v) for v in node]",
  "    return node",
  "with open(sys.argv[1], 'w') as f: json.dump(normalize(app.openapi()), f, sort_keys=True)",
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
