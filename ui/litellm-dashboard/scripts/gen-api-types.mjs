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
 *
 * The dump itself lives in scripts/dump_openapi.py at the repo root, shared
 * with the OpenAPI breaking-change CI gate. --include-internal forces the
 * internal UI routes (hidden from the served /openapi.json) into the spec so
 * they get typed here.
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
const dumpScript = join(repoRoot, "scripts", "dump_openapi.py");

try {
  execFileSync(python[0], [...python.slice(1), dumpScript, "--include-internal", specPath], {
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
