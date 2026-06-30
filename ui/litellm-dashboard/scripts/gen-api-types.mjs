/**
 * Regenerates src/lib/http/schema.d.ts from the proxy's OpenAPI spec.
 *
 * Two hops, because the backend is the source of truth: the FastAPI app emits
 * the spec from its route decorators (app.openapi()), then openapi-typescript
 * turns that spec into TypeScript types. There is no live server in the loop —
 * the spec is read straight off the app object, so this runs in CI without a
 * database or proxy boot.
 *
 * The Python interpreter must have litellm installed and be at least the
 * version litellm requires. Override which one via LITELLM_PYTHON (CI passes
 * "uv run --no-sync python"); otherwise the repo's .venv is preferred, falling
 * back to python3.
 */
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const MIN_PYTHON = { major: 3, minor: 10 };

function venvInterpreter(root) {
  return [join(root, "bin", "python"), join(root, "Scripts", "python.exe")];
}

export function resolvePythonCommand(env, repoRoot, exists = existsSync) {
  if (env.LITELLM_PYTHON) return env.LITELLM_PYTHON.split(" ").filter(Boolean);
  const candidates = [
    ...(env.VIRTUAL_ENV ? venvInterpreter(env.VIRTUAL_ENV) : []),
    ...venvInterpreter(join(repoRoot, ".venv")),
  ];
  const found = candidates.find((candidate) => exists(candidate));
  return found ? [found] : ["python3"];
}

export function parsePythonVersion(stdout) {
  const match = stdout.match(/(\d+)\.(\d+)/);
  return match ? { major: Number(match[1]), minor: Number(match[2]) } : null;
}

export function isSupportedPython(version) {
  if (!version) return false;
  return version.major > MIN_PYTHON.major || (version.major === MIN_PYTHON.major && version.minor >= MIN_PYTHON.minor);
}

export function unsupportedPythonMessage(pythonCommand, version) {
  const interpreter = pythonCommand.join(" ");
  const detected = version ? `${version.major}.${version.minor}` : "unknown";
  return (
    `litellm requires Python >=${MIN_PYTHON.major}.${MIN_PYTHON.minor}, ` +
    `but \`${interpreter}\` reports ${detected}. ` +
    "Point gen:api at a supported interpreter (e.g. `uv venv && uv sync`, or set " +
    "LITELLM_PYTHON to one with litellm installed), then re-run `npm run gen:api`."
  );
}

export function probePythonVersion(pythonCommand, cwd, exec = execFileSync) {
  const probe = "import sys; print('.'.join(map(str, sys.version_info[:2])))";
  const stdout = exec(pythonCommand[0], [...pythonCommand.slice(1), "-c", probe], {
    cwd,
    encoding: "utf8",
  });
  return parsePythonVersion(stdout);
}

function assertSupportedPython(pythonCommand, cwd) {
  const version = probePythonVersion(pythonCommand, cwd);
  if (!isSupportedPython(version)) {
    throw new Error(unsupportedPythonMessage(pythonCommand, version));
  }
}

function main() {
  const dashboardDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
  const repoRoot = resolve(dashboardDir, "..", "..");
  const outPath = join(dashboardDir, "src", "lib", "http", "schema.d.ts");
  const specDir = mkdtempSync(join(tmpdir(), "litellm-openapi-"));
  const specPath = join(specDir, "openapi.json");

  const python = resolvePythonCommand(process.env, repoRoot);
  // The dashboard calls internal UI routes that the public /openapi.json hides via
  // include_in_schema=False. Force them in so they get typed here; this mutates a
  // throwaway interpreter, so the spec the proxy actually serves is unchanged.
  const dumpSpec = [
    "import json, sys",
    "from litellm.proxy.proxy_server import app",
    "from fastapi.routing import APIRoute",
    "for route in app.routes:",
    "    if isinstance(route, APIRoute):",
    "        route.include_in_schema = True",
    "app.openapi_schema = None",
    "with open(sys.argv[1], 'w') as f: json.dump(app.openapi(), f, sort_keys=True)",
  ].join("\n");

  try {
    assertSupportedPython(python, repoRoot);

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
}

const isMain = process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) main();
