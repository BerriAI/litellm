import { createRequire } from "node:module";
import { readFileSync, writeFileSync, mkdirSync, rmSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

const root = dirname(fileURLToPath(import.meta.url));
const repo = resolve(process.env.LITELLM_REPO || "../litellm");
const ui = resolve(repo, "ui/litellm-dashboard");
const require = createRequire(resolve(ui, "package.json"));
const { build } = require("esbuild");
const { compile } = require("@tailwindcss/node");
const { Scanner } = require("@tailwindcss/oxide");
const scratch = resolve(root, "scratch");
mkdirSync(scratch, { recursive: true });
const result = await build({
  entryPoints: [resolve(root, "report-shell.tsx")],
  bundle: true,
  platform: "node",
  format: "cjs",
  packages: "external",
  jsx: "automatic",
  alias: { "@": resolve(ui, "src") },
  outfile: resolve(scratch, "report-shell.cjs"),
  metafile: true,
});
const inputs = Object.keys(result.metafile.inputs).map((path) => resolve(path));
const shared = [
  ...inputs.filter((path) => path.startsWith(`${ui}/src/`)),
  resolve(ui, "src/app/globals.css"),
];
for (const path of shared) {
  const relative = path.slice(repo.length + 1);
  const expected = execFileSync("git", ["show", `origin/main:${relative}`], {
    cwd: repo,
  });
  if (!expected.equals(readFileSync(path)))
    throw new Error(`${relative} differs from origin/main`);
}
execFileSync(process.execPath, [resolve(scratch, "report-shell.cjs")], {
  env: {
    ...process.env,
    REPORT_ROOT: root,
    LITELLM_REPO: repo,
    NODE_PATH: resolve(ui, "node_modules"),
  },
  stdio: "inherit",
});
const compiler = await compile(
  readFileSync(resolve(ui, "src/app/globals.css"), "utf8"),
  {
    base: resolve(ui, "src/app"),
    onDependency() {},
  },
);
const scanner = new Scanner({});
const candidates = scanner.scanFiles(
  inputs.map((path) => ({
    content: readFileSync(path, "utf8"),
    extension: "tsx",
  })),
);
writeFileSync(
  resolve(root, "artifacts/dashboard.css"),
  compiler.build(candidates),
);
writeFileSync(
  resolve(root, "artifacts/component-provenance.json"),
  JSON.stringify(
    {
      main_sha: execFileSync("git", ["rev-parse", "origin/main"], {
        cwd: repo,
        encoding: "utf8",
      }).trim(),
      components: shared.map((path) => path.slice(repo.length + 1)),
      verification:
        "Imported directly from dashboard source. Every imported source and globals.css byte-matches origin/main.",
      rendering:
        "React static rendering with the report's existing local interactions. No authenticated gateway state is bundled.",
    },
    null,
    2,
  ),
);
rmSync(resolve(scratch, "report-shell.cjs"));
console.log(
  "Built report shell from verified main components and compiled dashboard globals.css",
);
