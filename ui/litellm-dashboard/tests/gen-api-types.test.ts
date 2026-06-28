import { describe, it, expect } from "vitest";
import {
  resolvePythonCommand,
  parsePythonVersion,
  isSupportedPython,
  unsupportedPythonMessage,
} from "../scripts/gen-api-types.mjs";

const repoRoot = "/repo";

describe("resolvePythonCommand", () => {
  it("honors LITELLM_PYTHON, splitting multi-word commands", () => {
    const cmd = resolvePythonCommand({ LITELLM_PYTHON: "uv run --no-sync python" }, repoRoot, () => false);
    expect(cmd).toEqual(["uv", "run", "--no-sync", "python"]);
  });

  it("prefers an activated $VIRTUAL_ENV over the repo's .venv", () => {
    const active = "/active/bin/python";
    const repoVenv = `${repoRoot}/.venv/bin/python`;
    const cmd = resolvePythonCommand({ VIRTUAL_ENV: "/active" }, repoRoot, (p) => p === active || p === repoVenv);
    expect(cmd).toEqual([active]);
  });

  it("prefers the repo's .venv interpreter over a bare python3", () => {
    const venv = `${repoRoot}/.venv/bin/python`;
    const cmd = resolvePythonCommand({}, repoRoot, (p) => p === venv);
    expect(cmd).toEqual([venv]);
  });

  it("finds the Windows .venv interpreter", () => {
    const venv = `${repoRoot}/.venv/Scripts/python.exe`;
    const cmd = resolvePythonCommand({}, repoRoot, (p) => p === venv);
    expect(cmd).toEqual([venv]);
  });

  it("falls back to python3 when no venv exists", () => {
    expect(resolvePythonCommand({}, repoRoot, () => false)).toEqual(["python3"]);
  });
});

describe("parsePythonVersion", () => {
  it("parses major.minor from the probe output", () => {
    expect(parsePythonVersion("3.9\n")).toEqual({ major: 3, minor: 9 });
    expect(parsePythonVersion("3.13\n")).toEqual({ major: 3, minor: 13 });
  });

  it("returns null on unparseable output", () => {
    expect(parsePythonVersion("not a version")).toBeNull();
  });
});

describe("isSupportedPython", () => {
  it("rejects the 3.9 interpreter that breaks on dataclass slots=True", () => {
    expect(isSupportedPython({ major: 3, minor: 9 })).toBe(false);
  });

  it("accepts 3.10 and newer", () => {
    expect(isSupportedPython({ major: 3, minor: 10 })).toBe(true);
    expect(isSupportedPython({ major: 3, minor: 13 })).toBe(true);
    expect(isSupportedPython({ major: 4, minor: 0 })).toBe(true);
  });

  it("rejects an unknown version", () => {
    expect(isSupportedPython(null)).toBe(false);
  });
});

describe("unsupportedPythonMessage", () => {
  it("names the interpreter, the detected version, and how to fix it", () => {
    const message = unsupportedPythonMessage(["python3"], { major: 3, minor: 9 });
    expect(message).toContain("python3");
    expect(message).toContain("3.9");
    expect(message).toContain("LITELLM_PYTHON");
    expect(message).toContain(">=3.10");
  });
});
