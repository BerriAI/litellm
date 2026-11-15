import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { buildSkillInstallCommand, findPublicSkill } from "./skillInstall";

const options = {
  gatewayUrl: "https://gateway.example.test/proxy",
  archiveUrl: "https://gateway.example.test/proxy/skill-hub/skills/review/archive",
  name: "review",
  agent: "codex" as const,
  global: false,
  authHeaderName: "X-LiteLLM-Key",
};

describe("skill install commands", () => {
  it("quotes the public index name without executing shell substitutions", () => {
    const name = "review' $(printf injected) `printf injected`";
    const command = buildSkillInstallCommand({ ...options, global: true, publicSkill: { name } });
    const result = spawnSync("bash", ["-c", `npx() { printf '%s\\n' "$DISABLE_TELEMETRY" "$@"; }; ${command}`], {
      encoding: "utf8",
    });
    expect(result.status).toBe(0);
    expect(result.stdout.trim().split("\n")).toEqual([
      "1",
      "skills@1.7.6",
      "add",
      options.gatewayUrl,
      "--skill",
      name,
      "-a",
      "codex",
      "-g",
    ]);
  });

  it.each(["success", "fail"])("executes a private install with safe credentials and cleanup on %s", (outcome) => {
    const command = buildSkillInstallCommand({ ...options, agent: "claude-code", global: true })!;
    const result = spawnSync(
      "python3",
      [fileURLToPath(new URL("./skillInstall.fixture.py", import.meta.url)), outcome],
      {
        input: command,
        encoding: "utf8",
        timeout: 10_000,
      },
    );
    expect(result.status, result.stderr).toBe(0);
    const expected = {
      status: outcome === "success" ? 0 : 22,
      header: "X-LiteLLM-Key: Bearer private-fixture-key\n",
      secretLeaked: false,
      temporaryRemoved: true,
      retained: outcome === "success" ? "fixture skill" : null,
      telemetry: outcome === "success" ? "1" : null,
    };
    const receipt = JSON.parse(result.stdout) as typeof expected & { installArgs: string[] | null };
    expect(receipt).toMatchObject(expected);
    expect(receipt.installArgs).toEqual(
      outcome === "success" ? expect.arrayContaining(["--skill", "review", "-a", "claude-code", "-g"]) : null,
    );
  });

  it.each([
    "localhost",
    "skills-install.localhost",
    "127.0.0.2",
    "[::1]",
    "gateway.example.test",
    "127.attacker.test",
    "localhost.attacker.test",
  ])("requires HTTPS for authenticated installs outside loopback at %s", (host) => {
    const local = ["localhost", "skills-install.localhost", "127.0.0.2", "[::1]"].includes(host);
    const httpOptions = {
      ...options,
      gatewayUrl: `http://${host}`,
      archiveUrl: `http://${host}/skill-hub/skills/review/archive`,
    };
    expect(buildSkillInstallCommand(httpOptions) === null).toBe(!local);
    expect(buildSkillInstallCommand({ ...httpOptions, publicSkill: { name: "review" } })).not.toBeNull();
  });

  it.each([
    { gatewayUrl: "https://user:password@gateway.example.test/proxy" },
    { archiveUrl: "https://other.example.test/skill.zip" },
    { archiveUrl: "https://gateway.example.test/proxy/skill-hub/submissions/pending/archive" },
    { authHeaderName: "Authorization\r\nX-Other: value" },
    { name: "review\nLITELLM_SKILL\nexit" },
    { gatewayUrl: "file:///private/skill" },
  ])("rejects credentials and values outside the managed archive contract", (invalid) => {
    expect(buildSkillInstallCommand({ ...options, ...invalid })).toBeNull();
  });

  it("uses archive identity instead of a colliding public skill name", () => {
    const entry = { name: "review-2", type: "archive" as const, description: "Review", digest: "sha256:abc" };
    const wrong = { ...entry, name: "review", url: "https://other.example.test/skill.zip" };
    const right = { ...entry, url: `${options.archiveUrl}?digest=sha256:abc` };
    expect(findPublicSkill({ $schema: "", skills: [wrong, right] }, options.archiveUrl)).toMatchObject({
      name: right.name,
      url: right.url,
    });
    expect(findPublicSkill({ $schema: "", skills: [wrong] }, options.archiveUrl)).toBeUndefined();
    expect(
      findPublicSkill({ skills: [{ ...right, url: `${right.url}&token=credential` }] }, options.archiveUrl),
    ).toBeUndefined();
    expect(findPublicSkill({ skills: null }, options.archiveUrl)).toBeUndefined();
  });
});
