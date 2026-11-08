import { expect, test } from "@playwright/test";
import { execFile, spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { promisify } from "node:util";

const LITE_CLI = process.env.E2E_LITE_CLI ?? "lite";
const SKIP_TEAM_SELECTION = "skip\n";
const execFileAsync = promisify(execFile);

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`${name} must be set for the OIDC suite`);
  return value;
}

test("CLI SSO login stores a session that lists models and completes a chat request", async ({ browser, baseURL }) => {
  test.setTimeout(180_000);
  const issuer = requiredEnv("JWT_ISSUER");
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "lite-cli-login-"));
  const browserUrlFile = path.join(home, "browser-url");
  const browserCommand = path.join(home, "browser.sh");
  fs.writeFileSync(browserCommand, `#!/bin/sh\nprintf '%s' "$1" > '${browserUrlFile}'\n`, { mode: 0o700 });
  const env = {
    ...process.env,
    HOME: home,
    LITELLM_CLI_DISABLE_KEYRING: "1",
    BROWSER: browserCommand,
    PYTHONUNBUFFERED: "1",
    FORCE_COLOR: undefined,
    NO_COLOR: "1",
    LITELLM_PROXY_URL: baseURL,
    LITELLM_PROXY_API_KEY: undefined,
  };
  const login = spawn(LITE_CLI, ["login"], { env });
  let loginOutput = "";
  login.stdout.on("data", (chunk: Buffer) => (loginOutput += chunk.toString()));
  login.stderr.on("data", (chunk: Buffer) => (loginOutput += chunk.toString()));
  const loginExit = new Promise<number | null>((resolve) => login.on("close", resolve));
  login.stdin.end(SKIP_TEAM_SELECTION);
  try {
    await expect.poll(() => fs.existsSync(browserUrlFile), { timeout: 30_000 }).toBe(true);
    await expect.poll(() => loginOutput).toMatch(/Verification code: \S+/);
    const userCode = /Verification code: (\S+)/.exec(loginOutput)?.[1] ?? "";

    const context = await browser.newContext({ storageState: { cookies: [], origins: [] } });
    try {
      const page = await context.newPage();
      await page.goto(fs.readFileSync(browserUrlFile, "utf8"));
      await expect(page).toHaveURL((url) => url.href.startsWith(`${issuer}/`));
      await page.getByLabel("Username or email").fill(requiredEnv("E2E_OIDC_USERNAME"));
      await page.getByLabel("Password", { exact: true }).fill(requiredEnv("E2E_OIDC_PASSWORD"));
      await page.getByRole("button", { name: "Sign In", exact: true }).click();
      await page.getByLabel("Verification code").fill(userCode);
      await page.getByRole("button", { name: "Continue", exact: true }).click();
      await expect(page.getByRole("heading", { name: "Authentication Successful!" })).toBeVisible();
    } finally {
      await context.close();
    }

    expect(await loginExit, loginOutput).toBe(0);
    expect(loginOutput).toContain("Login successful!");
    const stored: { key?: unknown } = JSON.parse(fs.readFileSync(path.join(home, ".litellm", "token.json"), "utf8"));
    expect(typeof stored.key).toBe("string");
    expect(stored.key, "CLI login issues a session token, not a virtual key").not.toMatch(/^sk-/);

    const { stdout: modelsJson } = await execFileAsync(LITE_CLI, ["models", "list", "--format", "json"], { env });
    const models: { id: string }[] = JSON.parse(modelsJson);
    expect(models.length, "the stack serves at least one model").toBeGreaterThan(0);

    const chatRequest = JSON.stringify({
      model: models[0].id,
      messages: [{ role: "user", content: "Reply with the single word: ok" }],
    });
    const { stdout: completionJson } = await execFileAsync(
      LITE_CLI,
      ["http", "request", "POST", "/chat/completions", "-j", chatRequest],
      { env },
    );
    const completion: { choices: { message: { content: string | null } }[] } = JSON.parse(completionJson);
    expect(completion.choices[0]?.message.content).toBeTruthy();
  } finally {
    login.kill();
    fs.rmSync(home, { recursive: true, force: true });
  }
});
