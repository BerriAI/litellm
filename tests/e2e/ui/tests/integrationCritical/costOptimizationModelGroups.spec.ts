import { test, expect } from "@playwright/test";
import { randomUUID } from "node:crypto";
import { execFileSync } from "node:child_process";
import * as path from "node:path";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";

test("cache leakage by model merges a deployment's resolved and requested model names into its model group", async ({
  page,
}) => {
  const master = process.env.LITELLM_MASTER_KEY ?? "sk-integration-master";
  const marker = `integration-browser-${randomUUID()}`;
  const group = `${marker}-public`;
  const deployment = `${marker}-backend`;
  const apiKey = marker;
  const support = (...args: string[]) =>
    execFileSync(
      process.env.INTEGRATION_PYTHON ?? "python",
      [
        path.resolve(
          __dirname,
          "../../../../integration/_support/daily_spend_rows.py",
        ),
        ...args,
      ],
      { encoding: "utf8", timeout: 10_000, killSignal: "SIGKILL" },
    );
  try {
    support("seed", apiKey, group, deployment);
    await page.goto("/ui/login");
    await page.getByPlaceholder("Enter your username").fill("admin");
    await page.getByPlaceholder("Enter your password").fill(master);
    await page.getByRole("button", { name: "Login", exact: true }).click();
    await expect(page).toHaveURL(
      (url) =>
        url.pathname.startsWith("/ui") && !url.pathname.includes("login"),
    );
    await navigateToPage(page, Page.CostOptimization);
    await page.getByRole("tab", { name: "Prompt Caching" }).click();
    await page.getByRole("tab", { name: "By model" }).click();
    const rows = page.getByRole("row").filter({ hasText: marker });
    await expect(rows).toHaveCount(1);
    await expect(rows.first()).toContainText(group);
    await expect(rows.first()).toContainText("275,000");
    await expect(
      page.getByRole("row").filter({ hasText: deployment }),
    ).toHaveCount(0);
  } finally {
    support("clear", apiKey);
  }
});
