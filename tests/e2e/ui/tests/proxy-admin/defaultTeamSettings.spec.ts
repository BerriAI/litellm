import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { writeProxySettings } from "../../helpers/proxySettings";
import { captureRequestBody, readBack } from "../../helpers/roundTrip";

const SETTINGS = "default_team_settings";

type DefaultTeamSettings = Record<string, unknown>;

const storedSettings = async (page: Parameters<typeof readBack>[0]): Promise<DefaultTeamSettings> =>
  (await readBack<{ values: DefaultTeamSettings }>(page, `/get/${SETTINGS}`)).values;

test.describe("Proxy Admin - Default team settings", () => {
  test.describe.configure({ mode: "serial" });
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Saving a new max budget persists it and keeps every other default as stored", async ({ page, request }) => {
    const before = await storedSettings(page);
    const maxBudget = 4242 + Math.floor(Math.random() * 1000);
    try {
      await navigateToPage(page, Page.Teams);
      await page.getByRole("tab", { name: "Default Team Settings" }).click();
      await page.getByRole("button", { name: "Edit Settings" }).click();
      await page.getByLabel("Max Budget").fill(String(maxBudget));

      const body = await captureRequestBody<DefaultTeamSettings>(
        page,
        { method: "PATCH", urlIncludes: `/update/${SETTINGS}` },
        () => page.getByRole("button", { name: "Save Changes" }).click(),
      );
      expect(body, "the save sends the stored defaults with only the edited budget changed").toEqual({
        ...before,
        max_budget: maxBudget,
      });
      await expect(page.getByRole("button", { name: "Edit Settings" })).toBeVisible();

      await expect
        .poll(async () => (await storedSettings(page)).max_budget, {
          message: "saved max budget never reached the proxy",
          timeout: 15_000,
        })
        .toBe(maxBudget);
    } finally {
      await writeProxySettings(request, SETTINGS, before);
    }
  });
});
