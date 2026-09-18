import { test, expect } from "@playwright/test";

import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page as DashboardPage } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { captureRequestBody, readBack } from "../../helpers/roundTrip";
import { masterKey, uniqueSuffix } from "../../helpers/traffic";

test.use({ storageState: ADMIN_STORAGE_PATH });

test.describe("Tag management", () => {
  test("creates, edits, reopens, and reads back a tag", async ({ page }) => {
    const tagName = `e2e-tag-${uniqueSuffix()}`;
    const description = "synthetic tag description";
    const updatedDescription = `${description} updated`;

    await navigateToPage(page, DashboardPage.TagManagement);
    await page.getByRole("button", { name: "+ Create New Tag" }).click();

    await expect(
      page.getByRole("dialog", { name: "Create New Tag" }),
    ).toBeVisible();
    await page.getByLabel("Tag Name").fill(tagName);
    await page.getByLabel("Description").fill(description);
    await page.getByRole("button", { name: "Create Tag" }).click();

    await expect
      .poll(async () => {
        const response = await readBack<
          Record<string, Record<string, unknown>>
        >(page, "/tag/list");
        return Object.values(response).some((tag) => tag.name === tagName);
      })
      .toBe(true);
    try {
      await expect(page.getByText(tagName, { exact: true })).toBeVisible();

      await page.getByText(tagName, { exact: true }).click();
      await expect(page.getByText("Tag Name:")).toBeVisible();
      await page.getByRole("button", { name: "Edit Tag" }).click();
      await page.getByLabel("Description").fill(updatedDescription);
      const updateBody = await captureRequestBody(
        page,
        { method: "POST", urlIncludes: "/tag/update" },
        () => page.getByRole("button", { name: "Save Changes" }).click(),
      );
      expect(updateBody).toMatchObject({
        name: tagName,
        description: updatedDescription,
      });

      await expect
        .poll(async () => {
          const infoResponse = await page.request.post("/tag/info", {
            headers: { Authorization: `Bearer ${masterKey()}` },
            data: { names: [tagName] },
          });
          expect(infoResponse.ok()).toBe(true);
          const info = (await infoResponse.json()) as Record<
            string,
            { description?: string }
          >;
          return info[tagName]?.description;
        })
        .toBe(updatedDescription);
    } finally {
      const deleteResponse = await page.request.post("/tag/delete", {
        headers: {
          Authorization: `Bearer ${masterKey()}`,
          "Content-Type": "application/json",
        },
        data: { name: tagName },
      });
      expect(deleteResponse.ok()).toBe(true);
    }
  });
});
