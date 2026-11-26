import { test, expect, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { readBack } from "../../helpers/roundTrip";
import { masterKey } from "../../helpers/traffic";

const ORG_ID = "e2e-org-main";

const auth = () => ({ Authorization: `Bearer ${masterKey()}` });

async function orgRoleOf(page: PlaywrightPage, userId: string): Promise<string | undefined> {
  const info = await readBack<{ members: { user_id: string; user_role: string | null }[] }>(
    page,
    `/organization/info?organization_id=${ORG_ID}`,
  );
  return info.members.find((member) => member.user_id === userId)?.user_role ?? undefined;
}

test.describe("Proxy Admin - Organization members", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Adding a user from an organization's Members tab makes them an internal user of it", async ({
    page,
    request,
  }) => {
    const stamp = `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
    const userId = `e2e-org-member-${stamp}`;
    const email = `${userId}@test.local`;
    const created = await request.post("/user/new", {
      headers: auth(),
      data: { user_id: userId, user_email: email, user_role: "internal_user", auto_create_key: false },
    });
    expect(created.ok(), `POST /user/new (${created.status()}): ${await created.text()}`).toBe(true);
    try {
      await navigateToPage(page, Page.Organizations);
      await page.locator("td").filter({ hasText: ORG_ID }).first().click();
      await page.getByRole("tab", { name: "Members", exact: true }).click();
      await page.getByRole("button", { name: "Add Member" }).click();
      const modal = page.getByRole("dialog");
      await expect(modal).toBeVisible();

      const emailSearch = modal.getByPlaceholder("Search by email");
      await emailSearch.click();
      await emailSearch.pressSequentially(userId);
      await page.getByRole("option", { name: email }).click();
      await modal.getByRole("button", { name: "Add Member" }).click();
      await expect(modal).toBeHidden();

      await expect
        .poll(async () => orgRoleOf(page, userId), {
          message: "added user never appeared in the organization's members",
          timeout: 15_000,
        })
        .toBe("internal_user");
    } finally {
      await request.delete("/organization/member_delete", {
        headers: auth(),
        data: { organization_id: ORG_ID, user_id: userId },
      });
      await request.post("/user/delete", { headers: auth(), data: { user_ids: [userId] } });
    }
  });
});
